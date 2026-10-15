# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""OpenTelemetry export, plus dialects for the backends that read OTLP.

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    async with Observer(engine, OTelSink(OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces"))):
        ...

    Observer(engine, LangfuseSink(public_key="pk-lf-...", secret_key="sk-lf-..."))     # Langfuse over OTLP
    Observer(engine, LangSmithSink(api_key="...", project="my-agents"))               # LangSmith over OTLP
    Observer(engine, PhoenixSink(endpoint="http://localhost:6006/v1/traces"))         # Arize Phoenix (OpenInference)

How it works: `Span`s (derived from the event log) are turned into SDK `ReadableSpan` objects with their EXPLICIT trace ids,
span ids and start/end times, and handed straight to an OTLP exporter. No tracer, no clock, no id generator is involved,
so what is exported is exactly what the log says happened, whenever the export runs.

A **dialect** decides the attribute names. The OTel GenAI semantic conventions are still in "Development" (and have
moved repositories), so `GenAIDialect` pins the names of a stated version and a test checks every `gen_ai.*` name it
emits against the installed `opentelemetry-semantic-conventions`. Backends differ (Langfuse, LangSmith and Phoenix each read
extra attributes), hence one dialect per backend, all built on the same neutral span attributes (`aaa.trace.A`)."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Protocol

from aaa.observe.base import ExportError
from aaa.trace import A, Span

SEMCONV_VERSION = "opentelemetry-semantic-conventions 0.66b1 (gen_ai.* in Development)"


@dataclass
class Converted:
    name: str
    kind: str  # "client" | "internal"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[tuple[str, datetime, dict[str, Any]]] = field(default_factory=list)


class Dialect(Protocol):
    name: str

    def convert(self, span: Span) -> Converted: ...


def _json(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _common(span: Span) -> dict[str, Any]:
    a: dict[str, Any] = {"aaa.run.id": span.run_id, "aaa.span.kind": span.kind}
    if span.tenant:
        a["aaa.tenant"] = span.tenant
    if span.user:
        a["user.id"] = span.user
    if (sid := span.attributes.get(A.SESSION_ID)) is not None:
        a["session.id"] = sid
    if span.attributes.get(A.PARTIAL):
        a["aaa.partial"] = True
    if span.status == "error" and (et := span.attributes.get(A.ERROR_TYPE)):
        a["error.type"] = str(et)
    return a


class GenAIDialect:
    """OpenTelemetry GenAI semantic conventions (names as in `SEMCONV_VERSION`). The default."""

    name = "gen_ai"

    def convert(self, span: Span) -> Converted:
        at = span.attributes
        a = _common(span)
        op = at.get(A.OPERATION)
        if op:
            a["gen_ai.operation.name"] = op
        if span.kind == "agent":
            a["gen_ai.agent.name"] = at.get(A.AGENT_NAME)
            if (sid := at.get(A.SESSION_ID)) is not None:
                a["gen_ai.conversation.id"] = sid
            if (o := at.get(A.OUTCOME)) is not None:
                a["aaa.outcome"] = o
        elif span.kind == "llm":
            if (m := at.get(A.MODEL)) is not None:
                a["gen_ai.request.model"] = a["gen_ai.response.model"] = m
            for key, name in (
                (A.INPUT_TOKENS, "gen_ai.usage.input_tokens"),
                (A.OUTPUT_TOKENS, "gen_ai.usage.output_tokens"),
                (A.CACHED_TOKENS, "gen_ai.usage.cache_read.input_tokens"),
            ):
                if at.get(key) is not None:
                    a[name] = at[key]
            if at.get(A.FINISH_REASON):
                a["gen_ai.response.finish_reasons"] = [at[A.FINISH_REASON]]
            if span.input is not None:
                a["gen_ai.input.messages"] = _json(span.input)
            if span.output is not None:
                a["gen_ai.output.messages"] = _json(span.output)
        elif span.kind == "tool":
            a["gen_ai.tool.name"] = at.get(A.TOOL_NAME)
            a["gen_ai.tool.call.id"] = at.get(A.TOOL_CALL_ID)
            a["gen_ai.tool.type"] = "function"
            a["aaa.tool.attempts"] = at.get(A.TOOL_ATTEMPTS)
            if span.input is not None:
                a["gen_ai.tool.call.arguments"] = _json(span.input)
            if span.output is not None:
                a["gen_ai.tool.call.result"] = _json(span.output)
        elif span.kind == "evaluator":
            a["gen_ai.evaluation.name"] = at.get(A.SCORE_NAME)
            v = at.get(A.SCORE_VALUE)
            if isinstance(v, int | float):
                a["gen_ai.evaluation.score.value"] = v
        elif span.kind == "session":
            a["aaa.session.turn"] = at.get(A.TURN)
        # everything AAA-specific that has no standard name
        for key, name in (
            (A.COST, "aaa.usage.cost"),
            (A.LLM_STATE, "aaa.llm.state"),
            (A.CONTEXT_BEFORE, "aaa.context.tokens_before"),
            (A.CONTEXT_AFTER, "aaa.context.tokens_after"),
            (A.CONTEXT_APPLIED, "aaa.context.applied"),
            (A.MEMORY_OP, "aaa.memory.op"),
            (A.MEMORY_ITEMS, "aaa.memory.items"),
            (A.GUARD_NAME, "aaa.guard.name"),
            (A.GUARD_STAGE, "aaa.guard.stage"),
            (A.GUARD_ACTION, "aaa.guard.action"),
        ):
            if at.get(key) is not None:
                a[name] = at[key]
        return Converted(
            span.name,
            "client" if span.kind == "llm" else "internal",
            {k: v for k, v in a.items() if v is not None},
            [(e.name, e.time, e.attributes) for e in span.events],
        )


_OI_KIND = {
    "agent": "AGENT", "llm": "LLM", "tool": "TOOL", "memory": "RETRIEVER", "retriever": "RETRIEVER",
    "guardrail": "GUARDRAIL", "evaluator": "EVALUATOR", "session": "CHAIN", "internal": "CHAIN",
}  # fmt: skip


class OpenInferenceDialect:
    """OpenInference conventions, which Arize Phoenix (and, secondarily, LangSmith and Langfuse) understand."""

    name = "openinference"

    def convert(self, span: Span) -> Converted:
        at = span.attributes
        a = _common(span)
        a["openinference.span.kind"] = _OI_KIND[span.kind]
        if span.kind == "llm":
            a["llm.model_name"] = at.get(A.MODEL)
            for key, name in (
                (A.INPUT_TOKENS, "llm.token_count.prompt"),
                (A.OUTPUT_TOKENS, "llm.token_count.completion"),
                (A.TOTAL_TOKENS, "llm.token_count.total"),
                (A.CACHED_TOKENS, "llm.token_count.prompt_details.cache_read"),
            ):
                if at.get(key) is not None:
                    a[name] = at[key]
        elif span.kind == "tool":
            a["tool.name"] = at.get(A.TOOL_NAME)
            a["tool.id"] = at.get(A.TOOL_CALL_ID)
        elif span.kind == "agent":
            a["graph.node.name"] = at.get(A.AGENT_NAME)
        if span.input is not None:
            a["input.value"] = _json(span.input)
            a["input.mime_type"] = (
                "text/plain" if isinstance(span.input, str) else "application/json"
            )
        if span.output is not None:
            a["output.value"] = _json(span.output)
            a["output.mime_type"] = (
                "text/plain" if isinstance(span.output, str) else "application/json"
            )
        if at.get(A.COST) is not None:
            a["aaa.usage.cost"] = at[A.COST]
        return Converted(
            span.name,
            "client" if span.kind == "llm" else "internal",
            {k: v for k, v in a.items() if v is not None},
            [(e.name, e.time, e.attributes) for e in span.events],
        )


class LangSmithDialect(GenAIDialect):
    """GenAI names plus what LangSmith reads: `langsmith.span.kind`, session and metadata, and OpenInference-style
    `input.value` / `output.value` for content (LangSmith maps those too)."""

    name = "langsmith"
    _KIND: ClassVar[dict[str, str]] = {
        "llm": "llm",
        "tool": "tool",
        "memory": "retriever",
        "retriever": "retriever",
    }

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider

    def convert(self, span: Span) -> Converted:
        c = super().convert(span)
        a = c.attributes
        a["langsmith.span.kind"] = self._KIND.get(span.kind, "chain")
        if span.kind == "llm":
            if self.provider:
                a["gen_ai.system"] = self.provider
            tot = span.attributes.get(A.TOTAL_TOKENS)
            if tot is not None:
                a["gen_ai.usage.total_tokens"] = tot
        if (sid := span.attributes.get(A.SESSION_ID)) is not None:
            a["langsmith.trace.session_id"] = sid
        if span.tenant:
            a["langsmith.metadata.tenant"] = span.tenant
        if span.user:
            a["langsmith.metadata.user"] = span.user
        if span.input is not None:
            a["input.value"] = _json(span.input)
        if span.output is not None:
            a["output.value"] = _json(span.output)
        return c


class LangfuseDialect(GenAIDialect):
    """GenAI names plus Langfuse's own attributes (observation type, usage/cost details, session, user, level, I/O).
    Names follow Langfuse's OpenTelemetry ingestion mapping. Session and user are set on EVERY span, because Langfuse
    filters on them per observation."""

    name = "langfuse"
    _TYPE: ClassVar[dict[str, str]] = {
        "agent": "agent", "llm": "generation", "tool": "tool", "memory": "retriever", "retriever": "retriever",
        "guardrail": "guardrail", "evaluator": "evaluator", "session": "chain", "internal": "span",
    }  # fmt: skip

    def __init__(self, environment: str | None = None, tags: Sequence[str] = ()) -> None:
        self.environment, self.tags = environment, list(tags)

    def convert(self, span: Span) -> Converted:
        c = super().convert(span)
        a, at = c.attributes, span.attributes
        a["langfuse.observation.type"] = self._TYPE[span.kind]
        if span.parent_id is None or span.kind == "agent" and span.parent_id is None:
            a["langfuse.trace.name"] = span.name
            if self.tags:
                a["langfuse.trace.tags"] = self.tags
        if (sid := at.get(A.SESSION_ID)) is not None:
            a["langfuse.session.id"] = sid
        if span.user:
            a["langfuse.user.id"] = span.user
        if self.environment:
            a["langfuse.environment"] = self.environment
        if span.kind == "llm":
            if at.get(A.MODEL):
                a["langfuse.observation.model.name"] = at[A.MODEL]
            usage = {
                k: at[src]
                for k, src in (
                    ("input", A.INPUT_TOKENS),
                    ("output", A.OUTPUT_TOKENS),
                    ("total", A.TOTAL_TOKENS),
                )
                if at.get(src) is not None
            }
            if usage:
                a["langfuse.observation.usage_details"] = json.dumps(usage)
        if at.get(A.COST) is not None:
            a["langfuse.observation.cost_details"] = json.dumps({"total": at[A.COST]})
        if span.status == "error":
            a["langfuse.observation.level"] = "ERROR"
            a["langfuse.observation.status_message"] = span.status_message or ""
        if span.input is not None:
            a["langfuse.observation.input"] = _json(span.input)
        if span.output is not None:
            a["langfuse.observation.output"] = _json(span.output)
        return c


DIALECTS: dict[str, type] = {
    "gen_ai": GenAIDialect,
    "openinference": OpenInferenceDialect,
    "langsmith": LangSmithDialect,
    "langfuse": LangfuseDialect,
}


def _ns(dt: datetime) -> int:
    return int(dt.timestamp()) * 1_000_000_000 + dt.microsecond * 1000


class OTelSink:
    """Export spans through any OpenTelemetry `SpanExporter` (OTLP/HTTP, console, in-memory, a vendor's)."""

    def __init__(
        self,
        exporter: Any = None,
        *,
        dialect: Dialect | None = None,
        service_name: str = "aaa",
        resource_attributes: dict[str, Any] | None = None,
        batch_size: int = 512,
    ) -> None:
        if exporter is None:  # the standard OTEL_EXPORTER_OTLP_* environment variables configure it
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter()
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.util.instrumentation import InstrumentationScope

        from aaa import __version__

        self.exporter, self.dialect, self.batch_size = (
            exporter,
            dialect or GenAIDialect(),
            batch_size,
        )
        self.resource = Resource.create(
            {"service.name": service_name, **(resource_attributes or {})}
        )
        self.scope = InstrumentationScope("aaa", __version__)

    def _readable(self, span: Span, trace_flags: Any = None) -> Any:
        from opentelemetry.sdk.trace import Event as SDKEvent
        from opentelemetry.sdk.trace import ReadableSpan
        from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags

        c = self.dialect.convert(span)
        tid = int(span.trace_id, 16)
        ctx = SpanContext(tid, int(span.span_id, 16), False, TraceFlags(TraceFlags.SAMPLED))
        parent = (
            SpanContext(tid, int(span.parent_id, 16), False, TraceFlags(TraceFlags.SAMPLED))
            if span.parent_id
            else None
        )
        status = {
            "ok": Status(StatusCode.OK),
            "error": Status(StatusCode.ERROR, span.status_message),
            "incomplete": Status(StatusCode.UNSET),
        }[span.status]
        end = span.end or span.start
        return ReadableSpan(
            name=c.name,
            context=ctx,
            parent=parent,
            resource=self.resource,
            attributes=c.attributes,
            events=[SDKEvent(n, attributes=attrs, timestamp=_ns(t)) for n, t, attrs in c.events],
            kind=SpanKind.CLIENT if c.kind == "client" else SpanKind.INTERNAL,
            status=status,
            start_time=_ns(span.start),
            end_time=_ns(end),
            instrumentation_scope=self.scope,
        )

    async def export(self, spans: Sequence[Span]) -> None:
        from opentelemetry.sdk.trace.export import SpanExportResult

        ready = [self._readable(s) for s in spans]
        for i in range(0, len(ready), self.batch_size):
            batch = ready[i : i + self.batch_size]
            result = await asyncio.to_thread(
                self.exporter.export, batch
            )  # exporters block on the network
            if result is SpanExportResult.FAILURE:
                raise ExportError(
                    f"{type(self.exporter).__name__} reported a failure exporting {len(batch)} span(s)"
                )

    async def aclose(self) -> None:
        shutdown = getattr(self.exporter, "shutdown", None)
        if shutdown is not None:
            await asyncio.to_thread(shutdown)


def _otlp(endpoint: str, headers: dict[str, str]) -> Any:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=endpoint, headers=headers)


class LangfuseSink(OTelSink):
    """Langfuse over OTLP/HTTP (`/api/public/otel`; Langfuse accepts HTTP only, not gRPC)."""

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
        *,
        environment: str | None = None,
        tags: Sequence[str] = (),
        **kw: Any,
    ) -> None:
        auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        exporter = _otlp(
            f"{host.rstrip('/')}/api/public/otel/v1/traces",
            {"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
        )
        super().__init__(exporter, dialect=LangfuseDialect(environment, tags), **kw)


class LangSmithSink(OTelSink):
    """LangSmith over OTLP/HTTP. Note LangSmith silently drops a span whose parent never arrives; the observer exports
    whole traces parents-first for this reason."""

    def __init__(
        self,
        api_key: str,
        project: str | None = None,
        endpoint: str = "https://api.smith.langchain.com/otel",
        *,
        provider: str | None = None,
        **kw: Any,
    ) -> None:
        headers = {"x-api-key": api_key, **({"Langsmith-Project": project} if project else {})}
        super().__init__(
            _otlp(f"{endpoint.rstrip('/')}/v1/traces", headers),
            dialect=LangSmithDialect(provider),
            **kw,
        )


class PhoenixSink(OTelSink):
    """Arize Phoenix over OTLP/HTTP, using OpenInference attributes."""

    def __init__(
        self,
        endpoint: str = "http://localhost:6006/v1/traces",
        project: str | None = None,
        api_key: str | None = None,
        **kw: Any,
    ) -> None:
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        res = {"openinference.project.name": project} if project else {}
        kw.setdefault("resource_attributes", {}).update(res)
        super().__init__(_otlp(endpoint, headers), dialect=OpenInferenceDialect(), **kw)
