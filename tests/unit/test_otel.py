"""OTel export through the real SDK: ids and times are the log's, dialects add the right names, failures surface."""

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.types import LLMResponse, Message, ToolCall, Usage
from aaa.observe import Observer
from aaa.observe.otel import (
    GenAIDialect,
    LangfuseDialect,
    LangSmithDialect,
    OpenInferenceDialect,
    OTelSink,
)
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def lookup(q: str) -> str:
    """look something up"""
    return f"answer for {q}"


def priced(content=None, calls=()):
    return LLMResponse(
        message=Message(role="assistant", content=content, tool_calls=list(calls)),
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost=0.001,
        model="gpt-test",
        finish_reason="tool_calls" if calls else "stop",
    )


def script(req):
    if req.messages[-1].role == "user":
        return priced(calls=[ToolCall(id="c", name="lookup", arguments={"q": "x"})])
    return priced(content="done")


async def run_with(dialect):
    exporter = InMemorySpanExporter()
    eng = Engine()
    LLMHandler(FakeLLMClient(script)).attach(eng)
    reg = ToolRegistry([lookup])
    ToolHandler(reg).attach(eng)
    algo = ToolLoop(reg)
    eng.register(algo)
    async with (
        eng,
        Observer(eng, OTelSink(exporter, dialect=dialect), capture_content=True, settle_s=0),
    ):
        await eng.tenant("t1").run(algo, {"task": "hi"}, meta={"user": "u1"})
    return exporter.get_finished_spans()


def by_name(spans, prefix):
    return next(s for s in spans if s.name.startswith(prefix))


async def test_spans_carry_the_logs_ids_and_form_one_tree():
    spans = await run_with(GenAIDialect())
    assert len({s.context.trace_id for s in spans}) == 1
    ids = {s.context.span_id for s in spans}
    roots = [s for s in spans if s.parent is None]
    assert len(roots) == 1 and roots[0].name.startswith("invoke_agent")
    assert all(s.parent is None or s.parent.span_id in ids for s in spans)
    assert all(s.end_time >= s.start_time for s in spans)


async def test_genai_attributes():
    spans = await run_with(GenAIDialect())
    llm = by_name(spans, "chat")
    a = llm.attributes
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.usage.input_tokens"] == 10 and a["gen_ai.usage.output_tokens"] == 5
    assert a["gen_ai.response.model"] == "gpt-test"
    tool_span = by_name(spans, "execute_tool")
    assert tool_span.attributes["gen_ai.tool.name"] == "lookup"
    assert by_name(spans, "invoke_agent").attributes["user.id"] == "u1"


async def test_every_gen_ai_name_we_emit_exists_in_the_installed_conventions():
    """Drift guard: the GenAI conventions are in Development and renamed between releases."""
    try:
        from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as g
    except ImportError:
        pytest.skip("semantic conventions package without the incubating gen_ai module")
    known = {v for k, v in vars(g).items() if k.startswith("GEN_AI_") and isinstance(v, str)}
    spans = await run_with(GenAIDialect())
    emitted = {k for s in spans for k in s.attributes if k.startswith("gen_ai.")}
    # names the installed release does not (yet) define; reviewed by hand, keep this list short
    allowed_extra = {
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "gen_ai.evaluation.score.value",
    }
    assert emitted - known - allowed_extra == set(), emitted - known


async def test_openinference_dialect():
    spans = await run_with(OpenInferenceDialect())
    llm = by_name(spans, "chat")
    assert llm.attributes["openinference.span.kind"] == "LLM"
    assert llm.attributes["llm.token_count.prompt"] == 10
    assert llm.attributes["llm.token_count.completion"] == 5
    assert by_name(spans, "execute_tool").attributes["openinference.span.kind"] == "TOOL"
    assert by_name(spans, "invoke_agent").attributes["openinference.span.kind"] == "AGENT"
    assert "input.value" in llm.attributes


async def test_langfuse_dialect():
    spans = await run_with(LangfuseDialect(environment="test", tags=["a"]))
    llm = by_name(spans, "chat")
    assert llm.attributes["langfuse.observation.type"] == "generation"
    assert llm.attributes["langfuse.observation.model.name"] == "gpt-test"
    assert '"input": 10' in llm.attributes["langfuse.observation.usage_details"]
    assert llm.attributes["langfuse.observation.cost_details"] == '{"total": 0.001}'
    root = by_name(spans, "invoke_agent")
    assert root.attributes["langfuse.trace.name"].startswith("invoke_agent")
    assert root.attributes["langfuse.user.id"] == "u1"
    assert by_name(spans, "execute_tool").attributes["langfuse.observation.type"] == "tool"


async def test_langsmith_dialect():
    spans = await run_with(LangSmithDialect(provider="openai"))
    llm = by_name(spans, "chat")
    assert llm.attributes["langsmith.span.kind"] == "llm"
    assert llm.attributes["gen_ai.system"] == "openai"
    assert by_name(spans, "execute_tool").attributes["langsmith.span.kind"] == "tool"


async def test_content_is_off_unless_asked():
    exporter = InMemorySpanExporter()
    eng = Engine()
    LLMHandler(FakeLLMClient(script)).attach(eng)
    reg = ToolRegistry([lookup])
    ToolHandler(reg).attach(eng)
    algo = ToolLoop(reg)
    eng.register(algo)
    async with eng, Observer(eng, OTelSink(exporter), settle_s=0):
        await eng.run(algo, {"task": "secret question"})
    for s in exporter.get_finished_spans():
        assert "secret question" not in str(dict(s.attributes))


async def test_exporter_failure_becomes_an_export_error():
    class Failing:
        def export(self, spans):
            return SpanExportResult.FAILURE

        def shutdown(self):
            pass

    eng = Engine()
    LLMHandler(FakeLLMClient(script)).attach(eng)
    reg = ToolRegistry([lookup])
    ToolHandler(reg).attach(eng)
    algo = ToolLoop(reg)
    eng.register(algo)
    obs = Observer(eng, OTelSink(Failing()), settle_s=0)
    async with eng, obs:
        await eng.run(algo, {"task": "hi"})
    assert obs.export_errors == 1 and obs.exported_spans == 0


def _headers(exporter):
    """Where the OTLP exporter keeps its headers differs between releases (`_session` before 1.4x, `_client` after)."""
    client = getattr(exporter, "_client", None)
    h = getattr(client, "_headers", None) or getattr(
        getattr(exporter, "_session", None), "headers", None
    )
    assert h is not None, "exporter internals changed: update this helper"
    return {k.lower(): v for k, v in dict(h).items()}


def test_presets_point_at_the_documented_endpoints_with_auth():
    from aaa.observe.otel import LangfuseSink, LangSmithSink, PhoenixSink

    lf = LangfuseSink("pk", "sk", host="https://lf.example/")
    assert lf.exporter._endpoint == "https://lf.example/api/public/otel/v1/traces"
    assert _headers(lf.exporter)["x-langfuse-ingestion-version"] == "4"
    assert _headers(lf.exporter)["authorization"].startswith("Basic ")
    ls = LangSmithSink("key", project="p")
    assert ls.exporter._endpoint == "https://api.smith.langchain.com/otel/v1/traces"
    assert (
        _headers(ls.exporter)["x-api-key"] == "key"
        and _headers(ls.exporter)["langsmith-project"] == "p"
    )
    px = PhoenixSink(project="demo")
    assert px.exporter._endpoint == "http://localhost:6006/v1/traces"
    assert px.resource.attributes["openinference.project.name"] == "demo"
