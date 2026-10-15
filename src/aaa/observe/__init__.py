"""Observability: export traces of agent runs to any backend.

    async with Observer(engine, OTelSink(OTLPSpanExporter(...))):
        await engine.run(algo, {"task": "..."})

The trace is derived from the event log (`aaa.trace.spans_from_events`), so it covers runs from any process and can be
re-exported later (`Observer.backfill`). Sinks: `InMemorySink`, `ConsoleSink`, `CallableSink`, and (with the
OpenTelemetry packages installed) `OTelSink` plus ready-made presets for Langfuse, LangSmith and Arize Phoenix."""

from aaa.observe.base import ExportError, TraceSink
from aaa.observe.observer import Observer
from aaa.observe.sinks import CallableSink, ConsoleSink, InMemorySink


def __getattr__(name: str):  # the OpenTelemetry sinks import their packages lazily
    if name in {"OTelSink", "LangfuseSink", "LangSmithSink", "PhoenixSink"}:
        from aaa.observe import otel

        return getattr(otel, name)
    raise AttributeError(name)


__all__ = [
    "CallableSink",
    "ConsoleSink",
    "ExportError",
    "InMemorySink",
    "Observer",
    "TraceSink",
]  # + OTelSink, LangfuseSink, LangSmithSink, PhoenixSink (lazy)
