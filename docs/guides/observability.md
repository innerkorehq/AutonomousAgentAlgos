# Observability

AAA already records everything a run does as events. The **observer** turns that log into a *trace* (spans for the run, each model call,
each tool call, memory and context steps, child runs, session turns) and hands it to one or more **sinks**. Nothing is instrumented in your
code, so it covers runs from any process, and it can be re-exported later for history (`Observer.backfill`).

```python
--8<-- "examples/observability.py"
```

```text
invoke_agent tool_loop [1ms]
  chat fake [0ms, 0+0 tok]
  execute_tool lookup [0ms]
  chat fake [0ms, 0+0 tok]
spans exported: 4 in 1 batch
kinds: ['agent', 'llm', 'tool', 'llm']
```

## Sinks

| Sink | Install | Sends to |
|---|---|---|
| `InMemorySink`, `ConsoleSink`, `CallableSink` | none | tests, a terminal, your own function |
| `OTelSink(exporter)` | `pip install "autonomous-agent-architecture[otel]"` | any OpenTelemetry `SpanExporter` (OTLP/HTTP, console, vendor exporters) |
| `LangfuseSink(public_key, secret_key, host=...)` | `[langfuse]` | Langfuse, over its OTLP endpoint |
| `LangSmithSink(api_key, project=...)` | `[langsmith]` | LangSmith, over its OTLP endpoint |
| `PhoenixSink(endpoint, project=...)` | `[phoenix]` | Arize Phoenix (OpenInference attributes) |

```python
from aaa.observe import Observer, LangfuseSink

async with Observer(engine, LangfuseSink("pk-lf-...", "sk-lf-...", environment="prod")):
    await engine.run(algo, {"task": "..."})
```

The extras only install the OpenTelemetry SDK and OTLP/HTTP exporter: the presets are an endpoint, headers and an attribute dialect.
Use `OTelSink(OTLPSpanExporter())` with the standard `OTEL_EXPORTER_OTLP_*` environment variables for any other collector (Jaeger,
Honeycomb, Datadog, MLflow, ...).

## What a trace contains

- One trace per root run (the trace id is derived from the root run id). Child runs of a `Supervisor` hang under the tool call that delegated
  to them; session turns hang under a `turn n` span.
- `invoke_agent <algo>` for each run; `chat <model>` for each model call (tokens, cached tokens, cost, finish reason, whether provider state
  and context compaction were used); `execute_tool <name>` for each tool call (attempts; approval, fencing and stale-lease events as span events);
  memory and summary steps; guardrail verdicts and evaluation scores when those features are used.
- Times are **event times**: a span covers queueing and any clock difference between processes, not just the work.

## Behaviours worth knowing

- **Whole traces, parents first.** LangSmith silently drops a span whose parent has not arrived, so the observer never sends a child first.
- **Exactly once per trace in a cluster.** Observers share a competing-consumer group (`group="aaa.observe"`); give each deployment its own group
  if two should each export.
- **Sessions never complete**, so `Observer(..., flush_interval_s=30)` exports closed spans incrementally and a provisional root span marked
  `aaa.partial`, replaced when the session closes.
- **It cannot fail a run.** The observer reads the log from its own bounded queue; a slow or failing sink is logged and counted
  (`observer.export_errors`, `observer.exported_spans`), and a full queue drops work rather than blocking.
- **Prompts and replies are not exported** unless you pass `capture_content=True` (clipped to `content_limit` characters). Redact before you
  turn that on; see [Tenancy, redaction, retention](privacy.md). Tenant and user ids are attached as attributes.
- **Idempotent.** Span ids are deterministic. After a restart a backend may see a span twice; all the backends above de-duplicate by id.
- `sample=0.1` keeps a deterministic 10 % of traces (by trace id, so every process picks the same ones); `tenant="acme"` exports one tenant only.

## Attribute names

The OpenTelemetry GenAI conventions are still marked *Development* and have been renamed between releases. `GenAIDialect` pins the names
of one release (`opentelemetry-semantic-conventions 0.66b1`) and a test checks every `gen_ai.*` name we emit against the installed package, so
the weekly job tells us when they move. Pass `dialect=` to change names: `OpenInferenceDialect` (Phoenix), `LangSmithDialect`, `LangfuseDialect`, or
your own object with `convert(span) -> Converted`.

## Verified, and not

Exported spans are checked with the real OpenTelemetry SDK (ids, parents, times, attributes, failure reporting). The Langfuse, LangSmith and
Phoenix presets are checked for endpoint and header shape and for the attribute names their documentation lists; they have **not** been run
against live accounts.
