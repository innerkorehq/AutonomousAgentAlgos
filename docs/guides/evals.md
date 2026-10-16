# Evaluating agents

An eval runs a **dataset** of cases through your agent, **scores** each run with **metrics**, and gives you a **report** you can gate a
build on. Each case is a real run in the engine, so the score can see what the agent actually did: the answer, the tools it called, the
steps, tokens and cost it used, and (for delegating agents) what its child runs did.

```python
--8<-- "examples/evals.py"
```

```text
pass rate 67% over 3 runs
  contains: mean 0.67
  tool_called: mean 0.67
  within_budget.steps: mean 1.00
failed: ['vague']
gate: pass rate 66.7% < 90.0%
```

## Datasets

`Dataset.from_list`, `from_jsonl`, `from_csv`. A case is `{id, input, expected, tags, metadata}`; a string `input` becomes `{"task": ...}`.
`dataset.where(tag="math")` selects a slice. Case ids must be unique.

## Metrics

| Metric | Checks |
|---|---|
| `ExactMatch`, `Contains`, `Regex`, `NumericClose`, `JsonValid(model)` | the answer text |
| `OutcomeIs("success")` | how the run ended |
| `ToolCalled("a", forbidden=["b"])`, `ToolSequence("a", "b")` | the tool-use trajectory |
| `Budgeted(steps=, tokens=, cost=, latency_s=)` | the run stayed inside an allowance |
| `SemanticSimilarity(embedder)` | embedding similarity to `expected` |
| `LLMJudge(client, rubric)` | a model grades against a rubric |
| `CallableMetric(fn)` | anything: `fn(result) -> bool / float / (value, reason) / Score` |

Write your own by implementing `name` and `async score(result) -> Score | list[Score]`. Scores are numbers in 0..1 or pass/fail booleans;
`Score.cost` and `Score.usage` record what a judge spent. A metric that raises counts as a **failed** score (never a silent pass), and a case the
harness could not start is an `error`.

The **judge runs outside the evaluated run**: its calls are not part of that run's trace or budget, and its cost is reported separately
(`report.judge_cost`). Use a different, preferably stronger, model than the agent's, and look at a few judged cases by hand before trusting it.

## Running

`EvalSuite(dataset, algo, metrics, concurrency=4, repeats=1, budget=None, tenant=None)`. `repeats=5` runs each case five times, because
agents are not deterministic; the report keeps every run. `budget=Budget(max_cost=0.10)` bounds each case. Eval runs are stamped
`meta={"aaa_eval": True}`, so an online evaluator never scores them.

## Reports and CI

```python
report.save("evals/baseline.json")                      # once, from a run you accept
report.assert_ok(min_pass_rate=0.9, baseline="evals/baseline.json")   # in CI; raises AssertionError listing failing cases
open("eval.xml", "w").write(report.to_junit())          # for CI dashboards
print(report.to_markdown())
```

`assert_ok` fails if the pass rate is below the bar or any metric's mean fell by more than `tolerance` (default 0.02) against the baseline.

## Scoring live traffic

```python
from aaa.eval import OnlineEvaluator, LLMJudge

async with OnlineEvaluator(engine, [LLMJudge(client, "Is the reply polite and correct?")], sample=0.05, max_cost=5.0, sinks=[langfuse_sink]):
    ...
```

It scores a deterministic sample (by trace id, so every process picks the same runs) after the run finishes, in its own consumer group,
so a slow or failing judge never touches the run. `max_cost` caps judge spend in this process. Each score is recorded in the run's log as an
`aaa.eval.score` event (so it appears as an evaluator span in later trace exports) and passed to any sink that supports scores.
Note that the sample sees your production inputs and outputs: a hosted judge receives them.

## Where scores go

A sink that can take scores (`Scorable`) gets each `OnlineEvaluator` score for the run's trace *and span*:

| Sink | Delivered as | Verified |
|---|---|---|
| `LangfuseSink` | `POST /api/public/scores` (numeric, boolean or categorical; the id is derived from trace, span and name, so a repeat updates) | against a real Langfuse 2.95 server; Langfuse 4 not tried (one score object per call, retried as a list if the server asks) |
| `PhoenixSink` | span annotations (trace annotations when no span), `identifier` makes a repeat an update; retried briefly if the span has not arrived | against a real Phoenix server |
| `LangSmithSink` | feedback on the run LangSmith built from the span id (the 8 span-id bytes right-aligned in a UUID, as LangSmith documents for OTel ingestion) | **mock transport only**: no LangSmith account was used |
| any `OTelSink` | an `evaluate <name>` evaluator span under the run's span | against the SDK's in-memory exporter |

Delivery failures are logged and never touch the run.

## Cases from real traffic

```python
cases = await Dataset.from_traces(engine, tenant="acme", algo="react", limit=200)        # inputs scrubbed by default
golden = await Dataset.from_traces(engine, expected="output")                           # pins CURRENT behaviour
```

Child runs, eval runs, open runs, session runs and runs whose payloads were erased are skipped. Inputs go through `standard_redactor()` unless you pass your own `Redactor` or `scrub=False`.
`expected="output"` records what the run answered, which makes a regression test, not a correctness test: review it. The case id is the run id.

## Red-team checks

`aaa.eval.redteam` ships a small corpus (prompt injection, indirect injection, system-prompt leaks, jailbreaks, harmful requests, PII extraction, tool abuse; written for this library) and look-alike benign prompts.

```python
report = await evaluate_guard(my_guard)                      # any Guardrail, including the adapters
report.assert_ok(max_attack_success=0.2, max_false_positive=0.05)

suite = attack_suite(my_agent, canary="CANARY-7f3a91")      # plant the canary in the agent's prompt
await suite.run(engine)                                      # metrics: resisted, no_leak
```

"Attack success" means nothing *prevented* the attack (`allow` or only `flag`). `Resisted` is a refusal-phrase heuristic unless you pass `judge=LLMJudge(...)`. This is a regression smoke test
for your defences, **not a benchmark and not a security audit**: passing it says the obvious attacks are covered.

## Simulated users

```python
customer = SimulatedUser(client, persona="an impatient customer", goal="get a refund for order A1")
report = await SessionEvalRunner(manager, customer, [Contains("refund")], max_turns=6).run(engine, dataset)
```

A model plays the user; the agent side is a real `Session`, so context, memory and tools behave as in production. A conversation ends at `max_turns`, when the simulator says `[DONE]`, when its `max_cost` (counted only
if the client reports cost) is spent, or when a turn fails. A case's `input` may set its own `persona`, `goal` and `opening`. Metrics see the transcript as `result.text` and the pairs as `result.turns`; the agent's
spans, steps, tokens and cost are summed over the turns. The sessions are stamped `aaa_eval`, so `OnlineEvaluator` ignores them. A simulated user is another model: its behaviour varies, so repeat important cases.

```python
--8<-- "examples/eval_extras.py"
```

```text
from traces: {'task': 'Refund me, my email is [REDACTED:email]'}
red team: attack success 90%, false positives 0%
conversation: 2 turns; contains help = True
```

## Third-party eval libraries

Each library's metrics become AAA metrics, and **their judge calls go through your own `LLMClient`** (your gateway, keys and cost
reporting) instead of the library's OpenAI default. Install the library you use; AAA imports it only when you import its adapter.

| Library | Adapter | Install |
|---|---|---|
| autoevals (Braintrust) | `aaa.eval.autoevals`: `AutoevalsMetric(scorer)`, `openai_shim(client)` | `pip install "autonomous-agent-algos[autoevals]"` (adds `openai`, which autoevals imports but does not declare) |
| DeepEval | `aaa.eval.deepeval`: `DeepEvalMetric(metric)`, `DeepEvalJudge(client)` | `pip install deepeval` |
| Ragas | `aaa.eval.ragas`: `RagasMetric(metric)`, `RagasJudge(client)` | `pip install ragas "langchain-community<0.4" "langchain-core<1" "langchain-openai<1"` |
| Inspect AI | `aaa.eval.inspect`: `aaa_solver(setup)` | `pip install inspect-ai` |

```python
from autoevals import Factuality, Levenshtein
from aaa.eval.autoevals import AutoevalsMetric, openai_shim

metrics = [AutoevalsMetric(Levenshtein()), AutoevalsMetric(Factuality(client=openai_shim(llm), model="judge-model"))]
```

```python
from deepeval.metrics import AnswerRelevancyMetric
from aaa.eval.deepeval import DeepEvalMetric, DeepEvalJudge

metrics = [DeepEvalMetric(AnswerRelevancyMetric(model=DeepEvalJudge(llm, model="judge-model")))]
```

```python
from ragas.metrics.collections import Faithfulness
from aaa.eval.ragas import RagasMetric, RagasJudge

metrics = [RagasMetric(Faithfulness(llm=RagasJudge(llm, model="judge-model")))]   # needs case.metadata["retrieval_context"]
```

- **What a case supplies.** DeepEval gets an `LLMTestCase` with the input, answer, expected output, the tools the run called (read from the
  trajectory), token cost and latency, plus `metadata["retrieval_context"]`, `["context"]` and `["expected_tools"]` when present. Ragas gets whatever its
  `ascore` asks for by name: `user_input`, `response`, `reference`, `retrieved_contexts`; anything else, or a missing field, fails the case
  with a message instead of a guess. autoevals gets `output`, `expected` and `input`.
- **Judge cost.** The judge objects count their calls, tokens and cost (`judge.calls`, `judge.tokens`, `judge.cost`; `shim.meter` for autoevals),
  as running totals, not per case.
- **Inspect AI** keeps its own loop, scorers, sandboxes and logs: `aaa_solver(setup)` runs one AAA run per sample (a fresh engine from
  `setup`, an async context manager yielding `(engine, algo)`) and returns the answer as the model output, so Inspect's `includes()`,
  `match()` or model-graded scorers work on it. Inspect's own `model` is used only by model-graded scorers.
- **Verified.** Each adapter runs the real library's flow in CI against a fake judge model (DeepEval 4.2, Ragas 0.4.3, autoevals 0.4,
  Inspect AI 0.3), including autoevals' two OpenAI code paths (chat completions and the Responses API it picks for `gpt-5*` names). Not run:
  live judge models, DeepEval/Ragas agent and multi-turn metrics, Ragas metrics that need embeddings, Inspect sandboxes.
- **Dependency warning.** Ragas 0.4.3 does not import against the newest LangChain (`langchain_community.chat_models.vertexai`, then
  `ContextOverflowError`), hence the pins above. DeepEval, Ragas and Inspect AI are large and pin clashing versions of pydantic, protobuf
  and LangChain: put them in a separate environment if they fight with the rest of your stack.
