# Guardrails

A guardrail looks at text at a **stage** and returns an **action**. AAA checks four places, because that is where an agent gets hurt:

| Stage | What is checked | Why |
|---|---|---|
| `input` | user messages, before they reach the model | secrets and personal data going out; abusive or off-policy requests |
| `output` | the model's reply, before it is recorded or shown | leaked data, unsafe content |
| `tool_call` | the arguments the model wants to call a tool with | a tool that should not run, or not with those arguments |
| `tool_result` | what a tool hands back | **indirect prompt injection**: a web page, file or email that tells the model what to do |

```python
--8<-- "examples/guardrails.py"
```

```text
run 1: success
  the model saw the SSN: False
  the model saw the injection: False
  the model saw instead: [tool result withheld by policy: prompt_injection]
run 2: success - Error (Blocked): blocked by policy: tool_not_allowed
```

## Using it

```python
from aaa.guard import GuardPolicy, RegexGuard, LLMJudgeGuard, ToolAllowlistGuard, LLAMA_GUARD_CATEGORIES

policy = GuardPolicy([
    RegexGuard({"pii.ssn": r"\b\d{3}-\d{2}-\d{4}\b"}, action="redact"),
    LLMJudgeGuard(judge_client, LLAMA_GUARD_CATEGORIES, model="llama-guard-3", stages=["input", "tool_result"]),
    ToolAllowlistGuard({"search", "read_file"}, escalate={"send_email"}),
])
LLMHandler(client, guards=policy).attach(engine)     # input and output
ToolHandler(registry, guards=policy).attach(engine)  # tool_call and tool_result
```

Each guard says which stages it applies to; a policy can narrow that (`GuardSpec(guard, stages=...)`). Guards at one stage run **concurrently**
and the strongest action wins: `block` > `approve` > `redact` > `flag` > `allow`. Several redactors compose (the second sees the first's output).

## What each action does

| Action | `input` / `output` | `tool_call` | `tool_result` |
|---|---|---|---|
| `block` | the model call fails with `GuardrailBlocked`; the run ends as **`OutcomeKind.BLOCKED`** | a normal tool failure, `blocked by policy: <category>`; the model sees it and the run goes on | replaced by `[tool result withheld by policy: <category>]` |
| `redact` | the text is replaced before it is sent (input) or recorded (output) | treated as a block (the log holds what the model asked for; arguments are not rewritten) | replaced by the redacted text (a string) |
| `approve` | treated as a block (no human is in that loop) | the call waits for an `aaa.approval.granted` event, as for `requires_approval` tools | treated as a block |
| `flag` | recorded only | recorded only | recorded only |

The model is never told *why* beyond the category name, and the category goes to the model only at the tool stages.

## Verdicts are facts on the log

Every check appends an `aaa.guard.verdict` event: guard, version, stage, action, categories, scores, cost, and a **hash** of the content
(never the content). They show up as guardrail spans in traces and their cost counts in the run's spend.

- **A guard is not called twice for the same content.** A model-based classifier is billable and not always repeatable, so the stored verdict wins: the
  same user message across the many model calls of one run, and a redelivery after a crash, reuse the verdict on the log. Two exceptions: a `redact`
  verdict is recomputed (the log has no text to give back), and a verdict produced because the **guard itself failed** is recorded but never reused, so a
  transient outage does not become a permanent block.
- **Failure policy.** `on_error="closed"` (default) blocks when a guard crashes or times out (`timeout_s`, default 10): safe, but a guard outage stops traffic.
  `"open"` lets content through: available, but unprotected during the outage. Choose per guard.
- **Observe before you enforce.** `GuardSpec(guard, mode="observe")` plus `GuardWatcher(engine, policy).attach()` runs the guard out of line on real traffic
  and records verdicts without ever changing a run. Compare them with what you would have blocked, then switch to `enforce`.

## Built-in guards

| Guard | Does |
|---|---|
| `RegexGuard(patterns, action=)` | match patterns; block, flag or redact |
| `DenyListGuard(phrases)` | block / flag listed phrases |
| `MaxLengthGuard(n)` | block over-long content |
| `ToolAllowlistGuard(allowed, escalate=, check_args=)` | only these tools, optionally only with acceptable arguments; `escalate` needs human approval |
| `LLMJudgeGuard(client, policies)` | any model classifies against named policies; `LLAMA_GUARD_CATEGORIES` is a ready set, so a served Llama Guard / Prompt Guard class model works |
| `CallableGuard(fn)` | your function: returns a `GuardResult`, an action, or a bool |

## Adapters

| Service | Adapter | Install |
|---|---|---|
| OpenAI Moderation | `aaa.guard.openai_moderation.OpenAIModerationGuard(api_key, thresholds=)` | none (plain HTTP) |
| Microsoft Presidio | `aaa.guard.presidio.PresidioGuard(entities=, action=)` and `PresidioRedactor` (an event redactor for `Engine(redactor=)`) | `pip install presidio-analyzer presidio-anonymizer` and `python -m spacy download en_core_web_sm` |
| AWS Bedrock Guardrails | `aaa.guard.bedrock.BedrockGuard(guardrail_id, version)`: `ApplyGuardrail` checks text without calling any model | `pip install "autonomous-agent-algos[bedrock]"` |
| Azure Prompt Shields | `aaa.guard.azure.AzurePromptShieldGuard(endpoint, api_key)`: jailbreaks in user input, injected instructions in tool results | none (plain HTTP) |
| Lakera Guard | `aaa.guard.lakera.LakeraGuard(api_key, project_id=)` | none (plain HTTP) |
| Guardrails AI | `aaa.guard.guardrails_ai.GuardrailsAIGuard(guard)`: any `Guard` / `AsyncGuard` and its validators | `pip install guardrails-ai` |
| NVIDIA NeMo Guardrails | `aaa.guard.nemo.NeMoGuard(rails)`: your configured input and output rails | `pip install nemoguardrails` |

Hosted guards (OpenAI, Bedrock, Azure, Lakera) send the text to the provider: only use them where that is acceptable. Presidio, Guardrails AI and NeMo run in your process.

- **Bedrock** treats `GUARDRAIL_INTERVENED` as a block, unless every policy that fired only masked PII and Bedrock returned replacement text, which becomes a redaction. Stage to `source`: input and tool results are `INPUT`; model output and tool-call arguments are `OUTPUT`.
- **Azure Prompt Shields** sends user input as `userPrompt` and tool results as `documents`, which is the "document attack" case it exists for.
- **Lakera** maps stages to message roles (`user`, `assistant`, `tool`; tool messages are screened as untrusted). In Lakera's *Detect* mode `flagged` is always false, so the guard never blocks: use the project's enforce mode.
- **Guardrails AI**: a failed validation blocks, a validator with `on_fail="fix"` becomes a redaction, `on_fail="exception"` is caught. Give validators `on_fail="noop"` or `"fix"` and let AAA decide. The library sends telemetry over OTLP unless `OTEL_SDK_DISABLED=true`.
- **NeMo** exposes only input and output rails (via `check_async`); input rails also screen tool results. Output rails that read the user's message see a placeholder, because a guard only receives the text being checked.
- **Anything else**: `CallableGuard` wraps any function.

**What was verified.** Presidio, Guardrails AI (0.11, custom validator, async and sync) and NeMo (0.24, a configuration whose rail is a Python action) run their real flows in CI; Bedrock's request and response are validated by botocore's own service model; the OpenAI Moderation reply is validated against the OpenAI SDK's types. Azure Prompt Shields and Lakera are built from the vendors' published request/response descriptions and tested with mock transports (Lakera's per-detector field names were not confirmed, so they are read defensively). **No hosted service was called live.**

## Limits you should know

- **Guardrails reduce risk; they are not a security boundary.** Classifiers miss things and can be talked around. Keep least-privilege tools,
  approval for irreversible actions, and the tool execution guard.
- **Streaming.** With an enforcing `output` guard the reply is buffered (not streamed), because a reply that has already reached the user cannot be
  taken back. Disable output enforcement if you need streaming.
- **Latency.** Every enforcing guard adds its own latency to each model call (guards at a stage run in parallel). `LLMJudgeGuard` and hosted guards add a network round trip.
- **Only user messages are checked at `input`.** System prompts are yours; tool output is covered by `tool_result`; and the model's own tool calls by `tool_call`.
- **Verified.** The engine integration is tested end to end offline, Presidio runs for real in CI, and the OpenAI Moderation reply is validated against the OpenAI
  SDK's response types. Not run: a live moderation endpoint, Presidio on non-English text.
