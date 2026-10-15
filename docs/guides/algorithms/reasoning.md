# Reflexion, Debate, Swarm and Tree-of-thoughts

Four ways to turn more model calls into a better answer, or more specialised behaviour. Each is an ordinary [algorithm](custom.md): state is a fold of the run's log, parallel calls may finish in any
order without changing the result, and every one is bounded twice: by limits checked at construction, and by the run's [`Budget`](../../concepts/runs-and-the-engine.md).

```python
--8<-- "examples/reasoning_algorithms.py"
```

```text
reflexion: 42 km after 2 trials; lesson: ['State the unit.']
debate: 12 from rounds [['12', '10', '12'], ['12', '12', '12']]
swarm: Done: refunded A1 | handoffs: [['triage', 'billing']] | answered by: billing
tree: (8 / (3 - 8 / 3)) | score 10.0 | at most 24 model calls
```

## Reflexion

`Reflexion(validator="name" | None, max_trials=3, keep_reflections=5)`: attempt, evaluate, and if rejected write a short **lesson**, then attempt again **from scratch** with the lessons in the prompt. Contrast
[`GeneratorCritic`](refine.md), which keeps one conversation and patches the answer. The evaluator is a deterministic validator (a `ValidationHandler` name) or, with no `validator`, an LLM critic.

- Result `{"candidate", "trials", "reflections"}`; if no trial passes the run ends `NO_SOLUTION` with the last attempt and the lessons, so nothing learned is lost.
- Lessons are plain events (`aaa.reflexion.feedback` and the `reflect` model call) and are in the result. To carry them to a later run, store them (for example as [memories](../memory.md)) and start the next run with `input["reflections"] = [...]`.
- Cost: up to `max_trials` attempts, evaluations and reflections; `max_trials` is limited to 10.

## Debate

`Debate(agents=3, rounds=1, judge=True, personas=None)`: every agent answers independently; for each round, each agent reads the others' previous answers and revises; then a **judge** call rules (or, with
`judge=False`, the majority of the final answers wins). `personas` gives each agent its own system prompt, which is what makes the debate more than sampling the same view repeatedly.

- Cost `agents * (rounds + 1)` calls (+1 for the judge). Bounded at construction: 2 to 7 agents, 0 to 4 rounds.
- An agent whose call fails drops out of later rounds; the debate fails only if every agent fails in a round.
- Result `{"answer", "rounds", "agents", "answers": [[round-0 answers], ...], "judged"}`: you can see who changed their mind.
- Use it where answers are checkable but the model is unreliable. For problems with one exact answer, [`SelfConsistency`](ensembles-rag.md) is cheaper; debate helps when the reasoning is long enough that reading a peer's argument catches errors.

## Swarm

`Swarm(agents, registry, start="triage", max_handoffs=6)` with `Agent(instructions, tools=[...], handoffs=[...], description=...)`: one conversation, several specialised agents. A handoff is a tool call
(`transfer_to_<agent>`) that the algorithm answers itself, recording `aaa.swarm.handoff`; from the next model call on, the target's tools are the only ones offered, and the transfer's result tells the model its new
instructions. `ToolHandler(registry)` runs the real tools as usual.

Because a model picks the transitions, the guarantees are enforced in code, not by the prompt:

- an agent can only call **its own** tools and transfers: another agent's tool, or an invented one, comes back as `UnknownTool`;
- **one transfer per turn**, and at most `max_handoffs` per run, so two agents cannot pass a task back and forth indefinitely (a model that insists is ended by `max_turns`);
- every real tool call keeps the registry's protections: argument validation, [approval](../human-approval.md), the [guard](../tool-guard.md).

Result `{"text", "turns", "agent", "handoffs": [[from, to], ...]}`. Swarm is a `ToolLoop` (it takes `max_turns`, `stream`, `recall` ...), so sessions and the server can run it like any other.

## Tree-of-thoughts

`TreeOfThoughts(depth=3, width=2, branching=3, stop_score=9.0)`: a **beam search** over reasoning steps. At each level every node of the beam gets `branching` proposed next steps, a separate call scores each 0-10, and
the best `width` survive. A step that starts with `FINAL:` is a complete answer. It stops when the best node is final, when its score reaches `stop_score` (`None` disables this), or at the last level.

- Cost is bounded by construction: at most `depth * width * branching` proposals and as many scores (`algorithm.max_calls`); `width * branching` is limited to 25 and `depth` to 8.
- Scores that are missing or malformed count as 0, and failed calls are skipped, so a flaky scorer degrades the search rather than stalling it; the run fails only if a level has no usable proposals.
- Result `{"answer", "score", "path", "depth", "explored"}`: the winning chain of steps, not just its end.
- Scores are a model's opinion. Where you can check an answer with code, [`GenerateTestRepair`](refine.md) is stronger and much cheaper.

## Verified, and not

Tested with scripted models: the happy paths; budget cut-off for each; bounds; that results and call counts are identical when parallel calls complete in different (randomised) orders and that no model call is issued
twice; the frozen v0.2.0 logs still fold to the same loop state (swarm fields are omitted from an ordinary loop's state). **Not measured**: whether these algorithms actually improve answers with real models on real
tasks; that depends on the model and task and is for your own [evals](../evals.md) to settle before you pay for the extra calls.
