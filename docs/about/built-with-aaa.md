# Built with AAA

Real projects built on AAA, with what each one uses. They are a good place to see the library outside of a tutorial.

## BotsHub Agent

[**BotsHub Agent**](https://innerkorehq.github.io/botshub-agent/) ([source](https://github.com/innerkorehq/botshub-agent), MIT)
is an autonomous agent for [botshub.com](https://botshub.com). It finds which AI agents are getting popular (GitHub, Hacker News, SearXNG)
and builds a directory listing for each one using only facts it fetched, with a verbatim quote behind every claim.

It is a worked example of:

| AAA feature | How BotsHub Agent uses it |
|---|---|
| [`WorkflowGraph`](../guides/algorithms/workflow-fsm.md) | parallel signal collection with `on_error="continue"`, so one failing source never stops a run; single-tool graphs for fan-out fetching |
| [`GenerateTestRepair`](../guides/algorithms/refine.md) | fact extraction with a custom validator that rejects any claim whose quote is not on a fetched page |
| [`GeneratorCritic`](../guides/algorithms/refine.md) and [`SelfConsistency`](../guides/algorithms/ensembles-rag.md) | fact-checked overview prose; majority vote on exact values such as prices |
| [`ToolLoop`](../guides/algorithms/tool-loop-react.md) | a bounded gap-filler whose tools can only read the product's own site, scoped by trusted run metadata |
| [Built-in tools](../guides/builtin-tools.md) | the SSRF-hardened fetcher and the SearXNG search provider |
| [Persistence](../guides/persistence.md) | `SQLiteEventStore` plus `engine.recover()` on one machine |
| [Vector stores](../guides/vector-stores.md) | sqlite-vec for conservative entity de-duplication |
| [LLM gateways](../guides/llm-gateways.md) | an OpenAI-compatible gateway, plus a router that sends local MLX model ids in-process |

Its [lessons from a live run](https://innerkorehq.github.io/botshub-agent/aaa/lessons/) collects the integration details we hit against a real gateway
(reasoning-model parameters, algorithm names as identity, output-marker drift), and its
[algorithm choices](https://innerkorehq.github.io/botshub-agent/aaa/algorithms/) explain why each stage uses the algorithm it does.

## Add yours

Built something on AAA? Open a pull request adding a section here: a name, a link, and which AAA features it leans on.
