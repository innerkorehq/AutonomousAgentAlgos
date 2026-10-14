# SelfConsistency and RAGAgent

## SelfConsistency

```python
class SelfConsistency:
    def __init__(
        self,
        n=5,
        *,
        temperature=0.8,
        model=None,
        system="Think step by step, then end with 'Answer: <final answer>'.",
        extract=default_extract,
    ): ...
```

Ask the same question `n` times **in parallel** at a higher temperature and return the majority answer.

```python
--8<-- "examples/ensembles.py"
```

```text
vote: {'answer': '42', 'votes': 3, 'samples': 5, 'agreement': 0.6}
critic loop: {'candidate': 'draft v2', 'iterations': 2}
```

- `extract(text) -> str` pulls the comparable answer from each sample. The default takes the last `Answer: ...` line, else the last line.
- Answers are **normalized** (lowercased, punctuation trimmed, trailing period dropped) before voting, so `42`, `42.` and `Answer: 42.` agree.
- Ties go to the earliest-seen answer. `agreement` is the winner's share.
- Individual sample failures are tolerated; if *all* fail the run fails.
- Cost is `n` times a single call, but latency is one call (they run concurrently).

Use it where answers are exactly comparable (arithmetic, classification, extracted fields). It does nothing useful for open-ended prose.

## RAGAgent

```python
class RAGAgent:
    def __init__(self, index, *, k=5, **tool_loop_kwargs): ...
```

A `ToolLoop` preconfigured with a `search_documents` tool over any [`Index`](../../concepts/data-model.md#retrieval-index)
and a system prompt that demands grounding and citations ("answer using only retrieved evidence; cite ids like [d1]; say so
if the evidence is insufficient").

```python
--8<-- "examples/rag_agent.py"
```

```text
Paris is the capital of France [d1].
```

- The model decides **when and what** to retrieve, and may search several times with different queries.
- Results carry `id`, `score`, `text` (truncated) and `meta`, so citations can point back to your source.
- Attach the agent's tool executor: `ToolHandler(agent.tools).attach(engine)`.
- Retrieval quality is the index's: BM25 by default, hybrid with an `Embedder`. For large corpora implement the `Index`
  protocol over your vector database.

!!! note "Agentic, not prefetched"
    This is *tool-based* retrieval. It does not retrieve before the first model call. If you want a fixed
    retrieve-then-read pipeline, build it with a [`WorkflowGraph`](workflow-fsm.md) (retrieve node, generate node).
