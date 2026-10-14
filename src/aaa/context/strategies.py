"""Ways to make a conversation fit a model's context window.

Every strategy keeps three invariants, because breaking them gets the request rejected or quietly ruins
the agent:

1. **Tool calls and their results stay together.** An assistant message with `tool_calls` and the `tool`
   messages answering it form one group; providers reject a result with no call (and vice versa), so
   strategies cut only between groups.
2. **The head is untouched**: the leading system messages (instructions, recalled memories) and the first
   user message (the task).
3. **The most recent turns are kept verbatim** (`keep_recent`).

Strategies are cheap-and-lossless first, lossy last: shorten huge tool results, blank out old tool results,
summarize the oldest turns, and as a last resort drop them."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from aaa.context.tokens import TokenCounter
from aaa.llm.client import LLMClient
from aaa.llm.types import LLMRequest, Message, ToolSpec

SUMMARY_PREFIX = "[Summary of the earlier conversation]\n"


def is_summary(m: Message) -> bool:
    return m.role == "system" and (m.content or "").startswith(SUMMARY_PREFIX)


def summary_message(text: str) -> Message:
    return Message(role="system", content=SUMMARY_PREFIX + text)


def head_end(msgs: Sequence[Message]) -> int:
    """Index of the first message that is NOT part of the protected head: the leading system messages
    (a stored summary counts as part of the tail, not the head) plus the first user message."""
    i = 0
    while i < len(msgs) and msgs[i].role == "system" and not is_summary(msgs[i]):
        i += 1
    if i < len(msgs) and msgs[i].role == "user":
        i += 1
    return i


def groups(msgs: Sequence[Message], start: int = 0) -> list[tuple[int, int]]:
    """Half-open index ranges of atomic turn groups from `start`: an assistant message with tool calls
    together with its tool results, or any other message on its own."""
    out: list[tuple[int, int]] = []
    i = start
    while i < len(msgs):
        j = i + 1
        if msgs[i].role == "assistant" and msgs[i].tool_calls:
            wanted = {c.id for c in msgs[i].tool_calls}
            while (
                j < len(msgs)
                and msgs[j].role == "tool"
                and (msgs[j].tool_call_id in wanted or not wanted)
            ):
                j += 1
        out.append((i, j))
        i = j
    return out


def prefix_hash(msgs: Sequence[Message]) -> str:
    """Content hash of a message prefix: a stored summary is reused only for text that matches exactly."""
    blob = json.dumps(
        [m.model_dump(mode="json") for m in msgs], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


@dataclass
class SummaryRecord:
    """A summary of `T[head_end:upto]` of the logical transcript T, valid for any transcript whose first
    `upto` messages hash to `prefix_hash`."""

    upto: int
    prefix_hash: str
    text: str


@dataclass
class Work:
    """The conversation as it is being reduced, plus what strategies need to know about it."""

    messages: list[Message]
    tools: Sequence[ToolSpec]
    counter: TokenCounter
    target: int  # tokens to get under
    keep_recent: int  # turn groups at the end that must survive verbatim
    summary_upto: (
        int  # logical-transcript index where the messages after the summary begin (0 = no summary)
    )
    original: Sequence[Message] = ()  # the logical transcript, for hashing a summary's prefix
    applied: list[str] = field(default_factory=list)
    new_summaries: list[SummaryRecord] = field(default_factory=list)
    tokens_spent: int = 0  # what summarizing cost, for the budget ledger
    cost: float = 0.0
    aligned: bool = (
        True  # False once a strategy changed the message COUNT (view index != transcript index)
    )

    def tokens(self) -> int:
        return self.counter.count(self.messages, self.tools)

    def fits(self) -> bool:
        return self.tokens() <= self.target

    @property
    def has_summary(self) -> bool:
        h = head_end(self.messages)
        return h < len(self.messages) and is_summary(self.messages[h])

    def tail_start(self) -> int:
        """First view index after the protected head (and after a stored summary, if any)."""
        return head_end(self.messages) + (1 if self.has_summary else 0)


@runtime_checkable
class Strategy(Protocol):
    name: str

    async def apply(self, work: Work) -> None:
        """Reduce `work.messages` in place (replace the list); record `work.applied` if it changed anything."""
        ...


def _protected_from(work: Work) -> int:
    """First message index covered by `keep_recent`."""
    gs = groups(work.messages, work.tail_start())
    return gs[-work.keep_recent][0] if len(gs) >= work.keep_recent else len(work.messages)


class TruncateToolResults:
    """Shorten any single tool result longer than `max_chars`, keeping its start and end. Lossless in
    structure, lossy only inside very large outputs (a whole file, a big query result)."""

    name = "truncate_tool_results"

    def __init__(self, max_chars: int = 8_000, head: float = 0.7) -> None:
        self.max_chars, self.head = max_chars, head

    async def apply(self, work: Work) -> None:
        limit, changed = _protected_from(work), False
        for i, m in enumerate(work.messages[:limit]):
            c = m.content or ""
            if m.role == "tool" and len(c) > self.max_chars:
                h = int(self.max_chars * self.head)
                cut = len(c) - self.max_chars
                work.messages[i] = m.model_copy(
                    update={
                        "content": f"{c[:h]}\n[... {cut} characters removed to save context ...]\n{c[-(self.max_chars - h) :]}"
                    }
                )
                changed = True
        if changed:
            work.applied.append(self.name)


class ClearToolResults:
    """Replace the content of old tool results (all but the most recent `keep_last`) with a placeholder.
    The calls stay, so the model still knows what it did and can call the tool again; only the bulky output goes."""

    name = "clear_tool_results"

    def __init__(
        self, keep_last: int = 4, placeholder: str = "[tool result removed to save context]"
    ) -> None:
        self.keep_last, self.placeholder = keep_last, placeholder

    async def apply(self, work: Work) -> None:
        tool_idx = [i for i, m in enumerate(work.messages) if m.role == "tool"]
        old = tool_idx[: max(0, len(tool_idx) - self.keep_last)]
        old = [
            i
            for i in old
            if i < _protected_from(work) and work.messages[i].content != self.placeholder
        ]
        if not old:
            return
        for i in old:
            work.messages[i] = work.messages[i].model_copy(update={"content": self.placeholder})
        work.applied.append(self.name)


def _cut_for(work: Work, keep_tokens: int) -> int:
    """View index where the verbatim tail starts: the oldest group boundary such that the tail still fits in
    `keep_tokens`, but never fewer than `keep_recent` groups and never inside the head or the stored summary."""
    start = work.tail_start()
    gs = groups(work.messages, start)
    if len(gs) <= work.keep_recent:
        return start
    cut, used = len(work.messages), 0
    for n, (s, e) in enumerate(reversed(gs), start=1):
        used += work.counter.count(work.messages[s:e])
        if n > work.keep_recent and used > keep_tokens:
            break
        cut = s
    return max(cut, start)


class DropOldest:
    """Last resort: delete the oldest turn groups (never the head or the recent tail) until the conversation
    fits, leaving a one-line marker so the model knows something is missing."""

    name = "drop_oldest"

    async def apply(self, work: Work) -> None:
        start = work.tail_start()
        dropped = 0
        while not work.fits():
            gs = groups(work.messages, start)
            if len(gs) <= work.keep_recent:
                break
            s, e = gs[0]
            if is_summary(
                work.messages[s]
            ):  # nothing droppable before a summary but the summary itself
                break
            del work.messages[s:e]
            dropped += e - s
        if dropped:
            marker = Message(
                role="system",
                content=f"[{dropped} earlier messages were removed to fit the context window]",
            )
            work.messages.insert(start, marker)
            work.aligned = False
            work.applied.append(f"{self.name}({dropped})")


_SUMMARY_SYSTEM = (
    "You compress an AI agent's working history so it can continue the task after older turns are removed. "
    "Write a dense, factual summary: the goal, what has been done, key results and data (exact names, numbers, "
    "ids, file paths, error messages), decisions and why, constraints the user gave, and what remains. "
    "Do not invent anything. No preamble."
)


def render_for_summary(msgs: Sequence[Message], *, tool_chars: int = 2_000) -> str:
    lines: list[str] = []
    for m in msgs:
        if is_summary(m):
            lines.append("(earlier summary included above)")
        elif m.role == "tool":
            c = m.content or ""
            lines.append(
                f"tool result [{m.name or m.tool_call_id}]: {c[:tool_chars]}{' …' if len(c) > tool_chars else ''}"
            )
        elif m.tool_calls:
            calls = "; ".join(
                f"{c.name}({json.dumps(c.arguments, default=str)[:300]})" for c in m.tool_calls
            )
            lines.append(
                f"assistant: {(m.content or '').strip()} [called: {calls}]".replace(" []", "")
            )
        else:
            lines.append(f"{m.role}: {m.content or ''}")
    return "\n".join(lines)


class Summarize:
    """Replace the oldest turns with a model-written summary, kept as `[head] [summary] [recent tail]`. The
    summary is incremental (each new one folds in the previous), is persisted by the `LLMHandler` as an
    `aaa.context.summary` event, and is reused by content hash, so a restart or another worker does not pay for it again.

    `summarizer` defaults to a plain call on `client`/`model`; supply `summarize=` to use something else."""

    name = "summarize"

    def __init__(
        self,
        client: LLMClient | None = None,
        model: str | None = None,
        *,
        keep_tokens_ratio: float = 0.4,
        chunk_tokens: int = 24_000,
        max_summary_tokens: int = 1_200,
        summarize: Callable[[str, str | None], Awaitable[str]] | None = None,
    ) -> None:
        if client is None and summarize is None:
            raise ValueError("Summarize needs an LLM client or a summarize= callable")
        self.client, self.model = client, model
        self.keep_ratio, self.chunk_tokens, self.max_summary_tokens = (
            keep_tokens_ratio,
            chunk_tokens,
            max_summary_tokens,
        )
        self._fn = summarize

    async def _summarize(self, text: str, previous: str | None, work: Work) -> str:
        if self._fn is not None:
            return await self._fn(text, previous)
        assert self.client is not None
        prior = f"Existing summary of even older history:\n{previous}\n\n" if previous else ""
        resp = await self.client.complete(
            LLMRequest(
                model=self.model,
                messages=[
                    Message(role="system", content=_SUMMARY_SYSTEM),
                    Message(
                        role="user",
                        content=f"{prior}New turns to fold in:\n{text}\n\nWrite the updated summary.",
                    ),
                ],
                params={"max_tokens": self.max_summary_tokens, "temperature": 0},
            )
        )
        work.tokens_spent += resp.usage.total_tokens
        work.cost += resp.cost or 0.0
        return (resp.message.content or "").strip()

    async def apply(self, work: Work) -> None:
        if not work.aligned:
            return
        cut = _cut_for(work, int(work.target * self.keep_ratio))
        start = work.tail_start()
        head = head_end(work.messages)
        if cut <= start:  # nothing older than the verbatim tail to compress
            return
        previous = (
            (work.messages[head].content or "").removeprefix(SUMMARY_PREFIX)
            if work.has_summary
            else None
        )
        chunk: list[Message] = []
        carried, used = previous, 0
        for s, e in groups(work.messages, start):
            if s >= cut:
                break
            size = work.counter.count(work.messages[s:e])
            if chunk and used + size > self.chunk_tokens:
                carried = await self._summarize(render_for_summary(chunk), carried, work)
                chunk, used = [], 0
            chunk += work.messages[s:e]
            used += size
        if chunk:
            carried = await self._summarize(render_for_summary(chunk), carried, work)
        assert carried
        t_upto = work.summary_upto + (cut - start) if work.summary_upto else cut
        rec = SummaryRecord(
            upto=t_upto, prefix_hash=prefix_hash(work.original[:t_upto]), text=carried
        )
        work.new_summaries.append(rec)
        work.messages[:] = [*work.messages[:head], summary_message(carried), *work.messages[cut:]]
        work.summary_upto = t_upto
        work.applied.append(self.name)
