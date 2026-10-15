"""Metrics: a metric turns a `CaseResult` into one or more `Score`s.

Numbers are 0..1 (higher is better), booleans are pass/fail. Deterministic metrics need nothing; `LLMJudge` and
`SemanticSimilarity` call a model and report what that cost on the `Score`."""

from __future__ import annotations

import inspect
import json
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from aaa.data.capabilities import Embedder
from aaa.eval.result import CaseResult
from aaa.llm.client import LLMClient
from aaa.llm.types import LLMRequest, Message
from aaa.score import Score


@runtime_checkable
class Metric(Protocol):
    name: str

    async def score(self, result: CaseResult, /) -> Score | Sequence[Score]: ...


def _expected_text(r: CaseResult) -> str:
    e = r.case.expected
    return (
        "" if e is None else e if isinstance(e, str) else json.dumps(e, sort_keys=True, default=str)
    )


class ExactMatch:
    def __init__(self, *, normalize: bool = True, name: str = "exact_match") -> None:
        self.normalize, self.name = normalize, name

    async def score(self, r: CaseResult) -> Score:
        a, b = r.text, _expected_text(r)
        if self.normalize:
            a, b = " ".join(a.lower().split()), " ".join(b.lower().split())
        return Score(
            name=self.name, value=a == b, reason=None if a == b else f"got {r.text[:80]!r}"
        )


class Contains:
    """Passes if the answer contains `expected` (or each of `needles`)."""

    def __init__(
        self,
        needles: Sequence[str] | None = None,
        *,
        case_sensitive: bool = False,
        name: str = "contains",
    ) -> None:
        self.needles, self.case_sensitive, self.name = needles, case_sensitive, name

    async def score(self, r: CaseResult) -> Score:
        needles = list(self.needles) if self.needles else [_expected_text(r)]
        hay = r.text if self.case_sensitive else r.text.lower()
        missing = [n for n in needles if (n if self.case_sensitive else n.lower()) not in hay]
        return Score(
            name=self.name, value=not missing, reason=f"missing {missing}" if missing else None
        )


class Regex:
    def __init__(self, pattern: str, *, name: str = "regex") -> None:
        self.rx, self.name = re.compile(pattern), name

    async def score(self, r: CaseResult) -> Score:
        return Score(name=self.name, value=self.rx.search(r.text) is not None)


class JsonValid:
    """The answer parses as JSON (and, if given, validates against `model` - a pydantic model class)."""

    def __init__(self, model: Any = None, *, name: str = "json_valid") -> None:
        self.model, self.name = model, name

    async def score(self, r: CaseResult) -> Score:
        try:
            data = json.loads(r.text)
            if self.model is not None:
                self.model.model_validate(data)
        except Exception as e:  # noqa: BLE001 - any parse/validation error is a failed score
            return Score(name=self.name, value=False, reason=str(e)[:200])
        return Score(name=self.name, value=True)


class NumericClose:
    """The first number in the answer is within `rel_tol`/`abs_tol` of `expected`."""

    def __init__(
        self, *, rel_tol: float = 1e-6, abs_tol: float = 0.0, name: str = "numeric_close"
    ) -> None:
        self.rel_tol, self.abs_tol, self.name = rel_tol, abs_tol, name

    async def score(self, r: CaseResult) -> Score:
        m = re.search(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", r.text)
        if not m:
            return Score(name=self.name, value=False, reason="no number in the answer")
        got = float(m.group().replace(",", ""))
        want = float(r.case.expected)
        ok = math.isclose(got, want, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        return Score(name=self.name, value=ok, reason=None if ok else f"got {got}, want {want}")


class OutcomeIs:
    def __init__(self, kind: str = "success", *, name: str = "outcome") -> None:
        self.kind, self.name = kind, name

    async def score(self, r: CaseResult) -> Score:
        got = r.outcome.kind.value if r.outcome else "harness_error"
        return Score(
            name=self.name,
            value=got == self.kind,
            reason=None if got == self.kind else f"outcome was {got}",
        )


class ToolCalled:
    """Every named tool was called (at least once), and none of `forbidden`."""

    def __init__(
        self, *tools: str, forbidden: Sequence[str] = (), name: str = "tool_called"
    ) -> None:
        self.tools, self.forbidden, self.name = tools, tuple(forbidden), name

    async def score(self, r: CaseResult) -> Score:
        used = set(r.tool_calls)
        bad = [t for t in self.tools if t not in used] + [
            f"called {t}" for t in self.forbidden if t in used
        ]
        return Score(
            name=self.name,
            value=not bad,
            reason="; ".join(f"missing {b}" if not b.startswith("called") else b for b in bad)
            or None,
        )


class ToolSequence:
    """The tools were called in this order (other calls may be interleaved)."""

    def __init__(self, *tools: str, name: str = "tool_sequence") -> None:
        self.tools, self.name = tools, name

    async def score(self, r: CaseResult) -> Score:
        it = iter(r.tool_calls)
        ok = all(any(t == c for c in it) for t in self.tools)
        return Score(name=self.name, value=ok, reason=None if ok else f"calls were {r.tool_calls}")


class Budgeted:
    """Within a cost/step/token/latency allowance; the score falls linearly past the limit to 0 at twice the limit."""

    def __init__(
        self,
        *,
        steps: int | None = None,
        tokens: int | None = None,
        cost: float | None = None,
        latency_s: float | None = None,
        name: str = "within_budget",
    ) -> None:
        self.limits = {"steps": steps, "tokens": tokens, "cost": cost, "latency_s": latency_s}
        self.name = name

    async def score(self, r: CaseResult) -> list[Score]:
        out = []
        for k, lim in self.limits.items():
            if lim is None:
                continue
            used = getattr(r, k)
            out.append(
                Score(
                    name=f"{self.name}.{k}",
                    value=max(0.0, min(1.0, 2 - used / lim)) if lim else 0.0,
                    metadata={"used": used, "limit": lim},
                    reason=None if used <= lim else f"{k} {used} > {lim}",
                )
            )
        return out


class CallableMetric:
    """Wrap a function `fn(result) -> Score | float | bool | (value, reason)`, sync or async."""

    def __init__(
        self, fn: Callable[[CaseResult], Any | Awaitable[Any]], name: str | None = None
    ) -> None:
        self.fn, self.name = fn, name or getattr(fn, "__name__", "callable")

    async def score(self, r: CaseResult) -> Score | Sequence[Score]:
        out = self.fn(r)
        if inspect.isawaitable(out):
            out = await out
        if isinstance(out, Score) or (
            isinstance(out, list) and all(isinstance(s, Score) for s in out)
        ):
            return out
        if isinstance(out, tuple):
            return Score(name=self.name, value=out[0], reason=out[1])
        return Score(name=self.name, value=out)  # pyright: ignore[reportArgumentType]


class SemanticSimilarity:
    """Cosine similarity of the answer's and the expected text's embeddings (0..1); passes at `threshold`."""

    def __init__(
        self, embedder: Embedder, *, threshold: float = 0.8, name: str = "semantic_similarity"
    ) -> None:
        self.embedder, self.threshold, self.name = embedder, threshold, name

    async def score(self, r: CaseResult) -> Score:
        a, b = await self.embedder.embed([r.text, _expected_text(r)])
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
        sim = max(0.0, dot / (na * nb)) if na and nb else 0.0
        return Score(
            name=self.name,
            value=round(sim, 4),
            metadata={"threshold": self.threshold, "passed": sim >= self.threshold},
        )


JUDGE_SYSTEM = (
    "You grade an AI agent's answer against a rubric. Reply with JSON only: "
    '{"score": <number 0..1>, "reason": "<one sentence>"}. 1 means the rubric is fully met, 0 not at all.'
)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
}


class LLMJudge:
    """A model grades the answer against a rubric. The judge runs OUTSIDE the evaluated run (its calls are not part of
    that run's trace or budget); what it cost is reported on the `Score` and summed in the report."""

    def __init__(
        self,
        client: LLMClient,
        rubric: str,
        *,
        model: str | None = None,
        name: str = "judge",
        use_expected: bool = True,
    ) -> None:
        self.client, self.rubric, self.model, self.name, self.use_expected = (
            client,
            rubric,
            model,
            name,
            use_expected,
        )

    async def score(self, r: CaseResult) -> Score:
        parts = [f"Rubric:\n{self.rubric}", f"Task input:\n{json.dumps(r.case.input, default=str)}"]
        if self.use_expected and r.case.expected is not None:
            parts.append(f"Reference answer:\n{_expected_text(r)}")
        parts.append(f"Agent answer:\n{r.text}")
        resp = await self.client.complete(
            LLMRequest(
                model=self.model,
                messages=[
                    Message(role="system", content=JUDGE_SYSTEM),
                    Message(role="user", content="\n\n".join(parts)),
                ],
                response_schema=JUDGE_SCHEMA,
                params={"temperature": 0},
            )
        )
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
        try:
            data = json.loads(_json_text(resp.message.content or ""))
            value = max(0.0, min(1.0, float(data["score"])))
            reason = str(data.get("reason", ""))
        except Exception:  # noqa: BLE001 - an unparseable judge reply is a zero score, flagged
            return Score(
                name=self.name,
                value=0.0,
                reason="judge reply was not valid JSON",
                metadata={"error": True},
                usage=usage,
                cost=resp.cost or 0.0,
            )
        return Score(name=self.name, value=value, reason=reason, usage=usage, cost=resp.cost or 0.0)


def _json_text(s: str) -> str:
    m = re.search(r"\{.*\}", s, re.DOTALL)
    return m.group() if m else s
