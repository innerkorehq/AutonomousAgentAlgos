"""Multi-agent debate: several answers, a few rounds of reading each other's, then a judge (or a vote).

Cost is `agents * (rounds + 1)` model calls plus one judge call, and it is bounded at construction (at most 7 agents, 4 rounds)
so a typo cannot turn into a bill. Within a round the agents run in parallel; rounds are sequential."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from aaa.algos.base import finish, llm_request, stable
from aaa.algos.ensemble import _norm, default_extract
from aaa.core.outcome import OutcomeKind
from aaa.events import Event, Projection
from aaa.llm.types import LLMRequest, LLMResponse, Message
from aaa.runtime import Ctx

MAX_AGENTS, MAX_ROUNDS = 7, 4
_JUDGE_SYSTEM = (
    "You are the judge of a debate. Read the final answers below, decide which is correct (or combine "
    "them if they are partly right), and end with a line 'Answer: <final answer>'."
)


class _S(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    answers: dict[str, str] = Field(default_factory=dict)  # "<round>:<agent>" -> text
    failed: list[str] = Field(default_factory=list)  # tags that failed
    judged: str | None = None


def _apply(st: _S, e: Event) -> _S:
    tag = str(e.data.get("tag", ""))
    if e.type == "aaa.run.started":
        st.input = e.data["input"]
    elif e.type == "aaa.llm.completed" and tag.startswith("d"):
        st.answers[tag[1:]] = LLMResponse.model_validate(e.data["response"]).message.content or ""
    elif e.type == "aaa.llm.completed" and tag == "judge":
        st.judged = LLMResponse.model_validate(e.data["response"]).message.content or ""
    elif e.type == "aaa.llm.failed" and tag.startswith("d"):
        st.failed.append(tag[1:])
    return st


DEBATE = Projection(_S, _apply, name="aaa.debate", version=1, state_type=_S)


class Debate:
    """`agents` debaters answer `rounds + 1` times (round 0 independently, later rounds after reading the others' previous
    answers). `judge=True` (default) asks one more model call to rule on the final answers; `judge=False` takes the majority
    of the extracted final answers. `personas` gives each debater its own system prompt.

    Result: `{"answer", "rounds", "agents", "answers": [[round-0 answers], ...], "judged": bool}`."""

    name = "debate"

    def __init__(
        self,
        agents: int = 3,
        rounds: int = 1,
        *,
        judge: bool = True,
        personas: list[str] | None = None,
        model: str | None = None,
        judge_model: str | None = None,
        system: str = "Think step by step, then end with a line 'Answer: <final answer>'.",
        extract: Callable[[str], str] = default_extract,
        temperature: float = 0.7,
    ) -> None:
        if not 2 <= agents <= MAX_AGENTS:
            raise ValueError(f"agents must be between 2 and {MAX_AGENTS}")
        if not 0 <= rounds <= MAX_ROUNDS:
            raise ValueError(f"rounds must be between 0 and {MAX_ROUNDS}")
        if personas is not None and len(personas) != agents:
            raise ValueError("give one persona per agent")
        self.n, self.rounds, self.judge, self.personas = agents, rounds, judge, personas
        self.model, self.judge_model, self.system = model, judge_model, system
        self.extract, self.temperature = extract, temperature

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed"]

    def _prompt(self, task: str, agent: int, r: int, st: _S) -> LLMRequest:
        system = (self.personas[agent] + "\n" if self.personas else "") + self.system
        user = str(task)
        if r > 0:
            others = [
                f"Agent {j + 1}: {self.extract(st.answers[f'{r - 1}:{j}'])}"
                for j in range(self.n)
                if j != agent and f"{r - 1}:{j}" in st.answers
            ]
            mine = st.answers.get(f"{r - 1}:{agent}")
            user += (
                f"\n\nYour previous answer: {self.extract(mine) if mine else '(none)'}\n"
                "Other agents answered:\n" + "\n".join(others) + "\n"
                "Consider their reasoning, keep what is right, fix what is wrong, and give your updated answer."
            )
        return LLMRequest(
            model=self.model,
            params={"temperature": self.temperature},
            messages=[Message(role="system", content=system), Message(role="user", content=user)],
        )

    def _speakers(self, st: _S, r: int) -> list[int]:
        """Who speaks in round `r`: everyone in round 0, afterwards only those who answered the round before."""
        if r == 0:
            return list(range(self.n))
        return [j for j in range(self.n) if f"{r - 1}:{j}" in st.answers]

    def _round_done(self, st: _S, r: int) -> bool:
        return all(
            f"{r}:{j}" in st.answers or f"{r}:{j}" in st.failed for j in self._speakers(st, r)
        )

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        if e.type == "aaa.run.started":
            st = await ctx.project(DEBATE)
            task = e.data["input"].get("task", "")
            return [llm_request(e, self._prompt(task, i, 0, st), f"d0:{i}") for i in range(self.n)]
        st = await ctx.project(DEBATE)
        tag = str(e.data.get("tag", ""))
        task = st.input.get("task", "")
        if tag == "judge":
            if e.type == "aaa.llm.failed":
                return [finish(e, OutcomeKind.FAILED, reason="the judge failed")]
            return [self._finish(e, st, self.extract(st.judged or ""), judged=True)]
        if not tag.startswith("d"):
            return []
        r = int(tag[1:].split(":")[0])
        if not self._round_done(st, r):
            return []
        live = [j for j in range(self.n) if f"{r}:{j}" in st.answers]
        if not live:
            return [finish(e, OutcomeKind.FAILED, reason=f"every debater failed in round {r}")]
        if r < self.rounds:
            # a debater that failed earlier is skipped: it neither speaks nor is quoted
            return [
                stable(
                    llm_request(e, self._prompt(task, j, r + 1, st), f"d{r + 1}:{j}"),
                    f"debate:d{r + 1}:{j}",
                )
                for j in range(self.n)
                if f"{r}:{j}" in st.answers
            ]
        finals = [self.extract(st.answers[f"{r}:{j}"]) for j in live]
        if self.judge:
            listing = "\n".join(
                f"Agent {j + 1}: {self.extract(st.answers[f'{r}:{j}'])}" for j in live
            )
            req = LLMRequest(
                model=self.judge_model or self.model,
                messages=[
                    Message(role="system", content=_JUDGE_SYSTEM),
                    Message(role="user", content=f"Question:\n{task}\n\nFinal answers:\n{listing}"),
                ],
            )
            return [stable(llm_request(e, req, "judge"), "debate:judge")]
        votes = Counter(_norm(a) for a in finals)
        best = max(votes.items(), key=lambda kv: (kv[1], -[_norm(a) for a in finals].index(kv[0])))[
            0
        ]
        return [self._finish(e, st, next(a for a in finals if _norm(a) == best), judged=False)]

    def _finish(self, e: Event, st: _S, answer: str, *, judged: bool) -> Event:
        history = [
            [self.extract(st.answers[k]) for j in range(self.n) if (k := f"{r}:{j}") in st.answers]
            for r in range(self.rounds + 1)
        ]
        return finish(
            e,
            OutcomeKind.SUCCESS,
            {
                "answer": answer,
                "rounds": self.rounds,
                "agents": self.n,
                "answers": history,
                "judged": judged,
            },
        )
