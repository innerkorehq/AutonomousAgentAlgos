from __future__ import annotations

from pydantic import BaseModel

from aaa.events.event import Event


class Budget(BaseModel):
    """Hard limits for a run. None = unlimited. Enforced by the Engine, not by algorithms."""

    max_steps: int | None = None
    max_tokens: int | None = None
    max_cost: float | None = None
    deadline_s: float | None = None


class Ledger(BaseModel):
    steps: int = 0
    tokens: int = 0
    cost: float = 0.0

    def fold(self, e: Event) -> Ledger:
        """Pure fold; also usable as a Projection to rebuild spend from the log. Child-run
        spend rolls up into the parent via `aaa.child.completed`."""
        if e.type in ("aaa.llm.completed", "aaa.tool.completed", "aaa.tool.failed"):
            usage = e.data.get("usage") or {}
            return Ledger(
                steps=self.steps + 1,
                tokens=self.tokens + int(usage.get("total_tokens", 0)),
                cost=self.cost + float(e.data.get("cost") or 0.0),
            )
        if e.type == "aaa.child.completed":
            sub = e.data.get("ledger") or {}
            return Ledger(
                steps=self.steps + int(sub.get("steps", 0)),
                tokens=self.tokens + int(sub.get("tokens", 0)),
                cost=self.cost + float(sub.get("cost", 0.0)),
            )
        return self

    def exceeded(self, b: Budget) -> str | None:
        if b.max_steps is not None and self.steps > b.max_steps:
            return f"steps {self.steps} > {b.max_steps}"
        if b.max_tokens is not None and self.tokens > b.max_tokens:
            return f"tokens {self.tokens} > {b.max_tokens}"
        if b.max_cost is not None and self.cost > b.max_cost:
            return f"cost {self.cost:.4f} > {b.max_cost}"
        return None
