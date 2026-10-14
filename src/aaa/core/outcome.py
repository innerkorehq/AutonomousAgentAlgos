from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class OutcomeKind(StrEnum):
    SUCCESS = "success"
    NO_SOLUTION = "no_solution"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    INVALID_INPUT = "invalid_input"
    CANCELLED = "cancelled"
    FAILED = "failed"


class Outcome(BaseModel, Generic[T]):
    """Terminal result of a run; failure modes are distinct, never conflated."""

    kind: OutcomeKind
    value: T | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.kind is OutcomeKind.SUCCESS
