from aaa.core.errors import AAAError


class LLMError(AAAError):
    retryable = False


class LLMConfigError(LLMError):
    """Bad or incomplete configuration (missing env var, unknown provider)."""


class TransientError(LLMError):
    """Timeouts, connection errors, 5xx: safe to retry."""

    retryable = True


class RateLimited(TransientError):
    def __init__(self, msg: str = "rate limited", retry_after: float | None = None) -> None:
        super().__init__(msg)
        self.retry_after = retry_after


class ContextOverflow(LLMError):
    """Prompt too long for the model; retrying unchanged will not help."""


class StateNotFound(LLMError):
    """The provider no longer holds the conversation state we tried to continue (expired, deleted, or
    never stored). Not a failure of the request: the handler resends the full conversation."""


class ContentFiltered(LLMError):
    pass


class AuthError(LLMError):
    pass


class FatalError(LLMError):
    pass


class CircuitOpen(LLMError):
    """Short-circuited by the circuit breaker; not retried."""


class GuardrailBlocked(LLMError):
    """A guardrail blocked the request or the reply. Never retried; algorithms end the run as `OutcomeKind.BLOCKED`."""

    def __init__(
        self, stage: str = "", categories: list[str] | None = None, guard: str | None = None
    ) -> None:
        self.stage, self.categories, self.guard = stage, list(categories or []), guard
        super().__init__(
            f"blocked by policy at {stage}: {', '.join(self.categories) or 'unspecified'}"
        )


class UnsupportedModality(LLMError):
    """The request carries an attachment (image, audio, file) that this client cannot send. Nothing is dropped silently."""

    def __init__(self, kinds: set[str] | list[str], supported: set[str] | frozenset[str]) -> None:
        self.kinds, self.supported = sorted(kinds), sorted(supported)
        super().__init__(
            f"this client cannot send {', '.join(self.kinds)} attachments (it supports: {', '.join(self.supported)})"
        )
