"""Configuration for the stalled-worker guard on tool execution.

Why a guard at all: delivery on the bus is at-least-once, and a worker can stall (GC pause, VM
freeze, network partition) past its lease while its tool call is still running. Another worker then
receives the same request. Without a guard the tool runs twice. The guard layers three defences, each
independently configurable:

1. EXCLUSION   an execution lease (+ heartbeat) so two workers never run the same call concurrently
               while the first is alive.
2. FENCING     every execution is a numbered attempt recorded durably BEFORE it runs
               (`aaa.tool.started`, id `<request>:attempt:<n>`). The number is the fencing token: a
               stale attempt's result is rejected, and its task is cancelled when it loses the lease.
3. EFFECT SAFETY  the tool receives `ctx.idempotency_key`, `ctx.attempt` and `ctx.fence` so it can
               make the external effect idempotent (Stripe Idempotency-Key, SES dedupe, conditional
               DB writes). The library cannot make an arbitrary external effect exactly-once; this is
               the part only the tool can do.

Configure globally (`ToolHandler(registry, guard=GuardConfig(...))`) and override per tool
(`@tool(delivery="at_most_once", max_attempts=1, ...)`)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Delivery(StrEnum):
    AT_LEAST_ONCE = "at_least_once"
    """If an attempt is lost (worker died/stalled), run it again, up to `max_attempts`. The tool may
    execute more than once: it must be idempotent, or honor `ctx.idempotency_key`."""

    AT_MOST_ONCE = "at_most_once"
    """Never run a call again once an attempt has started. A lost attempt becomes an
    `OutcomeUnknown` failure (the effect may or may not have happened) for the algorithm or a human
    to reconcile. Use for sends/payments that cannot be made idempotent."""


class LostLease(StrEnum):
    CANCEL = "cancel"
    """A worker that discovers it lost its lease cancels the running tool and records nothing."""

    FINISH = "finish"
    """Let the tool finish anyway. Its result is still subject to `fence_results`."""


class GuardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    enabled: bool = True
    """False restores the minimal behavior: no lease, no attempt records (the bus still redelivers a
    request lost with its worker, so tools can run twice and nothing detects it)."""

    delivery: Delivery = Delivery.AT_LEAST_ONCE
    max_attempts: int = Field(default=3, ge=1)
    """Upper bound on executions of one call under AT_LEAST_ONCE. Exceeding it fails the call with
    `ExecutionAbandoned` (a poison call that keeps killing workers must not loop forever)."""

    lease_ttl_s: float = Field(default=15.0, gt=0)
    """How long a silent worker keeps its exclusive claim. Lower = faster takeover, but a worker
    that is merely slow to heartbeat (long GC pause) loses its claim sooner."""

    heartbeat_s: float | None = Field(default=None, gt=0)
    """Renewal period; default `lease_ttl_s / 3`."""

    wait_poll_s: float = Field(default=0.5, gt=0)
    """How often a worker that found the call busy re-checks (to take over, or to see it finished)."""

    on_lost_lease: LostLease = LostLease.CANCEL
    fence_results: bool = True
    """Drop (and audit as `aaa.tool.stale_result`) a result from an attempt that a newer attempt has
    superseded."""

    @model_validator(mode="after")
    def _check(self) -> GuardConfig:
        if self.heartbeat_s is not None and self.heartbeat_s >= self.lease_ttl_s:
            raise ValueError("heartbeat_s must be smaller than lease_ttl_s")
        return self

    @property
    def beat(self) -> float:
        return self.heartbeat_s if self.heartbeat_s is not None else self.lease_ttl_s / 3

    def merged(self, overrides: dict[str, Any] | GuardConfig | None) -> GuardConfig:
        """This config with a tool's explicit overrides applied (validated)."""
        if not overrides:
            return self
        if isinstance(overrides, GuardConfig):
            return overrides
        return GuardConfig.model_validate({**self.model_dump(), **overrides})


def check_overrides(overrides: dict[str, Any]) -> None:
    """Validate a tool's PARTIAL overrides field by field (types, ranges, unknown keys).

    Cross-field rules (e.g. heartbeat < lease TTL) depend on the handler's default, which is not known
    when the tool is defined, so they are checked when the tool meets its ToolHandler (`attach`)."""
    probe = dict(overrides)
    if probe.get("heartbeat_s") is not None and "lease_ttl_s" not in probe:
        probe["lease_ttl_s"] = (
            float(probe["heartbeat_s"]) * 2
        )  # only to satisfy the cross-field rule here
    GuardConfig.model_validate(probe)
