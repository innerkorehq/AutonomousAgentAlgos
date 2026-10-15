"""Deprecation helper. See docs/policy/stability.md: a deprecated API keeps working for at least two minor releases."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(*, since: str, removed_in: str, use: str | None = None) -> Callable[[F], F]:
    """Mark a function or method deprecated: calling it warns (once per call site) and names what to use instead.

    @deprecated(since="0.3", removed_in="0.5", use="new_name")
    def old_name(...): ...
    """

    def wrap(fn: F) -> F:
        name = getattr(fn, "__qualname__", repr(fn))
        msg = f"{name} is deprecated since {since} and will be removed in {removed_in}" + (
            f"; use {use} instead" if use else ""
        )

        @functools.wraps(fn)
        def inner(*a: Any, **kw: Any) -> Any:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return fn(*a, **kw)

        inner.__deprecated__ = msg  # type: ignore[attr-defined]
        return inner  # type: ignore[return-value]

    return wrap
