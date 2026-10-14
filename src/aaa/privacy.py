"""Redaction: scrub sensitive values from an event BEFORE it is persisted or delivered.

`Engine(redactor=...)` applies a `Redactor` to every event as it is appended (after schema
validation, before the large-payload claim-check), so neither the log, the blob store, the bus nor
any handler downstream ever sees the original. That is the point and the cost: a model that is
handed a tool result sees the redacted text. Narrow the blast radius with `only=`/`skip=` (event-type
globs) when some events must stay intact for the run to work.

Redactors are pure, deterministic and idempotent (redacting redacted text changes nothing), which
the engine relies on: an append that is retried after an outage must produce the same event.
They rewrite only string leaves and well-known secret-bearing keys, never ids, numbers or the
structural keys the library itself depends on (usage, cost, call ids, tags...)."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from aaa.events.event import Event

# Never rewritten, whatever their value: the library routes, accounts and correlates on them.
# Top-level keys of an event's `data` only (a user field called `name` deeper down IS scrubbed)...
PROTECTED_KEYS = frozenset(
    {
        "call_id",
        "child_run_id",
        "user",  # scoping metadata (run meta): rewriting it would silently change whose data it is
        "agent",
        "parent",
        "run_tag",
        "tag",
        "algo",
        "kind",
        "name",
        "model",
        "attempt",
        "fence",
        "idempotency_key",
    }
)
# ...and these anywhere (message roles and accounting subtrees are never user content).
NESTED_PROTECTED_KEYS = frozenset({"role", "tool_call_id", "usage", "cost", "budget"})

# Whole-key matches (after lower-casing and dropping punctuation). Deliberately NOT substring
# matches: `prompt_tokens` is accounting data, `token` is a credential.
SECRET_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "clientsecret",
        "apikey",
        "xapikey",
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "authtoken",
        "token",
        "authorization",
        "proxyauthorization",
        "cookie",
        "setcookie",
        "privatekey",
        "sessionid",
        "otp",
        "pin",
    }
)


@runtime_checkable
class Redactor(Protocol):
    def redact(self, event: Event) -> Event:
        """Return the event with sensitive values replaced. Must be pure and idempotent."""
        ...


def _norm(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _luhn(digits: str) -> bool:
    total, flip = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if flip:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total, flip = total + d, not flip
    return total % 10 == 0


def _card(match: re.Match[str]) -> bool:
    digits = re.sub(r"\D", "", match.group(0))
    return 13 <= len(digits) <= 19 and _luhn(digits)


class Pattern:
    """A named regex, with an optional `check` that can veto a match (e.g. a Luhn check)."""

    def __init__(
        self,
        name: str,
        regex: str | re.Pattern[str],
        check: Callable[[re.Match[str]], bool] | None = None,
    ) -> None:
        self.name = name
        self.regex = re.compile(regex) if isinstance(regex, str) else regex
        self.check = check


EMAIL = Pattern("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
CARD = Pattern("card", r"\b(?:\d[ -]?){12,18}\d\b", _card)
US_SSN = Pattern("ssn", r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
SECRET = Pattern(
    "secret",
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{30,})\b",
)
JWT = Pattern("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
BEARER = Pattern("bearer", r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}")

DEFAULT_PATTERNS = (EMAIL, CARD, US_SSN, SECRET, JWT, BEARER)


class _Base:
    """Common event-type scoping and the tree walk."""

    def __init__(self, *, only: Iterable[str] = ("*",), skip: Iterable[str] = ()) -> None:
        self.only, self.skip = tuple(only), tuple(skip)

    def applies(self, event_type: str) -> bool:
        return any(fnmatch.fnmatchcase(event_type, p) for p in self.only) and not any(
            fnmatch.fnmatchcase(event_type, p) for p in self.skip
        )

    def _walk(self, value: Any, depth: int = 0) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    @staticmethod
    def _protected(key: Any, depth: int) -> bool:
        return key in NESTED_PROTECTED_KEYS or (depth == 0 and key in PROTECTED_KEYS)

    def redact(self, event: Event) -> Event:
        if not self.applies(event.type):
            return event
        data, meta = self._walk(event.data), self._walk(event.meta)
        if data == event.data and meta == event.meta:
            return event
        return event.model_copy(update={"data": data, "meta": meta})


class PatternRedactor(_Base):
    """Replace matches of regular expressions inside any string value with `[REDACTED:<name>]`.

    `PatternRedactor()` covers e-mail addresses, payment-card numbers (Luhn-checked), US SSNs, common
    API keys/tokens, JWTs and bearer tokens. Pass your own `Pattern`s for anything else (national
    ids, phone formats, account numbers): redaction is only as good as its patterns, so treat this
    as a safety net and keep sensitive data out of events where you can."""

    def __init__(
        self,
        patterns: Iterable[Pattern] = DEFAULT_PATTERNS,
        *,
        only: Iterable[str] = ("*",),
        skip: Iterable[str] = (),
    ) -> None:
        super().__init__(only=only, skip=skip)
        self.patterns = tuple(patterns)

    def scrub_text(self, text: str) -> str:
        for p in self.patterns:

            def repl(m: re.Match[str], p: Pattern = p) -> str:
                return m.group(0) if p.check and not p.check(m) else f"[REDACTED:{p.name}]"

            text = p.regex.sub(repl, text)
        return text

    def _walk(self, value: Any, depth: int = 0) -> Any:
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, Mapping):
            return {
                k: v if self._protected(k, depth) else self._walk(v, depth + 1)
                for k, v in value.items()
            }
        if isinstance(value, list | tuple):
            return [self._walk(v, depth + 1) for v in value]
        return value


class KeyRedactor(_Base):
    """Replace the VALUE of well-known secret-bearing keys (`password`, `api_key`, `authorization`,
    `cookie`...) wherever they appear, however deeply nested. Keys match whole (case- and
    punctuation-insensitively), so accounting fields such as `prompt_tokens` are never touched."""

    def __init__(
        self,
        keys: Iterable[str] = SECRET_KEYS,
        *,
        replacement: str = "[REDACTED]",
        only: Iterable[str] = ("*",),
        skip: Iterable[str] = (),
    ) -> None:
        super().__init__(only=only, skip=skip)
        self.keys, self.replacement = frozenset(_norm(k) for k in keys), replacement

    def _walk(self, value: Any, depth: int = 0) -> Any:
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for k, v in value.items():
                if self._protected(k, depth):
                    out[k] = v
                elif (
                    isinstance(k, str)
                    and _norm(k) in self.keys
                    and isinstance(v, str | Mapping | list | tuple)
                ):
                    out[k] = self.replacement
                else:
                    out[k] = self._walk(v, depth + 1)
            return out
        if isinstance(value, list | tuple):
            return [self._walk(v, depth + 1) for v in value]
        return value


class ChainRedactor:
    """Apply several redactors in order."""

    def __init__(self, *redactors: Redactor) -> None:
        self.redactors = redactors

    def redact(self, event: Event) -> Event:
        for r in self.redactors:
            event = r.redact(event)
        return event


def standard_redactor(*, only: Iterable[str] = ("*",), skip: Iterable[str] = ()) -> ChainRedactor:
    """Secret-bearing keys + the default patterns: a sensible starting point."""
    return ChainRedactor(KeyRedactor(only=only, skip=skip), PatternRedactor(only=only, skip=skip))


__all__ = [
    "BEARER",
    "CARD",
    "DEFAULT_PATTERNS",
    "EMAIL",
    "JWT",
    "NESTED_PROTECTED_KEYS",
    "PROTECTED_KEYS",
    "SECRET",
    "SECRET_KEYS",
    "US_SSN",
    "ChainRedactor",
    "KeyRedactor",
    "Pattern",
    "PatternRedactor",
    "Redactor",
    "standard_redactor",
]
