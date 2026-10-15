"""A small metadata filter language every vector store understands (or refuses loudly).

    from aaa.vector import F

    F("year") >= 2020
    (F("lang") == "en") & F("tag").is_in(["a", "b"])
    ~F("archived").exists() | (F("score") < 0.5)

Fields are keys of the record's `meta`; dotted paths reach into nested objects (`F("author.country") == "PT"`).
Semantics (identical in every store, and enforced by the shared contract tests):

* `==` / `is_in` match only when the field is present and equal (no type coercion: `1` does not equal `"1"`).
* `!=` is true when the field is **absent or different** (it is exactly `~(F(x) == v)`).
* Ranges (`<`, `<=`, `>`, `>=`) match only numbers against numbers, or strings against strings.
* `exists()` is true when the field is present and not null.

A store that cannot express a filter raises `UnsupportedFilter` instead of fetching more and filtering afterwards, which would
quietly return fewer than `k` results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

Scalar = str | int | float | bool


class UnsupportedFilter(ValueError):
    """The store cannot evaluate this filter natively."""


class Filter:
    def __and__(self, other: Filter) -> Filter:
        return And((self, other))

    def __or__(self, other: Filter) -> Filter:
        return Or((self, other))

    def __invert__(self) -> Filter:
        return Not(self)

    @staticmethod
    def parse(spec: Filter | Mapping[str, Any] | None) -> Filter | None:
        """Accept a `Filter`, or a dict: `{"lang": "en", "year": {"$gte": 2020}, "tag": {"$in": ["a"]}, "$or": [{...}, {...}]}`."""
        if spec is None or isinstance(spec, Filter):
            return spec
        parts: list[Filter] = []
        for key, val in spec.items():
            if key == "$and":
                parts.append(And(tuple(Filter.parse(v) for v in val)))  # type: ignore[misc]
            elif key == "$or":
                parts.append(Or(tuple(Filter.parse(v) for v in val)))  # type: ignore[misc]
            elif key == "$not":
                parts.append(Not(Filter.parse(val)))  # type: ignore[arg-type]
            elif isinstance(val, Mapping):
                for op, arg in val.items():
                    parts.append(_OPS[op](key, arg))
            else:
                parts.append(Eq(key, val))
        return parts[0] if len(parts) == 1 else And(tuple(parts))


@dataclass(frozen=True)
class Eq(Filter):
    field: str
    value: Scalar | None


@dataclass(frozen=True)
class Ne(Filter):
    field: str
    value: Scalar | None


@dataclass(frozen=True)
class In(Filter):
    field: str
    values: tuple[Scalar, ...]


@dataclass(frozen=True)
class Range(Filter):
    field: str
    gt: Scalar | None = None
    gte: Scalar | None = None
    lt: Scalar | None = None
    lte: Scalar | None = None


@dataclass(frozen=True)
class Exists(Filter):
    field: str


@dataclass(frozen=True)
class And(Filter):
    items: tuple[Filter, ...]


@dataclass(frozen=True)
class Or(Filter):
    items: tuple[Filter, ...]


@dataclass(frozen=True)
class Not(Filter):
    item: Filter


_OPS: dict[str, Any] = {
    "$eq": Eq,
    "$ne": Ne,
    "$in": lambda f, v: In(f, tuple(v)),
    "$gt": lambda f, v: Range(f, gt=v),
    "$gte": lambda f, v: Range(f, gte=v),
    "$lt": lambda f, v: Range(f, lt=v),
    "$lte": lambda f, v: Range(f, lte=v),
    "$exists": lambda f, v: Exists(f) if v else Not(Exists(f)),
}


class F:
    """Field builder: `F("year") >= 2020`."""

    def __init__(self, field: str) -> None:
        self.field = field

    def __eq__(self, value: object) -> Filter:  # type: ignore[override]
        return Eq(self.field, value)  # type: ignore[arg-type]

    def __ne__(self, value: object) -> Filter:  # type: ignore[override]
        return Ne(self.field, value)  # type: ignore[arg-type]

    def __gt__(self, v: Scalar) -> Filter:
        return Range(self.field, gt=v)

    def __ge__(self, v: Scalar) -> Filter:
        return Range(self.field, gte=v)

    def __lt__(self, v: Scalar) -> Filter:
        return Range(self.field, lt=v)

    def __le__(self, v: Scalar) -> Filter:
        return Range(self.field, lte=v)

    def is_in(self, values: Iterable[Scalar]) -> Filter:
        return In(self.field, tuple(values))

    def exists(self) -> Filter:
        return Exists(self.field)

    __hash__ = None  # type: ignore[assignment]


def lookup(meta: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    """(present, value) for a dotted path."""
    cur: Any = meta
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def _same_type(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool)
    if isinstance(a, int | float) and isinstance(b, int | float):
        return True
    return isinstance(a, str) and isinstance(b, str)


def _eq(present: bool, got: Any, want: Any) -> bool:
    return present and got is not None and _same_type(got, want) and got == want


def matches(f: Filter | None, meta: Mapping[str, Any]) -> bool:
    """Reference evaluation of a filter against one record's metadata (the oracle the contract tests compare stores to)."""
    if f is None:
        return True
    if isinstance(f, Eq):
        p, v = lookup(meta, f.field)
        return (p and v is None) if f.value is None else _eq(p, v, f.value)
    if isinstance(f, Ne):
        return not matches(Eq(f.field, f.value), meta)
    if isinstance(f, In):
        p, v = lookup(meta, f.field)
        return any(_eq(p, v, w) for w in f.values)
    if isinstance(f, Exists):
        p, v = lookup(meta, f.field)
        return p and v is not None
    if isinstance(f, Range):
        p, v = lookup(meta, f.field)
        if not p or v is None or isinstance(v, bool):
            return False
        for bound, ok in (
            (f.gt, lambda a, b: a > b),
            (f.gte, lambda a, b: a >= b),
            (f.lt, lambda a, b: a < b),
            (f.lte, lambda a, b: a <= b),
        ):
            if bound is None:
                continue
            if not _same_type(v, bound) or not ok(v, bound):
                return False
        return True
    if isinstance(f, And):
        return all(matches(i, meta) for i in f.items)
    if isinstance(f, Or):
        return any(matches(i, meta) for i in f.items)
    if isinstance(f, Not):
        return not matches(f.item, meta)
    raise UnsupportedFilter(f"unknown filter node {type(f).__name__}")
