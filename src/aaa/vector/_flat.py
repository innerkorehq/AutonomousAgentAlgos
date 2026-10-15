"""Shared by stores whose metadata is flat (Chroma, Pinecone): flattening nested metadata, and compiling filters to the common
Mongo-like dialect (`$eq/$ne/$in/$nin/$gt.../$and/$or`) with negation pushed to the leaves, since that dialect has no NOT."""

from __future__ import annotations

from typing import Any

from aaa.vector.filters import And, Eq, Exists, Filter, In, Ne, Not, Or, Range, UnsupportedFilter


def flatten(meta: dict[str, Any]) -> dict[str, Any]:
    """Scalar leaves of the metadata under dotted keys (what a flat store can filter on). Nulls and lists are left out."""
    out: dict[str, Any] = {}

    def walk(prefix: str, v: Any) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                walk(f"{prefix}.{k}" if prefix else str(k), x)
        elif isinstance(v, str | int | float | bool):
            out[prefix] = v

    walk("", meta)
    return out


def _num(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def to_where(
    f: Filter,
    *,
    store: str,
    supports_exists: bool,
    never: dict[str, Any],
    mixed_in: bool = True,
    negate: bool = False,
) -> dict[str, Any] | None:
    """Compile `f`; None means 'no condition'. Anything the dialect cannot express raises `UnsupportedFilter`, naming `store`."""

    def rec(x: Filter, neg: bool) -> dict[str, Any] | None:
        return to_where(
            x,
            store=store,
            supports_exists=supports_exists,
            never=never,
            mixed_in=mixed_in,
            negate=neg,
        )

    if isinstance(f, Not):
        return rec(f.item, not negate)
    if isinstance(f, Eq):
        if f.value is None:
            raise UnsupportedFilter(f"{store} cannot store null, so it cannot filter for it")
        return {f.field: {"$ne": f.value}} if negate else {f.field: f.value}
    if isinstance(f, Ne):
        return rec(Eq(f.field, f.value), not negate)
    if isinstance(f, In):
        if not f.values:
            return None if negate else never
        if len({type(v) for v in f.values}) == 1:
            return {f.field: {"$nin" if negate else "$in": list(f.values)}}
        return rec(
            And(tuple(Ne(f.field, v) for v in f.values))
            if negate
            else Or(tuple(Eq(f.field, v) for v in f.values)),
            False,
        )
    if isinstance(f, Exists):
        if not supports_exists:
            raise UnsupportedFilter(f"{store} cannot test whether a metadata field exists")
        return {f.field: {"$exists": not negate}}
    if isinstance(f, Range):
        if negate:
            raise UnsupportedFilter(
                f"{store} cannot negate a range (that would include records where the field is missing or not a number)"
            )
        used = {
            k: v
            for k, v in {"$gt": f.gt, "$gte": f.gte, "$lt": f.lt, "$lte": f.lte}.items()
            if v is not None
        }
        if not all(_num(v) for v in used.values()):
            raise UnsupportedFilter(
                f"{store} ranges are numeric: a range over strings cannot be evaluated by the store"
            )
        conds = [{f.field: {k: v}} for k, v in used.items()]
        return conds[0] if len(conds) == 1 else {"$and": conds}
    if isinstance(f, And | Or):
        is_and = isinstance(f, And) != negate  # De Morgan: not(and) = or(not)
        parts = []
        for item in f.items:
            w = rec(item, negate)
            if w is None:
                if is_and:
                    continue
                return None
            parts.append(w)
        if not parts:
            return None if is_and else never
        return parts[0] if len(parts) == 1 else {"$and" if is_and else "$or": parts}
    raise UnsupportedFilter(f"{store} adapter cannot express {type(f).__name__}")
