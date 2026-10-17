"""Plain `{name}` templates for agent inputs and prompts. Deliberately tiny and closed: only plain argument names, no attribute
access, indexing, conversions or format specs, so a template cannot reach into anything but the validated arguments it is given."""

from __future__ import annotations

import json
import re
import string
from collections.abc import Mapping
from typing import Any

from aaa.mcp.config import SpecError

_FORMATTER = string.Formatter()
_PLAIN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def template_fields(node: Any, *, where: str = "template") -> set[str]:
    """Every placeholder in a template (a string, or a dict/list of them), checked to be a plain argument name."""
    found: set[str] = set()
    if isinstance(node, str):
        try:
            parsed = list(_FORMATTER.parse(node))
        except ValueError as e:
            raise SpecError(f"{where}: {e} (write a literal brace as {{{{ or }}}})") from None
        for _, field, spec, conv in parsed:
            if field is None:
                continue
            if not _PLAIN.match(field) or spec or conv:
                raise SpecError(
                    f"{where}: {{{field}}} is not allowed; placeholders are plain argument names like {{input}} "
                    "(no attributes, indexes, conversions or format specs)"
                )
            found.add(field)
    elif isinstance(node, Mapping):
        for v in node.values():
            found |= template_fields(v, where=where)
    elif isinstance(node, list):
        for v in node:
            found |= template_fields(v, where=where)
    return found


def render(node: Any, args: Mapping[str, Any]) -> Any:
    """Fill a template. A string that is exactly one placeholder becomes the argument itself (keeping its type); inside longer
    text, non-strings are written as JSON."""
    if isinstance(node, str):
        parsed = list(_FORMATTER.parse(node))
        if len(parsed) == 1 and parsed[0][0] == "" and parsed[0][1] is not None:
            return args.get(parsed[0][1])
        out = []
        for lit, field, _, _ in parsed:
            out.append(lit)
            if field is not None:
                v = args.get(field, "")
                out.append(v if isinstance(v, str) else json.dumps(v, default=str))
        return "".join(out)
    if isinstance(node, Mapping):
        return {k: render(v, args) for k, v in node.items()}
    if isinstance(node, list):
        return [render(v, args) for v in node]
    return node
