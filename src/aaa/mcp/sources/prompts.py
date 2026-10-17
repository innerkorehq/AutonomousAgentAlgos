"""MCP prompts: named, typed message templates a client can offer to its user."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aaa.mcp.model import MCPPrompt, MCPPromptArgument, MCPPromptResult
from aaa.mcp.spec import PromptSpec
from aaa.mcp.templating import render


class PromptError(ValueError):
    """The arguments do not fit the prompt. The message says which one and why (it contains no server internals)."""


class PromptDef:
    def __init__(self, spec: PromptSpec) -> None:
        self.spec = spec

    def describe(self) -> MCPPrompt:
        s = self.spec
        return MCPPrompt(
            s.name,
            s.description,
            s.title,
            tuple(MCPPromptArgument(a.name, a.description, a.required) for a in s.arguments),
        )

    def _coerce(self, name: str, kind: str, choices: list[str], raw: Any) -> Any:
        text = raw if isinstance(raw, str) else str(raw)
        if kind == "integer":
            try:
                return int(text)
            except ValueError:
                raise PromptError(f"argument {name!r} must be an integer") from None
        if kind == "boolean":
            if text.lower() in ("true", "false"):
                return text.lower()
            raise PromptError(f"argument {name!r} must be true or false")
        if kind == "enum" and text not in choices:
            raise PromptError(f"argument {name!r} must be one of {choices}")
        return text

    def get(self, arguments: Mapping[str, Any] | None) -> MCPPromptResult:
        given = dict(arguments or {})
        known = {a.name: a for a in self.spec.arguments}
        if unknown := sorted(set(given) - set(known)):
            raise PromptError(f"unknown argument(s) {unknown}; this prompt takes {sorted(known)}")
        values: dict[str, Any] = {}
        for a in self.spec.arguments:
            if a.name not in given:
                if a.required:
                    raise PromptError(f"missing required argument {a.name!r}")
                values[a.name] = ""
                continue
            values[a.name] = self._coerce(a.name, a.type, a.choices, given[a.name])
        msgs = tuple((m.role, str(render(m.text, values))) for m in self.spec.messages)
        return MCPPromptResult(self.spec.description, msgs)
