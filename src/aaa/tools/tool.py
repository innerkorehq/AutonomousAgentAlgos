from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from pydantic_core import to_jsonable_python

from aaa.llm.types import ToolSpec
from aaa.tools.guard import GuardConfig, check_overrides


class ToolError(Exception):
    """Raise from a tool to report a clean, model-readable failure."""


class ToolArgsError(ToolError):
    pass


@dataclass
class ToolContext:
    """Injected into tools that declare a `ctx: ToolContext` parameter.

    `idempotency_key` is stable across every attempt and every worker for the same logical call
    (default `<run>/<call id>`, or whatever the tool's `idempotency_key=` function returns). Pass it
    to the system you are calling so a repeated attempt cannot repeat the effect. `attempt`/`fence`
    is the 1-based, monotonically increasing execution number of this call."""

    engine: Any
    run_id: str
    call_id: str
    attempt: int = 1
    fence: int = 1
    idempotency_key: str = ""
    tenant: str | None = None  # the run's tenant (from the log, never from the model)
    meta: dict[str, Any] = field(default_factory=dict)  # the run's metadata, e.g. {"user": ...}


def _strip_titles(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            k: _strip_titles(v) for k, v in schema.items() if k != "title" or not isinstance(v, str)
        }
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


class Tool:
    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        requires_approval: bool = False,
        timeout_s: float = 60.0,
        pool: str = "default",
        delivery: str | None = None,
        idempotency_key: Callable[[dict[str, Any]], str] | None = None,
        guard: dict[str, Any] | GuardConfig | None = None,
    ) -> None:
        """`delivery`, `idempotency_key` and `guard` configure the stalled-worker guard for this
        tool (see aaa.tools.guard); anything unset inherits the ToolHandler's default."""
        self.fn, self.name = fn, name or fn.__name__
        self.description = (description or inspect.getdoc(fn) or "").strip()
        self.requires_approval, self.timeout_s, self.pool = requires_approval, timeout_s, pool
        self.idempotency_key = idempotency_key
        if isinstance(guard, GuardConfig):
            self.guard: dict[str, Any] | GuardConfig = guard
        else:
            self.guard = {**(guard or {}), **({"delivery": delivery} if delivery else {})}
            check_overrides(self.guard)  # field-level errors surface at definition time, not at 3am
        sig, hints = inspect.signature(fn), get_type_hints(fn)
        self.wants_ctx = "ctx" in sig.parameters
        fields: dict[str, Any] = {}
        for pname, p in sig.parameters.items():
            if pname == "ctx":
                continue
            fields[pname] = (hints.get(pname, Any), ... if p.default is p.empty else p.default)
        self.input_model: type[BaseModel] = create_model(
            f"{self.name}_args", __config__=ConfigDict(extra="forbid"), **fields
        )
        schema = _strip_titles(self.input_model.model_json_schema())
        self.parameters = schema

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext | None = None) -> Any:
        try:
            parsed = self.input_model.model_validate(arguments)
        except ValidationError as e:
            msgs = "; ".join(
                f"{'.'.join(map(str, x['loc'])) or 'arguments'}: {x['msg']}" for x in e.errors()
            )
            raise ToolArgsError(f"invalid arguments for {self.name}: {msgs}") from e
        kwargs = {k: getattr(parsed, k) for k in type(parsed).model_fields}
        if self.wants_ctx:
            kwargs["ctx"] = ctx
        if inspect.iscoroutinefunction(self.fn):
            result = await self.fn(**kwargs)
        else:
            result = await asyncio.to_thread(self.fn, **kwargs)
        return to_jsonable_python(result)


def tool(fn: Callable[..., Any] | None = None, **opts: Any) -> Any:
    """Decorator: `@tool` or `@tool(requires_approval=True)`. Schema comes from type hints,
    description from the docstring."""
    if fn is not None:
        return Tool(fn, **opts)
    return lambda f: Tool(f, **opts)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t: Tool | Callable[..., Any]) -> Tool:
        t = t if isinstance(t, Tool) else Tool(t)
        if t.name in self._tools:
            raise ValueError(f"duplicate tool {t.name!r}")
        self._tools[t.name] = t
        return t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        return [t.spec() for n, t in self._tools.items() if names is None or n in names]

    def extend(self, other: ToolRegistry) -> None:
        for t in other._tools.values():
            if self._tools.get(t.name) is t:
                continue
            self.add(t)

    def merged(self, other: ToolRegistry) -> ToolRegistry:
        return ToolRegistry([*self._tools.values(), *other._tools.values()])
