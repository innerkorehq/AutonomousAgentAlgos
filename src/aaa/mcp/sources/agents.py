"""Algorithms as MCP tools: each configured agent becomes a tool with its own name, input schema, budget and rules.

Safety, because a remote client now triggers model spend:
* every agent has a mandatory budget with a deadline, enforced by the engine;
* at most `limits.max_concurrent_agents` agent runs are in flight across all agent tools; the others wait briefly, then are refused;
* the tenant and user come from the host (the stdio configuration, or the authenticated principal behind `aaa-server`), never from the
  arguments, and the input template can only insert the validated arguments of the tool's own schema."""

from __future__ import annotations

import asyncio
import string
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aaa.eval.result import default_extract
from aaa.mcp.config import SpecError
from aaa.mcp.spec import AgentToolSpec
from aaa.runtime.budget import Budget
from aaa.tools.tool import SchemaTool, ToolArgsError, ToolContext, ToolError, validate_arguments

if TYPE_CHECKING:
    from aaa.runtime.engine import Algorithm, Engine

_FIELD = string.Formatter()
_PLAIN = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BUSY_WAIT_S = 5.0


def template_fields(node: Any) -> set[str]:
    """Every placeholder in a template, validated to be a plain argument name."""
    found: set[str] = set()
    if isinstance(node, str):
        for _, field, spec, conv in _FIELD.parse(node):
            if field is None:
                continue
            if not _PLAIN.match(field) or spec or conv:
                raise SpecError(
                    f"input_template: {{{field}}} is not allowed; placeholders are plain argument names like {{input}} "
                    "(no attributes, indexes, conversions or format specs)"
                )
            found.add(field)
    elif isinstance(node, Mapping):
        for v in node.values():
            found |= template_fields(v)
    elif isinstance(node, list):
        for v in node:
            found |= template_fields(v)
    return found


def render(node: Any, args: Mapping[str, Any]) -> Any:
    """Fill a template. A string that is exactly one placeholder becomes the argument itself (keeping its type)."""
    if isinstance(node, str):
        parsed = list(_FIELD.parse(node))
        if len(parsed) == 1 and parsed[0][0] == "" and parsed[0][1] is not None:
            return args.get(parsed[0][1])
        out = []
        for lit, field, _, _ in parsed:
            out.append(lit)
            if field is not None:
                v = args.get(field, "")
                out.append(v if isinstance(v, str) else __import__("json").dumps(v, default=str))
        return "".join(out)
    if isinstance(node, Mapping):
        return {k: render(v, args) for k, v in node.items()}
    if isinstance(node, list):
        return [render(v, args) for v in node]
    return node


def check_spec(spec: AgentToolSpec) -> None:
    schema = spec.input_schema
    if schema.get("type") != "object" or not isinstance(schema.get("properties", {}), dict):
        raise SpecError(
            f"agents.{spec.name}.input_schema: must be a JSON schema of type object with properties"
        )
    unknown = template_fields(spec.input_template) - set(schema.get("properties", {}))
    if unknown:
        raise SpecError(
            f"agents.{spec.name}.input_template uses {sorted(unknown)}, which are not properties of its input_schema"
        )


class AgentTool(SchemaTool):
    """A tool whose call runs an algorithm to completion on the engine, as the caller's tenant."""

    def __init__(
        self,
        spec: AgentToolSpec,
        algo: Algorithm,
        engine: Engine,
        gate: asyncio.Semaphore,
    ) -> None:
        async def _never(_: dict[str, Any]) -> None: ...

        super().__init__(
            _never,
            name=spec.name,
            description=spec.description,
            parameters=spec.input_schema,
            timeout_s=spec.budget.deadline_s + 15,
        )
        self.spec_, self.algo, self.engine, self.gate = spec, algo, engine, gate
        self.budget = Budget(
            deadline_s=spec.budget.deadline_s,
            max_steps=spec.budget.max_steps,
            max_tokens=spec.budget.max_tokens,
            max_cost=spec.budget.max_cost,
        )

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext | None = None) -> Any:
        if problems := validate_arguments(self.parameters, arguments):
            raise ToolArgsError(f"invalid arguments for {self.name}: {'; '.join(problems)}")
        tenant = (ctx.tenant if ctx else None) or "mcp"
        user = (ctx.meta.get("user") if ctx else None) or None
        try:
            await asyncio.wait_for(self.gate.acquire(), BUSY_WAIT_S)
        except TimeoutError:
            raise ToolError(
                "the server is running as many agents as it allows; try again shortly"
            ) from None
        try:
            inp = render(self.spec_.input_template, arguments)
            out = await self.engine.tenant(tenant).run(
                self.algo, inp, budget=self.budget, user=user
            )
        finally:
            self.gate.release()
        if not out.ok:
            raise ToolError(
                f"the agent ended {out.kind.value}" + (f": {out.reason}" if out.reason else "")
            )
        if self.spec_.output == "json":
            return {"kind": out.kind.value, "value": out.value, "reason": out.reason}
        return default_extract(out.value)
