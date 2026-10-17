"""Algorithms as MCP tools: each configured agent becomes a tool with its own name, input schema, budget and rules.

Safety, because a remote client now triggers model spend:
* every agent has a mandatory budget with a deadline, enforced by the engine;
* at most `limits.max_concurrent_agents` agent runs are in flight across all agent tools; the others wait briefly, then are refused;
* the tenant and user come from the host (the stdio configuration, or the authenticated principal behind `aaa-server`), never from the
  arguments, and the input template can only insert the validated arguments of the tool's own schema."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from aaa.eval.result import default_extract
from aaa.mcp.config import SpecError
from aaa.mcp.spec import AgentToolSpec
from aaa.mcp.templating import render, template_fields
from aaa.runtime.budget import Budget
from aaa.tools.tool import SchemaTool, ToolArgsError, ToolContext, ToolError, validate_arguments

if TYPE_CHECKING:
    from aaa.runtime.engine import Algorithm, Engine

BUSY_WAIT_S = 5.0


def check_spec(spec: AgentToolSpec) -> None:
    schema = spec.input_schema
    if schema.get("type") != "object" or not isinstance(schema.get("properties", {}), dict):
        raise SpecError(
            f"agents.{spec.name}.input_schema: must be a JSON schema of type object with properties"
        )
    unknown = template_fields(
        spec.input_template, where=f"agents.{spec.name}.input_template"
    ) - set(schema.get("properties", {}))
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
