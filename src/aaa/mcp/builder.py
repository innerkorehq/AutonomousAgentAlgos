"""`MCPServerBuilder`: the same configuration as the file, written as typed Python.

    core = await (
        MCPServerBuilder("desk")
        .title("Support desk")
        .add_tools(registry, tags=["core"])
        .exclude("admin_*")
        .override("search", title="Search", read_only=True)
        .agent("research", algorithm=ReAct(tools), description="Research a question.", deadline_s=60, max_cost=0.50)
        .engine(engine)
        .acompile()
    )

Every method edits an `MCPSpec` (`builder.spec()`), so whatever you build can be written to a file (`builder.dump("server.toml")`),
unless it holds live Python objects, which a file cannot carry. Methods return the builder, so they chain."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self

from aaa.mcp.compile import ApprovalCallback, CompiledServer, acompile_spec, compile_spec
from aaa.mcp.config import dump_spec
from aaa.mcp.spec import MCPSpec

if TYPE_CHECKING:
    import os

    from aaa.runtime.engine import Algorithm, Engine


class MCPServerBuilder:
    def __init__(self, name: str = "aaa") -> None:
        self._data: dict[str, Any] = {"server": {"name": name}}
        self._live: dict[str, Any] = {}
        self._engine: Engine | None = None
        self._approval: ApprovalCallback | None = None

    # -- the server ------------------------------------------------------------------------
    def _put(self, section: str, **kw: Any) -> Self:
        self._data.setdefault(section, {}).update({k: v for k, v in kw.items() if v is not None})
        return self

    def title(self, title: str) -> Self:
        return self._put("server", title=title)

    def version(self, version: str) -> Self:
        return self._put("server", version=version)

    def instructions(self, text: str) -> Self:
        return self._put("server", instructions=text)

    def tenant(self, tenant: str, user: str | None = None) -> Self:
        return self._put("tenancy", tenant=tenant, user=user)

    def limits(
        self,
        *,
        max_result_chars: int | None = None,
        max_tools: int | None = None,
        structured_output: bool | None = None,
        max_concurrent_agents: int | None = None,
    ) -> Self:
        return self._put(
            "limits",
            max_result_chars=max_result_chars,
            max_tools=max_tools,
            structured_output=structured_output,
            max_concurrent_agents=max_concurrent_agents,
        )

    # -- sources ---------------------------------------------------------------------------
    def add_tools(
        self, tools: Any, *, tags: list[str] | None = None, prefix: str = "", key: str | None = None
    ) -> Self:
        """Tools you already hold in Python: a `ToolRegistry`, a `Tool`, a list of tools or an `MCPToolServer`."""
        k = key or f"tools{len(self._live)}"
        self._live[k] = tools
        self._data.setdefault("sources", []).append(
            {"type": "live", "key": k, "tags": tags or [], "prefix": prefix}
        )
        return self

    def add_ref(self, ref: str, *, tags: list[str] | None = None, prefix: str = "") -> Self:
        """Tools from `package.module:attribute`, imported when the server is compiled."""
        self._data.setdefault("sources", []).append(
            {"type": "objects", "ref": ref, "tags": tags or [], "prefix": prefix}
        )
        return self

    def add_module(
        self,
        module: str,
        *,
        allow: list[str] | None = None,
        exclude: list[str] | None = None,
        path: str | os.PathLike[str] | None = None,
        strict: bool = True,
        default_approval: bool = True,
        no_approval: list[str] | None = None,
        read_only: list[str] | None = None,
        tags: list[str] | None = None,
        prefix: str = "",
    ) -> Self:
        """Expose a module's functions (its `__all__`, or `allow`). They require approval, and so stay hidden, unless listed in
        `no_approval` / `read_only`. See the generator guide for the rules."""
        self._data.setdefault("sources", []).append(
            {
                "type": "module",
                "module": module,
                "allow": allow or [],
                "exclude": exclude or [],
                "path": None if path is None else str(path),
                "strict": strict,
                "default_approval": default_approval,
                "no_approval": no_approval or [],
                "read_only": read_only or [],
                "tags": tags or [],
                "prefix": prefix,
            }
        )
        self._data["sources"][-1] = {
            k: v for k, v in self._data["sources"][-1].items() if v is not None
        }
        return self

    # -- the tool policy -------------------------------------------------------------------
    def _policy(self, key: str, values: list[str]) -> Self:
        self._data.setdefault("tools", {}).setdefault(key, []).extend(values)
        return self

    def include(self, *globs: str) -> Self:
        t = self._data.setdefault("tools", {})
        t["include"] = [*(g for g in t.get("include", []) if g != "*"), *globs]
        return self

    def exclude(self, *globs: str) -> Self:
        return self._policy("exclude", list(globs))

    def only_tags(self, *tags: str) -> Self:
        return self._policy("tags", list(tags))

    def prefix(self, prefix: str) -> Self:
        self._data.setdefault("tools", {})["prefix"] = prefix
        return self

    def rename(self, old: str, new: str) -> Self:
        self._data.setdefault("tools", {}).setdefault("rename", {})[old] = new
        return self

    def expose_approval_tools(self, on: bool = True) -> Self:
        self._data.setdefault("tools", {})["expose_approval_tools"] = on
        return self

    def override(
        self,
        name: str,
        *,
        title: str | None = None,
        description: str | None = None,
        requires_approval: bool | None = None,
        output_schema: dict[str, Any] | None = None,
        read_only: bool | None = None,
        destructive: bool | None = None,
        idempotent: bool | None = None,
        open_world: bool | None = None,
    ) -> Self:
        """Change how the tool exposed as `name` looks. Hints given here replace the ones derived from the tool."""
        o = self._data.setdefault("tools", {}).setdefault("overrides", {}).setdefault(name, {})
        o.update(
            {
                k: v
                for k, v in {
                    "title": title,
                    "description": description,
                    "requires_approval": requires_approval,
                    "output_schema": output_schema,
                }.items()
                if v is not None
            }
        )
        hints = {
            "read_only": read_only,
            "destructive": destructive,
            "idempotent": idempotent,
            "open_world": open_world,
        }
        if any(v is not None for v in hints.values()):
            o.setdefault("annotations", {}).update(
                {k: v for k, v in hints.items() if v is not None}
            )
        return self

    # -- agents ----------------------------------------------------------------------------
    def agent(
        self,
        name: str,
        *,
        algorithm: Algorithm | str,
        description: str,
        deadline_s: float,
        max_steps: int | None = None,
        max_tokens: int | None = None,
        max_cost: float | None = None,
        input_schema: Mapping[str, Any] | None = None,
        input_template: Mapping[str, Any] | None = None,
        output: str = "text",
        tags: list[str] | None = None,
        read_only: bool | None = None,
        open_world: bool | None = None,
    ) -> Self:
        """Expose an algorithm as a tool. `algorithm` is the object or a `module:attribute` reference; a deadline is mandatory."""
        if isinstance(algorithm, str):
            ref = algorithm
        else:
            ref = f"agent_{name}"
            self._live[ref] = algorithm
        entry: dict[str, Any] = {
            "name": name,
            "algorithm": ref,
            "description": description,
            "budget": {
                k: v
                for k, v in {
                    "deadline_s": deadline_s,
                    "max_steps": max_steps,
                    "max_tokens": max_tokens,
                    "max_cost": max_cost,
                }.items()
                if v is not None
            },
            "output": output,
            "tags": tags or [],
        }
        if input_schema is not None:
            entry["input_schema"] = dict(input_schema)
        if input_template is not None:
            entry["input_template"] = dict(input_template)
        hints = {
            k: v
            for k, v in {"read_only": read_only, "open_world": open_world}.items()
            if v is not None
        }
        if hints:
            entry["annotations"] = hints
        self._data.setdefault("agents", []).append(entry)
        return self

    # -- resources and prompts --------------------------------------------------------------
    def _add(self, section: str, entry: dict[str, Any]) -> Self:
        self._data.setdefault(section, []).append({k: v for k, v in entry.items() if v is not None})
        return self

    def static_resource(
        self,
        name: str,
        text: str,
        *,
        mime_type: str = "text/plain",
        title: str | None = None,
        description: str | None = None,
    ) -> Self:
        return self._add(
            "resources",
            {
                "type": "static",
                "name": name,
                "text": text,
                "mime_type": mime_type,
                "title": title,
                "description": description,
            },
        )

    def files(
        self,
        name: str,
        root: str | os.PathLike[str],
        *,
        pattern: str = "**/*",
        shared: bool = False,
        allow_hidden: bool = False,
        title: str | None = None,
        description: str | None = None,
    ) -> Self:
        """Files under `root` as resources, one subdirectory per tenant unless `shared=True`."""
        return self._add(
            "resources",
            {
                "type": "files",
                "name": name,
                "root": str(root),
                "pattern": pattern,
                "shared": shared,
                "allow_hidden": allow_hidden,
                "title": title,
                "description": description,
            },
        )

    def index(
        self,
        name: str,
        index: Any,
        *,
        tenant_key: str | None = None,
        shared: bool = False,
        k: int = 5,
        max_chars: int = 800,
    ) -> Self:
        """Search over an `Index` (an object, or a `module:attribute` reference). Say how documents are tenant-scoped (`tenant_key`) or `shared=True`."""
        ref = index if isinstance(index, str) else f"index_{name}"
        if not isinstance(index, str):
            self._live[ref] = index
        return self._add(
            "resources",
            {
                "type": "index",
                "name": name,
                "ref": ref,
                "tenant_key": tenant_key,
                "shared": shared,
                "k": k,
                "max_chars": max_chars,
            },
        )

    def memory(self, name: str, memory: Any, *, k: int = 5, max_chars: int = 800) -> Self:
        ref = memory if isinstance(memory, str) else f"memory_{name}"
        if not isinstance(memory, str):
            self._live[ref] = memory
        return self._add(
            "resources",
            {"type": "memory", "name": name, "ref": ref, "k": k, "max_chars": max_chars},
        )

    def prompt(
        self,
        name: str,
        messages: list[str | tuple[str, str]],
        *,
        arguments: list[dict[str, Any]] | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> Self:
        """A prompt template. Messages are texts (user role) or `(role, text)` pairs; `{argument}` placeholders must be declared in `arguments`."""
        msgs = [
            {"role": "user", "text": m} if isinstance(m, str) else {"role": m[0], "text": m[1]}
            for m in messages
        ]
        return self._add(
            "prompts",
            {
                "name": name,
                "title": title,
                "description": description,
                "arguments": arguments or [],
                "messages": msgs,
            },
        )

    # -- runtime ---------------------------------------------------------------------------
    def engine(self, engine: Engine | str) -> Self:
        """The engine agent tools run on: an object, or a `module:attribute` reference the server builds itself."""
        if isinstance(engine, str):
            self._data.setdefault("runtime", {})["engine"] = engine
        else:
            self._engine = engine
        return self

    def approval(self, callback: ApprovalCallback) -> Self:
        """Decide, per call, whether a tool that requires approval may run (see `MCPToolServer`). Fails closed."""
        self._approval = callback
        return self

    # -- results ---------------------------------------------------------------------------
    def spec(self) -> MCPSpec:
        return MCPSpec.model_validate(self._data)

    @property
    def live(self) -> dict[str, Any]:
        return dict(self._live)

    def dump(self, path: str | os.PathLike[str] | None = None) -> str:
        """Write the configuration as JSON/TOML/YAML (by extension). Fails if it holds live Python objects."""
        return dump_spec(self.spec(), path)

    async def acompile(self, *, allow_import: bool | None = None) -> CompiledServer:
        return await acompile_spec(
            self.spec(),
            live=self._live,
            engine=self._engine,
            approval=self._approval,
            allow_import=allow_import,
        )

    def compile(self, *, allow_import: bool | None = None) -> CompiledServer:
        return compile_spec(
            self.spec(),
            live=self._live,
            engine=self._engine,
            approval=self._approval,
            allow_import=allow_import,
        )


__all__ = ["MCPServerBuilder"]
