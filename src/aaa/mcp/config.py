"""Loading an `MCPSpec`: file -> environment overlay -> flag overlay -> one validation, with errors that name the field.

Formats: JSON and TOML need nothing; YAML needs PyYAML (`pip install "autonomous-agent-architecture[mcp-config]"`), read with
`safe_load` only. Secrets never live in the spec: a field that holds one takes an `env:NAME` / `${NAME}` reference, resolved into a
`Secret` whose `repr` is redacted."""

from __future__ import annotations

import json
import os
import re
import stat
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aaa.mcp.spec import MCPSpec

ENV_PREFIX = "AAA_MCP__"


class SpecError(ValueError):
    """The configuration is invalid or cannot be used. The message names the field and says how to fix it."""


def deep_merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Dictionaries merge key by key; anything else (scalars, lists) in `over` replaces what `base` had."""
    out = dict(base)
    for k, v in over.items():
        out[k] = (
            deep_merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), dict) else v
        )
    return out


def _coerce(raw: str) -> Any:
    s = raw.strip()
    if s[:1] in "[{" or s in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?", s):
        try:
            return json.loads(s)
        except ValueError:
            return raw
    return raw


def env_overlay(environ: Mapping[str, str]) -> dict[str, Any]:
    """`AAA_MCP__SERVER__NAME=x` -> `{"server": {"name": "x"}}` (double underscore separates levels; JSON for lists/numbers/bools)."""
    out: dict[str, Any] = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = [p.lower() for p in key[len(ENV_PREFIX) :].split("__") if p]
        if not path:
            continue
        node = out
        for p in path[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise SpecError(f"{key}: conflicts with another AAA_MCP__ variable")
        node[path[-1]] = _coerce(raw)
    return out


def _read(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise SpecError(f"{path}: cannot read the configuration file ({e.strerror})") from None
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(raw)
        elif suffix == ".toml":
            data = tomllib.loads(raw.decode())
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml  # pyright: ignore[reportMissingModuleSource]
            except ImportError:
                raise SpecError(
                    f"{path}: YAML needs PyYAML: pip install 'autonomous-agent-architecture[mcp-config]' "
                    "(or write the configuration as JSON or TOML)"
                ) from None
            data = yaml.safe_load(raw)
        else:
            raise SpecError(
                f"{path}: unknown configuration format {suffix!r} (use .yaml, .yml, .json or .toml)"
            )
    except SpecError:
        raise
    except Exception as e:  # noqa: BLE001 - a parser error is a configuration error, whatever the parser
        raise SpecError(f"{path}: not valid {suffix.lstrip('.').upper()}: {str(e)[:200]}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SpecError(f"{path}: the top level must be a mapping of settings")
    return data


def format_errors(err: ValidationError, where: str = "") -> str:
    lines = []
    for e in err.errors(include_input=False, include_url=False, include_context=False):
        loc = ".".join(str(p) for p in e["loc"]) or "(top level)"
        lines.append(f"{loc}: {e['msg']}")
    head = (
        f"{where}: invalid MCP server configuration"
        if where
        else "invalid MCP server configuration"
    )
    return head + "\n  " + "\n  ".join(lines)


def load_spec(
    source: str | os.PathLike[str] | Mapping[str, Any] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    overlay: Mapping[str, Any] | None = None,
) -> MCPSpec:
    """Build the spec: the file or mapping (lowest), then `AAA_MCP__*` environment variables, then `overlay` (command-line flags)."""
    where = ""
    if source is None:
        data: dict[str, Any] = {}
    elif isinstance(source, Mapping):
        data = dict(source)
    else:
        path = Path(source)
        where = str(path)
        data = _read(path)
    data = deep_merge(data, env_overlay(os.environ if environ is None else environ))
    if overlay:
        data = deep_merge(data, overlay)
    try:
        spec = MCPSpec.model_validate(data)
    except ValidationError as e:
        raise SpecError(format_errors(e, where)) from None
    if where and spec.imports_code():
        mode = Path(where).stat().st_mode
        if mode & stat.S_IWOTH:
            raise SpecError(
                f"{where}: this configuration imports code and is writable by everyone; "
                f"restrict it (chmod o-w) or the server would run whatever someone else puts in it"
            )
    return spec


# -- secrets -------------------------------------------------------------------------------------
_REF = re.compile(r"^(?:env:|\$\{)([A-Za-z_][A-Za-z0-9_]*)\}?$")


class Secret:
    """A value read from the environment when needed. Its `repr` and `str` never show it."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    @classmethod
    def parse(cls, ref: str, *, field: str = "secret") -> Secret:
        m = _REF.match(ref.strip())
        if not m:
            raise SpecError(
                f"{field}: give a reference to an environment variable ('env:NAME' or '${{NAME}}'), never the value itself"
            )
        return cls(m.group(1))

    @property
    def name(self) -> str:
        return self._name

    def value(self, environ: Mapping[str, str] | None = None) -> str:
        v = (os.environ if environ is None else environ).get(self._name)
        if not v:
            raise SpecError(
                f"the environment variable {self._name} is not set (it holds a secret this server needs)"
            )
        return v

    def __repr__(self) -> str:
        return f"Secret(env:{self._name})"

    __str__ = __repr__


__all__ = [
    "ENV_PREFIX",
    "Secret",
    "SpecError",
    "deep_merge",
    "env_overlay",
    "format_errors",
    "load_spec",
]
