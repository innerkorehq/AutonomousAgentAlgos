"""Tools from the public functions of a Python module.

The rules are strict because importing a module runs it and exposing a function hands it to a remote caller:

* what is exposed is an explicit list (`allow`, or the module's own `__all__`); there is no "everything public" default;
* only functions DEFINED in the module, never `_private` names, classes, generators, lambdas, or `*args/**kwargs`;
* every parameter needs a type hint (`strict`), because the hints ARE the argument schema the caller is validated against;
* the module is only inspected, never called, and the tools require approval unless the operator lists them as safe."""

from __future__ import annotations

import fnmatch
import importlib
import inspect
import re
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aaa.mcp.config import SpecError
from aaa.mcp.model import MCPAnnotations
from aaa.mcp.spec import ModuleSource
from aaa.tools.tool import Tool

_SECTION = re.compile(r"^(Args|Arguments|Parameters|Params)\s*:\s*$")
_END = re.compile(r"^(Returns?|Yields?|Raises?|Examples?|Notes?|See Also|Attributes)\s*:\s*$")
_PARAM = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


@dataclass
class Loaded:
    tools: list[Tool] = field(default_factory=list)
    annotations: dict[str, MCPAnnotations] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """(summary paragraph, {parameter: description}) from a Google-style docstring. Other styles keep the summary and lose the parameters."""
    text = inspect.cleandoc(doc or "")
    if not text:
        return "", {}
    lines = text.splitlines()
    summary: list[str] = []
    for ln in lines:
        if not ln.strip():
            if summary:
                break
            continue
        if _SECTION.match(ln.strip()) or _END.match(ln.strip()):
            break
        summary.append(ln.strip())
    params: dict[str, str] = {}
    i = 0
    while i < len(lines) and not _SECTION.match(lines[i].strip()):
        i += 1
    cur: str | None = None
    base = 0
    for ln in lines[i + 1 :]:
        if not ln.strip():
            continue
        if not ln.startswith((" ", "\t")) or _END.match(ln.strip()):
            break
        indent = len(ln) - len(ln.lstrip())
        m = _PARAM.match(ln.strip())
        if m and (cur is None or indent <= base):
            cur, base = m.group(1), indent
            params[m.group(1)] = m.group(2).strip()
        elif cur is not None:
            params[cur] = (params[cur] + " " + ln.strip()).strip()
    return " ".join(summary), params


def _check_function(name: str, fn: Any, module: str, strict: bool, notes: list[str]) -> None:
    where = f"module {module}: {name}"
    if not (inspect.isfunction(fn) or inspect.ismethod(fn)):
        raise SpecError(
            f"{where} is a {type(fn).__name__}, not a function; only functions can be exposed"
        )
    if fn.__name__ == "<lambda>":
        raise SpecError(f"{where} is a lambda; give it a name with def")
    if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
        raise SpecError(f"{where} is a generator; a tool must return a value")
    sig = inspect.signature(fn)
    bad = [p.name for p in sig.parameters.values() if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    if bad:
        raise SpecError(
            f"{where} takes {', '.join('*' + b for b in bad)}; a tool needs named, typed parameters"
        )
    missing = [
        p.name
        for p in sig.parameters.values()
        if p.name != "ctx" and p.annotation is inspect.Parameter.empty
    ]
    if missing:
        if strict:
            raise SpecError(
                f"{where}: parameters {missing} have no type hint. The hints are the argument schema callers are validated "
                "against: add them, or set strict = false to treat them as Any"
            )
        notes.append(f"{module}.{name}: unannotated parameters {missing} accept any value")


def load_module(src: ModuleSource) -> Loaded:
    """Import `src.module` and build a `Tool` for each function the source exposes. Only called after the import kill-switch passed."""
    if src.path is not None:
        d = Path(src.path).resolve()
        if not d.is_dir():
            raise SpecError(f"module {src.module}: path {src.path!r} is not a directory")
        if d.stat().st_mode & stat.S_IWOTH:
            raise SpecError(
                f"module {src.module}: path {src.path!r} is writable by everyone; refusing to import from it"
            )
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    try:
        mod = importlib.import_module(src.module)
    except Exception as e:  # noqa: BLE001 - whatever the module raised on import, report it as a configuration problem
        raise SpecError(
            f"module {src.module}: cannot import ({type(e).__name__}: {str(e)[:200]})"
        ) from None

    explicit = bool(src.allow)
    names = list(src.allow) if explicit else list(getattr(mod, "__all__", []) or [])
    if not names:
        raise SpecError(
            f"module {src.module} defines no __all__ and the source has no allow list: name the functions to expose with "
            "allow = [...] (there is no 'everything public' default)"
        )
    out = Loaded()
    for name in names:
        if name.startswith("_"):
            raise SpecError(f"module {src.module}: {name!r} is private and cannot be exposed")
        if any(fnmatch.fnmatchcase(name, p) for p in src.exclude):
            continue
        if not hasattr(mod, name):
            raise SpecError(f"module {src.module} has no attribute {name!r}")
        fn = getattr(mod, name)
        if not explicit and not (inspect.isfunction(fn) or inspect.ismethod(fn)):
            out.notes.append(f"{src.module}.{name} is in __all__ but is not a function; skipped")
            continue
        if getattr(fn, "__module__", None) != src.module and not explicit:
            out.notes.append(
                f"{src.module}.{name} is re-exported from {getattr(fn, '__module__', '?')}; skipped (list it in allow to expose it)"
            )
            continue
        _check_function(name, fn, src.module, src.strict, out.notes)

        summary, pdocs = parse_docstring(inspect.getdoc(fn))
        safe = any(fnmatch.fnmatchcase(name, p) for p in src.no_approval) or name in src.read_only
        tool = Tool(
            fn,
            name=name,
            description=summary or None,
            requires_approval=src.default_approval and not safe,
        )
        for pname, text in pdocs.items():
            prop = tool.parameters.get("properties", {}).get(pname)
            if prop is not None and text:
                prop["description"] = text
        out.tools.append(tool)
        if name in src.read_only:
            out.annotations[name] = MCPAnnotations(read_only=True)
    unknown = [
        n
        for n in src.read_only + src.no_approval
        if not any(fnmatch.fnmatchcase(t, n) for t in (x.name for x in out.tools))
    ]
    if unknown:
        raise SpecError(
            f"module {src.module}: read_only / no_approval name functions that are not exposed: {unknown}"
        )
    return out
