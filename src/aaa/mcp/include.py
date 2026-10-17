"""Composing one configuration from several files: `include = [...]` inside a file, and overlays on top of the result.

The rules are narrow because an included file can name code to import or a process to launch:

* an included path is relative to the file that names it, is resolved (symlinks too) and must stay inside the directory of the
  top configuration; absolute paths and `..` escapes are refused, so a configuration cannot pull in a file from elsewhere;
* nesting is capped at `MAX_DEPTH`, the number of files and each file's size are capped, and a cycle is an error naming the files;
* file paths inside each file (`document`, module `path`, files `root`) are anchored to THAT file's directory;
* merging: `sources`, `agents`, `resources`, `prompts` and `tools.policies` are concatenated in order; dictionaries merge key by
  key; scalars and other lists are replaced by the later file. An including file is "later" than the files it includes.

Overlays are chosen by the operator (a flag or `AAA_MCP_OVERLAY`), not by a configuration, so they may live anywhere; each is
composed on its own tree rules (its own includes stay inside its own directory)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aaa.mcp.config import SpecError, _anchor, _read

MAX_DEPTH = 5
MAX_FILES = 64
MAX_BYTES = 1_000_000
CONCAT = {("sources",), ("agents",), ("resources",), ("prompts",), ("tools", "policies")}


@dataclass
class Composed:
    data: dict[str, Any]
    files: list[Path] = field(default_factory=list)  # every file that contributed, top file first


def merge(
    base: dict[str, Any], over: Mapping[str, Any], _path: tuple[str, ...] = ()
) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        here = (*_path, k)
        cur = out.get(k)
        if here in CONCAT and isinstance(cur, list) and isinstance(v, list):
            out[k] = [*cur, *v]
        elif isinstance(v, Mapping) and isinstance(cur, dict):
            out[k] = merge(cur, v, here)
        else:
            out[k] = v
    return out


def _load(path: Path, root: Path, chain: tuple[Path, ...], files: list[Path]) -> dict[str, Any]:
    real = path.resolve()
    if real in chain:
        loop = " -> ".join(str(p) for p in (*chain, real))
        raise SpecError(f"include cycle: {loop}")
    if len(chain) >= MAX_DEPTH:
        raise SpecError(
            f"{chain[0]}: includes are nested deeper than {MAX_DEPTH} levels (at {real})"
        )
    if len(files) >= MAX_FILES:
        raise SpecError(f"{files[0]}: more than {MAX_FILES} files are included")
    try:
        size = real.stat().st_size
    except OSError as e:
        raise SpecError(f"{path}: cannot read the configuration file ({e.strerror})") from None
    if size > MAX_BYTES:
        raise SpecError(f"{real}: {size} bytes; configuration files are limited to {MAX_BYTES}")
    files.append(real)
    data = _read(real)
    _anchor(data, real.parent)
    names = data.pop("include", [])
    if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
        raise SpecError(f"{real}: include must be a list of relative file paths")
    merged: dict[str, Any] = {}
    for n in names:
        if Path(n).is_absolute():
            raise SpecError(f"{real}: include {n!r} is absolute; use a path relative to this file")
        target = (real.parent / n).resolve()
        if not target.is_relative_to(root):
            raise SpecError(
                f"{real}: include {n!r} resolves to {target}, outside {root}; "
                "an included file must stay inside the top configuration's directory"
            )
        merged = merge(merged, _load(target, root, (*chain, real), files))
    return merge(merged, data)


def compose(
    path: str | os.PathLike[str], overlays: Sequence[str | os.PathLike[str]] = ()
) -> Composed:
    """The top file with its includes, then each overlay (later wins). Raises `SpecError` naming the file at fault."""
    files: list[Path] = []
    top = Path(path).resolve()
    data = _load(top, top.parent, (), files)
    for o in overlays:
        op = Path(o).resolve()
        data = merge(data, _load(op, op.parent, (), files))
    return Composed(data, files)


def overlays_from_env(environ: Mapping[str, str]) -> list[str]:
    raw = environ.get("AAA_MCP_OVERLAY", "")
    return [p for p in raw.split(os.pathsep) if p]


__all__ = ["Composed", "compose", "merge", "overlays_from_env"]
