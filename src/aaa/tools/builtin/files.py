"""Read files from one directory tree, and nothing else. See docs/security/tools.md."""

from __future__ import annotations

import fnmatch
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from aaa.tools.tool import ToolError

DENY = (
    ".env*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    "*.p12",
    "*.pfx",
    "*.kdbx",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials*",
    "secrets*",
)


@dataclass(frozen=True)
class FileAccess:
    """A confined view of a directory. Every path is resolved against `root` and must stay inside it after resolution (symlinks included).

    Hidden files and directories (`.git`, `.env`) are refused unless `allow_hidden`, and names matching `deny` (keys, credentials) are refused always."""

    root: Path
    max_bytes: int = 200_000
    allow_hidden: bool = False
    deny: tuple[str, ...] = field(default=DENY)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    def resolve(self, path: str) -> Path:
        if "\x00" in path:
            raise ToolError("invalid path")
        p = Path(path)
        target = (p if p.is_absolute() else self.root / p).resolve()
        try:
            rel = target.relative_to(self.root)
        except ValueError:
            raise ToolError("path is outside the allowed directory") from None
        for part in rel.parts:
            if not self.allow_hidden and part.startswith("."):
                raise ToolError("hidden files are not accessible")
            if any(fnmatch.fnmatch(part.lower(), pat) for pat in self.deny):
                raise ToolError("this file is not accessible")
        return target

    def read(self, path: str, *, offset: int = 0, limit: int | None = None) -> dict[str, object]:
        """`offset`/`limit` are line numbers (0-based start, count). The result says whether it was cut."""
        target = self.resolve(path)
        try:
            fd = os.open(
                target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )  # the final component may not be a symlink swapped in since resolve()
        except OSError as e:
            raise ToolError(f"cannot read {path!r}: {e.strerror or 'error'}") from None
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ToolError(f"{path!r} is not a file")
            with os.fdopen(fd, "rb", closefd=False) as f:
                raw = f.read(self.max_bytes + 1)
        finally:
            os.close(fd)
        truncated = len(raw) > self.max_bytes
        raw = raw[: self.max_bytes]
        if b"\x00" in raw[:8192]:
            raise ToolError("the file looks binary")
        lines = raw.decode("utf-8", errors="replace").splitlines()
        total = len(lines)
        end = total if limit is None else min(total, offset + limit)
        text = "\n".join(lines[offset:end])
        return {
            "path": str(target.relative_to(self.root)),
            "text": text,
            "lines": [offset, end],
            "total_lines": total,
            "truncated": truncated or end < total,
        }

    def list(self, path: str = ".", *, pattern: str = "*", limit: int = 200) -> dict[str, object]:
        target = self.resolve(path)
        if not target.is_dir():
            raise ToolError(f"{path!r} is not a directory")
        out: list[str] = []
        for entry in sorted(target.iterdir()):
            name = entry.name
            if not self.allow_hidden and name.startswith("."):
                continue
            if any(fnmatch.fnmatch(name.lower(), pat) for pat in self.deny) or not fnmatch.fnmatch(
                name, pattern
            ):
                continue
            try:
                entry.resolve().relative_to(self.root)  # a symlink out of the root is not listed
            except ValueError:
                continue
            out.append(name + ("/" if entry.is_dir() else ""))
            if len(out) >= limit:
                break
        return {
            "path": str(target.relative_to(self.root)) or ".",
            "entries": out,
            "truncated": len(out) >= limit,
        }
