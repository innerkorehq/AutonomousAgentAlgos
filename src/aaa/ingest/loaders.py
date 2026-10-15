"""Loaders: turn files into `Document`s. Local files only; there is deliberately no URL loader (fetching untrusted URLs is the job of the guarded `http_fetch` tool)."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
from collections.abc import AsyncIterator, Iterable, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

from aaa.ingest.types import Document

log = logging.getLogger("aaa.ingest")
TEXT_SUFFIXES = (
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".html",
    ".htm",
    ".log",
    ".sql",
    ".sh",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".h",
    ".cpp",
)


class _Strip(HTMLParser):
    SKIP: ClassVar[set[str]] = {"script", "style", "noscript", "template", "svg"}
    BLOCK: ClassVar[set[str]] = {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "section",
        "article",
        "pre",
        "ul",
        "ol",
        "table",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.out.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    """(text, title). Scripts, styles and templates are dropped; block elements become line breaks."""
    p = _Strip()
    p.feed(html)
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.out))
    text = re.sub(r"\n\s*\n\s*", "\n\n", text).strip()
    return text, p.title.strip()


def _read_text(path: Path, max_bytes: int) -> tuple[str | None, str | None]:
    """(text, why not). Refuses oversized and binary files rather than indexing noise."""
    size = path.stat().st_size
    if size > max_bytes:
        return None, f"larger than {max_bytes} bytes"
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        return None, "looks binary"
    return raw.decode("utf-8", errors="replace"), None


def _document(path: Path, source: str, text: str) -> Document:
    meta: dict[str, object] = {"source": source, "uri": path.resolve().as_uri()}
    if path.suffix.lower() in (".html", ".htm"):
        text, title = html_to_text(text)
        if title:
            meta["title"] = title
    elif path.suffix.lower() in (".md", ".markdown"):
        m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if m:
            meta["title"] = m.group(1).strip()
    meta.setdefault("title", path.stem)
    return Document(source, text, meta)


class FileLoader:
    def __init__(
        self, path: str | Path, *, source: str | None = None, max_bytes: int = 5_000_000
    ) -> None:
        self.path, self.source, self.max_bytes = Path(path), source, max_bytes
        self.skipped: list[tuple[str, str]] = []

    async def load(self) -> AsyncIterator[Document]:
        text, why = await asyncio.to_thread(_read_text, self.path, self.max_bytes)
        src = self.source or self.path.name
        if text is None:
            self.skipped.append((src, why or "unreadable"))
            return
        yield _document(self.path, src, text)


class DirectoryLoader:
    """Every text file under `root` (by extension), with the path relative to `root` as the document's source.

    Safe by default: symlinks are not followed, hidden files and directories (`.git`, `.env`) are skipped, only `extensions` are read, files over `max_bytes` and binary files
    are skipped (and listed in `.skipped` with the reason), and a file that resolves outside `root` is refused. `max_files` stops a runaway tree."""

    def __init__(
        self,
        root: str | Path,
        *,
        glob: str = "**/*",
        extensions: Iterable[str] = TEXT_SUFFIXES,
        exclude: Sequence[str] = (),
        include_hidden: bool = False,
        follow_symlinks: bool = False,
        max_bytes: int = 5_000_000,
        max_files: int = 100_000,
    ) -> None:
        self.root = Path(root).resolve()
        self.glob, self.extensions = glob, tuple(e.lower() for e in extensions)
        self.exclude, self.include_hidden, self.follow_symlinks = (
            tuple(exclude),
            include_hidden,
            follow_symlinks,
        )
        self.max_bytes, self.max_files = max_bytes, max_files
        self.skipped: list[tuple[str, str]] = []

    def _paths(self) -> list[Path]:
        out: list[Path] = []
        for p in sorted(self.root.glob(self.glob)):
            rel = p.relative_to(self.root)
            if not p.is_file() and not p.is_symlink():
                continue
            if not self.include_hidden and any(part.startswith(".") for part in rel.parts):
                continue
            if p.suffix.lower() not in self.extensions or any(
                fnmatch.fnmatch(rel.as_posix(), pat) for pat in self.exclude
            ):
                continue
            if p.is_symlink() and not self.follow_symlinks:
                self.skipped.append((rel.as_posix(), "symlink"))
                continue
            try:
                p.resolve().relative_to(self.root)
            except ValueError:
                self.skipped.append((rel.as_posix(), "resolves outside the root"))
                continue
            out.append(p)
            if len(out) >= self.max_files:
                log.warning("DirectoryLoader stopped at max_files=%d", self.max_files)
                break
        return out

    async def load(self) -> AsyncIterator[Document]:
        self.skipped = []
        for p in await asyncio.to_thread(self._paths):
            rel = p.relative_to(self.root).as_posix()
            text, why = await asyncio.to_thread(_read_text, p, self.max_bytes)
            if text is None:
                self.skipped.append((rel, why or "unreadable"))
                continue
            yield _document(p, rel, text)


class PDFLoader:
    """A PDF, one document per page (`meta["page"]` is 1-based). Needs `pip install pypdf`; text only (no OCR, so a scanned PDF yields nothing)."""

    def __init__(
        self, path: str | Path, *, source: str | None = None, max_bytes: int = 50_000_000
    ) -> None:
        self.path, self.source, self.max_bytes = Path(path), source, max_bytes

    def _pages(self) -> list[str]:
        from pypdf import PdfReader  # pyright: ignore[reportMissingImports]

        if self.path.stat().st_size > self.max_bytes:
            raise ValueError(f"{self.path.name} is larger than {self.max_bytes} bytes")
        return [(p.extract_text() or "") for p in PdfReader(str(self.path)).pages]

    async def load(self) -> AsyncIterator[Document]:
        base = self.source or self.path.name
        for n, text in enumerate(await asyncio.to_thread(self._pages), 1):
            if text.strip():
                yield Document(
                    f"{base}#page={n}",
                    text,
                    {
                        "source": base,
                        "page": n,
                        "title": self.path.stem,
                        "uri": self.path.resolve().as_uri(),
                    },
                )
