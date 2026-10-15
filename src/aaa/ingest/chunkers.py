"""Cutting documents into pieces that fit an embedding model and answer a question on their own."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from aaa.ingest.types import Chunk, Document

_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


class RecursiveChunker:
    """Split on the largest natural boundary that keeps pieces under `size` (paragraphs, then lines, sentences, words, characters), then merge neighbours back up to
    `size`, carrying `overlap` of the previous chunk into the next so a sentence cut at a boundary is whole in one of them.

    `size` and `overlap` are measured by `length` (characters by default). For token budgets use `RecursiveChunker.by_tokens(256)`: it measures with the same
    estimate the context manager uses, which over-counts a little (the safe direction)."""

    def __init__(
        self,
        size: int = 1000,
        overlap: int = 100,
        *,
        separators: Sequence[str] = _SEPARATORS,
        length: Callable[[str], int] = len,
    ) -> None:
        if size <= 0 or not 0 <= overlap < size:
            raise ValueError("need size > 0 and 0 <= overlap < size")
        self.size, self.overlap, self.separators, self.length = (
            size,
            overlap,
            tuple(separators),
            length,
        )

    @classmethod
    def by_tokens(
        cls,
        max_tokens: int = 256,
        overlap_tokens: int = 32,
        counter: object | None = None,
        **kw: object,
    ) -> RecursiveChunker:
        from aaa.context import HeuristicCounter

        c = counter or HeuristicCounter()
        return cls(max_tokens, overlap_tokens, length=c.text, **kw)  # type: ignore[attr-defined, arg-type]

    # -- splitting --------------------------------------------------------------------------
    def _split(self, text: str, seps: Sequence[str]) -> list[str]:
        if self.length(text) <= self.size:
            return [text]
        for i, sep in enumerate(seps):
            if sep == "":  # last resort: characters
                return self._hard(text)
            if sep in text:
                out: list[str] = []
                parts = text.split(sep)
                for j, part in enumerate(parts):
                    piece = part + (sep if j < len(parts) - 1 else "")
                    if piece:
                        out += (
                            self._split(piece, seps[i + 1 :])
                            if self.length(piece) > self.size
                            else [piece]
                        )
                return out
        return self._hard(text)

    def _hard(self, text: str) -> list[str]:
        out, cur = [], ""
        for ch in text:
            if cur and self.length(cur + ch) > self.size:
                out.append(cur)
                cur = ""
            cur += ch
        return out + ([cur] if cur else [])

    def split_text(self, text: str) -> list[str]:
        pieces = self._split(text, self.separators)
        chunks: list[str] = []
        cur: list[str] = []
        for p in pieces:
            if cur and self.length("".join(cur) + p) > self.size:
                chunks.append("".join(cur))
                keep: list[
                    str
                ] = []  # the tail of the chunk just closed, up to `overlap`, becomes the start of the next
                for q in reversed(cur):
                    if self.length("".join([q, *keep])) > self.overlap:
                        break
                    keep.insert(0, q)
                cur = keep
                while cur and self.length("".join(cur) + p) > self.size:
                    cur.pop(0)
            cur.append(p)
        if cur:
            chunks.append("".join(cur))
        return [c.strip() for c in chunks if c.strip()]

    def chunk(self, doc: Document) -> list[Chunk]:
        return [Chunk(t) for t in self.split_text(doc.text)]


_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


class MarkdownChunker:
    """One chunk per section, cut at headings, each carrying `meta["heading"]` (the path, `Guide > Install > Linux`); a section longer than `max_chars`
    is cut further with `RecursiveChunker`. Fenced code blocks are not mistaken for headings."""

    def __init__(self, max_chars: int = 1500, overlap: int = 100) -> None:
        self.inner = RecursiveChunker(max_chars, min(overlap, max_chars - 1))

    def chunk(self, doc: Document) -> list[Chunk]:
        path: list[tuple[int, str]] = []
        sections: list[tuple[str, list[str]]] = [("", [])]
        fenced = False
        for line in doc.text.splitlines():
            if line.lstrip().startswith(("```", "~~~")):
                fenced = not fenced
            m = None if fenced else _HEADING.match(line)
            if m:
                level = len(m.group(1))
                while path and path[-1][0] >= level:
                    path.pop()
                path.append((level, m.group(2)))
                sections.append((" > ".join(t for _, t in path), [line]))
            else:
                sections[-1][1].append(line)
        out: list[Chunk] = []
        for heading, lines in sections:
            body = "\n".join(lines).strip()
            if not body:
                continue
            for t in self.inner.split_text(body):
                out.append(Chunk(t, {"heading": heading} if heading else {}))
        return out
