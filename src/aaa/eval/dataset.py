"""Eval datasets: a list of cases, loadable from JSONL / CSV / Python."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Case(BaseModel):
    id: str
    input: dict[str, Any]  # the algorithm's input, e.g. {"task": "..."}
    expected: Any = (
        None  # what a good answer is (a string, a number, a dict...; metrics decide how to use it)
    )
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Dataset:
    def __init__(self, cases: Iterable[Case], name: str = "dataset") -> None:
        self.cases = list(cases)
        self.name = name
        ids = [c.id for c in self.cases]
        if len(set(ids)) != len(ids):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate case ids: {dup}")

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[Case]:
        return iter(self.cases)

    @classmethod
    def from_list(cls, rows: Sequence[dict[str, Any]], name: str = "dataset") -> Dataset:
        """Rows are `{"input": ..., "expected": ...}`; `input` may be a plain string (becomes `{"task": ...}`)."""
        cases = []
        for n, r in enumerate(rows):
            inp = r["input"]
            cases.append(
                Case(
                    id=str(r.get("id", f"case-{n + 1}")),
                    input={"task": inp} if isinstance(inp, str) else inp,
                    expected=r.get("expected"),
                    tags=list(r.get("tags", [])),
                    metadata=dict(r.get("metadata", {})),
                )
            )
        return cls(cases, name)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Dataset:
        p = Path(path)
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        return cls.from_list(rows, p.stem)

    @classmethod
    def from_csv(
        cls, path: str | Path, *, input_col: str = "input", expected_col: str = "expected"
    ) -> Dataset:
        p = Path(path)
        with p.open(newline="") as f:
            rows = [
                {
                    "id": r.get("id") or f"case-{n + 1}",
                    "input": r[input_col],
                    "expected": r.get(expected_col) or None,
                    "tags": [t for t in (r.get("tags") or "").split(",") if t],
                }
                for n, r in enumerate(csv.DictReader(f))
            ]
        return cls.from_list(rows, p.stem)

    def to_jsonl(self, path: str | Path) -> None:
        Path(path).write_text("".join(c.model_dump_json() + "\n" for c in self.cases))

    def where(self, *, tag: str | None = None) -> Dataset:
        return Dataset([c for c in self.cases if tag is None or tag in c.tags], self.name)
