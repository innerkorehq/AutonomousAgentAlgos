"""Eval reports: aggregates, comparison with a baseline, and machine-readable output."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from aaa.eval.result import CaseResult


@dataclass
class MetricSummary:
    name: str
    n: int
    mean: float
    pass_rate: float | None  # None for metrics with no pass/fail reading

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": round(self.mean, 4),
            "pass_rate": None if self.pass_rate is None else round(self.pass_rate, 4),
        }


@dataclass
class Regression:
    metric: str
    baseline: float
    current: float

    def __str__(self) -> str:
        return f"{self.metric}: {self.baseline:.3f} -> {self.current:.3f}"


@dataclass
class EvalReport:
    name: str
    results: list[CaseResult] = field(default_factory=list)

    # -- aggregates ---------------------------------------------------------------------------
    def summary(self) -> dict[str, MetricSummary]:
        by: dict[str, list[Any]] = defaultdict(list)
        for r in self.results:
            for s in r.scores:
                by[s.name].append(s)
        out = {}
        for name, scores in sorted(by.items()):
            nums = [s.numeric for s in scores if s.numeric is not None]
            passes = [s.passed for s in scores if s.passed is not None]
            out[name] = MetricSummary(
                name,
                len(scores),
                mean(nums) if nums else 0.0,
                (sum(passes) / len(passes)) if passes else None,
            )
        return out

    @property
    def pass_rate(self) -> float:
        """Share of case runs where every pass/fail-like score passed (and the harness ran the case)."""
        if not self.results:
            return 0.0
        return sum(self._case_passed(r) for r in self.results) / len(self.results)

    @staticmethod
    def _case_passed(r: CaseResult) -> bool:
        return r.error is None and all(s.passed is not False for s in r.scores)

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not self._case_passed(r)]

    @property
    def cost(self) -> float:
        return sum(r.cost for r in self.results)

    @property
    def judge_cost(self) -> float:
        return sum(s.cost for r in self.results for s in r.scores)

    @property
    def tokens(self) -> int:
        return sum(r.tokens for r in self.results)

    # -- baseline -----------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cases": len({r.case.id for r in self.results}),
            "runs": len(self.results),
            "pass_rate": round(self.pass_rate, 4),
            "cost": round(self.cost, 6),
            "judge_cost": round(self.judge_cost, 6),
            "metrics": {k: v.as_dict() for k, v in self.summary().items()},
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def regressions(
        self, baseline: dict[str, Any] | str | Path, *, tolerance: float = 0.02
    ) -> list[Regression]:
        """Metrics whose mean fell by more than `tolerance` against a saved baseline, plus a fall in the overall pass rate."""
        base = baseline if isinstance(baseline, dict) else json.loads(Path(baseline).read_text())
        out = []
        cur = self.summary()
        for name, b in base.get("metrics", {}).items():
            if name in cur and cur[name].mean < b["mean"] - tolerance:
                out.append(Regression(name, b["mean"], cur[name].mean))
        if self.pass_rate < base.get("pass_rate", 0.0) - tolerance:
            out.append(Regression("pass_rate", base["pass_rate"], self.pass_rate))
        return out

    def assert_ok(
        self,
        *,
        min_pass_rate: float = 1.0,
        baseline: dict[str, Any] | str | Path | None = None,
        tolerance: float = 0.02,
    ) -> None:
        """For CI: raise `AssertionError` with the failing cases if the run is below the bar or regressed."""
        problems = []
        if self.pass_rate < min_pass_rate:
            lines = []
            for r in self.failures[:10]:
                bad = [
                    f"{s.name}" + (f" ({s.reason})" if s.reason else "")
                    for s in r.scores
                    if s.passed is False
                ]
                lines.append(f"  {r.case.id}: " + ("; ".join(bad) or r.error or "failed"))
            problems.append(
                f"pass rate {self.pass_rate:.1%} < {min_pass_rate:.1%}\n" + "\n".join(lines)
            )
        if baseline is not None:
            problems += [f"regression {x}" for x in self.regressions(baseline, tolerance=tolerance)]
        if problems:
            raise AssertionError(f"eval {self.name!r} failed:\n" + "\n".join(problems))

    # -- output -------------------------------------------------------------------------------
    def to_markdown(self) -> str:
        d = self.to_dict()
        rows = ["| metric | n | mean | pass rate |", "|---|---|---|---|"]
        for k, v in d["metrics"].items():
            pr = "-" if v["pass_rate"] is None else f"{v['pass_rate']:.0%}"
            rows.append(f"| {k} | {v['n']} | {v['mean']:.3f} | {pr} |")
        head = f"**{self.name}**: {d['runs']} runs over {d['cases']} cases, pass rate {self.pass_rate:.0%}, agent cost ${self.cost:.4f}, judge cost ${self.judge_cost:.4f}\n\n"
        return head + "\n".join(rows) + "\n"

    def to_junit(self) -> str:
        cases = []
        for r in self.results:
            name = quoteattr(f"{r.case.id}#{r.repeat}")
            body = ""
            if not self._case_passed(r):
                msg = (
                    "; ".join(
                        f"{s.name}: {s.reason or s.value}" for s in r.scores if s.passed is False
                    )
                    or r.error
                    or "failed"
                )
                body = f"<failure message={quoteattr(msg[:500])}>{escape(r.text[:2000])}</failure>"
            cases.append(
                f'<testcase classname={quoteattr(self.name)} name={name} time="{r.latency_s:.3f}">{body}</testcase>'
            )
        failed = len(self.failures)
        return (
            f'<?xml version="1.0" encoding="UTF-8"?>\n<testsuite name={quoteattr(self.name)} tests="{len(self.results)}" failures="{failed}">'
            + "".join(cases)
            + "</testsuite>\n"
        )
