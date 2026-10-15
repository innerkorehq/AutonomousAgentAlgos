"""Evaluation: datasets, metrics, a suite that runs them through your algorithm, reports, and online scoring.

    suite = EvalSuite(Dataset.from_jsonl("cases.jsonl"), algo, [Contains(), ToolCalled("search"), Budgeted(cost=0.05)])
    report = await suite.run(engine)
    report.assert_ok(min_pass_rate=0.9, baseline="evals/baseline.json")     # in CI

Third-party metrics (DeepEval, Ragas, autoevals) plug in as `Metric`s; see the guide."""

from aaa.eval.dataset import Case, Dataset
from aaa.eval.metrics import (
    Budgeted,
    CallableMetric,
    Contains,
    ExactMatch,
    JsonValid,
    LLMJudge,
    Metric,
    NumericClose,
    OutcomeIs,
    Regex,
    SemanticSimilarity,
    ToolCalled,
    ToolSequence,
)
from aaa.eval.online import OnlineEvaluator
from aaa.eval.redteam import NoLeak, RedTeamReport, Resisted, attack_suite, evaluate_guard
from aaa.eval.report import EvalReport, Regression
from aaa.eval.result import CaseResult
from aaa.eval.simulate import SessionEvalRunner, SimulatedUser
from aaa.eval.suite import EvalSuite
from aaa.score import Score

__all__ = [
    "Budgeted",
    "CallableMetric",
    "Case",
    "CaseResult",
    "Contains",
    "Dataset",
    "EvalReport",
    "EvalSuite",
    "ExactMatch",
    "JsonValid",
    "LLMJudge",
    "Metric",
    "NoLeak",
    "NumericClose",
    "OnlineEvaluator",
    "OutcomeIs",
    "RedTeamReport",
    "Regex",
    "Regression",
    "Resisted",
    "Score",
    "SemanticSimilarity",
    "SessionEvalRunner",
    "SimulatedUser",
    "ToolCalled",
    "ToolSequence",
    "attack_suite",
    "evaluate_guard",
]
