"""Evals: datasets, metrics, the suite (real runs), reports, and online scoring."""

import json

import pytest

from aaa.algos import ToolLoop
from aaa.eval import (
    Budgeted,
    CallableMetric,
    Case,
    CaseResult,
    Contains,
    Dataset,
    EvalSuite,
    ExactMatch,
    JsonValid,
    LLMJudge,
    NumericClose,
    OnlineEvaluator,
    OutcomeIs,
    Regex,
    SemanticSimilarity,
    ToolCalled,
    ToolSequence,
)
from aaa.eval.suite import EVAL_META
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.observe import InMemorySink
from aaa.runtime import Budget, Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def add(a: int, b: int) -> int:
    """add numbers"""
    return a + b


def agent_script(req):
    """Adds the two numbers in the task using the tool, then answers."""
    last = req.messages[-1]
    if last.role == "user":
        nums = [int(x) for x in (last.content or "").split() if x.isdigit()]
        if "boom" in (last.content or ""):
            return text("I refuse.")
        return tool_call("add", a=nums[0], b=nums[1])
    return text(f"The answer is {last.content}")


def make():
    eng = Engine()
    LLMHandler(FakeLLMClient(agent_script)).attach(eng)
    reg = ToolRegistry([add])
    ToolHandler(reg).attach(eng)
    algo = ToolLoop(reg)
    eng.register(algo)
    return eng, algo


DATA = Dataset.from_list(
    [
        {"id": "a", "input": "add 2 3", "expected": "5", "tags": ["math"]},
        {"id": "b", "input": "add 10 32", "expected": "42"},
        {"id": "c", "input": "boom", "expected": "7"},
    ]
)


def result(text_="x", expected=None, calls=()):
    from datetime import UTC, datetime

    from aaa.trace import A, Span

    now = datetime.now(UTC)
    spans = [
        Span("t", f"s{i}", None, "execute_tool", "tool", now, now, attributes={A.TOOL_NAME: c})
        for i, c in enumerate(calls)
    ]
    return CaseResult(Case(id="c", input={}, expected=expected), None, text_, spans=spans)


# -- dataset ----------------------------------------------------------------------------------
def test_dataset_loaders_and_duplicate_ids(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        '{"input": "q1", "expected": "a1", "tags": ["x"]}\n{"id": "z", "input": {"task": "q2"}}\n'
    )
    ds = Dataset.from_jsonl(p)
    assert [c.id for c in ds] == ["case-1", "z"] and ds.cases[0].input == {"task": "q1"}
    assert len(ds.where(tag="x")) == 1
    ds.to_jsonl(tmp_path / "o.jsonl")
    assert [c.id for c in Dataset.from_jsonl(tmp_path / "o.jsonl")] == ["case-1", "z"]
    c = tmp_path / "d.csv"
    c.write_text("id,input,expected\nk,hello,world\n")
    assert Dataset.from_csv(c).cases[0].expected == "world"
    with pytest.raises(ValueError, match="duplicate"):
        Dataset.from_list([{"id": "a", "input": "x"}, {"id": "a", "input": "y"}])


# -- metrics ----------------------------------------------------------------------------------
async def test_deterministic_metrics():
    r = result("The answer is 42.", expected="42", calls=["search", "add"])
    assert (await ExactMatch().score(result(" Hello  World ", "hello world"))).value is True
    assert (await ExactMatch().score(r)).value is False
    assert (await Contains().score(r)).value is True
    assert (await Contains(["answer", "nope"]).score(r)).value is False
    assert (await Regex(r"\d+").score(r)).value is True
    assert (await NumericClose().score(r)).value is True
    assert (await NumericClose().score(result("about 41", "42"))).value is False
    assert (await JsonValid().score(result('{"a": 1}'))).value is True
    assert (await JsonValid().score(result("nope"))).value is False
    assert (await ToolCalled("add").score(r)).value is True
    assert (await ToolCalled("add", forbidden=["search"]).score(r)).value is False
    assert (await ToolSequence("search", "add").score(r)).value is True
    assert (await ToolSequence("add", "search").score(r)).value is False


async def test_budget_metric_scores_fall_past_the_limit():
    r = result()
    r.steps, r.cost = 4, 0.01
    scores = {s.name: s for s in await Budgeted(steps=4, cost=0.005).score(r)}
    assert scores["within_budget.steps"].value == 1.0
    assert (
        scores["within_budget.cost"].value == 0.0 and scores["within_budget.cost"].passed is False
    )


async def test_callable_metric_accepts_several_return_shapes():
    assert (await CallableMetric(lambda r: True, "m").score(result())).value is True
    assert (await CallableMetric(lambda r: (0.2, "meh"), "m").score(result())).reason == "meh"

    async def amet(r):
        return 0.9

    assert (await CallableMetric(amet).score(result())).value == 0.9


async def test_semantic_similarity_uses_the_embedder():
    class E:
        async def embed(self, texts):
            return [[1.0, 0.0] if "cat" in t else [0.0, 1.0] for t in texts]

    s = await SemanticSimilarity(E()).score(result("a cat", "the cat"))
    assert s.value == 1.0
    assert (await SemanticSimilarity(E()).score(result("a cat", "a dog"))).passed is False


async def test_llm_judge_parses_scores_and_reports_cost():
    from aaa.llm.types import LLMResponse, Message, Usage

    seen = []

    def judge(req):
        seen.append(req)
        return LLMResponse(
            message=Message(role="assistant", content='noise {"score": 0.75, "reason": "mostly"} '),
            usage=Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            cost=0.002,
            model="j",
        )

    s = await LLMJudge(FakeLLMClient(judge), "Be polite").score(result("hi", "hello"))
    assert (
        s.value == 0.75
        and s.reason == "mostly"
        and s.cost == 0.002
        and s.usage["total_tokens"] == 10
    )
    assert "Be polite" in seen[0].messages[1].content and seen[0].response_schema

    bad = await LLMJudge(FakeLLMClient(lambda r: text("no json here")), "x").score(result())
    assert bad.value == 0.0 and bad.metadata["error"]


# -- suite ------------------------------------------------------------------------------------
async def test_suite_runs_real_cases_and_reads_the_trajectory():
    eng, algo = make()
    suite = EvalSuite(DATA, algo, [Contains(), ToolCalled("add"), OutcomeIs()], concurrency=2)
    async with eng:
        report = await suite.run(eng)
    by = {r.case.id: r for r in report.results}
    assert by["a"].tool_calls == ["add"] and by["a"].steps == 3 and by["a"].tokens >= 0
    assert by["a"].text.startswith("The answer is")
    assert all(r.run_id for r in report.results)
    failed = {r.case.id for r in report.failures}
    assert failed == {"c"}  # "boom": never called the tool and answered wrongly
    s = report.summary()
    assert s["contains"].n == 3 and round(s["contains"].pass_rate, 2) == 0.67
    assert round(report.pass_rate, 2) == 0.67


async def test_eval_runs_are_marked_and_repeats_multiply():
    eng, algo = make()
    async with eng:
        report = await EvalSuite(DATA.where(tag="math"), algo, [Contains()], repeats=3).run(eng)
        assert len(report.results) == 3 and {r.repeat for r in report.results} == {0, 1, 2}
        started = (await eng.store.read(report.results[0].run_id))[0]
        assert started.meta[EVAL_META] is True


async def test_a_metric_that_raises_is_a_failed_score_not_a_crashed_suite():
    eng, algo = make()

    def bad(r):
        raise RuntimeError("kaput")

    async with eng:
        report = await EvalSuite(
            DATA.where(tag="math"), algo, [CallableMetric(bad, "bad"), Contains()]
        ).run(eng)
    r = report.results[0]
    assert r.scores[0].passed is False and "kaput" in (r.scores[0].reason or "")
    assert report.pass_rate == 0.0


async def test_a_case_that_cannot_start_is_an_error():
    eng, algo = make()

    def boom(case):
        raise ValueError("no input for you")

    async with eng:
        report = await EvalSuite(DATA, algo, [Contains()], make_input=boom).run(eng)
    assert all(r.error and "no input" in r.error for r in report.results)
    assert report.pass_rate == 0.0 and all(r.scores[0].passed is False for r in report.results)


async def test_per_case_budget_stops_a_runaway_case():
    eng, algo = make()
    async with eng:
        report = await EvalSuite(
            DATA.where(tag="math"), algo, [OutcomeIs()], budget=Budget(max_steps=1)
        ).run(eng)
    assert report.results[0].outcome.kind.value == "budget_exceeded"
    assert report.results[0].scores[0].passed is False


# -- report -----------------------------------------------------------------------------------
async def test_report_outputs_baseline_and_gate(tmp_path):
    eng, algo = make()
    async with eng:
        report = await EvalSuite(DATA, algo, [Contains(), ToolCalled("add")]).run(eng)
    d = report.to_dict()
    assert d["cases"] == 3 and set(d["metrics"]) == {"contains", "tool_called"}
    assert "| contains |" in report.to_markdown()
    junit = report.to_junit()
    assert 'tests="3"' in junit and 'failures="1"' in junit and "<failure" in junit

    path = tmp_path / "base.json"
    report.save(path)
    assert report.regressions(path) == []
    better = json.loads(path.read_text())
    better["metrics"]["contains"]["mean"] = 1.0
    better["pass_rate"] = 1.0
    names = {x.metric for x in report.regressions(better)}
    assert names == {"contains", "pass_rate"}

    with pytest.raises(AssertionError, match="pass rate 66.7% < 90.0%") as ei:
        report.assert_ok(min_pass_rate=0.9)
    assert "c:" in str(ei.value)
    report.assert_ok(min_pass_rate=0.6, baseline=path)
    with pytest.raises(AssertionError, match="regression"):
        report.assert_ok(min_pass_rate=0.0, baseline=better)


# -- online -----------------------------------------------------------------------------------
async def test_online_evaluator_scores_real_runs_not_eval_runs_and_records_scores():
    from aaa.trace import trace_id_of

    eng, algo = make()
    sink = InMemorySink()
    async with eng:
        async with OnlineEvaluator(
            eng, [Contains(["answer"], name="has_answer")], sinks=[sink]
        ) as ev:
            handle = await eng.start(algo, {"task": "add 1 2"})
            await handle.result()
            await EvalSuite(DATA.where(tag="math"), algo, [Contains()]).run(
                eng
            )  # an eval run: must be skipped
        assert ev.evaluated == 1
        events = await eng.store.read(handle.run_id)
    scored = [e for e in events if e.type == "aaa.eval.score"]
    assert (
        len(scored) == 1
        and scored[0].data["name"] == "has_answer"
        and scored[0].data["value"] is True
    )
    assert [(t, s.name) for t, _, s in sink.scores] == [(trace_id_of(handle.run_id), "has_answer")]


async def test_online_evaluator_sampling_and_cost_cap():
    from aaa.llm.types import LLMResponse, Message, Usage

    def judge(req):
        return LLMResponse(
            message=Message(role="assistant", content='{"score": 1, "reason": "ok"}'),
            usage=Usage(),
            cost=0.5,
            model="j",
        )

    eng, algo = make()
    async with eng:
        async with OnlineEvaluator(eng, [LLMJudge(FakeLLMClient(judge), "r")], max_cost=0.5) as ev:
            for _ in range(3):
                await eng.run(algo, {"task": "add 1 2"})
        assert ev.evaluated == 1 and ev.skipped_over_budget == 2 and ev.spent == 0.5

    eng2, algo2 = make()
    async with eng2:
        async with OnlineEvaluator(eng2, [Contains(["x"])], sample=0.0) as ev2:
            await eng2.run(algo2, {"task": "add 1 2"})
        assert ev2.evaluated == 0


async def test_a_failing_metric_or_sink_never_fails_the_scored_run():
    eng, algo = make()

    class Bad:
        async def score(self, trace_id, scores, *, span_id=None):
            raise RuntimeError("sink down")

    async with eng:
        async with OnlineEvaluator(eng, [Contains(["answer"])], sinks=[Bad()]) as ev:
            out = await eng.run(algo, {"task": "add 1 2"})
        assert out.ok and ev.evaluated == 1
