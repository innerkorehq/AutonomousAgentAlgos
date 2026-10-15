"""Scores reaching Langfuse/Phoenix/LangSmith, datasets from traces, red-team suites, simulated users."""

import json
import os
import uuid

import httpx
import pytest

from aaa.algos import ToolLoop
from aaa.core import OutcomeKind
from aaa.eval import (
    Case,
    Contains,
    Dataset,
    NoLeak,
    OnlineEvaluator,
    Resisted,
    SessionEvalRunner,
    SimulatedUser,
    attack_suite,
    evaluate_guard,
    redteam,
)
from aaa.guard import RegexGuard
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.observe import InMemorySink
from aaa.observe.base import ExportError
from aaa.observe.scoring import (
    langfuse_body,
    langfuse_score,
    langsmith_body,
    langsmith_run_id,
    langsmith_score,
    phoenix_item,
    phoenix_score,
    stable_id,
)
from aaa.runtime import Budget, Engine
from aaa.score import Score
from aaa.session import SessionManager
from aaa.tools import ToolRegistry
from aaa.trace import span_id_of, trace_id_of

TID, SID = trace_id_of("run-1"), span_id_of("run", "run-1")


def mock(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def no_sleep(_):
    return None


# ===================================================================== score delivery
def test_langfuse_bodies_encode_each_kind_of_score():
    num = langfuse_body(
        TID, SID, Score(name="acc", value=0.8, reason="mostly right"), environment="prod"
    )
    assert num == {
        "id": stable_id(TID, SID, "acc"),
        "traceId": TID,
        "observationId": SID,
        "name": "acc",
        "value": 0.8,
        "dataType": "NUMERIC",
        "comment": "mostly right",
        "environment": "prod",
    }
    b = langfuse_body(TID, None, Score(name="ok", value=True))
    assert b["value"] == 1 and b["dataType"] == "BOOLEAN" and "observationId" not in b
    c = langfuse_body(TID, None, Score(name="tone", value="polite"))
    assert c["value"] == "polite" and c["dataType"] == "CATEGORICAL"
    assert (
        stable_id(TID, SID, "acc") == num["id"] != stable_id(TID, SID, "other")
    )  # repeat = upsert


async def test_langfuse_sends_one_object_per_score_with_basic_auth_and_adapts_to_list_servers():
    seen = []

    def handler(req):
        seen.append((req.url.path, req.headers["authorization"], json.loads(req.content)))
        return httpx.Response(200, json={"id": "x"})

    async with mock(handler) as c:
        await langfuse_score(
            c,
            "https://lf.example",
            "QUJD",
            TID,
            [Score(name="a", value=1.0), Score(name="b", value=False)],
            SID,
            None,
        )
    assert [s[0] for s in seen] == ["/api/public/scores"] * 2 and seen[0][1] == "Basic QUJD"
    assert isinstance(seen[0][2], dict) and seen[1][2]["dataType"] == "BOOLEAN"

    calls = []

    def list_only(req):
        body = json.loads(req.content)
        calls.append(body)
        if isinstance(body, dict):
            return httpx.Response(400, json={"message": "Expected array, received object"})
        return httpx.Response(200, json=[{"id": "x"}])

    async with mock(list_only) as c:
        await langfuse_score(
            c, "https://lf.example", "QUJD", TID, [Score(name="a", value=1.0)], None, None
        )
    assert isinstance(calls[0], dict) and isinstance(calls[1], list)


async def test_delivery_retries_transient_errors_and_reports_the_rest():
    n = {"i": 0}

    def flaky(req):
        n["i"] += 1
        return httpx.Response(503) if n["i"] < 3 else httpx.Response(200, json={})

    async with mock(flaky) as c:
        from aaa.observe.scoring import post_json

        await post_json(c, "https://x/y", {}, sleep=no_sleep)
    assert n["i"] == 3

    async with mock(lambda r: httpx.Response(401, text="nope")) as c:
        with pytest.raises(ExportError, match="HTTP 401"):
            await langfuse_score(
                c, "https://lf.example", "x", TID, [Score(name="a", value=1.0)], None, None
            )

    def boom(req):
        raise httpx.ConnectError("secret-host.internal refused")

    async with mock(boom) as c:
        with pytest.raises(ExportError) as e:
            await langfuse_score(
                c,
                "https://lf.example/p?token=abc",
                "x",
                TID,
                [Score(name="a", value=1.0)],
                None,
                None,
            )
    assert "token=abc" not in str(e.value)  # a query string never reaches the log


def test_phoenix_items():
    a = phoenix_item(TID, SID, Score(name="ok", value=True, reason="fine"))
    assert a == {
        "name": "ok",
        "annotator_kind": "CODE",
        "result": {"score": 1.0, "label": "pass", "explanation": "fine"},
        "identifier": "aaa:ok",
        "span_id": SID,
    }
    b = phoenix_item(TID, None, Score(name="tone", value="polite", usage={"total_tokens": 5}))
    assert b["trace_id"] == TID and "span_id" not in b and b["annotator_kind"] == "LLM"
    assert b["result"] == {"label": "polite"}


async def test_phoenix_waits_for_the_span_to_arrive():
    seen = []

    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(404) if len(seen) < 3 else httpx.Response(200, json={"data": []})

    async with mock(handler) as c:
        await phoenix_score(
            c,
            "http://phx:6006/",
            {"authorization": "Bearer k"},
            TID,
            [Score(name="a", value=1.0)],
            SID,
            sleep=no_sleep,
        )
        await phoenix_score(
            c, "http://phx:6006", {}, TID, [Score(name="a", value=1.0)], None, sleep=no_sleep
        )
    assert seen == ["/v1/span_annotations"] * 3 + ["/v1/trace_annotations"]


def test_langsmith_run_ids_come_from_span_ids():
    assert (
        langsmith_run_id("9222d40b53eb8eec") == "00000000-0000-0000-9222-d40b53eb8eec"
    )  # LangSmith's documented mapping
    uuid.UUID(langsmith_run_id(SID))
    body = langsmith_body(SID, Score(name="correct", value=0.5, reason="half"))
    assert (
        body["run_id"] == langsmith_run_id(SID)
        and body["key"] == "correct"
        and body["score"] == 0.5
    )
    assert (
        body["comment"] == "half"
        and body["id"] == langsmith_body(SID, Score(name="correct", value=0.1))["id"]
    )


async def test_langsmith_needs_a_span_and_tolerates_a_repeat():
    async with mock(lambda r: httpx.Response(409)) as c:
        await langsmith_score(
            c,
            "https://api.smith.langchain.com",
            {"x-api-key": "k"},
            [Score(name="a", value=1.0)],
            SID,
        )
        with pytest.raises(ExportError, match="span"):
            await langsmith_score(c, "https://x", {}, [Score(name="a", value=1.0)], None)


otlp = pytest.importorskip.__name__  # (sink-level tests below need the OpenTelemetry exporter)


async def test_the_sinks_deliver_through_their_score_method():
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    from aaa.observe.otel import LangfuseSink, LangSmithSink, PhoenixSink

    seen = []

    def handler(req):
        seen.append((str(req.url), json.loads(req.content)))
        return httpx.Response(200, json={"data": []})

    async with mock(handler) as c:
        lf = LangfuseSink("pk", "sk", "https://lf.example", environment="prod", http=c)
        ph = PhoenixSink("http://phx:6006/v1/traces", http=c, api_key="k")
        ls = LangSmithSink("key", http=c)
        for sink in (lf, ph, ls):
            await sink.score(TID, [Score(name="q", value=0.7)], span_id=SID)
    assert (
        seen[0][0] == "https://lf.example/api/public/scores" and seen[0][1]["environment"] == "prod"
    )
    assert seen[1][0].startswith("http://phx:6006/v1/span_annotations")
    assert seen[2][0] == "https://api.smith.langchain.com/feedback"


async def test_generic_otel_sinks_keep_scores_as_evaluator_spans():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from aaa.observe.otel import OTelSink

    exporter = InMemorySpanExporter()
    sink = OTelSink(exporter)
    await sink.score(TID, [Score(name="q", value=0.7, reason="r")], span_id=SID)
    await sink.score(TID, [Score(name="q", value=0.7, reason="r")], span_id=SID)
    spans = exporter.get_finished_spans()
    assert (
        len(spans) == 2 and spans[0].context.span_id == spans[1].context.span_id
    )  # same id: a repeat is the same span
    assert spans[0].name == "evaluate q" and spans[0].parent.span_id == int(SID, 16)


async def test_online_evaluator_tells_the_sink_which_span_the_score_is_about():
    sink = InMemorySink()
    async with Engine() as eng:
        LLMHandler(FakeLLMClient(lambda r: text("Paris"))).attach(eng)
        async with OnlineEvaluator(eng, [Contains("Paris")], sinks=[sink]):
            await eng.run(ToolLoop(ToolRegistry([])), {"task": "capital?"}, run_id="run-1")
    assert [(t, sp, s.name, s.value) for t, sp, s in sink.scores] == [(TID, SID, "contains", True)]


# ----- the real backends (CI starts them; locally set the variables to try) -----------------------
@pytest.mark.skipif(not os.environ.get("AAA_TEST_PHOENIX_URL"), reason="needs a Phoenix server")
async def test_phoenix_for_real():
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    from aaa.observe import Observer
    from aaa.observe.otel import PhoenixSink

    base = os.environ["AAA_TEST_PHOENIX_URL"].rstrip("/")
    rid = f"phx-{uuid.uuid4().hex[:8]}"
    sink = PhoenixSink(f"{base}/v1/traces", project="aaa-tests")
    async with Engine() as eng:
        LLMHandler(FakeLLMClient(lambda r: text("Paris"))).attach(eng)
        async with Observer(eng, sink):  # noqa: SIM117
            async with OnlineEvaluator(eng, [Contains("Paris")], sinks=[sink]):
                await eng.run(ToolLoop(ToolRegistry([])), {"task": "capital?"}, run_id=rid)
    sid = span_id_of("run", rid)
    await sink.score(
        trace_id_of(rid), [Score(name="contains", value=0.25, reason="changed")], span_id=sid
    )  # an update
    await sink.score(trace_id_of(rid), [Score(name="tone", value="polite")], span_id=sid)
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{base}/v1/projects/aaa-tests/span_annotations", params={"span_ids": sid})
    got = {a["name"]: a["result"] for a in r.json()["data"]}
    assert set(got) == {"contains", "tone"}  # one annotation per name: the repeat updated in place
    assert got["contains"]["score"] == 0.25 and got["tone"]["label"] == "polite"
    await sink.aclose()


@pytest.mark.skipif(not os.environ.get("AAA_TEST_LANGFUSE_URL"), reason="needs a Langfuse server")
async def test_langfuse_for_real():
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    from aaa.observe.otel import LangfuseSink

    host = os.environ["AAA_TEST_LANGFUSE_URL"].rstrip("/")
    sink = LangfuseSink("pk-lf-test", "sk-lf-test", host)
    tid = trace_id_of(f"lf-{uuid.uuid4().hex[:8]}")
    await sink.score(
        tid,
        [
            Score(name="acc", value=0.9),
            Score(name="ok", value=True),
            Score(name="tone", value="polite", reason="because"),
        ],
    )
    await sink.score(tid, [Score(name="acc", value=0.2)])  # same id: updated, not duplicated
    async with httpx.AsyncClient() as c:
        r = await c.get(
            f"{host}/api/public/scores", params={"traceId": tid}, auth=("pk-lf-test", "sk-lf-test")
        )
    mine = [x for x in r.json()["data"] if x["traceId"] == tid]
    rows = {x["name"]: x for x in mine}
    assert len(mine) == 3 and rows["acc"]["value"] == 0.2
    assert (
        rows["ok"]["dataType"] == "BOOLEAN"
        and rows["tone"]["stringValue"] == "polite"
        and rows["tone"]["comment"] == "because"
    )
    await sink.aclose()


# ===================================================================== datasets from traces
async def seeded():
    eng = Engine()
    LLMHandler(FakeLLMClient(lambda r: text("answer"))).attach(eng)
    return eng


async def test_from_traces_selects_scrubs_and_skips_what_it_must():
    async with await seeded() as eng:
        algo = ToolLoop(ToolRegistry([]))
        scope = eng.tenant("acme")
        await scope.run(algo, {"task": "my email is ann@example.com, what is 2+2?"}, run_id="r1")
        await scope.run(algo, {"task": "second"}, run_id="r2")
        await eng.tenant("globex").run(algo, {"task": "other tenant"}, run_id="r3")
        await eng.run(algo, {"task": "an eval"}, run_id="r4", meta={"aaa_eval": True})
        erased = await scope.run(algo, {"task": "to be erased"}, run_id="r5")
        assert erased.ok
        await scope.erase("r5")
        handle = await scope.start(
            ToolLoop(ToolRegistry([]), name="slow"), {"task": "open"}, run_id="r6"
        )  # never answered below
        await handle.cancel()
        await handle.result()
        ds = await Dataset.from_traces(eng, tenant="acme")
        assert sorted(c.id for c in ds) == [
            "acme/r1",
            "acme/r2",
        ]  # not the other tenant, the eval, the erased or the cancelled run
        first = next(c for c in ds if c.id == "acme/r1")
        assert "ann@example.com" not in json.dumps(first.input) and "2+2" in first.input["task"]
        assert first.tags == ["tool_loop", "success"] and "tenant" not in first.metadata
        assert first.expected is None
        raw = await Dataset.from_traces(eng, tenant="acme", scrub=False)
        assert "ann@example.com" in next(c for c in raw if c.id == "acme/r1").input["task"]
        golden = await Dataset.from_traces(
            eng, tenant="acme", expected="output", limit=1, include_tenant=True
        )
        assert [c.id for c in golden] == ["acme/r2"]  # the newest one
        assert golden.cases[0].expected == "answer" and golden.cases[0].metadata["tenant"] == "acme"
        cancelled = await Dataset.from_traces(eng, tenant="acme", outcomes=None)
        assert "acme/r6" in {c.id for c in cancelled}
        only = await Dataset.from_traces(eng, filter=lambda info: info.stream.endswith("r2"))
        assert [c.id for c in only] == ["acme/r2"]


async def test_from_traces_skips_child_runs_and_sessions():
    async with await seeded() as eng:
        mgr = SessionManager(eng, ToolLoop(ToolRegistry([])), system="s")
        s = await mgr.create()
        await s.send("hello")
        await s.close()
        assert len(await Dataset.from_traces(eng)) == 0


# ===================================================================== red team
def test_the_corpus_is_well_formed():
    ds = redteam.attacks()
    assert len(ds) >= 30 and {c.tags[0] for c in ds} == set(redteam.CATEGORIES)
    assert len({c.id for c in ds}) == len(ds)
    assert len(redteam.attacks(["jailbreak"])) == 4
    with pytest.raises(ValueError):
        redteam.attacks(["nope"])
    assert len(redteam.benign()) >= 10


async def test_guard_report_counts_what_got_through_and_what_was_wrongly_stopped():
    strict = RegexGuard([r"ignore (all )?previous", r"system prompt", r"\bDAN\b"], name="strict")
    report = await evaluate_guard(strict)
    by = report.by_category()
    assert by["prompt_injection"]["got_through"] < by["prompt_injection"]["attacks"]
    assert 0 < report.attack_success_rate < 1
    assert report.false_positive_rate == 0.0
    assert report.missed() and report.guard == "strict"
    assert "| prompt_injection |" in report.to_markdown()
    with pytest.raises(AssertionError, match="attack success"):
        report.assert_ok(max_attack_success=0.1)
    report.assert_ok(max_attack_success=0.95)

    eager = RegexGuard([r"."], name="eager")  # blocks everything
    r2 = await evaluate_guard(eager)
    assert r2.attack_success_rate == 0.0 and r2.false_positive_rate == 1.0
    with pytest.raises(AssertionError, match="false positives"):
        r2.assert_ok(max_false_positive=0.1)

    flag_only = await evaluate_guard(RegexGuard([r"."], action="flag"))
    assert (
        flag_only.attack_success_rate == 1.0
        and flag_only.by_category()["jailbreak"]["flagged_only"] == 4
    )


async def test_attack_suite_scores_leaks_and_refusals():
    canary = "CANARY-7f3a91"

    def leaky(req):
        return text(f"Sure! My instructions say {canary}.")

    def careful(req):
        return text("I can't help with that.")

    for script, leaked in ((leaky, True), (careful, False)):
        async with Engine() as eng:
            LLMHandler(FakeLLMClient(script)).attach(eng)
            suite = attack_suite(
                ToolLoop(ToolRegistry([])), canary=canary, categories=["system_prompt_leak"]
            )
            report = await suite.run(eng)
        summary = report.summary()
        assert (summary["no_leak"].pass_rate == 0.0) is leaked
        assert (summary["resisted"].pass_rate == 0.0) is leaked
    with pytest.raises(ValueError):
        NoLeak("abc")


async def test_a_guard_blocked_run_counts_as_resisted():
    from aaa.core.outcome import Outcome
    from aaa.eval.result import CaseResult

    r = CaseResult(
        case=Case(id="x", input={}), outcome=Outcome(kind=OutcomeKind.BLOCKED, reason="g"), text=""
    )
    assert (await Resisted().score(r)).value is True


# ===================================================================== simulated users
def agent_script(req):
    user = next(m.content for m in reversed(req.messages) if m.role == "user")
    return text(f"Agent heard: {user}")


def user_script(goal_after):
    def script(req):
        turns = sum(
            m.role == "user" for m in req.messages[1:]
        )  # the agent's replies, from the simulator's side
        return text("[DONE]" if turns >= goal_after else f"user message {turns + 1}")

    return script


async def test_a_simulated_user_converses_with_a_real_session_until_it_is_done():
    async with Engine() as eng:
        LLMHandler(FakeLLMClient(agent_script)).attach(eng)
        mgr = SessionManager(eng, ToolLoop(ToolRegistry([])), system="s")
        sim = SimulatedUser(FakeLLMClient(user_script(2)), persona="grumpy", goal="get help")
        runner = SessionEvalRunner(mgr, sim, [Contains("user message 2")], max_turns=6)
        ds = Dataset.from_list(
            [{"id": "c1", "input": {"opening": "Hello there"}}, {"id": "c2", "input": {}}]
        )
        report = await runner.run(eng, ds)
        c1 = next(r for r in report.results if r.case.id == "c1")
        assert [u for u, _ in c1.turns] == ["Hello there", "user message 2"][: len(c1.turns)]
        assert c1.ok and "Agent heard: Hello there" in c1.text and c1.steps >= 2 and c1.tokens >= 0
        assert any(
            s.kind == "llm" for s in c1.spans
        )  # the agent's trajectory is there for trajectory metrics
        assert c1.scores[0].name == "contains"
        # the conversations are stamped as evals and closed, so online evaluators ignore them
        runs = await eng.store.run_index()
        sessions = [
            r for r in runs if str(r.started.data.get("algo", "")).startswith("aaa.session:")
        ]
        assert sessions and all(r.started.meta.get("aaa_eval") and r.completed for r in sessions)


async def test_conversations_are_bounded_and_failures_end_them():
    async with Engine() as eng:
        LLMHandler(FakeLLMClient(agent_script)).attach(eng)
        mgr = SessionManager(eng, ToolLoop(ToolRegistry([])), system="s")
        endless = SimulatedUser(FakeLLMClient(lambda r: text("again")), goal="x")
        report = await SessionEvalRunner(mgr, endless, [], max_turns=3).run(
            eng, Dataset.from_list([{"input": {}}])
        )
        assert len(report.results[0].turns) == 3
        costly = SimulatedUser(
            FakeLLMClient(
                lambda r: __import__("aaa.llm.types", fromlist=["x"]).LLMResponse(
                    message=__import__("aaa.llm.types", fromlist=["x"]).Message(
                        role="assistant", content="more"
                    ),
                    cost=0.6,
                )
            ),
            max_cost=1.0,
        )
        r2 = await SessionEvalRunner(mgr, costly, [], max_turns=10).run(
            eng, Dataset.from_list([{"input": {}}])
        )
        assert len(r2.results[0].turns) == 2 and costly.spent == pytest.approx(1.2)
    with pytest.raises(ValueError):
        SessionEvalRunner(mgr, endless, [], max_turns=0)

    from aaa.llm.errors import FatalError

    def failing(req):
        raise FatalError("down")

    async with Engine() as eng:
        LLMHandler(FakeLLMClient(failing)).attach(eng)
        mgr = SessionManager(eng, ToolLoop(ToolRegistry([])), system="s")
        sim = SimulatedUser(FakeLLMClient(lambda r: text("hi")), goal="x")
        res = (
            await SessionEvalRunner(mgr, sim, [], max_turns=4).run(
                eng, Dataset.from_list([{"input": {}}])
            )
        ).results[0]
        assert len(res.turns) == 1 and res.outcome is not None and not res.outcome.ok


async def test_a_case_can_set_its_own_persona_and_goal():
    seen = []

    def user(req):
        seen.append(req.messages[0].content)
        return text("[DONE]")

    async with Engine() as eng:
        LLMHandler(FakeLLMClient(agent_script)).attach(eng)
        mgr = SessionManager(eng, ToolLoop(ToolRegistry([])), system="s")
        sim = SimulatedUser(
            FakeLLMClient(user), persona="default persona", goal="default goal", opening="hi"
        )
        ds = Dataset.from_list([{"input": {"persona": "a lawyer", "goal": "cancel the plan"}}])
        await SessionEvalRunner(mgr, sim, [], max_turns=2).run(eng, ds)
    assert "a lawyer" in seen[0] and "cancel the plan" in seen[0] and "default" not in seen[0]
    _ = Budget
