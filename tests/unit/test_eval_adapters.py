"""Third-party eval libraries run their REAL flows here, judged by a fake model behind our LLMClient.
Each class is skipped where its library is not installed (they are heavy; CI installs them in their own legs)."""

import pytest

from aaa.eval import Case, CaseResult
from aaa.llm import FakeLLMClient
from aaa.llm.fake import tool_call
from helpers.eval_stubs import SchemaJudge


def result(text="Paris is the capital.", expected="Paris", **meta):
    return CaseResult(
        Case(id="1", input={"task": "capital of France?"}, expected=expected, metadata=meta),
        None,
        text,
    )


class TestAutoevals:
    @pytest.fixture(autouse=True)
    def _need(self):
        pytest.importorskip("autoevals")

    async def test_heuristic_scorers_need_no_model(self):
        from autoevals import ExactMatch, Levenshtein

        from aaa.eval.autoevals import AutoevalsMetric

        assert (await AutoevalsMetric(Levenshtein()).score(result("Paris"))).value == 1.0
        s = await AutoevalsMetric(ExactMatch(), name="em").score(result("Paris", "Rome"))
        assert s.name == "em" and s.value == 0.0

    @pytest.mark.parametrize(
        "model", ["judge-model", "gpt-5-mini"]
    )  # gpt-5* makes autoevals use the Responses API
    async def test_llm_scorer_is_judged_by_our_client_on_both_openai_apis(self, model):
        from autoevals import Factuality

        from aaa.eval.autoevals import AutoevalsMetric, openai_shim

        seen = []

        def judge(req):
            seen.append(req)
            return tool_call("select_choice", reasoning="same", choice="A")

        shim = openai_shim(FakeLLMClient(judge))
        s = await AutoevalsMetric(Factuality(client=shim, model=model)).score(result())
        assert 0.0 <= s.value <= 1.0 and s.metadata["choice"] == "A"
        assert shim.meter.calls == 1
        assert seen[0].model == model and [t.name for t in seen[0].tools] == ["select_choice"]
        assert "Paris" in seen[0].messages[-1].content

    async def test_embedding_scorers_use_our_embed(self):
        from autoevals import EmbeddingSimilarity

        from aaa.eval.autoevals import AutoevalsMetric, openai_shim

        shim = openai_shim(FakeLLMClient(lambda r: None))
        s = await AutoevalsMetric(EmbeddingSimilarity(client=shim)).score(result("Paris", "Paris"))
        assert s.value > 0.99


class TestDeepEval:
    @pytest.fixture(autouse=True)
    def _need(self):
        pytest.importorskip("deepeval")

    async def test_metrics_run_with_our_judge(self):
        from deepeval.metrics import AnswerRelevancyMetric

        from aaa.eval.deepeval import DeepEvalJudge, DeepEvalMetric

        stub = SchemaJudge()
        judge = DeepEvalJudge(FakeLLMClient(stub), model="jm")
        s = await DeepEvalMetric(AnswerRelevancyMetric(model=judge, threshold=0.5)).score(result())
        assert (
            s.name == "Answer Relevancy" and 0.0 <= s.value <= 1.0 and s.metadata["success"] is True
        )
        assert judge.calls >= 2 and all(r.model == "jm" for r in stub.requests)
        assert any(
            "capital of France" in (m.content or "") for r in stub.requests for m in r.messages
        )

    async def test_the_test_case_carries_tools_and_context(self):
        from datetime import UTC, datetime

        from deepeval.metrics import AnswerRelevancyMetric

        from aaa.eval.deepeval import DeepEvalJudge, DeepEvalMetric
        from aaa.trace import A, Span

        now = datetime.now(UTC)
        r = result(retrieval_context=["Paris is in France."], expected_tools=["lookup"])
        r.spans = [
            Span(
                "t", "s", None, "execute_tool", "tool", now, now, attributes={A.TOOL_NAME: "lookup"}
            )
        ]
        tc = DeepEvalMetric(
            AnswerRelevancyMetric(model=DeepEvalJudge(FakeLLMClient(lambda q: None)))
        )._test_case(r)
        assert tc.input == "capital of France?" and tc.actual_output.startswith("Paris")
        assert tc.retrieval_context == ["Paris is in France."]
        assert [t.name for t in tc.tools_called] == ["lookup"] and [
            t.name for t in tc.expected_tools
        ] == ["lookup"]

    def test_sync_generate_works_even_inside_a_running_loop(self):
        import asyncio

        from aaa.eval.deepeval import DeepEvalJudge
        from aaa.llm.fake import text

        judge = DeepEvalJudge(FakeLLMClient(lambda q: text("hi")))

        async def inside():
            return judge.generate("hello")

        assert asyncio.run(inside()) == "hi"


class TestRagas:
    @pytest.fixture(autouse=True)
    def _need(self):
        pytest.importorskip("ragas.metrics.collections")

    async def test_metrics_map_case_fields_and_use_our_judge(self):
        from ragas.metrics.collections import ContextPrecision, ExactMatch, Faithfulness

        from aaa.eval.ragas import RagasJudge, RagasMetric

        r = result("Paris", retrieval_context=["Paris is in France."])
        assert (await RagasMetric(ExactMatch()).score(r)).value == 1.0
        stub = SchemaJudge()
        judge = RagasJudge(FakeLLMClient(stub), model="jm")
        f = await RagasMetric(Faithfulness(llm=judge)).score(r)
        assert f.name == "faithfulness" and 0.0 <= f.value <= 1.0 and judge.calls >= 1
        assert (await RagasMetric(ContextPrecision(llm=judge)).score(r)).value > 0.9
        assert all(q.model == "jm" for q in stub.requests)

    async def test_missing_inputs_are_a_clear_failure_not_a_guess(self):
        from ragas.metrics.collections import Faithfulness

        from aaa.eval.ragas import RagasJudge, RagasMetric

        m = RagasMetric(Faithfulness(llm=RagasJudge(FakeLLMClient(SchemaJudge()))))
        with pytest.raises(ValueError, match="no \\['retrieved_contexts'\\]"):
            await m.score(result())


class TestInspect:
    @pytest.fixture(autouse=True)
    def _need(self):
        pytest.importorskip("inspect_ai")

    def _setup(self):
        from contextlib import asynccontextmanager

        from aaa.algos import ToolLoop
        from aaa.llm import LLMHandler
        from aaa.llm.fake import text
        from aaa.runtime import Engine
        from aaa.tools import ToolHandler, ToolRegistry, tool

        @tool
        def add(a: int, b: int) -> int:
            """add"""
            return a + b

        def script(req):
            last = req.messages[-1]
            if last.role == "user":
                nums = [int(w) for w in (last.content or "").split() if w.isdigit()]
                return tool_call("add", a=nums[0], b=nums[1]) if len(nums) == 2 else text("no idea")
            return text(f"The answer is {last.content}")

        @asynccontextmanager
        async def agent():
            reg = ToolRegistry([add])
            algo = ToolLoop(reg)
            async with Engine() as eng:
                LLMHandler(FakeLLMClient(script)).attach(eng)
                ToolHandler(reg).attach(eng)
                eng.register(algo)
                yield eng, algo

        return agent

    def test_an_aaa_agent_is_the_solver_of_an_inspect_task(self, tmp_path):
        from inspect_ai import Task, eval
        from inspect_ai.dataset import Sample
        from inspect_ai.scorer import includes

        from aaa.eval.inspect import aaa_solver

        task = Task(
            dataset=[
                Sample(input="add 2 3", target="5", id="a"),
                Sample(input="add 40 2", target="42", id="b"),
                Sample(input="add things", target="7", id="c"),
            ],
            solver=aaa_solver(self._setup()),
            scorer=includes(),
        )
        [log] = eval(task, model="mockllm/model", display="none", log_dir=str(tmp_path))
        assert log.status == "success"
        verdicts = {s.id: s.scores["includes"].value for s in log.samples}
        assert verdicts == {"a": "C", "b": "C", "c": "I"}
        assert "The answer is 5" in log.samples[0].output.completion
