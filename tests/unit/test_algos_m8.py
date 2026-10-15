"""Reflexion, Debate, Swarm handoffs and Tree-of-thoughts: behaviour, bounds, budget cut-off and replay-stability."""

import asyncio
import json
import random

import pytest

from aaa.algos import Agent, Debate, Reflexion, Swarm, ToolLoop, TreeOfThoughts
from aaa.core import OutcomeKind
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Budget, Engine
from aaa.tools import ToolHandler, ToolRegistry, tool
from aaa.validate import ValidationHandler


class JitterClient(FakeLLMClient):
    """Completes calls in a seed-dependent order, to prove results never depend on which parallel call lands first."""

    def __init__(self, script, seed):
        super().__init__(script)
        self.rng = random.Random(seed)

    async def complete(self, request):
        await asyncio.sleep(self.rng.random() * 0.03)
        return await super().complete(request)


def sys_of(req):
    return next((m.content or "" for m in req.messages if m.role == "system"), "")


def user_of(req):
    return next((m.content or "" for m in reversed(req.messages) if m.role == "user"), "")


def wire(eng, llm, registry=None, validators=None):
    LLMHandler(llm).attach(eng)
    if registry is not None:
        ToolHandler(registry).attach(eng)
    if validators:
        ValidationHandler(validators).attach(eng)


def shape(events):
    """Order-insensitive fingerprint of a finished run (parallel completions may land in any order)."""
    return sorted((e.type, e.data.get("tag", "")) for e in events)


# ===================================================================== Reflexion
async def test_reflexion_retries_from_scratch_with_the_lessons():
    def script(req):
        s = sys_of(req)
        if "reviewing a failed attempt" in s:
            return text("Check the units before answering.")
        has_lesson = any("Check the units" in (m.content or "") for m in req.messages)
        return text("42 meters" if has_lesson else "42")

    llm = FakeLLMClient(script)
    async with Engine() as eng:
        wire(eng, llm, validators={"v": lambda c, ctx: ("meters" in c, "give the unit")})
        out = await eng.run(Reflexion(validator="v"), {"task": "How far?"})
    assert out.ok and out.value["candidate"] == "42 meters" and out.value["trials"] == 2
    assert out.value["reflections"] == ["Check the units before answering."]
    attempts = [r for r in llm.requests if "reviewing" not in sys_of(r)]
    assert len(attempts) == 2
    second = attempts[1]
    assert any("Lessons from earlier" in (m.content or "") for m in second.messages)
    assert not any(
        m.content == "42" and m.role == "assistant" for m in second.messages
    )  # a fresh attempt, not a patch


async def test_reflexion_gives_up_after_max_trials_and_returns_the_lessons():
    n = {"i": 0}

    def script(req):
        if "reviewing" in sys_of(req):
            n["i"] += 1
            return text(f"lesson {n['i']}")
        return text("wrong")

    async with Engine() as eng:
        wire(eng, FakeLLMClient(script), validators={"v": lambda c, ctx: (False, "still wrong")})
        out = await eng.run(Reflexion(validator="v", max_trials=3), {"task": "t"})
    assert out.kind is OutcomeKind.NO_SOLUTION and "still wrong" in out.reason
    assert out.value["trials"] == 3 and out.value["reflections"] == ["lesson 1", "lesson 2"]


async def test_reflexion_can_start_with_lessons_from_a_previous_run():
    llm = FakeLLMClient(lambda r: text("fine"))
    async with Engine() as eng:
        wire(eng, llm, validators={"v": lambda c, ctx: (True, "")})
        out = await eng.run(Reflexion(validator="v"), {"task": "t", "reflections": ["Be brief."]})
    assert out.ok and any("Be brief." in (m.content or "") for m in llm.requests[0].messages)


async def test_reflexion_with_an_llm_critic():
    def script(req):
        s = sys_of(req)
        if "strict evaluator" in s:
            ok = "v2" in user_of(req)
            return text(json.dumps({"ok": ok, "feedback": "" if ok else "say v2"}))
        if "reviewing" in s:
            return text("Always say v2.")
        return text(
            "v2" if any("Always say v2" in (m.content or "") for m in req.messages) else "v1"
        )

    async with Engine() as eng:
        wire(eng, FakeLLMClient(script))
        out = await eng.run(Reflexion(), {"task": "write"})
    assert out.ok and out.value["trials"] == 2


async def test_reflexion_is_cut_off_by_the_budget():
    async with Engine() as eng:
        wire(
            eng, FakeLLMClient(lambda r: text("x")), validators={"v": lambda c, ctx: (False, "no")}
        )
        out = await eng.run(
            Reflexion(validator="v", max_trials=10), {"task": "t"}, budget=Budget(max_steps=4)
        )
    assert out.kind is OutcomeKind.BUDGET_EXCEEDED


def test_reflexion_bounds():
    for bad in (0, 11):
        with pytest.raises(ValueError):
            Reflexion(max_trials=bad)


# ===================================================================== Debate
def debate_script(answers_by_round):
    def script(req):
        s = sys_of(req)
        if "judge of a debate" in s:
            return text("Weighing them up.\nAnswer: 12")
        u = user_of(req)
        r = 1 if "Other agents answered" in u else 0
        agent = next(a for a in "ABC" if f"[{a}]" in s)
        return text(f"reasoning\nAnswer: {answers_by_round[r][agent]}")

    return script


PERSONAS = ["[A] You are careful.", "[B] You are fast.", "[C] You are skeptical."]


async def test_debate_agents_read_each_other_and_a_judge_rules():
    answers = [{"A": "12", "B": "10", "C": "12"}, {"A": "12", "B": "12", "C": "12"}]
    llm = FakeLLMClient(debate_script(answers))
    async with Engine() as eng:
        wire(eng, llm)
        out = await eng.run(Debate(agents=3, rounds=1, personas=PERSONAS), {"task": "6*2?"})
    assert out.ok and out.value["answer"] == "12" and out.value["judged"] is True
    assert out.value["answers"] == [["12", "10", "12"], ["12", "12", "12"]]
    assert len(llm.requests) == 3 * 2 + 1  # agents * (rounds + 1) + judge
    round2 = [r for r in llm.requests if "Other agents answered" in user_of(r)]
    assert len(round2) == 3
    b = next(r for r in round2 if "[B]" in sys_of(r))
    assert (
        "Your previous answer: 10" in user_of(b)
        and "Agent 1: 12" in user_of(b)
        and "Agent 2" not in user_of(b)
    )


async def test_debate_by_majority_vote_needs_no_judge():
    answers = [{"A": "12", "B": "10", "C": "12"}] * 2
    llm = FakeLLMClient(debate_script(answers))
    async with Engine() as eng:
        wire(eng, llm)
        out = await eng.run(Debate(3, 1, judge=False, personas=PERSONAS), {"task": "q"})
    assert out.ok and out.value["answer"] == "12" and out.value["judged"] is False
    assert len(llm.requests) == 6


async def test_debate_survives_a_failing_debater_and_fails_when_all_do():
    def script(req):
        if "[B]" in sys_of(req):
            raise RuntimeError("boom")
        return text("Answer: 7")

    from aaa.llm.errors import FatalError

    def script2(req):
        if "[B]" in sys_of(req):
            raise FatalError("bad model")
        return text("Answer: 7")

    async with Engine() as eng:
        wire(eng, FakeLLMClient(script2))
        out = await eng.run(Debate(3, 1, judge=False, personas=PERSONAS), {"task": "q"})
    assert out.ok and out.value["answer"] == "7" and len(out.value["answers"][1]) == 2

    def allfail(req):
        raise FatalError("down")

    async with Engine() as eng:
        wire(eng, FakeLLMClient(allfail))
        out = await eng.run(Debate(2, 1, judge=False), {"task": "q"})
    assert out.kind is OutcomeKind.FAILED


async def test_debate_is_cut_off_by_the_budget():
    async with Engine() as eng:
        wire(eng, FakeLLMClient(lambda r: text("Answer: 1")))
        out = await eng.run(Debate(3, 4, judge=False), {"task": "q"}, budget=Budget(max_steps=5))
    assert out.kind is OutcomeKind.BUDGET_EXCEEDED


def test_debate_bounds():
    for kw in ({"agents": 1}, {"agents": 8}, {"rounds": 5}, {"rounds": -1}):
        with pytest.raises(ValueError):
            Debate(**kw)
    with pytest.raises(ValueError):
        Debate(3, personas=["only one"])


async def test_debate_is_stable_when_calls_finish_in_any_order():
    answers = [{"A": "12", "B": "10", "C": "12"}, {"A": "12", "B": "12", "C": "12"}]
    seen = []
    for seed in (1, 2, 3):
        async with Engine() as eng:
            wire(eng, JitterClient(debate_script(answers), seed))
            out = await eng.run(Debate(3, 1, personas=PERSONAS), {"task": "q"})
            history = await eng.store.read((await eng.store.run_index())[0].stream)
        seen.append((out.value, shape(history)))
        requested = [e for e in history if e.type == "aaa.llm.requested"]
        assert len(requested) == len({e.id for e in requested}) == 7  # no round was started twice
    assert seen[0] == seen[1] == seen[2]


# ===================================================================== Swarm
@tool
def refund(order: str) -> str:
    """Refund an order."""
    return f"refunded {order}"


@tool
def search_docs(q: str) -> str:
    """Search the docs."""
    return f"docs about {q}"


AGENTS = {
    "triage": Agent("You are triage.", handoffs=["billing", "support"]),
    "billing": Agent(
        "You handle billing.", tools=["refund"], handoffs=["triage"], description="Money questions."
    ),
    "support": Agent("You handle support.", tools=["search_docs"]),
}


async def swarm_run(script, **kw):
    reg = ToolRegistry([refund, search_docs])
    llm = FakeLLMClient(script)
    async with Engine() as eng:
        wire(eng, llm, reg)
        out = await eng.run(
            Swarm(AGENTS, reg, start="triage", **kw), {"task": "I want my money back for A1"}
        )
        events = await eng.store.read((await eng.store.run_index())[0].stream)
    return out, llm, events


async def test_handoff_switches_instructions_and_tools():
    def script(req):
        names = {t.name for t in req.tools}
        last = req.messages[-1]
        if names == {"transfer_to_billing", "transfer_to_support"}:
            return tool_call("transfer_to_billing")
        if (
            names == {"refund", "transfer_to_triage"}
            and last.role == "tool"
            and "Transferred" in last.content
        ):
            assert "You handle billing" in last.content
            return tool_call("refund", "c2", order="A1")
        return text(f"Done: {last.content}")

    out, llm, events = await swarm_run(script)
    assert out.ok and out.value["text"] == "Done: refunded A1"
    assert out.value["agent"] == "billing" and out.value["handoffs"] == [["triage", "billing"]]
    assert [{t.name for t in r.tools} for r in llm.requests] == [
        {"transfer_to_billing", "transfer_to_support"},
        {"refund", "transfer_to_triage"},
        {"refund", "transfer_to_triage"},
    ]
    assert sum(e.type == "aaa.swarm.handoff" for e in events) == 1


async def test_an_agent_cannot_use_another_agents_tool():
    def script(req):
        last = req.messages[-1]
        if last.role == "user":
            return tool_call("refund", order="A1")  # triage has no refund tool
        return text(f"saw: {last.content}")

    out, _, _ = await swarm_run(script)
    assert out.ok and "UnknownTool" in out.value["text"] and "refunded" not in out.value["text"]


async def test_a_transfer_not_in_the_agents_list_is_refused():
    def script(req):
        last = req.messages[-1]
        if last.role == "user":
            return tool_call("transfer_to_nowhere")
        return text(f"saw: {last.content}")

    out, _, _ = await swarm_run(script)
    assert "UnknownTool" in out.value["text"] and out.value["handoffs"] == []


async def test_one_transfer_per_turn_and_a_cap_on_ping_pong():
    from aaa.llm.types import LLMResponse, Message, ToolCall

    def double(req):
        if req.messages[-1].role == "user":
            calls = [
                ToolCall(id="a", name="transfer_to_billing", arguments={}),
                ToolCall(id="b", name="transfer_to_support", arguments={}),
            ]
            return LLMResponse(message=Message(role="assistant", tool_calls=calls), model="fake")
        return text("ok")

    out, _, events = await swarm_run(double)
    assert out.ok and out.value["handoffs"] == [["triage", "billing"]]
    failed = [e for e in events if e.type == "aaa.tool.failed"]
    assert len(failed) == 1 and failed[0].data["error"]["type"] == "TransferRefused"

    def pingpong(req):
        names = {t.name for t in req.tools}
        if "transfer_to_billing" in names:
            return tool_call("transfer_to_billing", f"c{len(req.messages)}")
        if "transfer_to_triage" in names:
            return tool_call("transfer_to_triage", f"c{len(req.messages)}")
        return text("gave up")

    out, _, events = await swarm_run(pingpong, max_handoffs=3, max_turns=12)
    # a model that insists on transferring is still stopped: three transfers happen, the rest are refused, max_turns ends it
    assert sum(e.type == "aaa.swarm.handoff" for e in events) == 3
    assert out.kind is OutcomeKind.NO_SOLUTION


def test_swarm_configuration_is_checked():
    reg = ToolRegistry([refund])
    with pytest.raises(ValueError):
        Swarm({"a": Agent("x", handoffs=["ghost"])}, reg, start="a")
    with pytest.raises(ValueError):
        Swarm({"a": Agent("x", tools=["ghost"])}, reg, start="a")
    with pytest.raises(ValueError):
        Swarm({"a": Agent("x")}, reg, start="b")


async def test_swarm_stops_at_max_turns():
    def forever(req):
        return (
            tool_call("transfer_to_billing", f"c{len(req.messages)}")
            if "transfer_to_billing" in {t.name for t in req.tools}
            else tool_call("refund", f"c{len(req.messages)}", order="x")
        )

    out, _, _ = await swarm_run(forever, max_turns=4)
    assert out.kind is OutcomeKind.NO_SOLUTION


# ===================================================================== Tree of thoughts
def tot_script(req):
    s, u = sys_of(req), user_of(req)
    if "Rate how promising" in s:
        # a path containing "good" is promising; ending in a FINAL step is best
        score = 10 if "FINAL: 24" in u else (8 if "good" in u else 2)
        return text(json.dumps({"score": score}))
    n = u.count("\n1. ") + u.count("\n2. ") + u.count("\n3. ")
    if "(none yet)" in u:
        return text("good start")  # all depth-0 proposals are identical text but different keys
    if n == 1:
        return text("good middle")
    return text("FINAL: 24")


async def test_tree_of_thoughts_finds_the_best_path_within_its_call_budget():
    llm = FakeLLMClient(tot_script)
    algo = TreeOfThoughts(depth=3, width=2, branching=2)
    async with Engine() as eng:
        wire(eng, llm)
        out = await eng.run(algo, {"task": "make 24"})
    assert out.ok and out.value["answer"] == "24" and out.value["score"] == 10
    assert out.value["path"] == ["good start", "good middle", "FINAL: 24"]
    assert len(llm.requests) <= algo.max_calls == 24


async def test_tree_stops_early_on_a_high_score_and_reports_failures():
    def script(req):
        if "Rate how promising" in sys_of(req):
            return text(json.dumps({"score": 9.5}))
        return text("a step")

    async with Engine() as eng:
        wire(eng, FakeLLMClient(script))
        out = await eng.run(
            TreeOfThoughts(depth=4, width=2, branching=2, stop_score=9), {"task": "t"}
        )
    assert out.ok and out.value["depth"] == 1

    from aaa.llm.errors import FatalError

    def down(req):
        raise FatalError("down")

    async with Engine() as eng:
        wire(eng, FakeLLMClient(down))
        out = await eng.run(TreeOfThoughts(depth=2, width=1, branching=2), {"task": "t"})
    assert out.kind is OutcomeKind.FAILED


async def test_bad_scores_count_as_zero_and_never_stall_the_search():
    def script(req):
        if "Rate how promising" in sys_of(req):
            return text("not json at all")
        return text("a step")

    async with Engine() as eng:
        wire(eng, FakeLLMClient(script))
        out = await eng.run(
            TreeOfThoughts(depth=2, width=1, branching=2, stop_score=None), {"task": "t"}
        )
    assert out.ok and out.value["score"] == 0.0 and out.value["depth"] == 2


async def test_tree_of_thoughts_is_cut_off_by_the_budget():
    async with Engine() as eng:
        wire(
            eng,
            FakeLLMClient(
                lambda r: text(json.dumps({"score": 1})) if "Rate how" in sys_of(r) else text("s")
            ),
        )
        out = await eng.run(
            TreeOfThoughts(depth=4, width=3, branching=3, stop_score=None),
            {"task": "t"},
            budget=Budget(max_steps=6),
        )
    assert out.kind is OutcomeKind.BUDGET_EXCEEDED


def test_tree_bounds():
    for kw in ({"depth": 0}, {"depth": 9}, {"width": 0}, {"width": 6, "branching": 5}):
        with pytest.raises(ValueError):
            TreeOfThoughts(**kw)


async def test_tree_result_does_not_depend_on_completion_order():
    seen = []
    for seed in (1, 2, 3):
        llm = JitterClient(tot_script, seed)
        async with Engine() as eng:
            wire(eng, llm)
            out = await eng.run(TreeOfThoughts(depth=3, width=2, branching=2), {"task": "make 24"})
            history = await eng.store.read((await eng.store.run_index())[0].stream)
        seen.append((out.value, shape(history)))
        asked = [e for e in history if e.type == "aaa.llm.requested"]
        assert (
            len(asked) == len({e.id for e in asked}) == len(llm.requests)
        )  # no call was made twice
    assert seen[0] == seen[1] == seen[2]


async def test_the_existing_loops_are_unchanged_by_the_new_hook():
    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    reg = ToolRegistry([add])

    def script(req):
        return (
            tool_call("add", a=1, b=2)
            if req.messages[-1].role == "user"
            else text(f"= {req.messages[-1].content}")
        )

    async with Engine() as eng:
        wire(eng, FakeLLMClient(script), reg)
        out = await eng.run(ToolLoop(reg), {"task": "1+2"})
    assert out.ok and out.value == {"text": "= 3", "turns": 2}
