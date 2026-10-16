"""Beyond a fixed dataset: cases from real traffic, a red-team check of a guard, and a simulated user holding a conversation. Offline."""

import asyncio

from aaa.algos import ToolLoop
from aaa.eval import Contains, Dataset, SessionEvalRunner, SimulatedUser, evaluate_guard
from aaa.guard import RegexGuard
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Engine
from aaa.session import SessionManager
from aaa.tools import ToolRegistry


async def main() -> None:
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(lambda r: text("Happy to help with that."))).attach(engine)
        algo = ToolLoop(ToolRegistry([]))
        await engine.tenant("acme").run(
            algo, {"task": "Refund me, my email is ann@example.com"}, run_id="r1"
        )

        # 1. a regression dataset from what really ran (inputs are scrubbed by default)
        cases = await Dataset.from_traces(engine, tenant="acme")
        print("from traces:", cases.cases[0].input)

        # 2. how much of a small attack corpus does a guard stop, and what does it wrongly block?
        guard = RegexGuard([r"ignore (all )?previous", r"system prompt"], name="my-guard")
        report = await evaluate_guard(guard)
        print(
            f"red team: attack success {report.attack_success_rate:.0%}, false positives {report.false_positive_rate:.0%}"
        )

        # 3. a model plays the customer against a real session
        manager = SessionManager(engine, algo, system="Be helpful.")
        customer = SimulatedUser(
            FakeLLMClient(
                lambda r: text("[DONE]" if len(r.messages) > 3 else "Where is my refund?")
            ),
            persona="an impatient customer",
            goal="find out about the refund",
        )
        run = await SessionEvalRunner(manager, customer, [Contains("help")], max_turns=4).run(
            engine, Dataset.from_list([{"id": "refund", "input": {}}])
        )
        result = run.results[0]
        print(
            "conversation:", len(result.turns), "turns;", "contains help =", result.scores[0].value
        )


asyncio.run(main())
