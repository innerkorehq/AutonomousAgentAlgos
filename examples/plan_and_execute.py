"""Plan once, run the steps as parallel child runs honoring dependencies, then synthesize."""

import asyncio
import json

from aaa.algos import PlanAndExecute, ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def population(city: str) -> int:
    """Population of a city (millions)."""
    return {"Tokyo": 14, "Delhi": 32, "Lima": 10}[city]


PLAN = {
    "steps": [
        {"id": "a", "description": "population of Tokyo", "depends_on": []},
        {"id": "b", "description": "population of Delhi", "depends_on": []},
        {"id": "c", "description": "add the two populations", "depends_on": ["a", "b"]},
    ]
}


def model(request):
    system = request.messages[0].content if request.messages[0].role == "system" else ""
    last = request.messages[-1]
    if "planner" in system:
        return text(json.dumps(PLAN))
    if "Combine" in system:
        return text("Tokyo + Delhi = 46 million people")
    if last.role == "user":  # an executor child: do its one step
        step = last.content.split("Your step (")[1].split(")")[0]
        if step in ("a", "b"):
            return tool_call("population", city="Tokyo" if step == "a" else "Delhi")
        return text("46")
    return text(last.content)


async def main() -> None:
    registry = ToolRegistry([population])
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        ToolHandler(registry).attach(engine)
        agent = PlanAndExecute(ToolLoop(registry), max_parallel=2)
        run = await engine.start(agent, {"task": "Total population of Tokyo and Delhi?"})
        outcome = await run.result()
        log = await engine.store.read(run.run_id)
    print(outcome.value["text"])
    print("steps run:", [e.data["step_id"] for e in log if e.type == "aaa.task.started"])


asyncio.run(main())
