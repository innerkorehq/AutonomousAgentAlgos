"""Stream tokens as they arrive: `stream=True` makes the LLM handler emit aaa.llm.delta events."""

import asyncio

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Engine
from aaa.tools import ToolRegistry


async def main() -> None:
    registry = ToolRegistry()
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(lambda request: text("Streaming works token by token."))).attach(
            engine
        )
        run = await engine.start(ToolLoop(registry, stream=True), {"task": "Say something"})
        async for event in run.events():  # replays the log, then tails it until the run ends
            if event.type == "aaa.llm.delta":
                print(event.data["text"], end="", flush=True)  # push this to your UI / websocket
        print()
        print("outcome:", (await run.result()).kind.value)


asyncio.run(main())
