"""Generate -> check with a deterministic validator -> feed the failure back -> regenerate."""

import asyncio
import json

from aaa.algos import GenerateTestRepair
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Engine
from aaa.validate import ValidationHandler, Verdict


def three_strings(candidate: str, context: dict) -> Verdict:
    """Your 'test': any function returning a Verdict, bool, or (bool, feedback)."""
    try:
        items = json.loads(candidate)
    except ValueError:
        return Verdict(ok=False, feedback="reply with ONLY a JSON array, no prose")
    ok = isinstance(items, list) and len(items) == 3 and all(isinstance(i, str) for i in items)
    return Verdict(ok=ok, feedback="" if ok else "need exactly 3 strings")


attempts = iter(["Sure! Here are some fruits.", '["apple", "pear"]', '["apple", "pear", "fig"]'])


async def main() -> None:
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(lambda request: text(next(attempts)))).attach(engine)
        ValidationHandler({"default": three_strings}).attach(engine)
        outcome = await engine.run(
            GenerateTestRepair(max_iterations=4), {"task": "List three fruits as a JSON array."}
        )
    print(
        outcome.kind.value,
        outcome.value["candidate"],
        "after",
        outcome.value["iterations"],
        "tries",
    )


asyncio.run(main())
