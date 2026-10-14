"""Majority vote over parallel samples (SelfConsistency), and a separate LLM critic loop."""

import asyncio
import json

from aaa.algos import GeneratorCritic, SelfConsistency
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Engine

samples = iter(["... Answer: 42", "Answer: 41", "so Answer: 42.", "Answer: 42", "Answer: 7"])


def critic_model(request):
    system = request.messages[0].content if request.messages[0].role == "system" else ""
    if "strict reviewer" in system:  # the critic: accept only the revised draft
        ok = "v2" in request.messages[-1].content
        return text(json.dumps({"ok": ok, "feedback": "" if ok else "mention v2"}))
    revised = any("mention v2" in (m.content or "") for m in request.messages)
    return text("draft v2" if revised else "draft v1")


async def main() -> None:
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(lambda request: text(next(samples)))).attach(engine)
        vote = await engine.run(SelfConsistency(n=5), {"task": "6 x 7?"})
    print("vote:", vote.value)

    async with Engine() as engine:
        LLMHandler(FakeLLMClient(critic_model)).attach(engine)
        out = await engine.run(GeneratorCritic(), {"task": "write a draft"})
    print("critic loop:", out.value)


asyncio.run(main())
