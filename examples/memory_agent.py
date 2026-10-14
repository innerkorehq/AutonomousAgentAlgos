"""Long-term memory: an agent that remembers across runs, per user, per tenant. Offline."""

import asyncio

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.memory import MemoryHandler, NativeMemory, memory_tools
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry


def scripted_model(request):  # stands in for a real LLM
    prompt = " ".join(m.content or "" for m in request.messages if m.role == "system")
    last = request.messages[-1]
    if last.role == "user" and "remember" in (last.content or ""):
        return tool_call("remember", text="Ann is vegetarian and lives in Lisbon")
    if last.role == "user" and "restaurant" in (last.content or ""):
        if "vegetarian" in prompt:  # the recalled memory is in the system prompt
            return text("A vegetarian place in Lisbon: Ao 26.")
        return text("Which cuisine do you like?")
    return text("Noted, I will remember that.")


async def main() -> None:
    # Keyword search only, so a recall needs a shared word (here: Lisbon). Pass
    # NativeMemory(embedder=LLMEmbedder(client)) for semantic recall, or use one of the adapters.
    memory = NativeMemory()  # swap for Mem0Memory / GraphitiMemory / SupermemoryMemory / ...
    registry = ToolRegistry(memory_tools(memory))  # the model may remember() and recall()
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(scripted_model)).attach(engine)
        ToolHandler(registry).attach(engine)
        MemoryHandler(memory).attach(engine)  # serves automatic recall as events
        algo = ToolLoop(registry, recall=True)  # recall memories before the first model call

        acme = engine.tenant("acme")
        await acme.run(algo, {"task": "please remember my diet"}, user="ann")
        print("first run stored:", [m.text for m in await _all(memory, "acme", "ann")])

        out = await acme.run(algo, {"task": "suggest a restaurant in Lisbon"}, user="ann")
        print("ann, new run:", out.value["text"])

        out = await acme.run(algo, {"task": "suggest a restaurant in Lisbon"}, user="bob")
        print("bob, same tenant:", out.value["text"])

        out = await engine.tenant("globex").run(
            algo, {"task": "suggest a restaurant in Lisbon"}, user="ann"
        )
        print("ann at another tenant:", out.value["text"])

        report = await acme.forget_memory(user="ann")
        print("forgot:", report.forgotten)
        out = await acme.run(algo, {"task": "suggest a restaurant in Lisbon"}, user="ann")
        print("ann after forgetting:", out.value["text"])


async def _all(memory, tenant, user):
    from aaa.memory import Scope

    return await memory.list(Scope(tenant=tenant, user=user))


asyncio.run(main())
