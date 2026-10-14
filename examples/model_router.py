"""One LLMHandler, several backends: dispatch on the request's model name."""

import asyncio

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Engine
from aaa.tools import ToolRegistry


class ModelRouter:
    """An LLMClient that forwards to a backend chosen by the model's prefix: "small/..." or "big/..."."""

    def __init__(self, backends: dict, default: str) -> None:
        self.backends, self.default = backends, default

    def _pick(self, model: str | None):
        prefix, _, rest = (model or "").partition("/")
        if prefix in self.backends:
            return self.backends[prefix], rest
        return self.backends[self.default], model

    async def complete(self, request):
        client, model = self._pick(request.model)
        return await client.complete(request.model_copy(update={"model": model}))

    async def embed(self, texts, model=None):
        client, model = self._pick(model)
        return await client.embed(texts, model)

    async def aclose(self) -> None:
        for client in self.backends.values():
            await client.aclose()


async def main() -> None:
    router = ModelRouter(
        {
            "small": FakeLLMClient(lambda r: text(f"[small:{r.model}] quick answer")),
            "big": FakeLLMClient(lambda r: text(f"[big:{r.model}] careful answer")),
        },
        default="small",
    )
    registry = ToolRegistry()
    async with Engine() as engine:
        LLMHandler(router).attach(engine)
        cheap = await engine.run(
            ToolLoop(registry, model="small/fast-1", name="cheap"), {"task": "hi"}
        )
        smart = await engine.run(
            ToolLoop(registry, model="big/deep-9", name="smart"), {"task": "hi"}
        )
    print(cheap.value["text"])
    print(smart.value["text"])


asyncio.run(main())
