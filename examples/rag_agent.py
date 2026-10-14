"""Agentic RAG: the model decides when to search your documents, then answers with citations."""

import asyncio

from aaa.algos import RAGAgent
from aaa.data import Doc, InMemoryIndex
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import ToolHandler


class WordEmbedder:  # any object with `async embed(list[str]) -> list[vector]`; use a real model in prod
    VOCAB = ("paris", "france", "tokyo", "japan", "capital")

    async def embed(self, texts):
        return [[float(w in t.lower()) for w in self.VOCAB] for t in texts]


def model(request):
    last = request.messages[-1]
    if last.role == "user":
        return tool_call("search_documents", query="capital of France")
    return text("Paris is the capital of France [d1].")


async def main() -> None:
    index = InMemoryIndex(WordEmbedder())  # BM25 keyword search fused with vector similarity
    await index.upsert(
        [
            Doc(id="d1", text="Paris is the capital of France.", meta={"source": "atlas"}),
            Doc(id="d2", text="Tokyo is the capital of Japan.", meta={"source": "atlas"}),
        ]
    )
    agent = RAGAgent(index, k=1)
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        ToolHandler(agent.tools).attach(engine)
        outcome = await engine.run(agent, {"task": "What is the capital of France?"})
    print(outcome.value["text"])


asyncio.run(main())
