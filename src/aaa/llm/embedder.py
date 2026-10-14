from __future__ import annotations

from aaa.llm.client import LLMClient


class LLMEmbedder:
    """Adapts any `LLMClient` to the `aaa.data.Embedder` protocol, so retrieval and memory can
    embed text through the same gateway (and credentials) as completions."""

    def __init__(self, client: LLMClient, model: str | None = None) -> None:
        self.client, self.model = client, model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return (await self.client.embed(texts, self.model)).vectors
