"""One agent, two native wire formats: Anthropic's Messages API and Google's Generative Language API. Offline (mock servers)."""

import asyncio
import json

import httpx

from aaa.algos import ToolLoop
from aaa.llm import LLMHandler
from aaa.llm.anthropic import AnthropicClient
from aaa.llm.config import LLMConfig
from aaa.llm.gemini import GeminiClient
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def anthropic_server(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if "tool_result" in json.dumps(body):
        content, stop = [{"type": "text", "text": "The sum is 5."}], "end_turn"
    else:
        content, stop = (
            [{"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 2, "b": 3}}],
            "tool_use",
        )
    usage = {"input_tokens": 20, "output_tokens": 8}
    return httpx.Response(
        200,
        json={
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "claude",
            "content": content,
            "stop_reason": stop,
            "usage": usage,
        },
    )


def gemini_server(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if "functionResponse" in json.dumps(body):
        parts = [{"text": "The sum is 5."}]
    else:
        parts = [{"functionCall": {"name": "add", "args": {"a": 2, "b": 3}}}]
    cand = {"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}
    return httpx.Response(
        200,
        json={
            "candidates": [cand],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 8,
                "totalTokenCount": 28,
            },
        },
    )


async def run(name: str, client) -> None:
    registry = ToolRegistry([add])
    algo = ToolLoop(registry)
    async with Engine() as engine:
        LLMHandler(client).attach(engine)
        ToolHandler(registry).attach(engine)
        outcome = await engine.run(algo, {"task": "What is 2 + 3?"})
    print(f"{name}: {outcome.value['text']}")


async def main() -> None:
    http = lambda handler: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    # In production: LLMConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="${ANTHROPIC_API_KEY}")
    #                LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key="${GEMINI_API_KEY}")  -> build_client(config)
    await run(
        "anthropic",
        AnthropicClient(
            LLMConfig(provider="anthropic", model="claude", api_key="k"),
            http_client=http(anthropic_server),
        ),
    )
    await run(
        "gemini",
        GeminiClient(
            LLMConfig(provider="gemini", model="gemini", api_key="k"),
            http_client=http(gemini_server),
        ),
    )


asyncio.run(main())
