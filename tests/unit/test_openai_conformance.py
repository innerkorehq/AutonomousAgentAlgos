"""The Responses client against the OFFICIAL openai SDK's types (skipped without the `openai` package; the CI `adapters`
job installs it). Our fixtures are only worth anything if the real SDK accepts them."""

import inspect
import json

import httpx
import pytest

openai = pytest.importorskip("openai")
from openai.resources.responses import Responses
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseInputParam,
    ResponseOutputMessage,
    ResponseUsage,
)
from pydantic import TypeAdapter

from aaa.llm import LLMConfig
from aaa.llm.openai_responses import OpenAIResponsesClient, input_items
from aaa.llm.types import LLMRequest, Message, ToolCall, ToolSpec


async def captured_payload(request: LLMRequest) -> dict:
    seen: list[dict] = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"id": "r", "status": "completed", "output": []})

    cfg = LLMConfig(
        provider="openai_responses",
        model="gpt-x",
        base_url="https://x.test/v1",
        params={"top_p": 0.9},
    )
    client = OpenAIResponsesClient(
        cfg,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        truncation="auto",
    )
    await client.complete(request)
    return seen[0]


async def test_every_request_field_and_item_is_one_the_sdk_defines():
    call = ToolCall(id="call_aaa_0", name="lookup", arguments={"q": "x"}, provider_id="call_prov_1")
    payload = await captured_payload(
        LLMRequest(
            messages=[
                Message(role="system", content="be brief"),
                Message(role="user", content="go"),
                Message(role="assistant", content="calling", tool_calls=[call]),
                Message(role="tool", tool_call_id="call_aaa_0", content="7"),
            ],
            tools=[
                ToolSpec(
                    name="lookup", description="d", parameters={"type": "object", "properties": {}}
                )
            ],
            params={"max_tokens": 10, "temperature": 0.1},
            response_schema={"type": "object", "properties": {"a": {"type": "string"}}},
            continue_from="resp_0",
        )
    )
    allowed = set(inspect.signature(Responses.create).parameters)
    assert set(payload) <= allowed, (
        f"fields the SDK does not know: {sorted(set(payload) - allowed)}"
    )
    TypeAdapter(ResponseInputParam).validate_python(payload["input"])
    TypeAdapter(list[FunctionToolParam]).validate_python(payload["tools"])
    assert (
        payload["text"]["format"]["type"] == "json_schema"
        and payload["previous_response_id"] == "resp_0"
    )


def test_input_items_validate_against_the_sdk_for_every_message_shape():
    msgs = [
        Message(role="user", content="u"),
        Message(role="assistant", content="a"),
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="t", arguments={})]),
        Message(role="tool", tool_call_id="c1", content="r"),
    ]
    TypeAdapter(ResponseInputParam).validate_python(input_items(msgs))


def test_the_response_shapes_we_parse_are_the_sdks_shapes():
    message = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "hi", "annotations": []}],
    }
    call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": "{}",
        "status": "completed",
    }
    usage = {
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
        "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    ResponseOutputMessage.model_validate(message)
    ResponseFunctionToolCall.model_validate(call)
    ResponseUsage.model_validate(usage)
