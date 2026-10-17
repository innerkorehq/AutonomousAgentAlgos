"""Generate MCP tools from an OpenAPI document: read operations are offered, state-changing ones wait for approval. Compile only, no network."""

import asyncio
import json
import os
import tempfile
from pathlib import Path

from aaa.mcp.compile import acompile_spec
from aaa.mcp.config import load_spec

DOCUMENT = {
    "openapi": "3.0.3",
    "info": {"title": "Orders", "version": "1"},
    "servers": [{"url": "https://api.example.com/v1"}],
    "paths": {
        "/orders/{orderId}": {
            "get": {
                "operationId": "getOrder",
                "summary": "Fetch one order",
                "parameters": [
                    {
                        "name": "orderId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "responses": {"200": {"description": "the order"}},
            },
            "delete": {
                "operationId": "cancelOrder",
                "summary": "Cancel an order",
                "parameters": [
                    {
                        "name": "orderId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "responses": {"204": {"description": "cancelled"}},
            },
        },
        "/orders": {
            "post": {
                "operationId": "createOrder",
                "summary": "Place an order",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["sku"],
                                "properties": {
                                    "sku": {"type": "string"},
                                    "quantity": {"type": "integer", "minimum": 1},
                                },
                            }
                        }
                    },
                },
                "responses": {"201": {"description": "created"}},
            }
        },
        "/uploads": {
            "post": {
                "operationId": "uploadReceipt",
                "requestBody": {"content": {"multipart/form-data": {"schema": {"type": "object"}}}},
                "responses": {"200": {"description": "ok"}},
            }
        },
    },
}


async def main() -> None:
    os.environ.setdefault(
        "ORDERS_API_TOKEN", "demo-token-not-used"
    )  # the demo has no real API; a real one sets this outside the program
    with tempfile.TemporaryDirectory() as tmp:
        document = Path(tmp) / "orders.json"
        document.write_text(json.dumps(DOCUMENT))
        spec = load_spec(
            {
                "sources": [
                    {
                        "type": "openapi",
                        "document": str(document),
                        "allowed_hosts": [
                            "api.example.com"
                        ],  # mandatory: the only host this server may call
                        "auth": {
                            "scheme": "bearer",
                            "secret": "env:ORDERS_API_TOKEN",
                        },  # a reference, never the value
                    }
                ]
            },
            environ={},
        )
        server = await acompile_spec(spec)
    for tool in server.list_tools():
        print("offered:", tool.name, "|", tool.description, "| read-only:", tool.read_only)
    print("hidden (state-changing, need approval):", sorted(server.hidden))
    for line in server.report:
        if "skipped" in line:
            print(line)


asyncio.run(main())
