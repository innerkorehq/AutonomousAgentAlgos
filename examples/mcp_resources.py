"""Serve documents and a prompt over MCP, with each tenant seeing only their own files. Offline."""

import asyncio
import json
import tempfile
from pathlib import Path

from aaa.data import Doc, InMemoryIndex
from aaa.mcp import MCPServerBuilder
from aaa.mcp.sources.resources import ResourceNotFound


async def main() -> None:
    index = InMemoryIndex()
    await index.upsert(
        [
            Doc(id="r1", text="Refunds are issued within 14 days.", meta={"tenant": "acme"}),
            Doc(id="r2", text="Refunds take 30 days.", meta={"tenant": "globex"}),
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for tenant, text in (("acme", "Acme handbook"), ("globex", "Globex handbook")):
            (root / tenant).mkdir()
            (root / tenant / "handbook.md").write_text(text)
        server = await (
            MCPServerBuilder("knowledge")
            .files("handbook", root, description="The tenant's handbook.")
            .index("faq", index, tenant_key="tenant")
            .prompt(
                "summarize",
                ["Summarize the {document} for a new hire in {words} words."],
                arguments=[{"name": "document"}, {"name": "words", "type": "integer"}],
                description="Ask for a summary.",
            )
            .acompile()
        )
        for tenant in ("acme", "globex"):
            listed = [r.uri for r in server.list_resources(tenant)]
            print(tenant, "sees:", listed)
        print(
            "acme reads:",
            (await server.read_resource("aaa://files/handbook/handbook.md", tenant="acme")).text,
        )
        try:
            await server.read_resource("aaa://files/handbook/../globex/handbook.md", tenant="acme")
        except ResourceNotFound:
            print("acme reading globex's file: resource not found")
        hits = await server.read_resource("aaa://index/faq/search?q=refunds", tenant="globex")
        print("globex searches the faq:", [h["id"] for h in json.loads(hits.text)])
        print(
            "prompt:",
            server.get_prompt("summarize", {"document": "handbook", "words": "50"}).messages[0][1],
        )


asyncio.run(main())
