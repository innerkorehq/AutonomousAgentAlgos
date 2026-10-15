"""Built-in tools for a model that can be steered by what it reads: SSRF-safe fetch, confined files, and code that runs only after a human says yes. Offline."""

import asyncio
import json
import tempfile
from pathlib import Path

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.sandbox import (
    SubprocessSandbox,
    code_exec_tool,
)  # use DockerSandbox() for anything exposed to untrusted input
from aaa.tools import ToolHandler, ToolRegistry
from aaa.tools.builtin import FetchPolicy, http_fetch_tool, read_file_tool
from aaa.tools.tool import ToolError


async def main() -> None:
    open_web = http_fetch_tool(
        FetchPolicy(allowed_hosts=())
    )  # no allow-list: any PUBLIC host (set allowed_hosts in production)
    only_docs = http_fetch_tool(FetchPolicy.strict(["docs.python.org"]))
    for tool_, url in (
        (open_web, "http://169.254.169.254/latest/meta-data/"),
        (open_web, "http://localhost/admin"),
        (open_web, "http://example.com:6379/"),
        (only_docs, "https://evil.example.com/"),
    ):
        try:
            await tool_.invoke({"url": url}, None)  # type: ignore[arg-type]
        except ToolError as e:
            print(f"fetch {url}: refused ({e})")

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "notes.txt").write_text("project notes")
        (Path(tmp).parent / "secret.txt").write_text("not for the model")
        read = read_file_tool(tmp)
        print("read notes.txt:", (await read.invoke({"path": "notes.txt"}, None))["text"])  # type: ignore[arg-type]
        try:
            await read.invoke({"path": "../secret.txt"}, None)  # type: ignore[arg-type]
        except ToolError as e:
            print("read ../secret.txt: refused (" + str(e) + ")")

    registry = ToolRegistry([code_exec_tool(SubprocessSandbox(unsafe=True))])

    def model(request):
        if request.messages[-1].role == "user":
            return tool_call("code_exec", language="python", code="print(sum(range(101)))")
        return text(
            "The program printed: "
            + str(json.loads(request.messages[-1].content or "{}").get("stdout", "")).strip()
        )

    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        ToolHandler(registry).attach(engine)
        handle = await engine.start(ToolLoop(registry), {"task": "add up 1..100"})
        wanted = await engine.await_event(
            handle.run_id, lambda e: e.type == "aaa.approval.requested"
        )
        print("code_exec waits for approval:", wanted.data["name"])
        await handle.signal("aaa.approval.granted", {"call_id": wanted.data["call_id"]})
        print((await handle.result()).value["text"])


asyncio.run(main())
