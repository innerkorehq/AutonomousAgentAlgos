from __future__ import annotations

from typing import Any

from aaa.sandbox.base import Limits, Sandbox, SandboxError
from aaa.tools.tool import Tool, ToolError


def code_exec_tool(
    sandbox: Sandbox,
    *,
    languages: tuple[str, ...] = ("python",),
    limits: Limits | None = None,
    requires_approval: bool = True,
    name: str = "code_exec",
) -> Tool:
    """A tool that runs model-written code in `sandbox`. **Every call waits for human approval by default** (the engine's approval flow); turn that off only for a sandbox
    you would trust with the worst code the model could write."""
    lim = limits or Limits()

    async def code_exec(language: str, code: str) -> dict[str, Any]:
        """Run a short program and return its output. Each run starts from nothing (no files, no network); the program has a few seconds and little memory. Print what you need to see."""
        if language not in languages:
            raise ToolError(f"unsupported language {language!r}; use one of {list(languages)}")
        try:
            r = await sandbox.run(code, language, limits=lim)
        except SandboxError as e:
            raise ToolError(f"the sandbox could not run the code: {e}") from None
        return {
            "exit_code": r.exit_code,
            "stdout": r.stdout,
            "stderr": r.stderr,
            "timed_out": r.timed_out,
            "killed": r.killed,
            "output_truncated": r.truncated,
        }

    return Tool(
        code_exec, name=name, requires_approval=requires_approval, timeout_s=lim.timeout_s + 60
    )
