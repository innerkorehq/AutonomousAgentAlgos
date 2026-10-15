"""Sandboxes for running code a model wrote.

sandbox = DockerSandbox()
registry = ToolRegistry([code_exec_tool(sandbox)])          # every call asks a human first
"""

from aaa.sandbox.base import ExecResult, Limits, Sandbox, SandboxError
from aaa.sandbox.docker import DockerSandbox
from aaa.sandbox.subproc import SubprocessSandbox
from aaa.sandbox.tool import code_exec_tool

__all__ = [
    "DockerSandbox",
    "ExecResult",
    "Limits",
    "Sandbox",
    "SandboxError",
    "SubprocessSandbox",
    "code_exec_tool",
]
