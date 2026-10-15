"""A child process with limits and a timeout. **This is not isolation**: the code runs as your user, with your files and your network.

It exists so agents that execute code can be developed and tested without Docker, and refuses to be created unless you pass `unsafe=True`, so it cannot be chosen by accident
for anything exposed to untrusted input. Use `DockerSandbox` (or a VM-backed sandbox) for that."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from aaa.sandbox.base import ExecResult, Limits, SandboxError, check_files, run_process

COMMANDS = {"python": [sys.executable, "-I", "-u", "-"], "bash": ["bash", "-s"], "sh": ["sh", "-s"]}


class SubprocessSandbox:
    def __init__(self, *, unsafe: bool = False) -> None:
        if not unsafe:
            raise SandboxError(
                "SubprocessSandbox provides NO isolation (the code runs with your user, files and network). Pass unsafe=True to use it for local development only."
            )

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(COMMANDS)

    @staticmethod
    def _limiter(limits: Limits):  # type: ignore[no-untyped-def]
        def apply() -> None:  # runs in the child between fork and exec (POSIX)
            import resource

            def lim(which: int, value: int) -> None:
                with contextlib.suppress(ValueError, OSError):
                    resource.setrlimit(which, (value, value))

            lim(resource.RLIMIT_CPU, int(limits.timeout_s) + 1)
            lim(resource.RLIMIT_FSIZE, limits.max_file_mb * 1024 * 1024)
            lim(resource.RLIMIT_NOFILE, 256)
            if sys.platform.startswith("linux"):  # the address-space limit is not honoured on macOS
                lim(resource.RLIMIT_AS, limits.memory_mb * 1024 * 1024)

        return apply

    async def run(
        self,
        code: str,
        language: str = "python",
        *,
        files: Mapping[str, bytes | str] | None = None,
        limits: Limits | None = None,
    ) -> ExecResult:
        limits = limits or Limits()
        if language not in COMMANDS:
            raise SandboxError(f"unsupported language {language!r}; supported: {sorted(COMMANDS)}")
        data = check_files(files)
        tmp = Path(tempfile.mkdtemp(prefix="aaa-sbx-")).resolve()
        try:
            for fname, raw in data.items():
                (tmp / fname).write_bytes(raw)
            env = {
                "PATH": os.environ.get("PATH", os.defpath),
                "HOME": str(tmp),
                "LANG": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
            }  # nothing else from this process
            proc = await asyncio.create_subprocess_exec(
                *COMMANDS[language], stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=tmp, env=env, start_new_session=True, preexec_fn=self._limiter(limits),
            )  # fmt: skip

            async def kill() -> None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(
                        proc.pid, signal.SIGKILL
                    )  # the whole process group, not just the shell

            try:
                return await run_process(proc, code.encode(), limits, kill)
            finally:
                await kill()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
