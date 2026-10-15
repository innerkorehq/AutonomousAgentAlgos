"""Run untrusted code somewhere it can do limited harm. See docs/security/tools.md for what each backend does and does not protect against."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
MAX_FILES_BYTES = 10 * 1024 * 1024


class SandboxError(Exception):
    """The sandbox itself failed (Docker missing, the engine refused to start the container); the CODE failing is an `ExecResult`, not this."""


@dataclass(frozen=True)
class Limits:
    timeout_s: float = 10.0  # wall clock; the container/process is killed when it elapses
    memory_mb: int = 256
    cpus: float = 1.0
    pids: int = 64
    tmp_mb: int = 64  # size of the scratch tmpfs
    max_output_bytes: int = 100_000  # stdout + stderr together; the run is killed when exceeded
    max_file_mb: int = 16  # largest file the code may write


@dataclass
class ExecResult:
    exit_code: int | None  # None when killed before it finished
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False  # output was cut at max_output_bytes (and the run killed)
    killed: str | None = None  # "timeout" | "output" | "oom"
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.killed


@runtime_checkable
class Sandbox(Protocol):
    async def run(
        self,
        code: str,
        language: str = "python",
        *,
        files: Mapping[str, bytes | str] | None = None,
        limits: Limits | None = None,
    ) -> ExecResult: ...


def check_files(files: Mapping[str, bytes | str] | None) -> dict[str, bytes]:
    """Validate input file names (plain names only: no directories, no dot-dot) and sizes."""
    out: dict[str, bytes] = {}
    total = 0
    for name, data in (files or {}).items():
        if not NAME.match(name):
            raise SandboxError(
                f"invalid file name {name!r}: use a plain name (letters, digits, . _ -)"
            )
        raw = data.encode() if isinstance(data, str) else data
        total += len(raw)
        out[name] = raw
    if total > MAX_FILES_BYTES:
        raise SandboxError(f"input files total {total} bytes; the limit is {MAX_FILES_BYTES}")
    return out


async def run_process(
    proc: asyncio.subprocess.Process,
    stdin: bytes,
    limits: Limits,
    kill: Callable[[], Awaitable[None]],
) -> ExecResult:
    """Feed `stdin`, collect output up to the cap, enforce the wall-clock timeout; `kill` stops the program (and whatever it started)."""
    start = time.monotonic()
    out, err = bytearray(), bytearray()
    state = {"over": False}

    async def pump(stream: asyncio.StreamReader | None, buf: bytearray) -> None:
        stopped = False
        while stream is not None:
            chunk = await stream.read(8192)
            if not chunk:
                return
            if stopped:
                continue  # keep draining after the cap: a child blocked on a full pipe could never exit
            room = limits.max_output_bytes - len(out) - len(err)
            buf += chunk[: max(room, 0)]
            if len(chunk) > room:
                state["over"] = True
                stopped = True
                await kill()

    async def feed() -> None:
        if proc.stdin is None:
            return
        try:
            proc.stdin.write(stdin)
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the program ended or was killed before reading everything

    timed_out = False
    try:
        await asyncio.wait_for(
            asyncio.gather(feed(), pump(proc.stdout, out), pump(proc.stderr, err), proc.wait()),
            limits.timeout_s,
        )
    except TimeoutError:
        timed_out = True
        await kill()
        await proc.wait()
    code = proc.returncode
    killed = "timeout" if timed_out else "output" if state["over"] else None
    return ExecResult(
        exit_code=None if killed else code,
        stdout=bytes(out).decode("utf-8", "replace"),
        stderr=bytes(err).decode("utf-8", "replace"),
        timed_out=timed_out,
        truncated=state["over"],
        killed=killed,
        duration_s=round(time.monotonic() - start, 3),
    )
