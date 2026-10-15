"""Run code in a locked-down Docker container: no network, read-only filesystem, no capabilities, no privilege escalation, bounded memory, processes, time and output,
and a fresh container per run.

    sandbox = DockerSandbox(images={"python": "python:3.12-slim@sha256:..."})      # pin by digest in production
    result = await sandbox.run("print(2 + 2)", "python", limits=Limits(timeout_s=5, memory_mb=128))

What each flag buys is in docs/security/tools.md. **Nothing from the host is mounted**: the program and any input files travel on the container's stdin and land in its
scratch tmpfs (`/tmp/work`), so the code has no path to the host filesystem at all, and it works on any Docker runtime (Docker Desktop, Colima, rootless, a remote `DOCKER_HOST`).
The image must contain `python3` (the small loader that unpacks stdin runs it); `bash` and `sh` programs run through the same image.

A container shares the host kernel, so this is defence in depth against buggy or careless code and a real barrier against casual abuse, but **not a boundary against a
determined attacker with a kernel exploit**: for hostile multi-tenant code run it on a host dedicated to the purpose."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import shutil
import uuid
from collections.abc import Mapping, Sequence

from aaa.sandbox.base import ExecResult, Limits, SandboxError, check_files, run_process

DEFAULT_IMAGE = "python:3.12-slim"
LANGUAGES = ("python", "bash", "sh")
BOOTSTRAP = """
import sys, os, json, base64
r = sys.stdin.buffer
h = json.loads(r.readline())
os.makedirs("/tmp/work", exist_ok=True)
for n, b in h["files"].items():
    with open("/tmp/work/" + n, "wb") as f:
        f.write(base64.b64decode(b))
code = r.read(h["n"])
os.chdir("/tmp/work")
if h["lang"] == "python":
    sys.path.insert(0, "/tmp/work")
    sys.argv = ["-"]
    exec(compile(code, "<stdin>", "exec"), {"__name__": "__main__"})
else:
    with open("/tmp/script", "wb") as f:
        f.write(code)
    os.execvp(h["lang"], [h["lang"], "/tmp/script"])
"""


class DockerSandbox:
    def __init__(
        self,
        *,
        images: Mapping[str, str] | None = None,
        docker: str = "docker",
        user: str = "65534:65534",
        extra_args: Sequence[str] = (),
    ) -> None:
        self.images = dict.fromkeys(LANGUAGES, DEFAULT_IMAGE) | dict(images or {})
        self.docker, self.user, self.extra_args = docker, user, tuple(extra_args)

    @property
    def languages(self) -> tuple[str, ...]:
        return LANGUAGES

    async def available(self) -> bool:
        if shutil.which(self.docker) is None:
            return False
        proc = await asyncio.create_subprocess_exec(
            self.docker,
            "version",
            "--format",
            "{{.Server.Version}}",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return await proc.wait() == 0

    def command(self, name: str, language: str, limits: Limits) -> list[str]:
        """The exact `docker run` invocation (exposed so the hardening can be asserted without Docker)."""
        if language not in LANGUAGES:
            raise SandboxError(f"unsupported language {language!r}; supported: {sorted(LANGUAGES)}")
        args = [
            self.docker, "run", "--name", name, "--label", "aaa.sandbox=1", "-i",
            "--network", "none",
            "--read-only",
            "--tmpfs", f"/tmp:rw,noexec,nosuid,nodev,size={limits.tmp_mb}m",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(limits.pids),
            "--memory", f"{limits.memory_mb}m",
            "--memory-swap", f"{limits.memory_mb}m",
            "--cpus", str(limits.cpus),
            "--user", self.user,
            "--ulimit", "nofile=256:256",
            "--ulimit", f"fsize={limits.max_file_mb * 1024 * 1024}",
            "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-w", "/tmp",
        ]  # fmt: skip
        return [
            *args,
            *self.extra_args,
            self.images[language],
            "python",
            "-I",
            "-u",
            "-c",
            BOOTSTRAP,
        ]

    async def _docker(self, *args: str) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            self.docker, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        out, _ = await proc.communicate()
        return proc.returncode or 0, out.decode("utf-8", "replace").strip()

    async def run(
        self,
        code: str,
        language: str = "python",
        *,
        files: Mapping[str, bytes | str] | None = None,
        limits: Limits | None = None,
    ) -> ExecResult:
        limits = limits or Limits()
        data = check_files(files)
        name = f"aaa-sbx-{uuid.uuid4().hex[:12]}"
        argv = self.command(name, language, limits)
        body = code.encode()
        header = json.dumps(
            {
                "lang": language,
                "files": {k: base64.b64encode(v).decode() for k, v in data.items()},
                "n": len(body),
            }
        ).encode()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise SandboxError(
                f"{self.docker!r} was not found; install Docker or use another sandbox"
            ) from e

        async def kill() -> None:
            # Detach our end first: a program flooding its output keeps the daemon and the CLI busy, and the kill request can queue behind that.
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await self._docker("kill", name)

        try:
            result = await run_process(proc, header + b"\n" + body, limits, kill)
            if result.exit_code in (137, None):  # killed by the kernel for memory?
                rc, out = await self._docker("inspect", "-f", "{{.State.OOMKilled}}", name)
                if rc == 0 and out == "true":
                    result.killed, result.exit_code = "oom", 137
            if result.exit_code in (125, 126, 127) and not result.stdout:
                raise SandboxError(
                    f"docker could not run the container: {result.stderr.strip()[:300]}"
                )
            return result
        finally:
            with contextlib.suppress(
                Exception
            ):  # always remove the container, even if we were cancelled
                await asyncio.shield(self._docker("rm", "-f", name))
