"""The code sandbox: what the Docker command locks down (asserted without Docker), the dev-only subprocess backend, and, where Docker exists, the real limits."""

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from aaa.core.outcome import OutcomeKind
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.sandbox import DockerSandbox, Limits, SandboxError, SubprocessSandbox, code_exec_tool
from aaa.sandbox.base import check_files
from aaa.tools import ToolHandler, ToolRegistry
from aaa.tools.tool import ToolError


# -- the Docker invocation, asserted as data ----------------------------------------------------
def test_the_docker_command_is_locked_down():
    argv = DockerSandbox().command(
        "aaa-sbx-x", "python", Limits(memory_mb=128, pids=32, cpus=0.5, tmp_mb=16)
    )
    joined = " ".join(argv)
    for flag in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--pids-limit 32",
        "--memory 128m",
        "--memory-swap 128m",
        "--cpus 0.5",
        "--user 65534:65534",
    ):
        assert flag in joined, flag
    assert "/tmp:rw,noexec,nosuid,nodev,size=16m" in joined
    for forbidden in (
        "--privileged",
        "docker.sock",
        "--network host",
        "--pid host",
        "--cap-add",
        "--device",
        "--volume",
        "--mount",
    ):
        assert forbidden not in joined, forbidden
    assert "-v" not in argv  # nothing from the host is ever mounted: files and code travel on stdin
    assert argv[argv.index("-w") + 1] == "/tmp" and argv[-5:-1] == ["python", "-I", "-u", "-c"]


def test_unsupported_languages_and_bad_file_names_are_refused():
    with pytest.raises(SandboxError, match="unsupported language"):
        DockerSandbox().command("n", "ruby", Limits())
    for bad in ("../x", "a/b", "/etc/passwd", ".", "", "x" * 200, "a b"):
        with pytest.raises(SandboxError, match="invalid file name"):
            check_files({bad: "x"})
    with pytest.raises(SandboxError, match="limit"):
        check_files({"big": b"x" * (11 * 1024 * 1024)})
    assert check_files({"ok.txt": "hi", "data-1.bin": b"\x00"}) == {
        "ok.txt": b"hi",
        "data-1.bin": b"\x00",
    }


def test_the_image_is_configurable_per_language():
    sb = DockerSandbox(images={"python": "python:3.12-slim@sha256:abc"})
    assert "python:3.12-slim@sha256:abc" in sb.command("n", "python", Limits())
    assert DockerSandbox().command("n", "bash", Limits())[-6] == "python:3.12-slim"


def test_the_subprocess_backend_cannot_be_created_by_accident():
    with pytest.raises(SandboxError, match="NO isolation"):
        SubprocessSandbox()
    SubprocessSandbox(unsafe=True)


# -- the dev-only subprocess backend ------------------------------------------------------------
@pytest.fixture
def sub():
    return SubprocessSandbox(unsafe=True)


async def test_subprocess_runs_code_and_reports_exit_codes(sub):
    r = await sub.run("print(6 * 7)")
    assert r.ok and r.stdout.strip() == "42" and r.exit_code == 0
    bad = await sub.run("import sys; print('oops', file=sys.stderr); sys.exit(3)")
    assert bad.exit_code == 3 and "oops" in bad.stderr and not bad.ok
    assert "NameError" in (await sub.run("undefined_name")).stderr
    sh = await sub.run("echo hello; exit 2", "bash")
    assert sh.stdout.strip() == "hello" and sh.exit_code == 2


async def test_subprocess_input_files_and_a_clean_environment(sub, monkeypatch):
    monkeypatch.setenv("AAA_TEST_SECRET", "hunter2")
    r = await sub.run(
        "import os; print(open('data.txt').read(), os.environ.get('AAA_TEST_SECRET'), sorted(os.environ))",
        files={"data.txt": "payload"},
    )
    assert r.stdout.startswith("payload None")  # the host's environment does not leak in
    assert "AAA_TEST_SECRET" not in r.stdout


async def test_subprocess_timeout_kills_the_whole_process_group(sub, tmp_path):
    marker = tmp_path / "child_alive"
    code = f"import subprocess, time\nsubprocess.Popen(['sh','-c','sleep 2; touch {marker}'])\nwhile True: time.sleep(0.05)"
    r = await sub.run(code, limits=Limits(timeout_s=0.7))
    assert r.timed_out and r.killed == "timeout" and r.exit_code is None
    await asyncio.sleep(2.5)
    assert not marker.exists()  # the grandchild died with the group


async def test_subprocess_output_is_capped_and_the_run_stopped(sub):
    r = await sub.run(
        "while True: print('x' * 1000)", limits=Limits(timeout_s=10, max_output_bytes=5000)
    )
    assert r.truncated and r.killed == "output" and len(r.stdout) <= 5000 and r.duration_s < 5


async def test_subprocess_file_size_limit(sub):
    r = await sub.run(
        "open('big.bin','wb').write(b'x' * (3 * 1024 * 1024))", limits=Limits(max_file_mb=1)
    )
    assert not r.ok  # RLIMIT_FSIZE stops the write


# -- the tool -----------------------------------------------------------------------------------
async def test_code_exec_requires_approval_by_default_and_reports_results(sub):
    t = code_exec_tool(sub)
    assert t.requires_approval is True and t.spec().name == "code_exec"
    out = await t.invoke({"language": "python", "code": "print('hi')"}, None)  # type: ignore[arg-type]
    assert out["stdout"] == "hi\n" and out["exit_code"] == 0 and out["killed"] is None
    with pytest.raises(ToolError, match="unsupported language"):
        await t.invoke({"language": "bash", "code": "echo hi"}, None)  # type: ignore[arg-type]
    assert code_exec_tool(sub, requires_approval=False).requires_approval is False


async def test_a_model_cannot_run_code_without_a_human_saying_yes(sub):
    def script(req):
        if req.messages[-1].role == "user":
            return tool_call("code_exec", language="python", code="open('ran.txt', 'w').write('x')")
        return text("done: " + (req.messages[-1].content or "")[:80])

    from aaa.algos import ToolLoop

    reg = ToolRegistry([code_exec_tool(sub)])
    eng = Engine()
    LLMHandler(FakeLLMClient(script)).attach(eng)
    ToolHandler(reg).attach(eng)
    async with eng:
        h = await eng.start(ToolLoop(reg), {"task": "run it"})
        ev = await eng.await_event(h.run_id, lambda e: e.type == "aaa.approval.requested")
        await asyncio.sleep(0.3)
        assert not Path("ran.txt").exists()  # nothing ran while it waited
        await h.signal("aaa.approval.denied", {"call_id": ev.data["call_id"], "reason": "no"})
        out = await h.result()
    assert (
        out.kind is OutcomeKind.SUCCESS and "ApprovalDenied" in out.value["text"]
    )  # the run went on, told that a human said no


# -- the real thing: Docker ---------------------------------------------------------------------
IMAGE = os.environ.get("AAA_TEST_DOCKER_IMAGE", "python:3.12-slim")
needs_docker = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")


def _ensure_image() -> None:
    if (
        subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True, check=False
        ).returncode
        != 0
    ):
        subprocess.run(["docker", "pull", IMAGE], check=True, capture_output=True, timeout=600)


@pytest.fixture
async def docker():
    sb = DockerSandbox(images={"python": IMAGE, "bash": IMAGE, "sh": IMAGE})
    if not await sb.available():
        if os.environ.get("AAA_REQUIRE_DOCKER"):
            raise AssertionError("Docker is required in this job but unavailable")
        pytest.skip("docker daemon not available")
    await asyncio.to_thread(
        _ensure_image
    )  # a first run would otherwise spend its timeout on the download
    return sb


def leftover() -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "-a", "--filter", "label=aaa.sandbox=1", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return out.split()


@needs_docker
async def test_docker_runs_code_as_an_unprivileged_user_with_a_clean_environment(
    docker, monkeypatch
):
    monkeypatch.setenv("AAA_TEST_SECRET", "hunter2")
    r = await docker.run(
        "import os, sys; print(os.getuid(), os.environ.get('AAA_TEST_SECRET'), os.path.exists('/var/run/docker.sock'))"
    )
    assert r.ok, r.stderr
    assert r.stdout.split() == ["65534", "None", "False"]
    assert leftover() == []


@needs_docker
async def test_docker_has_no_network(docker):
    code = "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), timeout=3)\n    print('CONNECTED')\nexcept OSError as e:\n    print('blocked', type(e).__name__)"
    r = await docker.run(code)
    assert r.ok and r.stdout.startswith("blocked"), (r.stdout, r.stderr)


@needs_docker
async def test_docker_filesystem_is_read_only_except_a_small_noexec_scratch(docker):
    code = (
        "import os, subprocess\n"
        "for p in ('/x', '/usr/x', '/work/x'):\n"
        "    try:\n        open(p, 'w').write('a'); print(p, 'WRITABLE')\n    except OSError as e:\n        print(p, 'read-only')\n"
        "open('/tmp/ok', 'w').write('fine'); print('tmp ok')\n"
        "open('/tmp/x.sh', 'w').write('#!/bin/sh\\necho ran'); os.chmod('/tmp/x.sh', 0o755)\n"
        "try:\n    subprocess.run(['/tmp/x.sh'], check=True); print('EXEC FROM TMP')\nexcept (OSError, subprocess.CalledProcessError):\n    print('tmp noexec')\n"
    )
    r = await docker.run(code)
    assert r.ok, r.stderr
    assert (
        "WRITABLE" not in r.stdout
        and "tmp ok" in r.stdout
        and "tmp noexec" in r.stdout
        and "EXEC FROM TMP" not in r.stdout
    )


@needs_docker
async def test_docker_timeout_kills_and_removes_the_container(docker):
    r = await docker.run("while True: pass", limits=Limits(timeout_s=2))
    assert r.timed_out and r.killed == "timeout" and r.exit_code is None
    assert leftover() == []


@needs_docker
async def test_docker_memory_limit_kills_the_program(docker):
    r = await docker.run(
        "x = bytearray(1024 * 1024 * 1024)\nprint('allocated')",
        limits=Limits(memory_mb=64, timeout_s=20),
    )
    assert "allocated" not in r.stdout and not r.ok
    assert r.killed == "oom" or r.exit_code in (137, 1), (r.exit_code, r.killed, r.stderr)
    assert leftover() == []


@needs_docker
async def test_docker_fork_bomb_is_contained_by_the_pid_limit(docker):
    code = "import os\nn = 0\nwhile n < 500:\n    try:\n        if os.fork() == 0:\n            import time; time.sleep(5); os._exit(0)\n        n += 1\n    except OSError:\n        print('limited at', n); break\nelse:\n    print('NOT LIMITED')"
    r = await docker.run(code, limits=Limits(pids=20, timeout_s=15))
    assert "NOT LIMITED" not in r.stdout and "limited at" in r.stdout, (r.stdout, r.stderr)


@needs_docker
async def test_docker_output_is_capped_and_the_run_killed(docker):
    r = await docker.run(
        "while True: print('x' * 1000)", limits=Limits(timeout_s=20, max_output_bytes=20_000)
    )
    assert r.truncated and r.killed == "output" and len(r.stdout) <= 20_000
    assert leftover() == []


@needs_docker
async def test_docker_input_files_arrive_without_any_host_mount_and_nothing_persists(docker):
    r = await docker.run(
        "print(open('data.txt').read(), open('/tmp/work/data.txt').read())\nimport helper\nprint(helper.VALUE)",
        files={"data.txt": "payload", "helper.py": "VALUE = 41 + 1"},
    )
    assert r.ok, r.stderr
    assert r.stdout.split() == [
        "payload",
        "payload",
        "42",
    ]  # in the working directory, and importable
    assert (await docker.run("open('/tmp/marker', 'w').write('x')")).ok
    again = await docker.run(
        "import os; print(os.path.exists('/tmp/marker'), os.path.exists('/tmp/work/data.txt'))"
    )
    assert again.stdout.split() == [
        "False",
        "False",
    ]  # a new container every time: nothing persists


@needs_docker
async def test_docker_cannot_see_the_host(docker, tmp_path):
    (tmp_path / "hostfile").write_text("host secret")
    r = await docker.run(
        f"import os; print(os.path.exists({str(tmp_path / 'hostfile')!r}), os.path.exists('/Users'), os.path.exists('/home'))"
    )
    assert r.stdout.split() == ["False", "False", "False"] or r.stdout.split()[:1] == ["False"]


@needs_docker
async def test_docker_reports_the_codes_failure_as_a_result_not_an_error(docker):
    r = await docker.run("raise SystemExit(7)")
    assert r.exit_code == 7 and not r.ok
    sh = await docker.run("echo from-bash; exit 1", "bash")
    assert sh.stdout.strip() == "from-bash" and sh.exit_code == 1


async def test_docker_missing_binary_is_a_clear_error():
    with pytest.raises(SandboxError, match="not found"):
        await DockerSandbox(docker="/nonexistent/docker").run("print(1)")
