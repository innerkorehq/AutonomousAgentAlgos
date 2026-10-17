"""Ephemeral Postgres / Redis for adapter tests.

We never touch the developer's own servers: each session launches throwaway instances on free
ports in a temp dir (initdb / redis-server), or uses AAA_TEST_POSTGRES_URL / AAA_TEST_REDIS_URL
if provided. Tests skip cleanly when the binaries are unavailable."""

from __future__ import annotations

import asyncio
import glob
import os
import shutil
import socket
import subprocess
import tempfile
import time
import uuid

import pytest

# golden project files include a generated `tests/test_server.py`: data to compare against, not tests to run here
collect_ignore_glob = ["golden/*"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float = 20) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"port {port} never opened")


def _pg_bin() -> str | None:
    cands = [os.environ.get("AAA_TEST_PG_BIN", "")]
    cands += glob.glob("/opt/homebrew/opt/postgresql*/bin") + glob.glob("/usr/lib/postgresql/*/bin")
    cands += glob.glob("/usr/local/opt/postgresql*/bin")
    for c in cands:
        if c and os.path.exists(os.path.join(c, "initdb")):
            return c
    which = shutil.which("initdb")
    return os.path.dirname(which) if which else None


@pytest.fixture(scope="session")
def pg_admin_dsn():
    if url := os.environ.get("AAA_TEST_POSTGRES_URL"):
        yield url
        return
    pgbin = _pg_bin()
    if not pgbin:
        pytest.skip("PostgreSQL binaries not found (set AAA_TEST_PG_BIN or AAA_TEST_POSTGRES_URL)")
    root = tempfile.mkdtemp(prefix="aaa-pg-")
    port = _free_port()
    env = {**os.environ, "LC_ALL": "C"}  # postgres >= 17 refuses to start without a valid locale
    subprocess.run(
        [
            os.path.join(pgbin, "initdb"),
            "-D",
            f"{root}/data",
            "-U",
            "aaa",
            "--auth=trust",
            "-E",
            "UTF8",
            "--locale=C",
        ],
        check=True,
        capture_output=True,
        env=env,
    )
    proc = subprocess.Popen(
        [
            os.path.join(pgbin, "postgres"),
            "-D",
            f"{root}/data",
            "-p",
            str(port),
            "-k",
            root,
            "-c",
            "listen_addresses=127.0.0.1",
            "-c",
            "fsync=off",
            "-c",
            "max_connections=200",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        _wait_port(port)
        time.sleep(0.5)
        yield f"postgresql://aaa@127.0.0.1:{port}/postgres"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
async def pg_dsn(pg_admin_dsn):
    """A fresh empty database per test."""
    import asyncpg

    name = "t_" + uuid.uuid4().hex[:12]
    admin = await asyncpg.connect(pg_admin_dsn)
    await admin.execute(f'CREATE DATABASE "{name}"')
    await admin.close()
    yield pg_admin_dsn.rsplit("/", 1)[0] + "/" + name
    admin = await asyncpg.connect(pg_admin_dsn)
    await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    await admin.close()


@pytest.fixture(scope="session")
def redis_base_url():
    if url := os.environ.get("AAA_TEST_REDIS_URL"):
        yield url
        return
    exe = shutil.which("redis-server")
    if not exe:
        pytest.skip("redis-server not found (set AAA_TEST_REDIS_URL)")
    root = tempfile.mkdtemp(prefix="aaa-redis-")
    port = _free_port()
    proc = subprocess.Popen(
        [
            exe,
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            root,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_port(port)
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def redis_url(redis_base_url):
    """Shared server, isolated per test by a unique key prefix (see `prefix`)."""
    return redis_base_url


@pytest.fixture
def prefix():
    return "aaa" + uuid.uuid4().hex[:10]


@pytest.fixture
def event_loop_policy():  # pragma: no cover - keeps anyio/pytest-asyncio happy across versions
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(params=["memory", "postgres", "redis"])
async def cluster(request, prefix):
    """The same cluster tests run against every backend."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from cluster import Cluster

    kind = request.param
    if kind == "memory":
        c = Cluster("memory")
    elif kind == "postgres":
        c = Cluster("postgres", admin_dsn=request.getfixturevalue("pg_admin_dsn"))
    else:
        c = Cluster("redis", url=request.getfixturevalue("redis_url"), prefix=prefix)
    await c.setup()
    yield c
    await c.close()


@pytest.fixture(autouse=True)
def _hang_watchdog():
    """If a test runs far longer than any legitimate one, print every pending asyncio task's stack
    (a hung event loop is invisible to faulthandler, which only shows the idle selector). Runs on a
    timer thread and only reads, so it cannot disturb the test."""
    import gc
    import sys
    import threading

    def dump() -> None:
        print("\n=== WATCHDOG: test still running after 150s; pending tasks ===", file=sys.stderr)
        for obj in gc.get_objects():
            if isinstance(obj, asyncio.Task) and not obj.done():
                print(f"--- {obj.get_name()} {obj.get_coro()!r}", file=sys.stderr)
                obj.print_stack(limit=6, file=sys.stderr)
        sys.stderr.flush()

    timer = threading.Timer(150, dump)
    timer.daemon = True
    timer.start()
    yield
    timer.cancel()
