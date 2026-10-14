"""Real OS processes, a shared database, a SIGKILL. No in-process simulation."""

import asyncio
import os
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from aaa.algos import ToolLoop
from aaa.events import ClientBus
from aaa.runtime import Engine
from aaa.tools import ToolRegistry
from cluster import Cluster, until

ROOT = Path(__file__).parents[2]
WORKER = ROOT / "tests" / "helpers" / "worker.py"


def spawn(kind, target, member, prefix, effects):
    return subprocess.Popen(
        [sys.executable, "-I", str(WORKER), kind, target, member, prefix, str(effects)],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


async def ready(proc):
    line = await asyncio.wait_for(asyncio.to_thread(proc.stdout.readline), 30)
    assert "READY" in line, f"worker failed to start: {line!r}"


@pytest.mark.parametrize("kind", ["postgres", "redis"])
async def test_sigkill_one_of_three_worker_processes_every_run_still_completes(
    kind, request, tmp_path, prefix
):
    cluster = Cluster(
        kind,
        partitions=8,
        admin_dsn=request.getfixturevalue("pg_admin_dsn") if kind == "postgres" else None,
        url=request.getfixturevalue("redis_url") if kind == "redis" else None,
        prefix=prefix,
    )
    await cluster.setup()
    target = cluster.dsn if kind == "postgres" else cluster.url
    effects = tmp_path / "effects.log"
    procs = [spawn(kind, target, f"w{i}", prefix, effects) for i in range(3)]
    client = None
    try:
        for p in procs:
            await ready(p)
        node = await cluster.node()  # the test process is a submit-only client
        client = Engine(store=node.store, bus=ClientBus())
        algo = ToolLoop(ToolRegistry())
        await asyncio.sleep(2.0)  # let the three workers rebalance
        handles = [
            await client.start(algo, {"task": f"compute {i}"}, run_id=f"mp-{i}")
            for i in range(1, 31)
        ]
        await asyncio.sleep(0.6)  # runs are now in flight on several workers
        procs[1].send_signal(signal.SIGKILL)  # hard-kill one worker: no cleanup of any kind
        procs[1].wait(timeout=5)

        outs = await asyncio.wait_for(asyncio.gather(*(h.result() for h in handles)), 90)
        assert all(o.ok for o in outs), [o for o in outs if not o.ok][:2]
        assert [o.value["text"] for o in outs][:3] == ["result 2", "result 4", "result 6"]

        assert len(await node.store.run_index()) == 30
        for h in handles:
            log = await node.store.read(h.run_id)
            c = Counter(e.type for e in log)
            assert c["aaa.run.completed"] == 1 and c["aaa.tool.completed"] == 1
            assert len({e.id for e in log}) == len(log)  # one result per request, no duplicates
        done = Counter(int(line.split()[0]) for line in effects.read_text().splitlines())
        assert set(done) == set(range(1, 31))  # every side effect happened...
        assert max(done.values()) <= 2  # ...at-least-once: only the kill can repeat one
        assert sum(done.values()) <= 30 + 8  # bounded by work in flight on the dead worker
        who = Counter(line.split()[1] for line in effects.read_text().splitlines())
        assert len(who) >= 2  # work was genuinely shared between processes
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=5)
            if p.stdout:
                p.stdout.close()
        if client:
            await client.__aexit__(None, None, None)
        await cluster.close()


async def _cluster_with_workers(kind, request, tmp_path, prefix, env):
    cluster = Cluster(
        kind,
        partitions=8,
        admin_dsn=request.getfixturevalue("pg_admin_dsn") if kind == "postgres" else None,
        url=request.getfixturevalue("redis_url") if kind == "redis" else None,
        prefix=prefix,
    )
    await cluster.setup()
    target = cluster.dsn if kind == "postgres" else cluster.url
    effects = tmp_path / "effects.log"
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        procs = [spawn(kind, target, f"w{i}", prefix, effects) for i in range(3)]
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    for p in procs:
        await ready(p)
    node = await cluster.node()
    client = Engine(store=node.store, bus=ClientBus())
    return cluster, procs, node, client, effects


def _lines(path):
    return [ln.split() for ln in path.read_text().splitlines()] if path.exists() else []


async def _kill_mid_tool(procs, effects):
    """SIGKILL the process that is in the middle of the first running tool call; return its member and
    the call numbers that were in flight on it."""
    starts = effects.parent / "effects.log.starts"
    await until(lambda: len(_lines(starts)) >= 3, 30)
    victim = _lines(starts)[0][1]
    procs[int(victim[1:])].send_signal(signal.SIGKILL)
    procs[int(victim[1:])].wait(timeout=5)
    done = {n for n, *_ in _lines(effects)}
    in_flight = {int(n) for n, m in _lines(starts) if m == victim and n not in done}
    return in_flight


@pytest.mark.parametrize("kind", ["postgres", "redis"])
async def test_sigkill_in_the_middle_of_a_tool_call_another_process_finishes_it(
    kind, request, tmp_path, prefix
):
    cluster, procs, node, client, effects = await _cluster_with_workers(
        kind, request, tmp_path, prefix, {"AAA_TOOL_SECONDS": "2.0", "AAA_LEASE_TTL": "1.5"}
    )
    try:
        algo = ToolLoop(ToolRegistry())
        await asyncio.sleep(2.0)
        handles = [
            await client.start(algo, {"task": f"compute {i}"}, run_id=f"k-{i}")
            for i in range(1, 10)
        ]
        in_flight = await _kill_mid_tool(procs, effects)
        assert in_flight, "the kill missed every running tool call"

        outs = await asyncio.wait_for(asyncio.gather(*(h.result() for h in handles)), 120)
        assert all(o.ok for o in outs), [o for o in outs if not o.ok][:2]
        starts = Counter(int(n) for n, _ in _lines(effects.parent / "effects.log.starts"))
        who = {
            n: {m for n2, m in _lines(effects.parent / "effects.log.starts") if int(n2) == n}
            for n in in_flight
        }
        assert all(
            starts[n] == 2 and len(who[n]) == 2 for n in in_flight
        )  # re-run once, on a DIFFERENT process
        assert all(
            starts[n] == 1 for n in set(range(1, 10)) - in_flight
        )  # everything else exactly once
        for h in handles:
            log = await node.store.read(h.run_id)
            c = Counter(e.type for e in log)
            assert c["aaa.tool.completed"] == 1 and c["aaa.run.completed"] == 1
            n = int(h.run_id.split("-")[1])
            assert [e.data["attempt"] for e in log if e.type == "aaa.tool.started"] == (
                [1, 2] if n in in_flight else [1]
            )
    finally:
        await _teardown(procs, client, cluster)


@pytest.mark.parametrize("kind", ["postgres", "redis"])
async def test_sigkill_mid_call_for_an_at_most_once_tool_is_reported_not_rerun(
    kind, request, tmp_path, prefix
):
    cluster, procs, _node, client, effects = await _cluster_with_workers(
        kind,
        request,
        tmp_path,
        prefix,
        {"AAA_TOOL_SECONDS": "2.0", "AAA_LEASE_TTL": "1.5", "AAA_TOOL_DELIVERY": "at_most_once"},
    )
    try:
        algo = ToolLoop(ToolRegistry())
        await asyncio.sleep(2.0)
        handles = [
            await client.start(algo, {"task": f"compute {i}"}, run_id=f"m-{i}")
            for i in range(1, 10)
        ]
        in_flight = await _kill_mid_tool(procs, effects)
        assert in_flight

        outs = {
            int(h.run_id.split("-")[1]): o
            for h, o in zip(
                handles,
                await asyncio.wait_for(asyncio.gather(*(h.result() for h in handles)), 120),
                strict=True,
            )
        }
        starts = Counter(int(n) for n, _ in _lines(effects.parent / "effects.log.starts"))
        for n, o in outs.items():
            assert o.ok
            if n in in_flight:
                assert (
                    "OutcomeUnknown" in o.value["text"] and starts[n] == 1
                )  # never executed a second time
            else:
                assert "OutcomeUnknown" not in o.value["text"] and starts[n] == 1
    finally:
        await _teardown(procs, client, cluster)


async def _teardown(procs, client, cluster):
    for p in procs:
        if p.poll() is None:
            p.kill()
        p.wait(timeout=5)
        if p.stdout:
            p.stdout.close()
    await client.__aexit__(None, None, None)
    await cluster.close()
