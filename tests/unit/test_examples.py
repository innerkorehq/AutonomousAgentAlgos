"""Every example embedded in the documentation is executed here, so the docs cannot drift from the
code. Examples run offline against a scripted model (or ephemeral local state); the two cluster
examples need a database and run forever, so they are only compile-checked (the behavior they show is
covered by test_multiprocess.py, which launches real worker processes)."""

import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
EXAMPLES = ROOT / "examples"

EXPECTED = {
    "quickstart.py": ["success - 17 x 23 = 391"],
    "inspect_log.py": [
        "0 aaa.run.started",
        "4 aaa.tool.started",
        "8 aaa.run.completed",
        "outcome: success",
    ],
    "data_codecs.py": [
        "ticket  f7199c89",
        "row     d578aa73",
        "event   81fa4071",
        "doc     2d87e6fc",
        "frame   78efaf7c",
        "col_a=1, col_b=x",
        "preflight: missing capabilities: scorer",
    ],
    "tool_approval.py": ["waiting for approval of: wire_money", "sent $250 to ACME"],
    "plan_and_execute.py": ["Tokyo + Delhi = 46 million people", "steps run: ['a', 'b', 'c']"],
    "generate_test_repair.py": ['success ["apple", "pear", "fig"] after 3 tries'],
    "router_supervisor.py": [
        "Refund issued.",
        "Findings: AAA is event-driven.",
        "Draft: AAA logs everything as events.",
    ],
    "workflow_fsm.py": ["workflow: 22", "fsm path: draft -> review -> draft -> review -> done"],
    "model_router.py": ["[small:fast-1] quick answer", "[big:deep-9] careful answer"],
    "streaming.py": ["Streaming works token by token.", "outcome: success"],
    "persistence_recovery.py": [
        "process 1 died with the run open",
        "process 2 recovered 1 run(s): result: value-of-x",
    ],
    "custom_algorithm.py": ["{'French': 'Bonjour', 'German': 'Hallo', 'Spanish': 'Hola'}"],
    "tool_guard.py": [
        "refund  -> at_most_once | max_attempts = 3",
        "search  -> at_least_once",
        "refunded A-100 (key=refund:A-100, attempt=1, fence=1)",
    ],
    "testing_replay.py": ["replayed identically: 5"],
    "rag_agent.py": ["Paris is the capital of France [d1]."],
    "large_payloads.py": [
        "got 60000 chars",
        "stored event keys: ['$claim', 'call_id', 'name']",
        "hydrated result length: 60000",
    ],
    "llm_gateways.py": [
        "POST https://api.portkey.ai/v1/chat/completions",
        "POST http://kong:8000/chat",
        "custom provider registered",
    ],
    "ensembles.py": [
        "vote: {'answer': '42', 'votes': 3, 'samples': 5, 'agreement': 0.6}",
        "critic loop: {'candidate': 'draft v2', 'iterations': 2}",
    ],
}
COMPILE_ONLY = {"cluster_worker.py", "cluster_client.py"}


def test_every_example_is_accounted_for():
    on_disk = {p.name for p in EXAMPLES.glob("*.py")}
    assert on_disk == set(EXPECTED) | COMPILE_ONLY, "add new examples to EXPECTED (and the docs)"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_example_runs_and_prints_what_the_docs_say(name):
    proc = subprocess.run(
        [sys.executable, "-I", str(EXAMPLES / name)],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert proc.returncode == 0, f"{name} failed:\n{proc.stdout}\n{proc.stderr}"
    for expected in EXPECTED[name]:
        assert expected in proc.stdout, f"{name}: missing {expected!r} in:\n{proc.stdout}"


@pytest.mark.parametrize("name", sorted(COMPILE_ONLY))
def test_cluster_examples_at_least_compile(name, tmp_path):
    py_compile.compile(str(EXAMPLES / name), cfile=str(tmp_path / "x.pyc"), doraise=True)


DOCS = ROOT / "docs"
MARKER = __import__("re").compile(r'^--8<-- "(?P<path>[^"]+)"$')


def test_docs_embed_markers_are_well_formed_and_point_at_real_files():
    """A mangled `--8<--` marker still builds (it is just text) but publishes a broken page."""
    problems = []
    for md in DOCS.rglob("*.md"):
        for n, line in enumerate(md.read_text().splitlines(), 1):
            if (
                line.startswith("--8")
                or "--8 <" in line
                or "8<--" in line
                and not line.lstrip().startswith("`")
            ):
                m = MARKER.match(line)
                if not m or not (ROOT / m["path"]).is_file():
                    problems.append(f"{md.relative_to(ROOT)}:{n}: {line!r}")
    assert not problems, "\n".join(problems)


def test_every_example_is_embedded_somewhere_in_the_docs():
    embedded = set()
    for md in DOCS.rglob("*.md"):
        embedded |= {
            m["path"].split("/")[-1]
            for line in md.read_text().splitlines()
            if (m := MARKER.match(line))
        }
    missing = {p.name for p in EXAMPLES.glob("*.py")} - embedded
    assert not missing, f"examples not shown in the docs: {sorted(missing)}"
