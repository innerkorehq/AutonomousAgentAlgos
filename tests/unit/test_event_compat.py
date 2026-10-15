"""The event log is the user's data: logs written by an OLD release must still be readable by this code.

`tests/fixtures/events_v0_2/` holds real logs produced by v0.2.0 (see `scripts/gen_event_fixtures.py`) and the state every Projection
folded out of them back then. Here the CURRENT code replays the same files and must produce identical state. If this fails you changed
how an existing event is read: make the change additive (see docs/policy/stability.md), or add a new event type and keep reading the old one.
These fixtures are frozen: never regenerate them for a released tag."""

import json
import re
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from aaa.algos.control import FSM_STATE, WF
from aaa.algos.ensemble import SAMPLES
from aaa.algos.loops import LOOP
from aaa.algos.plan import PLAN
from aaa.algos.refine import REFINE
from aaa.events import Event, InMemoryEventStore
from aaa.runtime.engine import RUNMETA
from aaa.session import SESSION

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "events_v0_2"
PROJECTIONS = {
    "runmeta": RUNMETA,
    "loop": LOOP,
    "samples": SAMPLES,
    "workflow": WF,
    "fsm": FSM_STATE,
    "plan": PLAN,
    "refine": REFINE,
    "session": SESSION,
}
SCENARIOS = sorted(p.name.removesuffix(".jsonl") for p in FIX.glob("*.jsonl"))
CATALOG = Path(__file__).resolve().parents[2] / "docs" / "reference" / "event-catalog.md"
# subscription patterns that appear in the catalog table but are not event types
PATTERNS = {"aaa.engine.cancel", "aaa.llm", "aaa.tool.request", "aaa.validate"}
# event types added after v0.2.0 (so no v0.2 fixture can exist); a later release freezes its own fixtures
SINCE_0_3 = {"aaa.swarm.handoff", "aaa.reflexion.feedback"}


def streams(name: str) -> dict[str, list[Event]]:
    out: dict[str, list[Event]] = {}
    for line in (FIX / f"{name}.jsonl").read_text().splitlines():
        row = json.loads(line)
        out.setdefault(row["stream"], []).append(Event.model_validate(row["event"]))
    return out


def test_the_fixtures_exist_and_cover_every_documented_event_type():
    assert len(SCENARIOS) >= 20
    manifest = json.loads((FIX / "MANIFEST.json").read_text())
    assert manifest["generated_with"] == "0.2.0"
    covered = {t for types in manifest["scenarios"].values() for t in types}
    documented = (
        set(re.findall(r"^\| `(aaa\.[a-z_.]+)`", CATALOG.read_text(), re.MULTILINE))
        - PATTERNS
        - SINCE_0_3
    )
    assert not (documented - covered), (
        f"event types with no v0.2 fixture: {sorted(documented - covered)}"
    )


@pytest.mark.parametrize("name", SCENARIOS)
def test_every_stored_event_still_parses_and_round_trips(name):
    for evs in streams(name).values():
        for e in evs:
            again = Event.model_validate_json(e.model_dump_json())
            assert again == e


@pytest.mark.parametrize("name", SCENARIOS)
async def test_current_projections_fold_old_logs_to_the_same_state(name):
    expected = json.loads((FIX / f"{name}.expected.json").read_text())
    store = InMemoryEventStore()
    for sid, evs in streams(name).items():
        await store.append(sid, [e.model_copy(update={"seq": -1}) for e in evs])
    checked = 0
    for sid, per_projection in expected.items():
        for pname, want in per_projection.items():
            st = (await PROJECTIONS[pname].load(store, sid, use_snapshots=False)).state
            got = TypeAdapter(type(st)).dump_python(st, mode="json")
            assert json.loads(json.dumps(got, sort_keys=True, default=str)) == json.loads(
                json.dumps(want, sort_keys=True, default=str)
            ), f"{name}/{sid}/{pname}"
            checked += 1
    assert checked > 0


async def test_a_run_log_with_every_event_type_is_readable_by_the_trace_and_cloudevents_exporters():
    """Observers read old logs too: derive spans and CloudEvents from every fixture."""
    from aaa.events.cloudevents import to_cloudevent
    from aaa.trace import spans_from_events

    for name in SCENARIOS:
        for evs in streams(name).values():
            spans_from_events(evs)  # must not raise on any historical shape
            for e in evs:
                assert to_cloudevent(e)["type"] == e.type
