"""Configuration includes and overlays: merge rules, containment, cycles, precedence, flatten."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aaa.mcp.config import SpecError, load_spec
from aaa.mcp.include import compose, merge


def w(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def objs(ref):
    return f'[[sources]]\ntype = "objects"\nref = "{ref}"\n'


# ================================================================ merge rules
def test_merge_concatenates_the_named_lists_and_replaces_the_rest():
    a = {
        "sources": [1],
        "tools": {"include": ["a"], "policies": [1]},
        "server": {"name": "x", "title": "t"},
    }
    b = {"sources": [2], "tools": {"include": ["b"], "policies": [2]}, "server": {"name": "y"}}
    assert merge(a, b) == {
        "sources": [1, 2],
        "tools": {"include": ["b"], "policies": [1, 2]},
        "server": {"name": "y", "title": "t"},
    }


def test_including_file_wins_over_what_it_includes(tmp_path):
    w(tmp_path / "base.toml", '[server]\nname = "base"\ntitle = "Base"\n' + objs("a:b"))
    top = w(
        tmp_path / "top.toml",
        'include = ["base.toml"]\n[server]\nname = "top"\n' + objs("c:d"),
    )
    spec = load_spec(top, environ={})
    assert spec.server.name == "top" and spec.server.title == "Base"
    assert [s.ref for s in spec.sources] == ["a:b", "c:d"]  # included sources come first


def test_includes_in_order_and_nested(tmp_path):
    w(tmp_path / "tools" / "c.toml", objs("c:c"))
    w(tmp_path / "tools" / "a.toml", 'include = ["c.toml"]\n' + objs("a:a"))
    w(tmp_path / "tools" / "b.toml", objs("b:b"))
    top = w(tmp_path / "top.toml", 'include = ["tools/a.toml", "tools/b.toml"]\n')
    assert [s.ref for s in load_spec(top, environ={}).sources] == ["c:c", "a:a", "b:b"]


def test_paths_are_anchored_to_the_file_that_names_them(tmp_path):
    w(
        tmp_path / "apis" / "orders.toml",
        '[[sources]]\ntype = "openapi"\ndocument = "orders.json"\nallowed_hosts = ["api.example.com"]\n',
    )
    top = w(tmp_path / "top.toml", 'include = ["apis/orders.toml"]\n')
    assert load_spec(top, environ={}).sources[0].document == str(
        (tmp_path / "apis" / "orders.json").resolve()
    )


# ================================================================ containment
def test_escape_absolute_and_symlink_are_refused(tmp_path):
    outside = w(tmp_path / "outside" / "x.toml", objs("x:x"))
    root = tmp_path / "root"
    top = w(root / "top.toml", 'include = ["../outside/x.toml"]\n')
    with pytest.raises(SpecError, match="outside"):
        load_spec(top, environ={})
    top = w(root / "top.toml", f'include = ["{outside}"]\n')
    with pytest.raises(SpecError, match="absolute"):
        load_spec(top, environ={})
    (root / "link.toml").symlink_to(outside)
    top = w(root / "top.toml", 'include = ["link.toml"]\n')
    with pytest.raises(SpecError, match="outside"):
        load_spec(top, environ={})


def test_cycle_depth_size_and_shape(tmp_path):
    w(tmp_path / "a.toml", 'include = ["b.toml"]\n')
    w(tmp_path / "b.toml", 'include = ["a.toml"]\n')
    with pytest.raises(SpecError, match=r"cycle: .*a\.toml -> .*b\.toml -> .*a\.toml"):
        load_spec(tmp_path / "a.toml", environ={})
    for i in range(7):
        w(tmp_path / "d" / f"{i}.toml", f'include = ["{i + 1}.toml"]\n')
    w(tmp_path / "d" / "7.toml", "")
    with pytest.raises(SpecError, match="deeper than"):
        load_spec(tmp_path / "d" / "0.toml", environ={})
    w(tmp_path / "big.toml", "# " + "x" * 1_100_000 + "\n")
    with pytest.raises(SpecError, match="limited to"):
        load_spec(w(tmp_path / "t.toml", 'include = ["big.toml"]\n'), environ={})
    with pytest.raises(SpecError, match="list of relative"):
        load_spec(w(tmp_path / "t2.toml", 'include = "x.toml"\n'), environ={})
    with pytest.raises(SpecError, match="cannot read"):
        load_spec(w(tmp_path / "t3.toml", 'include = ["missing.toml"]\n'), environ={})
    with pytest.raises(SpecError, match="configuration file"):
        load_spec({"include": ["x.toml"]}, environ={})


def test_errors_name_the_file_that_is_at_fault(tmp_path):
    w(tmp_path / "bad.toml", "this is = not [valid")
    top = w(tmp_path / "top.toml", 'include = ["bad.toml"]\n')
    with pytest.raises(SpecError, match=r"bad\.toml"):
        load_spec(top, environ={})


def test_world_writable_included_file_that_imports_code_is_refused(tmp_path):
    inc = w(tmp_path / "inc.toml", objs("a:b"))
    top = w(tmp_path / "top.toml", 'include = ["inc.toml"]\n')
    os.chmod(top, 0o644)
    os.chmod(inc, 0o666)
    with pytest.raises(SpecError, match=r"inc\.toml.*writable by everyone"):
        load_spec(top, environ={})


# ================================================================ overlays
def test_precedence_file_then_overlays_then_env_then_flags(tmp_path):
    top = w(tmp_path / "top.toml", '[server]\nname = "file"\ntitle = "file"\n' + objs("a:b"))
    o1 = w(tmp_path / "o1.toml", '[server]\nname = "o1"\ntitle = "o1"\n')
    o2 = w(tmp_path / "ov" / "o2.toml", '[server]\nname = "o2"\n')
    spec = load_spec(top, environ={}, overlays=[o1, o2])
    assert (spec.server.name, spec.server.title) == ("o2", "o1")  # later overlay wins, per field
    spec = load_spec(
        top,
        environ={"AAA_MCP__SERVER__NAME": "env"},
        overlays=[o1, o2],
        overlay={"server": {"title": "flag"}},
    )
    assert (spec.server.name, spec.server.title) == ("env", "flag")


def test_overlay_from_the_environment_variable(tmp_path):
    top = w(tmp_path / "top.toml", '[server]\nname = "file"\n' + objs("a:b"))
    o1 = w(tmp_path / "o1.toml", '[server]\nname = "o1"\n')
    o2 = w(tmp_path / "o2.toml", '[server]\nname = "o2"\n')
    env = {"AAA_MCP_OVERLAY": os.pathsep.join([str(o1), str(o2)])}
    assert load_spec(top, environ=env).server.name == "o2"
    assert (
        load_spec(top, environ=env, overlays=[o1]).server.name == "o1"
    )  # explicit ones come after


def test_overlay_can_add_sources_and_its_own_includes_stay_in_its_directory(tmp_path):
    top = w(tmp_path / "top.toml", objs("a:b"))
    w(tmp_path / "ov" / "extra.toml", objs("e:e"))
    ov = w(tmp_path / "ov" / "prod.toml", 'include = ["extra.toml"]\n' + objs("p:p"))
    assert [s.ref for s in load_spec(top, environ={}, overlays=[ov]).sources] == [
        "a:b",
        "e:e",
        "p:p",
    ]
    bad = w(tmp_path / "ov" / "bad.toml", 'include = ["../top.toml"]\n')
    with pytest.raises(SpecError, match="outside"):
        load_spec(top, environ={}, overlays=[bad])


def test_compose_lists_every_contributing_file(tmp_path):
    w(tmp_path / "inc.toml", objs("a:b"))
    top = w(tmp_path / "top.toml", 'include = ["inc.toml"]\n')
    ov = w(tmp_path / "o.toml", "")
    names = [f.name for f in compose(top, [ov]).files]
    assert names == ["top.toml", "inc.toml", "o.toml"]


# ================================================================ flatten and the plain case
def run_cli(*args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "aaa.cli", "mcp", *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_flatten_round_trips_to_the_same_spec(tmp_path):
    w(
        tmp_path / "apis" / "orders.toml",
        '[[sources]]\ntype = "openapi"\ndocument = "orders.json"\nallowed_hosts = ["api.example.com"]\n',
    )
    top = w(
        tmp_path / "top.toml",
        'include = ["apis/orders.toml"]\n[server]\nname = "s"\n' + objs("a:b"),
    )
    ov = w(tmp_path / "prod.toml", '[server]\ntitle = "Prod"\n')
    out = tmp_path / "out" / "flat.toml"
    out.parent.mkdir()
    r = run_cli("flatten", "--config", str(top), "--overlay", str(ov), "-o", str(out), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    flat = out.read_text()
    assert "include" not in flat and 'document = "../apis/orders.json"' in flat
    a = load_spec(top, environ={}, overlays=[ov]).model_dump(mode="json")
    b = load_spec(out, environ={}).model_dump(mode="json")
    assert a == b


def test_flatten_to_stdout_is_json(tmp_path):
    top = w(tmp_path / "top.toml", '[server]\nname = "s"\n' + objs("a:b"))
    r = run_cli("flatten", "--config", str(top), cwd=tmp_path)
    assert r.returncode == 0 and json.loads(r.stdout)["server"]["name"] == "s"


def test_a_file_without_includes_is_unchanged(tmp_path):
    top = w(tmp_path / "top.toml", '[server]\nname = "s"\n' + objs("a:b"))
    assert load_spec(top, environ={}).server.name == "s"
