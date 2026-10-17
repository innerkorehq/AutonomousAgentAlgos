"""Per-tool policy: timeouts, rate limits, concurrency, and who may see a tool."""

import asyncio

import pytest

from aaa.mcp import MCPToolServer
from aaa.mcp.compile import acompile_spec, describe
from aaa.mcp.config import SpecError
from aaa.mcp.policy import Limiter, ToolRule
from aaa.mcp.spec import MCPSpec
from aaa.tools import ToolRegistry, tool


@tool
async def slow(seconds: float = 0.5) -> str:
    """Sleep."""
    await asyncio.sleep(seconds)
    return "done"


@tool
def ping() -> str:
    """Answer."""
    return "pong"


@tool
def admin_only() -> str:
    """Privileged."""
    return "secret"


def spec(**tools):
    return MCPSpec.model_validate(
        {"sources": [{"type": "live", "key": "t"}], "tools": tools, "tenancy": {"tenant": "acme"}}
    )


async def build(raw, clock=None, **kw):
    core = await acompile_spec(raw, live={"t": ToolRegistry([slow, ping, admin_only])}, **kw)
    if clock is not None:
        core._limiter.clock = clock
    return core


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ================================================================ limiter
def test_bucket_refills_with_time_and_is_per_key():
    c = Clock()
    lim = Limiter(c)
    assert [lim.allow("a", 60, 2) for _ in range(3)] == [True, True, False]
    assert lim.allow("b", 60, 2)  # another caller has its own bucket
    c.t += 1.0  # 60/min = one token a second
    assert lim.allow("a", 60, 2)
    assert not lim.allow("a", 60, 2)


def test_limiter_forgets_old_keys():
    lim = Limiter(Clock(), max_keys=3)
    for i in range(10):
        lim.allow(f"k{i}", 60, 1)
    assert len(lim._buckets) == 3


def test_rule_visibility():
    r = ToolRule(scopes=frozenset({"admin"}), tenants=frozenset({"acme"}))
    assert r.visible_to("acme", frozenset({"admin", "run"}))
    assert r.visible_to("acme", None)  # a caller the host did not restrict
    assert not r.visible_to("acme", frozenset({"run"}))
    assert not r.visible_to("globex", frozenset({"admin"}))


# ================================================================ compiled behaviour
async def test_timeout_comes_from_policy_and_override_beats_it():
    core = await build(
        spec(
            policies=[{"match": ["s*"], "timeout_s": 0.05}],
            overrides={"slow": {"timeout_s": 0.01}},
        )
    )
    r = await core.call("slow", {"seconds": 1})
    assert r.is_error and "timed out after 0.01s" in r.text
    core = await build(spec(policies=[{"match": ["s*"], "timeout_s": 0.05}]))
    assert "timed out after 0.05s" in (await core.call("slow", {"seconds": 1})).text


async def test_first_matching_policy_wins_per_setting():
    core = await build(
        spec(
            policies=[
                {"match": ["slow"], "timeout_s": 0.02},
                {"match": ["*"], "timeout_s": 5, "rate_limit": {"per_minute": 60, "burst": 1}},
            ]
        )
    )
    rule = core.tool_policy["slow"]
    assert (
        rule.timeout_s == 0.02 and rule.rate_per_minute == 60
    )  # timeout from the first, limit from the second
    assert core.tool_policy["ping"].timeout_s == 5


async def test_rate_limit_is_per_principal_and_tool_with_a_generic_message():
    c = Clock()
    core = await build(
        spec(policies=[{"match": ["ping"], "rate_limit": {"per_minute": 60, "burst": 2}}]), c
    )
    ok = [(await core.call("ping", {}, user="ann")).is_error for _ in range(3)]
    assert ok == [False, False, True]
    refused = await core.call("ping", {}, user="ann")
    assert refused.text == "ping is being called too often; retry shortly"
    assert "60" not in refused.text
    assert not (await core.call("ping", {}, user="bob")).is_error  # another caller
    assert not (await core.call("ping", {}, tenant="globex", user="ann")).is_error
    c.t += 2
    assert not (await core.call("ping", {}, user="ann")).is_error


async def test_concurrency_cap_refuses_instead_of_queueing_and_releases():
    core = await build(spec(policies=[{"match": ["slow"], "max_concurrency": 1}]))
    first = asyncio.create_task(core.call("slow", {"seconds": 0.3}))
    await asyncio.sleep(0.05)
    busy = await core.call("slow", {"seconds": 0.0})
    assert busy.is_error and busy.text == "slow is busy; retry shortly"
    assert not (await first).is_error
    assert not (await core.call("slow", {"seconds": 0.0})).is_error  # released
    # a failure also releases
    core2 = await build(spec(policies=[{"match": ["slow"], "max_concurrency": 1}]))
    assert (await core2.call("slow", {"seconds": "x"})).is_error
    assert not (await core2.call("slow", {"seconds": 0.0})).is_error


async def test_scope_hidden_tool_is_indistinguishable_from_unknown():
    core = await build(spec(policies=[{"match": ["admin_only"], "scopes": ["admin"]}]))
    names = lambda s: {t.name for t in core.list_tools(scopes=s)}
    assert "admin_only" in names(None)  # unrestricted caller
    assert "admin_only" in names({"admin"})
    assert "admin_only" not in names({"run"})
    hidden = await core.call("admin_only", {}, scopes={"run"})
    missing = await core.call("nothing_here", {}, scopes={"run"})
    assert hidden.is_error and hidden.text.replace("admin_only", "X") == missing.text.replace(
        "nothing_here", "X"
    )
    assert (await core.call("admin_only", {}, scopes={"admin"})).text == "secret"


async def test_tenancy_scopes_describe_the_stdio_caller():
    core = await build(
        MCPSpec.model_validate(
            {
                "sources": [{"type": "live", "key": "t"}],
                "tenancy": {"scopes": ["run"]},
                "tools": {"policies": [{"match": ["admin_only"], "scopes": ["admin"]}]},
            }
        )
    )
    assert "admin_only" not in {t.name for t in core.list_tools()}
    assert (await core.call("admin_only", {})).is_error
    assert "admin_only" not in core.hidden  # hidden by scope, not by approval


async def test_tenant_allow_list():
    core = await build(spec(overrides={"ping": {"tenants": ["globex"]}}))
    assert "ping" not in {t.name for t in core.list_tools()}  # server tenant is acme
    assert (await core.call("ping", {})).is_error
    assert (await core.call("ping", {}, tenant="globex")).text == "pong"


async def test_a_glob_that_selects_nothing_is_an_error():
    with pytest.raises(SpecError, match="selects no exposed tool"):
        await build(spec(policies=[{"match": ["pnig"], "timeout_s": 1}]))


def test_spec_rejects_nonsense():
    with pytest.raises(ValueError):
        spec(policies=[{"match": ["*"], "timeout_s": -1}])
    with pytest.raises(ValueError):
        spec(policies=[{"match": ["*"], "rate_limit": {"per_minute": 0}}])
    with pytest.raises(ValueError):
        spec(policies=[{"match": [], "timeout_s": 1}])
    with pytest.raises(ValueError):
        spec(policies=[{"match": ["*"], "scopes": []}])
    with pytest.raises(ValueError):
        spec(policies=[{"match": ["*"], "timeuot_s": 1}])


async def test_describe_lists_the_rules():
    core = await build(spec(policies=[{"match": ["ping"], "timeout_s": 3, "scopes": ["a"]}]))
    assert describe(core)["policies"] == {"ping": {"timeout_s": 3, "scopes": ["a"]}}


async def test_plain_server_is_unchanged():
    core = MCPToolServer(ToolRegistry([ping]))
    assert [t.name for t in core.list_tools()] == ["ping"]
    assert (await core.call("ping", {})).text == "pong"
