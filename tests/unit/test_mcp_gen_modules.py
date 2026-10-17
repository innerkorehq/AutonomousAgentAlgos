"""The Python-module source: explicit exposure, introspection without execution, approval by default, and the kill-switch."""

import itertools
import sys
import textwrap

import pytest

from aaa.mcp.compile import acompile_spec
from aaa.mcp.config import SpecError
from aaa.mcp.sources.pymodule import parse_docstring
from aaa.mcp.spec import MCPSpec

_counter = itertools.count()


def write_module(tmp_path, body: str) -> str:
    """Write a module with a fresh name into tmp_path and return the name."""
    name = f"gentools_{next(_counter)}_{id(tmp_path) % 10_000}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body))
    return name


async def compile_module(tmp_path, body: str, **src):
    name = write_module(tmp_path, body)
    spec = MCPSpec.model_validate(
        {"sources": [{"type": "module", "module": name, "path": str(tmp_path), **src}]}
    )
    try:
        return await acompile_spec(spec), name
    finally:
        sys.modules.pop(name, None)


BASIC = '''
    __all__ = ["add", "greet", "wipe", "stats"]

    def add(a: int, b: int) -> int:
        """Add two integers.

        Args:
            a: the first number
            b (int): the second number,
                which may wrap onto a second line

        Returns:
            the sum
        """
        return a + b

    async def greet(name: str, excited: bool = False) -> str:
        """Greet someone."""
        return f"Hello {name}" + ("!" if excited else "")

    def wipe() -> str:
        """Delete everything."""
        return "wiped"

    def stats(xs: list[float]) -> dict:
        """Mean and count."""
        return {"mean": sum(xs) / len(xs), "n": len(xs)}
'''


# ===================================================================== exposure rules
async def test_nothing_is_exposed_without_approval_by_default(tmp_path):
    core, _ = await compile_module(tmp_path, BASIC)
    assert core.list_tools() == [] and sorted(core.hidden) == ["add", "greet", "stats", "wipe"]
    assert (
        await core.call("add", {"a": 1, "b": 2})
    ).is_error  # hidden = indistinguishable from unknown


async def test_listing_safe_functions_exposes_them_with_schemas_and_descriptions(tmp_path):
    core, _ = await compile_module(
        tmp_path, BASIC, no_approval=["add", "greet"], read_only=["stats"]
    )
    by = {t.name: t for t in core.list_tools()}
    assert sorted(by) == ["add", "greet", "stats"]  # wipe still needs approval
    assert by["add"].description == "Add two integers."
    props = by["add"].input_schema["properties"]
    assert props["a"]["description"] == "the first number"
    assert props["b"]["description"] == "the second number, which may wrap onto a second line"
    assert by["stats"].annotations is not None and by["stats"].annotations.read_only is True
    assert (
        await core.call("add", {"a": 2, "b": 3})
    ).text == "5"  # sync function, validated by its hints
    assert (
        await core.call("greet", {"name": "Ann", "excited": True})
    ).text == "Hello Ann!"  # async function
    assert (await core.call("stats", {"xs": [1, 2, 3]})).text == '{"mean": 2.0, "n": 3}'
    bad = await core.call("add", {"a": "x", "b": 2})
    assert bad.is_error and "invalid arguments" in bad.text
    assert (await core.call("add", {"a": 1, "b": 2, "c": 3})).is_error  # unknown arguments refused


async def test_default_approval_can_be_turned_off_for_a_trusted_module(tmp_path):
    core, _ = await compile_module(tmp_path, BASIC, default_approval=False)
    assert sorted(t.name for t in core.list_tools()) == ["add", "greet", "stats", "wipe"]


async def test_return_types_become_output_schemas(tmp_path):
    body = '''
        from pydantic import BaseModel
        __all__ = ["locate"]
        class Point(BaseModel):
            x: float
            y: float
        def locate(name: str) -> Point:
            """Find it."""
            return Point(x=1.0, y=2.0)
    '''
    core, _ = await compile_module(tmp_path, body, default_approval=False)
    (t,) = core.list_tools()
    assert t.output_schema is not None and t.output_schema["properties"]["x"]["type"] == "number"
    assert (await core.call("locate", {"name": "n"})).structured == {"x": 1.0, "y": 2.0}


async def test_a_function_can_receive_the_hosts_tenant_through_ctx(tmp_path):
    body = '''
        from aaa.tools.tool import ToolContext
        __all__ = ["whoami"]
        def whoami(ctx: ToolContext) -> str:
            """Who is asking."""
            return f"{ctx.tenant}/{ctx.meta.get('user')}"
    '''
    core, _ = await compile_module(tmp_path, body, default_approval=False)
    assert (await core.call("whoami", tenant="acme", user="ann")).text == "acme/ann"


# ===================================================================== what is refused
async def test_without_all_or_allow_there_is_no_default(tmp_path):
    with pytest.raises(SpecError, match="no __all__"):
        await compile_module(tmp_path, "def f(x: int) -> int:\n    return x\n")


async def test_allow_overrides_all_and_private_or_missing_names_are_errors(tmp_path):
    core, _ = await compile_module(tmp_path, BASIC, allow=["add"], default_approval=False)
    assert [t.name for t in core.list_tools()] == ["add"]
    with pytest.raises(SpecError, match="private"):
        await compile_module(
            tmp_path, "def _hidden(x: int) -> int:\n    return x\n", allow=["_hidden"]
        )
    with pytest.raises(SpecError, match="no attribute 'nope'"):
        await compile_module(tmp_path, BASIC, allow=["nope"])


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ("class C:\n    pass\n", "not a function"),
        ("f = lambda x: x\n", "lambda"),
        ("def f(x: int):\n    yield x\n", "generator"),
        ("def f(*args: int) -> int:\n    return 1\n", "named, typed"),
        ("def f(**kw: int) -> int:\n    return 1\n", "named, typed"),
        ("def f(x) -> int:\n    return 1\n", "no type hint"),
        ("VALUE = 3\n", "not a function"),
    ],
)
async def test_unsuitable_callables_are_refused_with_a_reason(tmp_path, body, why):
    with pytest.raises(SpecError, match=why):
        await compile_module(
            tmp_path,
            body,
            allow=[
                "C" if "class" in body else "f" if "def" in body or "lambda" in body else "VALUE"
            ],
        )


async def test_non_strict_mode_accepts_unannotated_parameters_with_a_warning(tmp_path):
    core, _ = await compile_module(
        tmp_path,
        "__all__ = ['f']\ndef f(x, y: int = 1) -> str:\n    return str(x) + str(y)\n",
        strict=False,
        default_approval=False,
    )
    assert any("unannotated parameters ['x']" in n for n in core.report)
    assert (await core.call("f", {"x": "a", "y": 2})).text == "a2"


async def test_reexports_are_skipped_from_all_but_available_through_allow(tmp_path):
    body = "from os.path import join\n__all__ = ['join', 'mine']\ndef mine(x: int) -> int:\n    return x\n"
    core, _ = await compile_module(tmp_path, body, default_approval=False)
    assert [t.name for t in core.list_tools()] == ["mine"] and any(
        "re-exported" in n for n in core.report
    )


async def test_exclude_and_unknown_safe_names(tmp_path):
    core, _ = await compile_module(tmp_path, BASIC, exclude=["wipe", "st*"], no_approval=["add"])
    assert [t.name for t in core.list_tools()] == ["add"] and "wipe" not in core.hidden
    with pytest.raises(SpecError, match="not exposed"):
        await compile_module(tmp_path, BASIC, no_approval=["nothing_like_this"])


# ===================================================================== inspection never executes anything
async def test_functions_are_never_called_while_the_module_is_inspected(tmp_path):
    marker = tmp_path / "called.txt"
    body = f'''
        from pathlib import Path
        __all__ = ["danger"]
        def danger(x: int) -> int:
            """Would write a file if it ran."""
            Path({str(marker)!r}).write_text("ran")
            return x
    '''
    core, _ = await compile_module(tmp_path, body, default_approval=False)
    assert not marker.exists() and core.list_tools()[0].name == "danger"
    assert marker.exists() is False


# ===================================================================== import safety
async def test_the_kill_switch_stops_the_import_itself(tmp_path, monkeypatch):
    marker = tmp_path / "imported.txt"
    name = write_module(
        tmp_path,
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('side effect')\n__all__ = []\n",
    )
    spec = MCPSpec.model_validate(
        {"sources": [{"type": "module", "module": name, "path": str(tmp_path)}]}
    )
    monkeypatch.setenv("AAA_MCP_ALLOW_IMPORT", "0")
    with pytest.raises(SpecError, match="importing is disabled"):
        await acompile_spec(spec)
    assert not marker.exists() and name not in sys.modules
    monkeypatch.delenv("AAA_MCP_ALLOW_IMPORT")
    with pytest.raises(SpecError, match="no __all__|defines no"):
        await acompile_spec(
            spec
        )  # without the switch it is imported (the side effect runs), then rejected for exposing nothing
    assert marker.exists()  # which is exactly why the kill-switch exists
    sys.modules.pop(name, None)


async def test_the_import_path_must_exist_and_not_be_world_writable(tmp_path):
    name = write_module(tmp_path, "__all__ = []\n")
    with pytest.raises(SpecError, match="not a directory"):
        await acompile_spec(
            MCPSpec.model_validate(
                {"sources": [{"type": "module", "module": name, "path": str(tmp_path / "missing")}]}
            )
        )
    loose = tmp_path / "loose"
    loose.mkdir()
    loose.chmod(0o777)
    with pytest.raises(SpecError, match="writable by everyone"):
        await acompile_spec(
            MCPSpec.model_validate(
                {"sources": [{"type": "module", "module": name, "path": str(loose)}]}
            )
        )


async def test_a_module_that_fails_to_import_is_a_configuration_error_without_a_stack(tmp_path):
    with pytest.raises(SpecError, match=r"cannot import \(ZeroDivisionError"):
        await compile_module(tmp_path, "1 / 0\n", allow=["x"])
    with pytest.raises(SpecError, match="cannot import"):
        await acompile_spec(
            MCPSpec.model_validate(
                {"sources": [{"type": "module", "module": "definitely_not_a_module_xyz"}]}
            )
        )


def test_module_names_are_validated_by_the_spec():
    for bad in ("os; import sys", "a..b", "../x", "a-b", ""):
        with pytest.raises(Exception, match="String should match|at least|pattern"):
            MCPSpec.model_validate({"sources": [{"type": "module", "module": bad}]})


# ===================================================================== docstrings
def test_docstring_parsing():
    s, p = parse_docstring("""Do a thing.

    More detail that is not part of the summary.

    Args:
        x: first
        y (str): second, long
            continued here

    Raises:
        ValueError: if bad
    """)
    assert s == "Do a thing." and p == {"x": "first", "y": "second, long continued here"}
    assert parse_docstring(None) == ("", {})
    assert parse_docstring("Only a summary line\nthat wraps.") == (
        "Only a summary line that wraps.",
        {},
    )
    assert parse_docstring("Summary.\n\nReturns:\n    the thing") == ("Summary.", {})
    assert parse_docstring("Summary.\n\n:param x: sphinx style") == (
        "Summary.",
        {},
    )  # other styles keep the summary only
