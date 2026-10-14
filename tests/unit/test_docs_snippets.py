"""Inline Python fragments in the docs must at least be valid Python.

Embedded examples (`--8<--`) are executed by test_examples.py; this covers the short fragments written
directly in the pages (which use top-level `await` and elide setup with `...`)."""

import ast
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[2] / "docs"
FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```python[^\n]*\n(?P<body>.*?)^(?P=indent)```", re.DOTALL | re.MULTILINE
)


def _blocks():
    for md in sorted(DOCS.rglob("*.md")):
        for m in FENCE.finditer(md.read_text()):
            body = m["body"]
            if "--8<--" in body:
                continue
            first = md.read_text()[: m.start()].count("\n") + 2
            yield pytest.param(md, first, body, id=f"{md.relative_to(DOCS)}:{first}")


@pytest.mark.parametrize(("md", "line", "body"), list(_blocks()))
def test_inline_python_fragment_is_valid_python(md, line, body):
    import textwrap

    code = textwrap.dedent(body)
    try:
        compile(code, str(md), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    except SyntaxError as e:
        raise AssertionError(f"{md.name}:{line + (e.lineno or 1) - 1}: {e.msg}: {e.text}") from e


def test_docs_contain_a_reasonable_number_of_checked_fragments():
    assert len(list(_blocks())) >= 30
