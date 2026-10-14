import dataclasses

import pytest
from pydantic import BaseModel

from aaa.data import CodecRegistry, FunctionCodec, MissingCapability, preflight


class Ticket(BaseModel):
    id: int
    title: str


@dataclasses.dataclass
class Row:
    a: int
    b: str


def test_same_api_over_arbitrary_data():
    reg = CodecRegistry()
    reg.register("ticket", Ticket)
    reg.register("row", Row)
    reg.register("raw", dict)
    reg.register("doc", str)
    reg.register(
        "frame",
        FunctionCodec(
            to_text=lambda r: ", ".join(f"{k}={v}" for k, v in r.items()),
            to_json=dict,
            from_json=dict,
        ),
    )
    items = [
        reg.wrap("ticket", Ticket(id=1, title="x")),
        reg.wrap("row", Row(1, "z")),
        reg.wrap("raw", {"k": [1, 2]}),
        reg.wrap("doc", "hello"),
        reg.wrap("frame", {"c": 3}),
    ]
    assert all(i.content_hash and reg.render(i) for i in items)
    assert reg.render(items[3], max_chars=3) == "he…"
    assert reg.get("row").from_json({"a": 2, "b": "q"}) == Row(2, "q")
    assert reg.wrap("raw", {"k": [1, 2]}).content_hash == items[2].content_hash


def test_preflight():
    with pytest.raises(MissingCapability):
        preflight({"scorer", "embedder"}, {"embedder": object()})
