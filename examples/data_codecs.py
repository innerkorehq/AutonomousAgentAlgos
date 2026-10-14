"""Arbitrary data: the same API over Pydantic models, dataclasses, dicts, text and custom types."""

import dataclasses

from pydantic import BaseModel

from aaa.data import CodecRegistry, FunctionCodec, MissingCapability, preflight


class Ticket(BaseModel):
    id: int
    title: str
    priority: str = "normal"


@dataclasses.dataclass
class Row:
    sku: str
    qty: int


codecs = CodecRegistry()
codecs.register("ticket", Ticket)  # a Pydantic model
codecs.register("row", Row)  # a dataclass
codecs.register("event", dict)  # a plain dict
codecs.register("doc", str)  # plain text
codecs.register(  # anything else: teach AAA how to render and serialize it
    "frame",
    FunctionCodec(
        to_text=lambda r: ", ".join(f"{k}={v}" for k, v in r.items()),
        to_json=dict,
        from_json=dict,
    ),
)

items = [
    codecs.wrap("ticket", Ticket(id=7, title="Printer on fire", priority="high")),
    codecs.wrap("row", Row(sku="A-1", qty=3)),
    codecs.wrap("event", {"type": "login", "ok": False}),
    codecs.wrap("doc", "A long document body " * 20),
    codecs.wrap("frame", {"col_a": 1, "col_b": "x"}),
]
for item in items:
    # every kind renders to LLM-ready text, can be truncated, and has a stable content hash
    print(f"{item.kind:7} {item.content_hash[:8]}  {codecs.render(item, max_chars=48)!r}")

# round-trip through JSON and back to the user's own type
restored = codecs.get("row").from_json(codecs.get("row").to_json(Row("A-1", 3)))
assert restored == Row("A-1", 3)
# identical content => identical hash, which is what dedup and caching key on
assert codecs.wrap("event", {"type": "login", "ok": False}).content_hash == items[2].content_hash

try:  # fail fast at startup if an algorithm needs something you did not provide
    preflight({"scorer", "embedder"}, {"embedder": object()})
except MissingCapability as e:
    print("preflight:", e)
