"""A fake judge model that answers any structured request with a schema-valid reply, so third-party eval libraries (which
build their own prompts and schemas) can run their real flows offline."""

import json
from typing import Any

from aaa.llm.fake import text


def _fab(s: dict[str, Any], defs: dict[str, Any]) -> Any:
    if "$ref" in s:
        return _fab(defs[s["$ref"].split("/")[-1]], defs)
    if "anyOf" in s:
        return _fab(s["anyOf"][0], defs)
    t = s.get("type")
    if t == "object":
        return {k: _fab(v, defs) for k, v in s.get("properties", {}).items()}
    if t == "array":
        return [_fab(s.get("items", {}), defs)]
    if t == "string":
        return s["enum"][0] if s.get("enum") else "yes"
    if t in ("integer", "number"):
        return 8
    if t == "boolean":
        return True
    return None


class SchemaJudge:
    """Callable for `FakeLLMClient`; `requests` records what the library sent."""

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def __call__(self, req):
        self.requests.append(req)
        sch = req.response_schema
        if sch:
            return text(json.dumps(_fab(sch, sch.get("$defs", {}))))
        return text("1. Compare the answer with the reference\n2. Decide")
