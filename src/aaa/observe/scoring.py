"""Delivering `Score`s to the backends that hold the traces (the optional `Scorable` capability of a sink).

Each function is the vendor's own REST call, plain httpx, no SDK:

* **Langfuse**  `POST {host}/api/public/scores`, Basic auth, one score per call (`traceId`, optional `observationId`, `name`, `value`,
  `dataType` NUMERIC | BOOLEAN | CATEGORICAL). The score `id` makes a repeat an upsert.
* **Phoenix**  `POST {base}/v1/span_annotations` (or `/v1/trace_annotations`), `data: [{name, annotator_kind, span_id|trace_id,
  result: {score, label, explanation}, identifier}]`. The `identifier` makes a repeat an update.
* **LangSmith**  `POST {api}/feedback` with `run_id`, `key`, `score`, `value`, `comment`. LangSmith builds a run's id from the OTLP
  span id (the 8 bytes right-aligned in a UUID), so feedback is addressed to that derived id.

Failures raise `ExportError`; the caller (an `OnlineEvaluator`) logs them and carries on."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import httpx

from aaa.observe.base import ExportError
from aaa.score import Score

TRANSIENT = {408, 425, 429, 500, 502, 503, 504}


def stable_id(*parts: str) -> str:
    """A 32-hex id for a (trace, span, score name): the same score sent twice is one score, not two."""
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:32]


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    body: Any,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
    retry_on: set[int] | None = None,
    attempts: int = 3,
    delay_s: float = 0.5,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    accept: set[int] | None = None,
) -> httpx.Response:
    """POST with a few retries on transient statuses (and on `retry_on`, for "the span has not arrived yet")."""
    waits = (TRANSIENT | (retry_on or set())) - (accept or set())
    for i in range(attempts):
        try:
            r = await client.post(url, json=body, headers=headers, params=params)
        except httpx.HTTPError as e:
            if i + 1 == attempts:
                raise ExportError(f"POST {_safe(url)} failed: {type(e).__name__}") from None
            await sleep(delay_s * 2**i)
            continue
        if r.status_code < 300 or r.status_code in (accept or set()):
            return r
        if r.status_code in waits and i + 1 < attempts:
            await sleep(delay_s * 2**i)
            continue
        raise ExportError(f"POST {_safe(url)} -> HTTP {r.status_code}: {r.text[:200]}")
    raise ExportError(f"POST {_safe(url)} gave up")  # pragma: no cover


def _safe(url: str) -> str:
    return url.split("?", 1)[0]


# -- Langfuse -------------------------------------------------------------------------------------
def langfuse_body(
    trace_id: str,
    span_id: str | None,
    s: Score,
    *,
    environment: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": stable_id(trace_id, span_id or "", s.name),
        "traceId": trace_id,
        "name": s.name,
    }
    if span_id:
        body["observationId"] = span_id
    if isinstance(s.value, bool):
        body["value"], body["dataType"] = int(s.value), "BOOLEAN"
    elif isinstance(s.value, int | float):
        body["value"], body["dataType"] = float(s.value), "NUMERIC"
    else:
        body["value"], body["dataType"] = str(s.value)[:500], "CATEGORICAL"
    if s.reason:
        body["comment"] = s.reason[:2000]
    if s.metadata:
        body["metadata"] = s.metadata
    if environment:
        body["environment"] = environment
    return body


async def langfuse_score(
    client: httpx.AsyncClient,
    host: str,
    auth: str,
    trace_id: str,
    scores: Sequence[Score],
    span_id: str | None,
    environment: str | None,
) -> None:
    url = f"{host.rstrip('/')}/api/public/scores"
    headers = {"Authorization": f"Basic {auth}"}
    for s in scores:
        body = langfuse_body(trace_id, span_id, s, environment=environment)
        try:
            await post_json(client, url, body, headers=headers)
        except ExportError as e:
            # Langfuse 2.x/3.x take one score object; newer servers take a list. Try the list once if the single was refused for its shape.
            if "HTTP 400" in str(e) and "array" in str(e):
                await post_json(client, url, [body], headers=headers)
            else:
                raise


# -- Phoenix --------------------------------------------------------------------------------------
def phoenix_item(trace_id: str, span_id: str | None, s: Score) -> dict[str, Any]:
    result: dict[str, Any] = {}
    n = s.numeric
    if n is not None:
        result["score"] = n
    if isinstance(s.value, bool):
        result["label"] = "pass" if s.value else "fail"
    elif isinstance(s.value, str):
        result["label"] = s.value
    if s.reason:
        result["explanation"] = s.reason
    item: dict[str, Any] = {
        "name": s.name,
        "annotator_kind": "LLM" if (s.usage or s.cost) else "CODE",
        "result": result,
        "identifier": f"aaa:{s.name}",
    }
    if s.metadata:
        item["metadata"] = s.metadata
    item["span_id" if span_id else "trace_id"] = span_id or trace_id
    return item


async def phoenix_score(
    client: httpx.AsyncClient,
    base: str,
    headers: Mapping[str, str],
    trace_id: str,
    scores: Sequence[Score],
    span_id: str | None,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    path = "span_annotations" if span_id else "trace_annotations"
    await post_json(
        client,
        f"{base.rstrip('/')}/v1/{path}",
        {"data": [phoenix_item(trace_id, span_id, s) for s in scores]},
        headers=headers,
        params={"sync": "true"},
        retry_on={404, 422},  # the span may not have been ingested yet
        attempts=5,
        sleep=sleep,
    )


# -- LangSmith ------------------------------------------------------------------------------------
def langsmith_run_id(span_id: str) -> str:
    """The LangSmith run id for an OTLP span id: its 8 bytes right-aligned in a UUID (per LangSmith's OTel ingestion)."""
    return str(uuid.UUID(int=int(span_id, 16)))


def langsmith_body(span_id: str, s: Score) -> dict[str, Any]:
    run_id = langsmith_run_id(span_id)
    body: dict[str, Any] = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"aaa:{run_id}:{s.name}")),
        "run_id": run_id,
        "key": s.name,
        "feedback_source": {"type": "model" if (s.usage or s.cost) else "api"},
    }
    n = s.numeric
    if n is not None:
        body["score"] = n
    if isinstance(s.value, str):
        body["value"] = s.value
    if s.reason:
        body["comment"] = s.reason
    return body


async def langsmith_score(
    client: httpx.AsyncClient,
    api_url: str,
    headers: Mapping[str, str],
    scores: Sequence[Score],
    span_id: str | None,
) -> None:
    if not span_id:
        raise ExportError(
            "LangSmith feedback needs the span (run) it is about: pass span_id= (OnlineEvaluator does)"
        )
    for s in scores:
        await post_json(
            client,
            f"{api_url.rstrip('/')}/feedback",
            langsmith_body(span_id, s),
            headers=headers,
            accept={409},  # same feedback id again: already recorded
        )
