"""Attachments (images, audio, files) on messages: validation, storage, erasure, provider wire formats, context, privacy, traces, guards, sessions."""

import base64
import json

import pytest
from pydantic import ValidationError

from aaa.algos import ToolLoop
from aaa.context import ClearMedia, ContextPolicy, HeuristicCounter
from aaa.context.strategies import Work, render_for_summary
from aaa.core.outcome import OutcomeKind
from aaa.data.blob import FileBlobStore, InMemoryBlobStore
from aaa.events import SQLiteEventStore
from aaa.guard import GuardPolicy, GuardResult
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.llm.parts import AudioPart, FilePart, ImagePart, MediaError, describe, salt, unsalt
from aaa.llm.types import Message
from aaa.privacy import PatternRedactor
from aaa.runtime import Engine
from aaa.session import SessionManager
from aaa.tools import ToolHandler, ToolRegistry

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 40  # ~10 KB: above the inline threshold
SMALL_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def build(*, blobs=None, store=None, llm=None, context=None, guards=None, redactor=None, **kw):
    llm = llm or FakeLLMClient(lambda r: text("I see it."))
    eng = Engine(blobs=blobs, store=store, redactor=redactor)
    LLMHandler(llm, context=context, guards=guards).attach(eng)
    reg = ToolRegistry([])
    ToolHandler(reg).attach(eng)
    return eng, ToolLoop(reg, **kw), llm


async def log_json(eng, run_id):
    return json.dumps([e.model_dump(mode="json") for e in await eng.store.read(run_id)])


# -- parts ------------------------------------------------------------------------------------
def test_parts_are_validated_at_the_edge():
    ok = ImagePart.from_bytes(SMALL_PNG, "image/png")
    assert ok.size == len(SMALL_PNG) and ok.type == "image"
    with pytest.raises(ValidationError, match="signature mismatch"):
        ImagePart.from_bytes(b"not a png at all", "image/png")
    with pytest.raises(ValidationError, match="cannot have media type"):
        ImagePart.from_bytes(SMALL_PNG, "audio/wav")
    with pytest.raises(ValidationError, match="not valid base64"):
        ImagePart(media_type="image/png", data="!!!")
    with pytest.raises(ValidationError, match="exactly one"):
        ImagePart(media_type="image/png")
    with pytest.raises(ValidationError, match="scheme"):
        ImagePart.from_url("file:///etc/passwd", "image/png")
    with pytest.raises(ValidationError, match="scheme"):
        ImagePart.from_url("data:image/png;base64,AAAA", "image/png")
    assert ImagePart.from_url("https://example.com/a.png", "image/png").url


def test_size_limit(monkeypatch):
    from aaa.core import parts

    monkeypatch.setattr(parts, "MAX_MEDIA_BYTES", 100)
    with pytest.raises(ValidationError, match="limit"):
        FilePart.from_bytes(b"x" * 101, "application/pdf")


def test_file_names_are_base_names_only():
    assert FilePart.from_bytes(b"x", "text/plain", name="../../etc/passwd").name == "passwd"
    assert FilePart.from_bytes(b"x", "text/plain", name="C:\\Users\\a\\doc.txt").name == "doc.txt"


def test_text_only_messages_serialize_exactly_as_before():
    assert Message(role="user", content="hi").model_dump() == {
        "role": "user",
        "content": "hi",
        "tool_calls": [],
        "tool_call_id": None,
        "name": None,
    }
    m = Message(role="user", content="see", parts=[ImagePart.from_bytes(SMALL_PNG, "image/png")])
    assert Message.model_validate(m.model_dump(mode="json")) == m


def test_salting_round_trips_and_differs_per_stream():
    assert unsalt(salt("run-a", b"bytes")) == b"bytes" and salt("run-a", b"x") != salt(
        "run-b", b"x"
    )
    with pytest.raises(MediaError):
        unsalt(b"plain bytes")


# -- storage ----------------------------------------------------------------------------------
async def test_attachment_bytes_leave_the_log_and_reach_the_provider_intact():
    blobs = InMemoryBlobStore()
    eng, algo, llm = build(blobs=blobs)
    async with eng:
        h = await eng.start(
            algo, {"task": "what is this?", "attachments": [ImagePart.from_bytes(PNG, "image/png")]}
        )
        out = await h.result()
        raw = await log_json(eng, h.run_id)
        events = await eng.store.read(h.run_id)
    assert out.ok
    b64 = base64.b64encode(PNG).decode()
    assert b64 not in raw  # the log holds a reference, never the base64
    assert any(e.blobs for e in events)
    sent = llm.requests[0].messages[-1].parts[0]
    assert sent.data == b64 and sent.blob is None  # the provider-bound copy has the bytes back
    assert len(blobs._d) == 1


async def test_small_attachments_stay_inline_and_need_no_blob_store():
    eng, algo, llm = build(blobs=None)
    async with eng:
        h = await eng.start(
            algo, {"task": "x", "attachments": [ImagePart.from_bytes(SMALL_PNG, "image/png")]}
        )
        assert (await h.result()).ok
        assert base64.b64encode(SMALL_PNG).decode() in await log_json(eng, h.run_id)
    assert llm.requests[0].messages[-1].parts[0].data


async def test_without_a_blob_store_large_attachments_are_refused_with_the_fix_named():
    eng, algo, _ = build(blobs=None)
    async with eng:
        with pytest.raises(MediaError, match="blob store"):
            await eng.start(
                algo,
                {
                    "task": "x",
                    "attachments": [
                        ImagePart.from_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 300_000, "image/png")
                    ],
                },
            )


async def test_attachments_survive_a_durable_store(tmp_path):
    eng, algo, _llm = build(
        blobs=FileBlobStore(tmp_path / "blobs"), store=SQLiteEventStore(tmp_path / "log.db")
    )
    async with eng:
        h = await eng.start(
            algo, {"task": "x", "attachments": [ImagePart.from_bytes(PNG, "image/png")]}
        )
        await h.result()
    eng2 = Engine(
        blobs=FileBlobStore(tmp_path / "blobs"), store=SQLiteEventStore(tmp_path / "log.db")
    )
    started = await eng2._started_of(h.run_id)
    part = ImagePart.model_validate(started.data["input"]["attachments"][0])
    assert part.blob is not None and await eng2.media_bytes(part) == PNG


# -- erasure and isolation --------------------------------------------------------------------
async def test_identical_media_in_two_runs_are_separate_blobs_and_erasure_is_exact():
    blobs = InMemoryBlobStore()
    eng, algo, _ = build(blobs=blobs)
    async with eng:
        a = await eng.start(
            algo, {"task": "a", "attachments": [ImagePart.from_bytes(PNG, "image/png")]}
        )
        b = await eng.start(
            algo, {"task": "b", "attachments": [ImagePart.from_bytes(PNG, "image/png")]}
        )
        await a.result()
        await b.result()
        assert (
            len(blobs._d) == 2
        )  # same bytes, two runs: two blobs, so one run cannot learn that another holds this image
        started_b = await eng._started_of(b.run_id)
        part_b = ImagePart.model_validate(started_b.data["input"]["attachments"][0])
        await eng.erase(a.run_id)
        assert len(blobs._d) == 1
        assert await eng.media_bytes(part_b) == PNG  # b is untouched
        await eng.erase(b.run_id)
        assert blobs._d == {}


# -- the provider boundary --------------------------------------------------------------------
async def test_a_client_that_cannot_take_an_attachment_fails_loudly_and_sends_nothing():
    llm = FakeLLMClient(lambda r: text("never"))
    llm.modalities = frozenset({"text"})
    eng, algo, _ = build(blobs=InMemoryBlobStore(), llm=llm)
    async with eng:
        out = await eng.run(
            algo, {"task": "x", "attachments": [ImagePart.from_bytes(PNG, "image/png")]}
        )
    assert (
        out.kind is OutcomeKind.FAILED
        and "UnsupportedModality" in (out.reason or "")
        and "image" in (out.reason or "")
    )
    assert llm.requests == []


async def test_audio_and_file_attachments_reach_a_capable_client():
    eng, algo, llm = build(blobs=InMemoryBlobStore())
    async with eng:
        out = await eng.run(
            algo,
            {
                "task": "x",
                "attachments": [
                    AudioPart.from_bytes(b"RIFF" + b"0" * 5000, "audio/wav"),
                    FilePart.from_bytes(b"%PDF-" + b"0" * 5000, "application/pdf", name="a.pdf"),
                ],
            },
        )
    assert out.ok
    kinds = [p.type for p in llm.requests[0].messages[-1].parts]
    assert kinds == ["audio", "file"] and all(p.data for p in llm.requests[0].messages[-1].parts)


def transport_capturing(box):
    import httpx

    def handler(request):
        box.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {},
            },
        )

    return httpx.MockTransport(handler)


async def test_openai_compat_wire_format():
    import httpx

    from aaa.llm.config import LLMConfig
    from aaa.llm.openai_compat import OpenAICompatClient

    box: list = []
    c = OpenAICompatClient(
        LLMConfig(model="m", base_url="http://x", modalities=["image", "audio", "file"]),
        http_client=httpx.AsyncClient(transport=transport_capturing(box)),
    )
    from aaa.llm.types import LLMRequest

    msg = Message(
        role="user",
        content="look",
        parts=[
            ImagePart.from_bytes(SMALL_PNG, "image/png", detail="low"),
            ImagePart.from_url("https://example.com/a.png", "image/png"),
            AudioPart.from_bytes(b"RIFF" + b"0" * 10, "audio/wav"),
            FilePart.from_bytes(b"%PDF-1", "application/pdf", name="doc.pdf"),
        ],
    )
    await c.complete(LLMRequest(messages=[msg]))
    content = box[0]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "look"}
    assert (
        content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        and content[1]["image_url"]["detail"] == "low"
    )
    assert content[2]["image_url"] == {"url": "https://example.com/a.png"}
    assert content[3]["type"] == "input_audio" and content[3]["input_audio"]["format"] == "wav"
    assert content[4]["type"] == "file" and content[4]["file"]["filename"] == "doc.pdf"
    # a plain text message is untouched: content stays a string
    await c.complete(LLMRequest(messages=[Message(role="user", content="plain")]))
    assert box[1]["messages"][0]["content"] == "plain"


def test_openai_wire_shapes_match_the_sdks_own_types():
    pytest.importorskip("openai")
    from openai.types.chat import ChatCompletionContentPartParam
    from openai.types.responses import ResponseInputFileParam, ResponseInputImageParam
    from pydantic import TypeAdapter

    from aaa.llm.wire_media import chat_content, responses_content

    msg = Message(
        role="user",
        content="look",
        parts=[
            ImagePart.from_bytes(SMALL_PNG, "image/png", detail="high"),
            AudioPart.from_bytes(b"RIFF0000", "audio/wav"),
            FilePart.from_bytes(b"%PDF-1", "application/pdf", name="d.pdf"),
        ],
    )
    ta = TypeAdapter(ChatCompletionContentPartParam)
    for item in chat_content(msg):
        ta.validate_python(item)
    items = responses_content(
        Message(role="user", content="look", parts=[msg.parts[0], msg.parts[2]])
    )
    TypeAdapter(ResponseInputImageParam).validate_python(items[1])
    TypeAdapter(ResponseInputFileParam).validate_python(items[2])


def test_responses_client_refuses_audio_instead_of_dropping_it():
    from aaa.llm.errors import FatalError
    from aaa.llm.wire_media import responses_content

    with pytest.raises(FatalError, match="audio"):
        responses_content(
            Message(
                role="user", content="x", parts=[AudioPart.from_bytes(b"RIFF0000", "audio/wav")]
            )
        )


# -- context ----------------------------------------------------------------------------------
def test_attachments_are_charged_to_the_token_budget():
    counter = HeuristicCounter()
    plain = Message(role="user", content="hi")
    img = plain.model_copy(update={"parts": [ImagePart.from_bytes(SMALL_PNG, "image/png")]})
    assert counter.count([img]) - counter.count([plain]) == 800


async def test_clear_media_drops_old_attachments_but_keeps_the_newest():
    png = ImagePart.from_bytes(SMALL_PNG, "image/png")
    msgs = [Message(role="system", content="s")] + [
        m
        for i in range(3)
        for m in (
            Message(role="user", content=f"q{i}", parts=[png]),
            Message(role="assistant", content="a"),
        )
    ]
    work = Work(
        messages=msgs,
        tools=[],
        counter=HeuristicCounter(),
        target=10**6,
        keep_recent=1,
        summary_upto=0,
    )
    await ClearMedia(keep_last=1).apply(work)
    carrying = [i for i, m in enumerate(work.messages) if m.parts]
    assert carrying == [5]  # only the most recent attachment survives
    assert (
        "image attachment removed" in (work.messages[1].content or "")
        and "ClearMedia" not in work.applied
        and "clear_media" in work.applied
    )


def test_summaries_mention_attachments_without_their_bytes():
    m = Message(role="user", content="see", parts=[ImagePart.from_bytes(SMALL_PNG, "image/png")])
    out = render_for_summary([m])
    assert "image attachment: image/png" in out and base64.b64encode(SMALL_PNG).decode() not in out
    assert describe(m.parts[0]).startswith("[image attachment")


async def test_a_default_context_policy_sheds_old_images_before_anything_else():
    big = ImagePart.from_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 2000, "image/png")
    policy = ContextPolicy(max_input_tokens=2200, reserve_output=0)
    msgs = [Message(role="system", content="s")]
    for i in range(3):
        msgs += [
            Message(role="user", content=f"q{i}", parts=[big]),
            Message(role="assistant", content="a"),
        ]
    prep = await policy.prepare(msgs)
    assert prep.report is not None and "clear_media" in prep.report["applied"]
    assert sum(1 for m in prep.messages if m.parts) < 3


# -- privacy, traces, guards ------------------------------------------------------------------
async def test_the_pattern_redactor_does_not_corrupt_attachment_bytes():
    from aaa.events import Event

    card_like = "4111111111111111"  # valid base64 text AND a Luhn-valid card number
    e = Event(
        type="aaa.llm.requested",
        data={
            "note": "pay 4111111111111111",
            "part": {"type": "file", "media_type": "text/plain", "data": card_like},
        },
    )
    out = PatternRedactor().redact(e)
    assert out.data["part"]["data"] == card_like and "4111111111111111" not in out.data["note"]


async def test_traces_never_carry_attachment_bytes():
    from aaa.trace import spans_from_events

    eng, algo, _ = build(blobs=InMemoryBlobStore())
    async with eng:
        h = await eng.start(
            algo, {"task": "x", "attachments": [ImagePart.from_bytes(SMALL_PNG, "image/png")]}
        )
        await h.result()
        spans = spans_from_events(
            [await eng.hydrate(e) for e in await eng.store.read(h.run_id)], capture_content=True
        )
    blob = json.dumps([s.input for s in spans if s.input is not None], default=str)
    assert "image/png" in blob and base64.b64encode(SMALL_PNG).decode() not in blob


async def test_guards_learn_that_a_message_carries_media_even_when_it_has_no_text():
    seen = []

    class Spy:
        name, stages, version, deterministic = "spy", frozenset({"input"}), "1", False

        async def check(self, content, ctx):
            seen.append((content, ctx.has_media))
            return GuardResult()

    eng, algo, _ = build(blobs=InMemoryBlobStore(), guards=GuardPolicy([Spy()]))
    async with eng:
        await eng.run(
            algo, {"task": "", "attachments": [ImagePart.from_bytes(SMALL_PNG, "image/png")]}
        )
        await eng.run(algo, {"task": "just text"})
    assert ("", True) in seen and ("just text", False) in seen


# -- sessions ---------------------------------------------------------------------------------
async def test_a_session_remembers_attachments_across_turns():
    blobs = InMemoryBlobStore()
    llm = FakeLLMClient(
        lambda r: text(f"{sum(len(m.parts or []) for m in r.messages)} attachments in view")
    )
    eng, algo, _ = build(blobs=blobs, llm=llm)
    async with eng:
        mgr = SessionManager(eng, algo, system="Be brief.")
        chat = await mgr.create(user="ann")
        r1 = await chat.send("what is this?", attachments=[ImagePart.from_bytes(PNG, "image/png")])
        r2 = await chat.send("and what colour is it?")
        hist = await chat.history()
    assert (
        r1.text == "1 attachments in view" and r2.text == "1 attachments in view"
    )  # turn 2 still carries the picture
    assert hist[0].attachments and not hist[1].attachments
    assert all(
        p.data for r in llm.requests for m in r.messages for p in m.parts or []
    )  # the provider always got bytes, not references
