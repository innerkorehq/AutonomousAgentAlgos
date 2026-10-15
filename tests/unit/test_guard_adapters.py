"""Third-party guard libraries exercised for real where they can run offline (skipped where not installed)."""

import json

import pytest

from aaa.guard import GuardContext, GuardPolicy

TEXT = "Contact John Smith at john@example.com, card 4111 1111 1111 1111."


class TestPresidio:
    @pytest.fixture(autouse=True)
    def _need(self):
        pytest.importorskip("presidio_analyzer")
        pytest.importorskip("presidio_anonymizer")
        spacy = pytest.importorskip("spacy")
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model en_core_web_sm not installed")

    async def test_guard_finds_and_redacts_pii(self):
        from aaa.guard.presidio import PresidioGuard

        g = PresidioGuard(entities=["EMAIL_ADDRESS", "CREDIT_CARD", "PERSON"])
        res = await g.check(TEXT, GuardContext("input"))
        assert res.action == "redact"
        assert {"pii.email_address", "pii.credit_card", "pii.person"} <= set(res.categories)
        assert "john@example.com" not in (res.text or "") and "[REDACTED:EMAIL_ADDRESS]" in (
            res.text or ""
        )
        assert (await g.check("Nothing sensitive here.", GuardContext("input"))).action == "allow"

    async def test_guard_in_block_mode_and_in_a_policy(self):
        from aaa.guard.presidio import PresidioGuard

        policy = GuardPolicy([PresidioGuard(entities=["CREDIT_CARD"], action="block")])
        assert (await policy.evaluate("input", TEXT, GuardContext("input"))).action == "block"
        assert (await policy.evaluate("input", "hello", GuardContext("input"))).action == "allow"

    def test_redactor_keeps_pii_out_of_events_and_is_idempotent(self):
        from aaa.events import Event
        from aaa.guard.presidio import PresidioRedactor

        r = PresidioRedactor(entities=["EMAIL_ADDRESS", "CREDIT_CARD"])
        e = Event(type="x", data={"note": TEXT, "n": 3, "nested": {"mail": "a@b.com"}})
        once = r.redact(e)
        assert "john@example.com" not in json.dumps(once.data) and "a@b.com" not in json.dumps(
            once.data
        )
        assert once.data["n"] == 3
        assert r.redact(once).data == once.data

    async def test_engine_redactor_hook(self):
        from aaa.algos import ToolLoop
        from aaa.guard.presidio import PresidioRedactor
        from aaa.llm import FakeLLMClient, LLMHandler
        from aaa.llm.fake import text
        from aaa.runtime import Engine
        from aaa.tools import ToolHandler, ToolRegistry

        eng = Engine(redactor=PresidioRedactor(entities=["EMAIL_ADDRESS"]))
        LLMHandler(FakeLLMClient(lambda r: text("ok"))).attach(eng)
        reg = ToolRegistry([])
        ToolHandler(reg).attach(eng)
        algo = ToolLoop(reg)
        eng.register(algo)
        async with eng:
            h = await eng.start(algo, {"task": "write to jane@example.com"})
            await h.result()
            blob = json.dumps([e.data for e in await eng.store.read(h.run_id)], default=str)
        assert "jane@example.com" not in blob


class TestOpenAIModerationShape:
    async def test_a_reply_in_the_sdks_own_shape_is_understood(self):
        """Build a full reply from the OpenAI SDK's response model (every category, real aliases such as `self-harm/intent`),
        prove the SDK accepts it, and run it through the guard."""
        pytest.importorskip("openai")
        import httpx
        from openai.types import Moderation, ModerationCreateResponse

        from aaa.guard.openai_moderation import OpenAIModerationGuard

        names = {
            n: (f.alias or n)
            for n, f in Moderation.model_fields["categories"].annotation.model_fields.items()
        }
        scores = {
            alias: (0.95 if alias == "violence/graphic" else 0.01) for alias in names.values()
        }
        body = {
            "id": "modr-1",
            "model": "omni-moderation-latest",
            "results": [
                {
                    "flagged": True,
                    "categories": {a: s > 0.5 for a, s in scores.items()},
                    "category_applied_input_types": {a: ["text"] for a in scores},
                    "category_scores": scores,
                }
            ],
        }
        parsed = ModerationCreateResponse.model_validate(body)
        assert parsed.results[0].categories.violence_graphic is True

        g = OpenAIModerationGuard(
            "k", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
        )
        res = await g.check("x", GuardContext("output"))
        assert res.action == "block" and res.categories == ["violence.graphic"]
        assert res.findings[0].score == 0.95
