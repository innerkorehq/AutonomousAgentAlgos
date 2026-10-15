# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
"""Microsoft Presidio: PII detection and anonymisation as a guard and as an event redactor.

    from aaa.guard.presidio import PresidioGuard, PresidioRedactor

    PresidioGuard(action="redact")                       # strip PII from model input / tool results
    PresidioGuard(entities=["US_SSN", "CREDIT_CARD"], action="block", stages=["output"])
    Engine(redactor=PresidioRedactor())                  # keep PII out of the event log itself

Presidio needs a spaCy model for names and places (`python -m spacy download en_core_web_sm`, or the larger `en_core_web_lg`); without
one only the pattern-based entities (email, card, SSN, IBAN, ...) are found. Detection is probabilistic: it misses things and
sometimes flags the wrong thing, so treat it as a safety net, not a guarantee. Analysis is CPU-bound and runs in a worker thread."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from aaa.guard.types import Action, Finding, GuardContext, GuardResult, Stage
from aaa.privacy import PatternRedactor

ALL_STAGES: tuple[Stage, ...] = ("input", "output", "tool_call", "tool_result")


def default_analyzer(model: str = "en_core_web_sm", language: str = "en") -> Any:
    """An `AnalyzerEngine` on a small spaCy model (the stock engine downloads the 400 MB `en_core_web_lg`)."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    nlp = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": model}],
        }
    ).create_engine()
    return AnalyzerEngine(nlp_engine=nlp, supported_languages=[language])


class _Presidio:
    def __init__(
        self,
        analyzer: Any,
        anonymizer: Any,
        entities: Iterable[str] | None,
        language: str,
        score_threshold: float,
    ) -> None:
        self.analyzer = analyzer if analyzer is not None else default_analyzer(language=language)
        if anonymizer is None:
            from presidio_anonymizer import AnonymizerEngine

            anonymizer = AnonymizerEngine()
        self.anonymizer = anonymizer
        self.entities, self.language, self.score_threshold = (
            (list(entities) if entities else None),
            language,
            score_threshold,
        )

    def analyze(self, text: str) -> list[Any]:
        return self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities,
            score_threshold=self.score_threshold,
        )

    def anonymize(self, text: str, results: list[Any]) -> str:
        from presidio_anonymizer.entities import OperatorConfig

        ops = {
            r.entity_type: OperatorConfig("replace", {"new_value": f"[REDACTED:{r.entity_type}]"})
            for r in results
        }
        return self.anonymizer.anonymize(text=text, analyzer_results=results, operators=ops).text


class PresidioGuard(_Presidio):
    deterministic = True  # the same text gives the same entities (for a given model and version)

    def __init__(
        self,
        analyzer: Any = None,
        anonymizer: Any = None,
        *,
        entities: Iterable[str] | None = None,
        language: str = "en",
        score_threshold: float = 0.5,
        action: Action = "redact",
        name: str = "presidio",
        stages: Iterable[Stage] = ALL_STAGES,
    ) -> None:
        super().__init__(analyzer, anonymizer, entities, language, score_threshold)
        self.action: Action = action
        self.name, self.stages = name, frozenset(stages)
        try:
            from importlib.metadata import version

            self.version = f"1:{version('presidio-analyzer')}"
        except Exception:  # noqa: BLE001 - the version is only a cache key
            self.version = "1"

    async def check(self, content: str, ctx: GuardContext, /) -> GuardResult:
        results = await asyncio.to_thread(self.analyze, content)
        if not results:
            return GuardResult()
        findings = [Finding(f"pii.{r.entity_type.lower()}", float(r.score)) for r in results]
        if self.action == "redact":
            return GuardResult(
                "redact", findings, await asyncio.to_thread(self.anonymize, content, results)
            )
        return GuardResult(self.action, findings)


class PresidioRedactor(PatternRedactor):
    """An event `Redactor` (see `aaa.privacy`) that uses Presidio instead of regular expressions."""

    def __init__(
        self,
        analyzer: Any = None,
        anonymizer: Any = None,
        *,
        entities: Iterable[str] | None = None,
        language: str = "en",
        score_threshold: float = 0.5,
        only: Iterable[str] = ("*",),
        skip: Iterable[str] = (),
    ) -> None:
        super().__init__((), only=only, skip=skip)
        self._p = _Presidio(analyzer, anonymizer, entities, language, score_threshold)

    def scrub_text(self, text: str) -> str:
        results = self._p.analyze(text)
        return self._p.anonymize(text, results) if results else text
