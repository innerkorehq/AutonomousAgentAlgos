"""Write your own algorithm: a fan-out/fan-in that translates into N languages in parallel.

An algorithm is just: which events it reacts to, and a function from (event, run state) to new
events. It holds no state of its own, so it survives restarts and can run on any worker."""

import asyncio

from pydantic import BaseModel, Field

from aaa.algos.base import finish, llm_request
from aaa.core import OutcomeKind
from aaa.events import Event, Projection
from aaa.llm import FakeLLMClient, LLMHandler, LLMRequest, LLMResponse, Message
from aaa.llm.fake import text
from aaa.runtime import Ctx, Engine


class Translations(BaseModel):  # run state = a fold over the run's events
    wanted: list[str] = Field(default_factory=list)
    done: dict[str, str] = Field(default_factory=dict)


def apply(state: Translations, e: Event) -> Translations:
    if e.type == "aaa.run.started":
        state.wanted = e.data["input"]["languages"]
    elif e.type == "aaa.llm.completed":
        state.done[e.data["tag"]] = (
            LLMResponse.model_validate(e.data["response"]).message.content or ""
        )
    elif e.type == "aaa.llm.failed":
        state.done[e.data["tag"]] = f"<failed: {e.data['error']['type']}>"
    return state


TRANSLATIONS = Projection(
    Translations, apply, name="example.translations", version=1, state_type=Translations
)


class Translate:
    name = "translate"

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed"]

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        if e.type == "aaa.run.started":  # fan out: one tagged LLM request per language
            body, langs = e.data["input"]["text"], e.data["input"]["languages"]
            return [
                llm_request(
                    e,
                    LLMRequest(messages=[Message(role="user", content=f"To {lang}: {body}")]),
                    tag=lang,
                )
                for lang in langs
            ]
        state = await ctx.project(TRANSLATIONS)  # fan in: wait until every language reported
        if len(state.done) < len(state.wanted):
            return []
        return [finish(e, OutcomeKind.SUCCESS, state.done)]


def model(request):
    lang = request.messages[-1].content.split(":")[0].replace("To ", "")
    return text({"French": "Bonjour", "German": "Hallo", "Spanish": "Hola"}[lang])


async def main() -> None:
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        outcome = await engine.run(
            Translate(), {"text": "Hello", "languages": ["French", "German", "Spanish"]}
        )
    print(dict(sorted(outcome.value.items())))


asyncio.run(main())
