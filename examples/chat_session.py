"""A chat session that remembers across turns, with the provider holding the conversation. Runs offline."""

import asyncio
import re

from aaa.algos import ToolLoop
from aaa.llm import FakeStatefulClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Engine
from aaa.session import SessionManager
from aaa.tools import ToolRegistry


def scripted_model(
    request,
):  # stands in for a real LLM; it sees the WHOLE conversation, like a real one
    said = [m.content or "" for m in request.messages if m.role == "user"]
    question = said[-1]
    if "my name" in question and question.endswith("?"):
        names = [m.group(1) for s in said if (m := re.search(r"my name is (\w+)", s))]
        return text(f"Your name is {names[-1]}." if names else "I don't know your name yet.")
    return text(f"Noted ({len(said)} things said so far).")


async def main() -> None:
    provider = FakeStatefulClient(
        scripted_model
    )  # swap for OpenAIResponsesClient to hold state at OpenAI
    async with Engine() as engine:
        LLMHandler(provider).attach(engine)
        sessions = SessionManager(
            engine, ToolLoop(ToolRegistry([])), system="You are a helpful assistant."
        )

        chat = await sessions.tenant("acme").create(user="ann")
        for line in ["Hi, my name is Ann.", "I work on payments.", "What is my name?"]:
            reply = await chat.send(line)
            print(f"you: {line}\nbot: {reply.text}")

        # a different process, an hour later: reattach by id; the conversation is just events in the log
        again = await sessions.tenant("acme").get(chat.id)
        print(f"reattached: {len(await again.history())} turns on record")

        sent = [n for n, _ in provider.sent]
        everything = [
            2 * i + 2 for i in range(len(sent))
        ]  # system prompt + all earlier turns + the new message
        print(
            f"messages sent to the provider per turn: {sent}  (resending the history would be {everything})"
        )
        print(
            f"turns continued from provider state: {[cont is not None for _, cont in provider.sent]}"
        )

        await chat.close()
        report = await engine.tenant("acme").erase_all()
        print(
            f"erased {len(report.streams)} runs; provider state deleted: {sum(r.provider_state for r in report.erased)}"
        )


asyncio.run(main())
