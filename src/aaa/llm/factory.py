from __future__ import annotations

from collections.abc import Callable

from aaa.llm.client import LLMClient
from aaa.llm.config import LLMConfig
from aaa.llm.errors import LLMConfigError
from aaa.llm.openai_compat import OpenAICompatClient
from aaa.llm.openai_responses import OpenAIResponsesClient

_PROVIDERS: dict[str, Callable[[LLMConfig], LLMClient]] = {
    "openai_compat": OpenAICompatClient,
    "openai_responses": OpenAIResponsesClient,
}


def register_provider(name: str, factory: Callable[[LLMConfig], LLMClient]) -> None:
    """Plug in a custom backend: `register_provider('mine', lambda cfg: MyClient(cfg))`."""
    _PROVIDERS[name] = factory


def build_client(config: LLMConfig) -> LLMClient:
    if config.provider == "litellm" and "litellm" not in _PROVIDERS:
        from aaa.llm.litellm_client import LiteLLMClient

        _PROVIDERS["litellm"] = LiteLLMClient
    try:
        return _PROVIDERS[config.provider](config)
    except KeyError:
        raise LLMConfigError(
            f"unknown provider {config.provider!r}; known: {sorted(_PROVIDERS)}"
        ) from None
