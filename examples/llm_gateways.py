"""Pointing AAA at an LLM gateway is configuration. This prints each preset; no network is used."""

import os

from aaa.llm import LLMConfig, OpenAICompatClient, build_client, gateways, register_provider

# Secrets are written as ${ENV_VAR} and resolved when the client is built. Dummy values here.
for name in (
    "PORTKEY_API_KEY",
    "PROVIDER_API_KEY",
    "CLOUDFLARE_API_TOKEN",
    "AI_GATEWAY_API_KEY",
    "LITELLM_API_KEY",
):
    os.environ.setdefault(name, "dummy")

presets = {
    "LiteLLM proxy": gateways.litellm_proxy("gpt-4o-mini"),
    "Portkey": gateways.portkey("gpt-4o-mini", provider="openai"),
    "Cloudflare AI Gateway": gateways.cloudflare("openai/gpt-4o-mini", account_id="ACCT"),
    "Vercel AI Gateway": gateways.vercel("openai/gpt-4o-mini"),
    "Bifrost": gateways.bifrost("openai/gpt-4o-mini"),
    "Kong AI Gateway": gateways.kong("gpt-4o-mini", route_url="http://kong:8000/chat"),
    "MLflow AI Gateway": gateways.mlflow("my-endpoint"),
}
for label, cfg in presets.items():
    client = build_client(cfg)  # a real OpenAI-compatible client; construction makes no request
    url = cfg.base_url.rstrip("/") + cfg.chat_path
    print(f"{label:22} POST {url}  headers={sorted(cfg.headers)}")
    assert isinstance(client, OpenAICompatClient)

# Anything else that speaks the OpenAI dialect (vLLM, Ollama, Azure-compatible proxies, ...):
local = LLMConfig(model="gemma4:latest", base_url="http://localhost:11434/v1")


# A fully custom backend: implement `complete()`, `embed()`, `aclose()` and register it by name.
class MyClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    async def complete(self, request): ...
    async def embed(self, texts, model=None): ...
    async def aclose(self): ...


register_provider("mine", MyClient)
assert isinstance(build_client(LLMConfig(provider="mine", model="x")), MyClient)
print("custom provider registered")
