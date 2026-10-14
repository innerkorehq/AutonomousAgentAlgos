"""Presets for common AI gateways. All speak the OpenAI chat-completions dialect, so a preset
is only a base URL + auth/routing headers. Conventions verified against vendor docs; the
two version-dependent ones are noted. Every field can be overridden afterwards."""

from __future__ import annotations

from typing import Any

from aaa.llm.config import LLMConfig


def litellm_proxy(model: str, base_url: str = "http://localhost:4000/v1", **kw: Any) -> LLMConfig:
    return LLMConfig(
        model=model, base_url=base_url, api_key=kw.pop("api_key", "${LITELLM_API_KEY}"), **kw
    )


def bifrost(model: str, base_url: str = "http://localhost:8080/v1", **kw: Any) -> LLMConfig:
    """Model is provider-prefixed, e.g. 'openai/gpt-4o-mini'. NOTE: path (/v1 vs /openai) has
    varied across Bifrost versions; override base_url if your deployment differs."""
    return LLMConfig(model=model, base_url=base_url, api_key=kw.pop("api_key", None), **kw)


def portkey(
    model: str, provider: str = "openai", base_url: str = "https://api.portkey.ai/v1", **kw: Any
) -> LLMConfig:
    """Portkey key authenticates to Portkey; the Bearer key is the upstream provider's."""
    headers = {"x-portkey-api-key": "${PORTKEY_API_KEY}", "x-portkey-provider": provider}
    headers.update(kw.pop("headers", {}))
    return LLMConfig(
        model=model,
        base_url=base_url,
        api_key=kw.pop("api_key", "${PROVIDER_API_KEY}"),
        headers=headers,
        **kw,
    )


def cloudflare(
    model: str, account_id: str, gateway_id: str = "default", authenticated: bool = True, **kw: Any
) -> LLMConfig:
    """Unified /compat endpoint; model is '{provider}/{model}'. Authenticated gateways need
    cf-aig-authorization (Cloudflare token); the Bearer key is the upstream provider's."""
    headers = {"cf-aig-authorization": "Bearer ${CLOUDFLARE_API_TOKEN}"} if authenticated else {}
    headers.update(kw.pop("headers", {}))
    return LLMConfig(
        model=model,
        base_url=f"https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/compat",
        api_key=kw.pop("api_key", "${PROVIDER_API_KEY}"),
        headers=headers,
        **kw,
    )


def vercel(model: str, base_url: str = "https://ai-gateway.vercel.sh/v1", **kw: Any) -> LLMConfig:
    """Model is 'creator/model' (case-sensitive), e.g. 'anthropic/claude-sonnet-4.5'."""
    return LLMConfig(
        model=model, base_url=base_url, api_key=kw.pop("api_key", "${AI_GATEWAY_API_KEY}"), **kw
    )


def kong(model: str, route_url: str, **kw: Any) -> LLMConfig:
    """Kong's ai-proxy (route_type llm/v1/chat) exposes the OpenAI format at *your route URL*
    (e.g. http://localhost:8000/chat), so no path suffix is appended."""
    return LLMConfig(
        model=model, base_url=route_url, chat_path="", api_key=kw.pop("api_key", None), **kw
    )


def mlflow(
    endpoint: str, base_url: str = "http://localhost:5000/gateway/mlflow/v1", **kw: Any
) -> LLMConfig:
    """`endpoint` is the gateway endpoint name (used as the model). NOTE: older MLflow 3.x
    docs use http://host:5000/v1; override base_url to match your server version."""
    return LLMConfig(model=endpoint, base_url=base_url, api_key=kw.pop("api_key", None), **kw)
