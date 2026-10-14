from __future__ import annotations

import os
import re
from typing import Any

from pydantic import BaseModel, Field

from aaa.llm.errors import LLMConfigError

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: str, env: dict[str, str] | None = None) -> str:
    """Expand ${VAR} placeholders; a missing variable is a loud config error."""
    src = env if env is not None else os.environ

    def sub(m: re.Match[str]) -> str:
        if m.group(1) not in src:
            raise LLMConfigError(f"environment variable {m.group(1)} is not set")
        return src[m.group(1)]

    return _VAR.sub(sub, value)


class LLMConfig(BaseModel):
    """Declarative LLM selection. Secrets are referenced as ${ENV_VAR}, never stored."""

    provider: str = "openai_compat"  # "openai_compat" | "litellm" | any registered name
    model: str
    base_url: str = ""
    api_key: str | None = None  # e.g. "${OPENAI_API_KEY}"; sent as Bearer
    headers: dict[str, str] = Field(default_factory=dict)  # values may use ${VAR}
    chat_path: str = "/chat/completions"  # "" when base_url is already the full endpoint
    embeddings_path: str = "/embeddings"
    timeout_s: float = 60.0
    params: dict[str, Any] = Field(default_factory=dict)  # default request params
    price_in_per_mtok: float | None = None  # optional client-side cost estimation
    price_out_per_mtok: float | None = None
