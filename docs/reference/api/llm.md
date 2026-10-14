# `aaa.llm`

## Protocols and handler

::: aaa.llm.LLMClient

::: aaa.llm.StreamingLLMClient

::: aaa.llm.LLMHandler

## Configuration and factories

::: aaa.llm.LLMConfig

::: aaa.llm.build_client

::: aaa.llm.register_provider

::: aaa.llm.OpenAICompatClient

## Gateway presets

::: aaa.llm.gateways
    options:
      members:
        - litellm_proxy
        - bifrost
        - portkey
        - cloudflare
        - vercel
        - kong
        - mlflow

## Reliability wrappers

::: aaa.llm.RetryingClient

::: aaa.llm.CircuitBreakerClient

## Messages

::: aaa.llm.LLMRequest

::: aaa.llm.LLMResponse

::: aaa.llm.Message

::: aaa.llm.ToolCall

::: aaa.llm.ToolSpec

::: aaa.llm.Usage

::: aaa.llm.StreamChunk

::: aaa.llm.EmbedResponse

::: aaa.llm.collect

## Test doubles

::: aaa.llm.FakeLLMClient

::: aaa.llm.ReplayLLMClient

## Errors

::: aaa.llm.errors
