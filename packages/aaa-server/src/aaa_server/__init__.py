"""HTTP/SSE server for AAA agents: authentication, tenant isolation, rate limits, quotas and approvals."""

from .a2a import A2AConfig
from .admission import (
    ConcurrencyLimiter,
    Denied,
    InMemoryRateLimiter,
    Quota,
    QuotaStore,
    RateDecision,
    RateLimiter,
    RedisRateLimiter,
)
from .app import ServerConfig, clamp_budget, create_app, default_event_view
from .auth import (
    ApiKey,
    ApiKeyAuthenticator,
    Authenticator,
    AuthError,
    JWKSKeys,
    JWTAuthenticator,
    Principal,
    hash_api_key,
)

__all__ = [
    "A2AConfig",
    "ApiKey",
    "ApiKeyAuthenticator",
    "AuthError",
    "Authenticator",
    "ConcurrencyLimiter",
    "Denied",
    "InMemoryRateLimiter",
    "JWKSKeys",
    "JWTAuthenticator",
    "Principal",
    "Quota",
    "QuotaStore",
    "RateDecision",
    "RateLimiter",
    "RedisRateLimiter",
    "ServerConfig",
    "clamp_budget",
    "create_app",
    "default_event_view",
    "hash_api_key",
]
