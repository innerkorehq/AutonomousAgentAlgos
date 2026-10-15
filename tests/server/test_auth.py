"""JWT/JWKS verification, API keys, and the Redis rate limiter."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import jwt
import pytest
from aaa_server import (
    ApiKeyAuthenticator,
    AuthError,
    InMemoryRateLimiter,
    JWKSKeys,
    JWTAuthenticator,
    RedisRateLimiter,
)
from cryptography.hazmat.primitives.asymmetric import rsa

ISS, AUD = "https://id.example", "aaa"


def make_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def jwks_of(*pairs):
    keys = []
    for kid, key in pairs:
        d = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
        keys.append({**d, "kid": kid, "use": "sig", "alg": "RS256"})
    return {"keys": keys}


class Idp:
    def __init__(self, *pairs):
        self.pairs, self.fetches = list(pairs), 0

    def handler(self, request):
        self.fetches += 1
        return httpx.Response(200, json=jwks_of(*self.pairs))

    def keys(self, **kw):
        client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        return JWKSKeys("https://id.example/jwks", client=client, **kw)


def token(key, *, kid="k1", alg="RS256", **claims):
    base = {"iss": ISS, "aud": AUD, "sub": "ann", "tenant": "acme", "exp": time.time() + 300}
    base.update({k: v for k, v in claims.items() if v is not None})
    for k, v in claims.items():
        if v is None:
            base.pop(k, None)
    return jwt.encode(base, key, algorithm=alg, headers={"kid": kid})


def auth(idp, **kw):
    return JWTAuthenticator(issuer=ISS, audience=AUD, keys=idp.keys(), **kw)


def bearer(t):
    return {"authorization": f"Bearer {t}"}


async def test_a_valid_token_yields_the_principal():
    key = make_key()
    a = auth(Idp(("k1", key)))
    p = await a.authenticate(bearer(token(key, scope="run approve nonsense")))
    assert (p.tenant, p.subject) == ("acme", "ann") and p.scopes == {"run", "approve"}
    p = await a.authenticate(bearer(token(key)))  # no scope claim -> defaults
    assert p.scopes == {"run", "session"}


@pytest.mark.parametrize(
    "bad",
    [
        {"exp": time.time() - 3600},
        {"aud": "someone-else"},
        {"iss": "https://evil.example"},
        {"tenant": None},
        {"tenant": "a/b"},
        {"tenant": 123},
        {"sub": None},
        {"scope": 5},
    ],
)
async def test_bad_claims_are_refused(bad):
    key = make_key()
    a = auth(Idp(("k1", key)))
    with pytest.raises(AuthError):
        await a.authenticate(bearer(token(key, **bad)))


async def test_a_token_without_exp_is_refused():
    key = make_key()
    a = auth(Idp(("k1", key)))
    t = jwt.encode(
        {"iss": ISS, "aud": AUD, "sub": "x", "tenant": "acme"},
        key,
        algorithm="RS256",
        headers={"kid": "k1"},
    )
    with pytest.raises(AuthError):
        await a.authenticate(bearer(t))


async def test_alg_none_and_key_confusion_are_refused():
    key = make_key()
    idp = Idp(("k1", key))
    a = auth(idp)
    unsigned = jwt.encode(
        {"iss": ISS, "aud": AUD, "sub": "x", "tenant": "acme", "exp": time.time() + 99},
        None,
        algorithm="none",
        headers={"kid": "k1"},
    )
    with pytest.raises(AuthError):
        await a.authenticate(bearer(unsigned))
    # HS256 signed with the PUBLIC key as the secret (the classic confusion attack), built by hand because
    # PyJWT itself refuses to produce it
    import base64
    import hmac

    from cryptography.hazmat.primitives import serialization

    pub_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )

    def b64(b):
        return base64.urlsafe_b64encode(b).rstrip(b"=")

    head = b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": "k1"}).encode())
    body = b64(
        json.dumps(
            {"iss": ISS, "aud": AUD, "sub": "x", "tenant": "acme", "exp": time.time() + 99}
        ).encode()
    )
    sig = b64(hmac.new(pub_pem, head + b"." + body, "sha256").digest())
    forged = (head + b"." + body + b"." + sig).decode()
    with pytest.raises(AuthError):
        await a.authenticate(bearer(forged))


async def test_a_token_signed_by_another_key_is_refused():
    with pytest.raises(AuthError):
        await auth(Idp(("k1", make_key()))).authenticate(bearer(token(make_key())))


async def test_unknown_kid_refetches_at_most_once_per_interval():
    key, rotated = make_key(), make_key()
    idp = Idp(("k1", key))
    now = [1000.0]
    a = JWTAuthenticator(
        issuer=ISS, audience=AUD, keys=idp.keys(min_refetch_s=30, clock=lambda: now[0])
    )
    await a.authenticate(bearer(token(key)))
    assert idp.fetches == 1
    for _ in range(5):  # inside the interval a made-up kid never reaches the network
        with pytest.raises(AuthError):
            await a.authenticate(bearer(token(rotated, kid="made-up")))
    assert idp.fetches == 1
    now[0] += 31
    for _ in range(5):  # after it, a flood of made-up kids
        with pytest.raises(AuthError):
            await a.authenticate(bearer(token(rotated, kid="made-up")))
    assert idp.fetches == 2  # costs one refetch, then the rate limit holds again
    idp.pairs.append(("k2", rotated))  # real rotation
    now[0] += 31
    assert (await a.authenticate(bearer(token(rotated, kid="k2")))).tenant == "acme"


async def test_a_token_cannot_choose_where_keys_come_from():
    key = make_key()
    idp = Idp(("k1", key))
    a = auth(idp)
    t = jwt.encode(
        {"iss": ISS, "aud": AUD, "sub": "x", "tenant": "acme", "exp": time.time() + 99},
        key,
        algorithm="RS256",
        headers={"kid": "k1", "jku": "https://evil.example/keys", "x5u": "https://evil.example/c"},
    )
    assert (await a.authenticate(bearer(t))).tenant == "acme"
    assert idp.fetches == 1  # only the configured URL was ever fetched


def test_configuration_mistakes_fail_loudly():
    with pytest.raises(ValueError):
        JWTAuthenticator(issuer=ISS, audience=AUD)  # neither keys nor secret
    with pytest.raises(ValueError):
        JWTAuthenticator(issuer=ISS, audience=AUD, keys=Idp().keys(), algorithms=("HS256",))
    with pytest.raises(ValueError):
        JWTAuthenticator(issuer=ISS, audience=AUD, keys=Idp().keys(), algorithms=("none",))
    with pytest.raises(ValueError):
        JWTAuthenticator(issuer=ISS, audience=AUD, secret="short")


@pytest.mark.filterwarnings("ignore::UserWarning")
async def test_shared_secret_mode_accepts_only_hs256():
    secret = "s" * 40
    a = JWTAuthenticator(issuer=ISS, audience=AUD, secret=secret)
    claims = {"iss": ISS, "aud": AUD, "sub": "x", "tenant": "acme", "exp": time.time() + 99}
    assert (
        await a.authenticate(bearer(jwt.encode(claims, secret, algorithm="HS256")))
    ).tenant == "acme"
    with pytest.raises(AuthError):
        await a.authenticate(bearer(jwt.encode(claims, secret, algorithm="HS512")))


async def test_api_keys():
    a = ApiKeyAuthenticator.from_keys({"k-1": ("acme", "ann"), "k-2": ("globex", "gus", ["admin"])})
    assert (await a.authenticate(bearer("k-1"))).tenant == "acme"
    assert (await a.authenticate(bearer("k-2"))).scopes == {"admin"}
    for bad in ("k-", "k-11", "", "K-1"):
        with pytest.raises(AuthError):
            await a.authenticate(bearer(bad))
    with pytest.raises(AuthError):
        await a.authenticate({})


async def test_in_memory_limiter_refills_smoothly():
    now = [0.0]
    lim = InMemoryRateLimiter(per_second=2, burst=4, clock=lambda: now[0])
    assert [(await lim.take("k")).allowed for _ in range(5)] == [True] * 4 + [False]
    d = await lim.take("k")
    assert not d.allowed and d.retry_after == pytest.approx(0.5)
    now[0] += 0.5
    assert (await lim.take("k")).allowed


async def test_redis_limiter_is_shared_between_processes(redis_url):
    import redis.asyncio as aredis

    a, b = aredis.from_url(redis_url), aredis.from_url(redis_url)
    try:
        la = RedisRateLimiter(a, per_second=0.01, burst=3, prefix="t:rl:")
        lb = RedisRateLimiter(b, per_second=0.01, burst=3, prefix="t:rl:")
        got = [(await (la if i % 2 else lb).take("tenant/ann")).allowed for i in range(5)]
        assert got == [True, True, True, False, False]
        assert (await la.take("tenant/other")).allowed  # keys are independent
        assert (await lb.take("tenant/ann")).retry_after != 0
        # concurrent callers never exceed the burst
        results = await asyncio.gather(*(la.take("burst") for _ in range(20)))
        assert sum(r.allowed for r in results) == 3
    finally:
        await a.aclose()
        await b.aclose()
