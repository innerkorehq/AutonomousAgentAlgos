async def test_ephemeral_postgres_and_redis_work(pg_dsn, redis_url):
    import asyncpg
    import redis.asyncio as aioredis

    c = await asyncpg.connect(pg_dsn)
    assert (await c.fetchval("select 41 + 1")) == 42
    await c.close()
    r = aioredis.from_url(redis_url)
    assert await r.ping()
    await r.aclose()
