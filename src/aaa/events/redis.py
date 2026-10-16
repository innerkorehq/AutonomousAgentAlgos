"""Redis EventStore + bus backend (redis-py asyncio).

* Each stream is a Redis Stream whose entry ids are `<seq+1>-0`, so ordering and uniqueness are
  native. A Lua script appends atomically: it checks OCC, dedupes by event id, writes the event
  AND its bus entry, so the bus can never miss a stored event. All keys of a partition share a
  hash tag `{p}`, which keeps the script valid on Redis Cluster.
* The bus is one Redis Stream per partition with explicit ids from an INCR counter. Consumer
  cursors are stored in a hash; partition leases are `SET NX PX` keys renewed by their owner.
* Durability is whatever your Redis is configured for (use AOF `appendfsync everysec`/`always`
  for an event log; without persistence a restart loses the log). Prefer Postgres when you need
  the event log to be the system of record.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import redis.asyncio as aioredis

from aaa.core.errors import ConcurrencyError
from aaa.events import migrate
from aaa.events.distributed import partition_of
from aaa.events.event import Event, scrubbed
from aaa.events.snapshot import StoredSnapshot
from aaa.events.store import ANY_SEQ, RunInfo

_APPEND = """
local n = tonumber(ARGV[4])
local expected = tonumber(ARGV[1])
local last = -1
local rev = redis.call('XREVRANGE', KEYS[1], '+', '-', 'COUNT', 1)
if #rev > 0 then last = tonumber(string.match(rev[1][1], '^(%d+)')) - 1 end
if expected ~= -2 and expected ~= last then return {-1, last} end
local res = {}
local nxt = last + 1
for i = 0, n - 1 do
  local id, typ, body = ARGV[5 + i * 3], ARGV[6 + i * 3], ARGV[7 + i * 3]
  local existing = redis.call('HGET', KEYS[2], id)
  if existing then
    local sid = (tonumber(existing) + 1) .. '-0'
    local ent = redis.call('XRANGE', KEYS[1], sid, sid)
    res[#res + 1] = {1, tonumber(existing), ent[1][2][4]}
  else
    redis.call('XADD', KEYS[1], (nxt + 1) .. '-0', 'id', id, 'body', body)
    redis.call('HSET', KEYS[2], id, nxt)
    if ARGV[3] == '1' then
      local bid = redis.call('INCR', KEYS[4])
      redis.call('XADD', KEYS[3], bid .. '-0', 'seq', nxt, 'body', body)
    end
    if typ == 'aaa.run.started' then redis.call('HSETNX', KEYS[5], ARGV[2], nxt .. ':' .. body) end
    if typ == 'aaa.run.completed' then redis.call('HSETNX', KEYS[6], ARGV[2], nxt .. ':' .. body) end
    res[#res + 1] = {0, nxt, ''}
    nxt = nxt + 1
  end
end
return res
"""

_LEASE = """
local cur = redis.call('GET', KEYS[1])
if not cur then redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2]) return 1 end
if cur == ARGV[1] then redis.call('PEXPIRE', KEYS[1], ARGV[2]) return 1 end
return 0
"""
_UNLEASE = (
    "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end return 0"
)
_CURSOR = """
local cur = tonumber(redis.call('HGET', KEYS[1], ARGV[1]) or '0')
if tonumber(ARGV[2]) > cur then redis.call('HSET', KEYS[1], ARGV[1], ARGV[2]) end
return 1
"""
_BEAT = """
local t = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[1])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
return 1
"""
_MEMBERS = """
local t = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
return redis.call('ZRANGEBYSCORE', KEYS[1], now + 1, '+inf')
"""
_SNAP = """
local cur = redis.call('HGET', KEYS[1], ARGV[1])
if cur then
  local c = cjson.decode(cur)
  if tonumber(c.seq) >= tonumber(ARGV[2]) then return 0 end
end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[3])
return 1
"""


def _s(x: Any) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


MIGRATIONS: dict[
    int, Any
] = {}  # n -> async callable(redis, prefix) migrating layout n-1 to n (empty: version 1 is the 0.1/0.2 layout)


class RedisEventStore:
    def __init__(self, r: aioredis.Redis, partitions: int, prefix: str, outbox: bool) -> None:
        # redis-py's asyncio client is typed as sync-or-awaitable unions that type checkers cannot
        # resolve, so the handle is deliberately `Any`; behavior is covered by the contract tests.
        self.r: Any = r
        self.partitions, self.prefix, self.outbox = partitions, prefix, outbox
        self._append = r.register_script(_APPEND)
        self._backend: RedisBusBackend | None = None

    @classmethod
    async def create(
        cls,
        url: str,
        *,
        partitions: int = 16,
        prefix: str = "aaa",
        outbox: bool = True,
        max_connections: int | None = None,
    ) -> RedisEventStore:
        """`max_connections` bounds the client pool. Every consumer group blocks on each partition it owns, so a worker needs
        roughly `groups * partitions` connections; the default (`max(100, 16 * partitions)`) covers the built-in groups plus
        a few algorithms, and a deployment with many algorithms or partitions should raise it."""
        r: Any = aioredis.from_url(
            url,
            decode_responses=False,
            max_connections=max_connections or max(100, 16 * partitions),
        )
        key = f"{prefix}:meta:partitions"
        await r.set(key, str(partitions), nx=True)
        have = int(await r.get(key))
        if have != partitions:
            await r.aclose()
            raise ValueError(
                f"prefix {prefix!r} was created with {have} partitions, not {partitions}; "
                "the partition count is fixed for the lifetime of a cluster"
            )
        vkey = f"{prefix}:meta:schema_version"
        await r.set(
            vkey, str(migrate.SCHEMA_VERSION), nx=True
        )  # a prefix without a version was created by 0.1/0.2: layout 1
        try:
            found = int(await r.get(vkey))
            for v in migrate.pending(found, migrate.SCHEMA_VERSION, MIGRATIONS):
                await MIGRATIONS[v](r, prefix)
                await r.set(vkey, str(v))
        except BaseException:
            await r.aclose()
            raise
        return cls(r, partitions, prefix, outbox)

    async def schema_version(self) -> int:
        return int(await self.r.get(f"{self.prefix}:meta:schema_version"))

    # key helpers (all keys of a partition share the hash tag {p})
    def _k(self, p: int, *parts: str) -> str:
        return f"{self.prefix}:{{{p}}}:" + ":".join(parts)

    def bus_backend(self) -> RedisBusBackend:
        if self._backend is None:
            self._backend = RedisBusBackend(self)
        return self._backend

    @staticmethod
    def _event(seq: int, body: bytes | str) -> Event:
        return Event.model_validate_json(body).model_copy(update={"seq": seq})

    async def append(
        self, stream: str, events: Sequence[Event], expected_seq: int = ANY_SEQ
    ) -> list[Event]:
        p = partition_of(stream, self.partitions)
        args: list[Any] = [expected_seq, stream, "1" if self.outbox else "0", len(events)]
        for e in events:
            stored = e.model_copy(update={"stream": stream, "seq": -1})
            args += [e.id, e.type, stored.model_dump_json()]
        res = await self._append(
            keys=[
                self._k(p, "ev", stream),
                self._k(p, "ids", stream),
                self._k(p, "bus"),
                self._k(p, "busn"),
                self._k(p, "started"),
                self._k(p, "done"),
            ],
            args=args,
        )
        if res and not isinstance(res[0], list):
            raise ConcurrencyError(f"{stream}: expected seq {expected_seq}, at {res[1]}")
        out: list[Event] = []
        for e, (dup, seq, body) in zip(events, res, strict=True):
            if dup:
                out.append(self._event(int(seq), body).model_copy(update={"dup": True}))
            else:
                out.append(e.model_copy(update={"stream": stream, "seq": int(seq)}))
        return out

    async def read(self, stream: str, from_seq: int = 0, limit: int | None = None) -> list[Event]:
        p = partition_of(stream, self.partitions)
        rows = await self.r.xrange(self._k(p, "ev", stream), f"{from_seq + 1}-0", "+", count=limit)
        return [self._event(int(_s(i).split("-")[0]) - 1, f[b"body"]) for i, f in rows]

    async def last_seq(self, stream: str) -> int:
        p = partition_of(stream, self.partitions)
        rows = await self.r.xrevrange(self._k(p, "ev", stream), "+", "-", count=1)
        return int(_s(rows[0][0]).split("-")[0]) - 1 if rows else -1

    async def has(self, stream: str, event_id: str) -> bool:
        p = partition_of(stream, self.partitions)
        return bool(await self.r.hexists(self._k(p, "ids", stream), event_id))

    async def subscribe(self, stream: str, from_seq: int = 0) -> AsyncIterator[Event]:
        p, pos = partition_of(stream, self.partitions), from_seq
        key = self._k(p, "ev", stream)
        while True:
            batch = await self.read(stream, pos)
            if not batch:
                await self.r.xread({key: f"{pos}-0"}, block=500, count=1)  # wakes on append
                continue
            for e in batch:
                yield e
            pos = batch[-1].seq + 1

    async def run_index(self) -> list[RunInfo]:
        out: list[RunInfo] = []
        for p in range(self.partitions):
            started = await self.r.hgetall(self._k(p, "started"))
            done = await self.r.hgetall(self._k(p, "done"))

            def parse(v: bytes) -> Event:
                seq, _, body = v.decode().partition(":")
                return self._event(int(seq), body)

            for stream, v in started.items():
                s = stream.decode()
                out.append(RunInfo(s, parse(v), parse(done[stream]) if stream in done else None))
        return out

    # -- snapshots --------------------------------------------------------------------------
    def _snap_key(self, stream: str) -> str:
        return self._k(partition_of(stream, self.partitions), "snap", stream)

    async def load_snapshot(self, stream: str, name: str) -> StoredSnapshot | None:
        raw = await self.r.hget(self._snap_key(stream), name)
        if raw is None:
            return None
        d = json.loads(raw)
        return StoredSnapshot(
            stream=stream, name=name, seq=d["seq"], version=d["version"], state=d["state"]
        )

    async def save_snapshot(self, snap: StoredSnapshot) -> None:
        await self.r.eval(
            _SNAP,
            1,
            self._snap_key(snap.stream),
            snap.name,
            snap.seq,
            json.dumps({"seq": snap.seq, "version": snap.version, "state": snap.state}),
        )

    async def _erase(self, stream: str, delete: bool) -> int:
        p = partition_of(stream, self.partitions)
        key = self._k(p, "ev", stream)
        rows = await self.r.xrange(key, "-", "+")
        started_key, done_key = self._k(p, "started"), self._k(p, "done")
        async with self.r.pipeline(transaction=True) as pipe:
            pipe.delete(key)
            if delete:
                pipe.delete(self._k(p, "ids", stream))
                pipe.hdel(started_key, stream)
                pipe.hdel(done_key, stream)
            else:
                for entry_id, f in rows:
                    seq = int(_s(entry_id).split("-")[0]) - 1
                    s = scrubbed(self._event(seq, f[b"body"]))
                    body = s.model_dump_json()
                    pipe.xadd(key, {"id": s.id, "body": body}, id=_s(entry_id))
                    if s.type == "aaa.run.started":
                        pipe.hset(started_key, stream, f"{seq}:{body}")
                    elif s.type == "aaa.run.completed":
                        pipe.hset(done_key, stream, f"{seq}:{body}")
            pipe.delete(self._snap_key(stream))
            await pipe.execute()
        # bus copies (a bounded, transient outbox) carry the payloads too: drop this stream's
        gone: list[Any] = []
        for entry_id, f in await self.r.xrange(self._k(p, "bus"), "-", "+"):
            if json.loads(f[b"body"]).get("stream") == stream:
                gone.append(entry_id)
        if gone:
            await self.r.xdel(self._k(p, "bus"), *gone)
        return len(rows)

    async def scrub(self, stream: str) -> int:
        return await self._erase(stream, False)

    async def delete_stream(self, stream: str) -> int:
        return await self._erase(stream, True)

    async def trim_outbox(self, keep_last: int = 100_000) -> int:
        """Trim each partition's bus stream to roughly the newest `keep_last` entries."""
        n = 0
        for p in range(self.partitions):
            n += await self.r.xtrim(self._k(p, "bus"), maxlen=keep_last, approximate=True)
        return n

    async def aclose(self) -> None:
        if self._backend:
            await self._backend.close()
        await self.r.aclose()


class RedisBusBackend:
    def __init__(self, store: RedisEventStore) -> None:
        self.store, self.partitions = store, store.partitions
        r = store.r
        self._lease, self._unlease = r.register_script(_LEASE), r.register_script(_UNLEASE)
        self._cursor, self._beat, self._members = (
            r.register_script(_CURSOR),
            r.register_script(_BEAT),
            r.register_script(_MEMBERS),
        )

    def _bus(self, p: int) -> str:
        return self.store._k(p, "bus")

    @staticmethod
    def _entry(eid: bytes, f: dict[bytes, bytes]) -> tuple[int, Event]:
        ev = Event.model_validate_json(f[b"body"]).model_copy(update={"seq": int(f[b"seq"])})
        return int(_s(eid).split("-")[0]), ev

    async def read(
        self, partition: int, after: int, limit: int, wait_s: float
    ) -> list[tuple[int, Event]]:
        r, key = self.store.r, self._bus(partition)
        if wait_s > 0:
            res = await r.xread({key: f"{after}-0"}, count=limit, block=max(1, int(wait_s * 1000)))
            rows = res[0][1] if res else []
        else:
            rows = await r.xrange(key, f"({after}-0", "+", count=limit)
        return [self._entry(i, f) for i, f in rows]

    async def tail(self, partition: int) -> int:
        v = await self.store.r.get(self.store._k(partition, "busn"))
        return int(v) if v else 0

    def _cur_key(self, group: str) -> str:
        return f"{self.store.prefix}:cursor:{group}"

    async def get_cursor(self, group: str, partition: int) -> int:
        v = await self.store.r.hget(self._cur_key(group), str(partition))
        return int(v) if v else 0

    async def set_cursor(self, group: str, partition: int, cursor: int) -> None:
        await self._cursor(keys=[self._cur_key(group)], args=[str(partition), cursor])

    def _lease_key(self, group: str, partition: int) -> str:
        return f"{self.store.prefix}:lease:{group}:{partition}"

    async def try_lease(self, group: str, partition: int, member: str, ttl_s: float) -> bool:
        return bool(
            await self._lease(
                keys=[self._lease_key(group, partition)], args=[member, int(ttl_s * 1000)]
            )
        )

    async def release_lease(self, group: str, partition: int, member: str) -> None:
        await self._unlease(keys=[self._lease_key(group, partition)], args=[member])

    def _mem_key(self, group: str) -> str:
        return f"{self.store.prefix}:members:{group}"

    async def heartbeat(self, group: str, member: str, ttl_s: float) -> None:
        await self._beat(keys=[self._mem_key(group)], args=[member, int(ttl_s * 1000)])

    async def members(self, group: str) -> list[str]:
        return sorted(_s(m) for m in await self._members(keys=[self._mem_key(group)]))

    async def leave(self, group: str, member: str) -> None:
        await self.store.r.zrem(self._mem_key(group), member)

    async def abandon(self) -> None:
        """Simulated crash: leases and membership are left to expire by TTL."""
        await asyncio.sleep(0)

    async def close(self) -> None:
        return None


__all__ = ["RedisBusBackend", "RedisEventStore"]
