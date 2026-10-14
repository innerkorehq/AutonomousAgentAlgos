import os
import threading
import time

_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_lock = threading.Lock()
_last_ms = 0
_last_rand = 0


def new_id() -> str:
    """Time-sortable, monotonic ULID (26 chars, Crockford base32)."""
    global _last_ms, _last_rand
    with _lock:
        ms = time.time_ns() // 1_000_000
        if ms <= _last_ms:
            ms = _last_ms
            _last_rand += 1
        else:
            _last_ms = ms
            _last_rand = int.from_bytes(os.urandom(10), "big")
        value = (ms << 80) | (_last_rand & ((1 << 80) - 1))
    return "".join(_B32[(value >> (5 * i)) & 31] for i in range(25, -1, -1))


def stable_id(*parts: object) -> str:
    """Deterministic id from its inputs: re-deriving the same effect after a crash yields the
    same id, so the store's idempotent append absorbs the repeat (effectively-once on the log)."""
    import hashlib

    h = hashlib.blake2b("\x1f".join(map(str, parts)).encode(), digest_size=13).digest()
    return "".join(_B32[b & 31] for b in h)[:20] + "".join(_B32[(b >> 3) & 31] for b in h[:6])
