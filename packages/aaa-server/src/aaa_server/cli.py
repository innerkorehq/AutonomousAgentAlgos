"""`aaa-server serve module:factory` - run the server for the configuration your factory builds.

The factory is a function (sync or async) returning a `ServerConfig`; it is where you choose the store, the LLM client, the
algorithms, the authenticator and the limits. The command only adds the network listener, and listens on loopback unless you
pass `--host`."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import sys
from collections.abc import Callable, Sequence
from typing import Any

from .app import ServerConfig, create_app


def load_factory(spec: str) -> Callable[[], Any]:
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise SystemExit("expected module:factory, e.g. myapp.server:build")
    sys.path.insert(0, "")
    fn = getattr(importlib.import_module(module), attr)
    return fn if callable(fn) else (lambda: fn)


async def _build(factory: Callable[[], Any]) -> ServerConfig:
    cfg = factory()
    if inspect.isawaitable(cfg):
        cfg = await cfg
    if not isinstance(cfg, ServerConfig):
        raise SystemExit(f"the factory must return a ServerConfig, got {type(cfg).__name__}")
    return cfg


def load_config(spec: str) -> ServerConfig:
    """Build the configuration now (for tests and sync code). The server itself builds it inside its own event loop."""
    return asyncio.run(_build(load_factory(spec)))


class LazyApp:
    """An ASGI app that builds the configuration INSIDE the server's event loop, at startup. A factory that opens connection
    pools (asyncpg, redis) must run in the loop that will use them, which `asyncio.run(factory())` before `uvicorn.run` is not."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory, self.app = factory, None

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    try:
                        cfg = await _build(self.factory)
                        stack = cfg.engine
                        await stack.__aenter__()
                        self.app = create_app(cfg)
                    except BaseException as e:  # noqa: BLE001 - report the failure to the server, which exits
                        await send(
                            {
                                "type": "lifespan.startup.failed",
                                "message": f"{type(e).__name__}: {e}",
                            }
                        )
                        return
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await stack.__aexit__(None, None, None)
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        elif self.app is None:
            raise RuntimeError("the server has not finished starting")
        else:
            await self.app(scope, receive, send)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aaa-server")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve", help="run the HTTP server")
    sv.add_argument("factory", help="module:function returning a ServerConfig")
    sv.add_argument("--host", default="127.0.0.1", help="default: loopback only")
    sv.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    import uvicorn

    app = LazyApp(load_factory(a.factory))
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: listening on {a.host}. Put TLS and a reverse proxy (body/header timeouts) in front; see the server guide.",
            file=sys.stderr,
        )
    uvicorn.run(app, host=a.host, port=a.port, log_level="info", server_header=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
