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
from collections.abc import Sequence

from .app import ServerConfig, create_app


def load_config(spec: str) -> ServerConfig:
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise SystemExit("expected module:factory, e.g. myapp.server:build")
    sys.path.insert(0, "")
    fn = getattr(importlib.import_module(module), attr)
    cfg = fn() if callable(fn) else fn
    if inspect.isawaitable(cfg):
        cfg = asyncio.run(_await(cfg))
    if not isinstance(cfg, ServerConfig):
        raise SystemExit(f"{spec} must return a ServerConfig, got {type(cfg).__name__}")
    return cfg


async def _await(x):  # pragma: no cover - trivial
    return await x


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aaa-server")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve", help="run the HTTP server")
    sv.add_argument("factory", help="module:function returning a ServerConfig")
    sv.add_argument("--host", default="127.0.0.1", help="default: loopback only")
    sv.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    import uvicorn

    app = create_app(load_config(a.factory), manage_engine=True)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: listening on {a.host}. Put TLS and a reverse proxy (body/header timeouts) in front; see the server guide.",
            file=sys.stderr,
        )
    uvicorn.run(app, host=a.host, port=a.port, log_level="info", server_header=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
