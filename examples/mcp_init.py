"""Generate a project around a server configuration, then see whether its files have drifted from their templates. Offline."""

import json
import tempfile
from pathlib import Path

from aaa.mcp import scaffold
from aaa.mcp.config import load_spec

CONFIG = {
    "server": {"name": "orders"},
    "sources": [
        {
            "type": "openapi",
            "document": "openapi/api.json",
            "allowed_hosts": ["api.example.com"],
            "auth": {"scheme": "bearer", "secret": "env:ORDERS_API_TOKEN"},
        }
    ],
}

with tempfile.TemporaryDirectory() as tmp:
    spec = load_spec(CONFIG, environ={})
    options = scaffold.Options("orders", transport="http", docker=True, fmt="toml")
    files = scaffold.render(spec, scaffold.config_text_for(spec, options), options)
    dest = Path(tmp) / "orders"
    scaffold.write_project(dest, files, options)
    print("project files:", ", ".join(sorted(files)))
    print("secret names (never values):", scaffold.secret_names(spec))
    print("drift right after generating:", scaffold.check_project(dest) or "none")
    (dest / "Dockerfile").write_text("FROM scratch\n")
    print("after editing the Dockerfile:", scaffold.check_project(dest))
    manifest = json.loads((dest / scaffold.MANIFEST).read_text())
    print(
        "recorded:",
        manifest["generator"].split()[0],
        "with",
        len(manifest["files"]),
        "hashed files",
    )
