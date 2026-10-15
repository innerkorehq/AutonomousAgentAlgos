# Built-in tools and the code sandbox

A model that can browse, read files and run code is useful, and dangerous: **anything it reads can steer it**. These tools are built on that assumption. The reasoning, threat by
threat, is on the [tools threat-model page](../security/tools.md); this page is how to use them.

```python
--8<-- "examples/builtin_tools.py"
```

```text
fetch http://169.254.169.254/latest/meta-data/: refused (the host resolves to a non-public address)
fetch http://localhost/admin: refused (the host resolves to a non-public address)
fetch http://example.com:6379/: refused (port 6379 is not allowed)
fetch https://evil.example.com/: refused (this host is not allowed)
read notes.txt: project notes
read ../secret.txt: refused (path is outside the allowed directory)
code_exec waits for approval: code_exec
The program printed: 5050
```

## `http_fetch`: fetch a page without becoming an SSRF gadget

```python
from aaa.tools.builtin import http_fetch_tool, FetchPolicy

http_fetch_tool(FetchPolicy.strict(["docs.python.org", "*.readthedocs.io"]))     # https only, those hosts only (recommended)
http_fetch_tool()                                                                  # any PUBLIC host (logs a warning once)
```

On every request and every redirect: the scheme must be http(s) (or https only), there must be no credentials in the URL, the port must be allowed (80 and 443 by default), the host must match
`allowed_hosts` and not `denied_hosts`, the host is resolved **once** and **every** address must be globally routable (loopback, private, link-local, CGNAT, multicast, reserved, and IPv6
forms that embed such an IPv4 address are refused), and the connection goes to the address that was checked, with the original name as `Host` and TLS server name, so DNS rebinding has no second
lookup to exploit. Redirects are followed by AAA, not httpx, so the whole check runs again for each; an https to http downgrade is refused. Proxy environment variables are ignored (a proxy
would resolve the name itself). Bodies are read as a stream and cut at `max_bytes` after decompression; only text-like content types are read; HTML is reduced to text. Refusals say why in
general terms and never echo an address.

**Without an allow-list any public host is reachable**, and a model can put data in a URL. Set `allowed_hosts` in production, and filter egress at the network too: this is one layer.
`FetchPolicy(allow_private_networks=True)` exists for tests and trusted intranets and is named so it reads as a decision.

## `read_file` and `list_files`

```python
from aaa.tools.builtin import read_file_tool, list_files_tool
registry = ToolRegistry([read_file_tool("./docs"), list_files_tool("./docs")])
```

Paths resolve against one root and must stay inside it after resolution (so `..`, absolute paths and symlinks out are refused); hidden files and directories (`.env`, `.git`) and names
like `*.pem` or `id_rsa*` are refused; reads are size-capped, binary files refused, and `offset`/`limit` read a range of lines. Keep the root unwritable by untrusted parties (a swapped
directory component is the one race this cannot close).

## `web_search`

```python
from aaa.tools.builtin import web_search_tool, SearXNGSearch, BraveSearch
web_search_tool(SearXNGSearch("http://localhost:8888"))        # a SearXNG you run
web_search_tool(BraveSearch(api_key="..."))                    # Brave Search API
```

`SearchProvider` is one method (`search(query, k) -> list[SearchResult]`). Results are untrusted text: snippets are capped at 500 characters and non-http(s) URLs dropped; run the
[`tool_result` guardrail](guardrails.md) over them. The query leaves your process, and the provider's key never reaches the model. SearXNG needs `json` in `search.formats` of its settings
(otherwise it answers 403 and the provider tells you so).

## Running code

```python
from aaa.sandbox import DockerSandbox, Limits, code_exec_tool
registry = ToolRegistry([code_exec_tool(DockerSandbox(), limits=Limits(timeout_s=10, memory_mb=256))])
```

**Every `code_exec` call waits for a human to approve it** (the engine's approval flow); pass `requires_approval=False` only for a sandbox you would trust with the worst code the model could write.

| | `DockerSandbox` | `SubprocessSandbox(unsafe=True)` |
|---|---|---|
| Isolation | a new container per run: `--network none`, read-only root, `--cap-drop ALL`, `no-new-privileges`, non-root user, memory (swap off), CPU and PID limits, a small `noexec` tmpfs, nothing from the host mounted | **none**: your user, files and network. Local development only; refuses to be created without `unsafe=True` |
| Limits enforced | wall-clock timeout (container killed), output cap (killed), memory (reported as `killed="oom"`), processes, file size | timeout (whole process group killed), output cap, CPU time, file size, open files; memory on Linux only |
| Languages | `python`, `bash`, `sh` (the image needs `python3`) | `python`, `bash`, `sh` |

Input files and the program travel on the container's stdin into its scratch space, so **no host path is mounted** and it works with Colima, rootless Docker and a remote `DOCKER_HOST`. Pin the
image by digest (`DockerSandbox(images={"python": "python:3.12-slim@sha256:..."})`). Results are an `ExecResult` (`exit_code`, `stdout`, `stderr`, `timed_out`, `killed`, `truncated`): the code failing
is a result, the sandbox failing (no Docker) is a `SandboxError`.

A container shares the host's kernel. For code from untrusted, multi-tenant users run the sandbox on a host or VM dedicated to it; VM-backed sandboxes (E2B, Modal, Firecracker) are a future adapter.

## What was verified

The SSRF table (addresses, spellings, redirects, rebinding, TLS name checks against a real local TLS server), the confinement of `read_file`, and the sandbox hardening ran against the real thing: a real
Docker engine for the network, filesystem, memory, fork-bomb, timeout and output limits, and a real SearXNG container for the search provider. Brave is built from its published request/response and
mock-tested. Not verified: a Docker engine other than Colima/Linux in CI, hosted Brave, and hostile code attempting a kernel escape (by design out of scope).
