# Threat model: built-in tools and the code sandbox

Written before the code (milestone M5). These tools hand a model the ability to reach the network, read files and run code. The model can be steered by anything it reads (a web page, a
file, a search snippet), so every tool is designed on the assumption that **its arguments are attacker-influenced**.

## Assets

the host's network position (internal services, cloud metadata), its files and secrets, its CPU and memory, other tenants' data, and the user's trust in what comes back.

## `http_fetch`: server-side request forgery is the whole game

| # | Threat | Decision |
|---|---|---|
| 1 | The model (or a page it read) asks for `http://169.254.169.254/` (cloud metadata), `localhost`, `10.x`, an internal admin port. | Every address a hostname resolves to must be **globally routable**; loopback, private, link-local, CGNAT, multicast, unspecified, reserved, and IPv6 unique-local are refused. Checked on **all** records, not the first. |
| 2 | Address spellings that dodge a string check: decimal/octal/hex IPv4 (`2130706433`, `0x7f.1`), IPv4-mapped IPv6 (`::ffff:127.0.0.1`), NAT64/6to4/Teredo forms, trailing dots, `localhost.`, userinfo tricks (`http://good.com@127.0.0.1`). | Parsing is done by the `ipaddress` module on the **resolved** addresses, never on the text; embedded IPv4 inside IPv6 transition forms is extracted and checked; URLs with credentials are refused. |
| 3 | **DNS rebinding**: the name resolves to a public address when checked and a private one when connected. | The connection goes to the **address we validated** (resolve once, connect to that IP, send the original name as `Host` and TLS server name). There is no second resolution. |
| 4 | **Redirects** to internal addresses, or from https to http. | Redirects are followed by us, one hop at a time, with the **full check repeated** on every hop; at most 5; no scheme downgrade; credentials never forwarded. |
| 5 | Schemes and ports: `file:`, `gopher:`, `ftp:`, port 22/6379/5432. | Only `http` and `https`; only ports 80 and 443 unless the policy lists more. |
| 6 | Resource exhaustion: an endless or huge body, a slow loris, a compression bomb. | Connect, read and total deadlines; the **decoded** body is cut at `max_bytes` while streaming (the rest is never read); content types are limited to text-like ones. |
| 7 | The response is **untrusted content that goes to the model** (prompt injection). | Not solvable here; flagged for the `tool_result` guard stage. Only the body text (HTML reduced to text) and a few safe headers are returned; cookies are not kept; no request headers from the model. |
| 8 | Using the tool as an exfiltration channel (secrets in the URL). | Documented. An `allowed_hosts` allow-list (exact names or `*.example.com`) is the control; **with no allow-list any public host is reachable** and the tool says so once at startup. Production should set one. |
| 9 | Error messages leaking internal topology. | Refusals say *why in general* ("resolves to a non-public address"), never the address or the resolver's answer. |

What it cannot do: a hostname that resolves to a public address which then proxies to an internal one (that is the remote site's own SSRF), or defend a host whose DNS an attacker controls
**and** which you allow-listed. Egress filtering at the network layer remains the strongest control and is recommended in addition.

## `read_file` and `list_files`

| # | Threat | Decision |
|---|---|---|
| 1 | Path traversal (`../../etc/passwd`), absolute paths, `~`. | Every path is resolved against a **root** and must stay inside it after resolution. |
| 2 | Symlinks pointing out of the root. | The resolved real path must be inside the root; the final component is opened with `O_NOFOLLOW`. |
| 3 | Reading secrets that live inside the root (`.env`, `.git`, keys). | Hidden files and directories are refused by default; a deny-list of name patterns (`*.pem`, `id_rsa*`, `.env*`) applies on top. |
| 4 | Huge or binary files flooding the context. | A byte cap, binary detection, and line-range reads. |
| 5 | Time-of-check/time-of-use swaps of a directory component. | Documented: the root must not be writable by untrusted parties. |

## `web_search`

The query leaves the process (to SearXNG you run, or a hosted API): do not put secrets in queries; the API key is held by the provider object and never reaches the model.
Results are untrusted text (snippet length is capped, non-http(s) URLs are dropped) and should pass the `tool_result` guard.

## The code sandbox

| # | Threat | `DockerSandbox` decision |
|---|---|---|
| 1 | Network access (exfiltration, scanning, downloading malware). | `--network none`. |
| 2 | Reading or writing the host. | **Nothing from the host is mounted**: the program and any input files travel on the container's stdin into its scratch tmpfs, so there is no host path to reach (this also works on remote and rootless Docker); read-only root filesystem; a small `noexec` tmpfs for scratch; no Docker socket. |
| 3 | Privilege escalation and kernel attack surface. | `--cap-drop ALL`, `--security-opt no-new-privileges`, Docker's default seccomp profile, non-root user, never `--privileged`. |
| 4 | Resource exhaustion: memory, CPU, processes, disk, output. | `--memory` with swap disabled, `--cpus`, `--pids-limit`, `ulimit` on files and processes, a tmpfs size cap, a **wall-clock timeout** that kills the container, and an output cap that kills it as well. |
| 5 | State leaking between runs or tenants. | A new container per run, removed afterwards; nothing persists. |
| 6 | Secrets in the environment. | The container gets an **empty** environment (only what the caller passes). |
| 7 | A malicious or compromised image. | The image is configurable and its tag/digest is the operator's responsibility; the docs recommend pinning by digest. |

**What this does not give you:** a container shares the host kernel, so a kernel or runtime vulnerability is an escape. For code from untrusted multi-tenant users run the sandbox on a
host or VM dedicated to it (or use a VM-backed provider, which is a future adapter). It is not a boundary against a determined attacker with a kernel exploit.

`SubprocessSandbox` is **not isolation at all**: it runs the code as a child process with CPU/file limits and a timeout, but with the host's network, files and user. It exists for local
development and refuses to be created unless you pass `unsafe=True`, so it cannot be picked by accident.

`code_exec` defaults to **requiring human approval** for every call, using the engine's existing approval flow.

## Out of scope

malware analysis of fetched content, content moderation of results, rate limiting of a provider's quota, and anything the operating system or network is supposed to enforce
(firewalls, egress proxies), which remain recommended.
