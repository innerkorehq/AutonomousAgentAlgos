"""A fetcher a model can be pointed at without it becoming an SSRF gadget. See docs/security/tools.md for the threat model.

The rules, in the order they run on every hop (the first request and every redirect):

1. the URL must be `http` or `https`, carry no credentials, and name a host and an allowed port;
2. the host must match `allowed_hosts` (when set) and not `denied_hosts`;
3. the host is resolved ONCE, and every address it resolves to must be globally routable (loopback, private, link-local, CGNAT, multicast, reserved and the IPv6
   transition forms that embed such an IPv4 address are refused);
4. the connection is made to the address that was validated, with the original name sent as `Host` and as the TLS server name, so there is no second DNS lookup to rebind;
5. the body is streamed and cut at `max_bytes`; only text-like content types are read.

Redirects are followed here (not by httpx) so that rules 1 to 4 run again for each one. Proxy environment variables are ignored: a proxy would resolve the name itself."""

from __future__ import annotations

import asyncio
import fnmatch
import ipaddress
import logging
import socket
import ssl
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx

from aaa.tools.tool import ToolError

log = logging.getLogger("aaa.tools")
Resolver = Callable[[str, int], Awaitable[list[str]]]
_warned = False

# beyond ipaddress.is_global: ranges that are "global" by the registry but never a legitimate public destination
_EXTRA_BLOCKED = tuple(
    ipaddress.ip_network(n)
    for n in (
        "192.0.0.0/24",  # IETF protocol assignments
        "198.18.0.0/15",  # benchmarking
        "192.88.99.0/24",  # deprecated 6to4 relay
        "2001::/32",  # Teredo
        "2001:db8::/32",  # documentation
        "3fff::/20",  # documentation
        "100::/64",  # discard-only
    )
)
DEFAULT_CONTENT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml+xml",
    "application/rss+xml",
    "application/atom+xml",
    "application/ld+json",
)


class FetchBlocked(ToolError):
    """The request was refused by policy (said plainly, without echoing internal addresses)."""


def blocked_reason(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Why `ip` may not be fetched, or None if it is a public address."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return blocked_reason(ip.ipv4_mapped)
        if ip in ipaddress.ip_network("64:ff9b::/96"):  # NAT64: the low 32 bits are an IPv4 address
            return blocked_reason(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
        if ip in ipaddress.ip_network("2002::/16"):  # 6to4: bits 16..48 are an IPv4 address
            return blocked_reason(ipaddress.IPv4Address((int(ip) >> 80) & 0xFFFFFFFF))
    if (
        not ip.is_global
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_loopback
        or ip.is_link_local
    ):
        return "a non-public address"
    if any(ip in net for net in _EXTRA_BLOCKED if net.version == ip.version):
        return "a reserved address"
    return None


async def system_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    out: list[str] = []
    for *_, sockaddr in infos:
        ip = str(sockaddr[0]).split("%", 1)[0]  # drop an IPv6 zone id
        if ip not in out:
            out.append(ip)
    return out


def _match(host: str, patterns: tuple[str, ...]) -> bool:
    return any(
        fnmatch.fnmatchcase(host, p.lower()) or (p.startswith("*.") and host == p[2:].lower())
        for p in patterns
    )


@dataclass(frozen=True)
class FetchPolicy:
    """What `http_fetch` may reach. With no `allowed_hosts` every PUBLIC host is reachable: set it in production."""

    allowed_hosts: tuple[str, ...] = ()  # exact names or "*.example.com"; empty = any public host
    denied_hosts: tuple[str, ...] = ()
    allowed_ports: tuple[int, ...] = (80, 443)
    allow_http: bool = True  # False: https only
    max_redirects: int = 5
    max_bytes: int = 1_000_000
    timeout_s: float = 20.0  # for the whole fetch, redirects included
    connect_timeout_s: float = 5.0
    content_types: tuple[str, ...] = DEFAULT_CONTENT_TYPES
    user_agent: str = "aaa-http-fetch/1.0"
    # Escape hatch for tests and trusted intranet use. Off unless you say so, and named so it reads as a decision.
    allow_private_networks: bool = False
    extra: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def strict(cls, allowed_hosts: tuple[str, ...] | list[str], **kw: object) -> FetchPolicy:
        """https only, to the listed hosts only."""
        return cls(allowed_hosts=tuple(allowed_hosts), allow_http=False, allowed_ports=(443,), **kw)  # type: ignore[arg-type]


@dataclass
class FetchResult:
    url: str  # the final URL after redirects
    status: int
    content_type: str
    text: str
    truncated: bool
    redirects: list[str] = field(default_factory=list)


class SafeFetcher:
    def __init__(
        self,
        policy: FetchPolicy | None = None,
        *,
        resolver: Resolver | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        verify: ssl.SSLContext | bool = True,
    ) -> None:
        self.policy = policy or FetchPolicy()
        self.resolver = resolver or system_resolver
        self._transport, self._verify = transport, verify
        global _warned
        if not self.policy.allowed_hosts and not _warned:
            _warned = True
            log.warning(
                "http_fetch has no allowed_hosts: every public host is reachable. Set FetchPolicy(allowed_hosts=...) in production."
            )

    # -- one hop ---------------------------------------------------------------------------
    async def _target(self, url: str) -> tuple[httpx.URL, str, str]:
        """(connect URL with the validated IP, original host, Host header value) or `FetchBlocked`."""
        try:
            u = httpx.URL(url)
        except httpx.InvalidURL as e:
            raise FetchBlocked("not a valid URL") from e
        scheme = u.scheme.lower()
        if scheme not in ("http", "https") or (scheme == "http" and not self.policy.allow_http):
            raise FetchBlocked(f"scheme {scheme!r} is not allowed")
        if u.userinfo:
            raise FetchBlocked("URLs with credentials are not allowed")
        host = u.host.rstrip(".").lower() if u.host else ""
        if not host:
            raise FetchBlocked("the URL has no host")
        port = u.port or (443 if scheme == "https" else 80)
        if port not in self.policy.allowed_ports:
            raise FetchBlocked(f"port {port} is not allowed")
        p = self.policy
        if (p.allowed_hosts and not _match(host, p.allowed_hosts)) or _match(host, p.denied_hosts):
            raise FetchBlocked("this host is not allowed")
        try:
            addrs = await asyncio.wait_for(self.resolver(host, port), p.connect_timeout_s)
        except (OSError, TimeoutError) as e:
            raise FetchBlocked("the host could not be resolved") from e
        if not addrs:
            raise FetchBlocked("the host could not be resolved")
        if not p.allow_private_networks:
            for a in addrs:
                try:
                    why = blocked_reason(ipaddress.ip_address(a))
                except ValueError:
                    why = "an unrecognised address"
                if why:
                    raise FetchBlocked(f"the host resolves to {why}")
        ip = addrs[0]
        shown = f"[{ip}]" if ":" in ip else ip
        default = 443 if scheme == "https" else 80
        host_header = host if port == default else f"{host}:{port}"
        return httpx.URL(f"{scheme}://{shown}:{port}{u.raw_path.decode()}"), host, host_header

    async def fetch(self, url: str, *, headers: Mapping[str, str] | None = None) -> FetchResult:
        try:
            async with asyncio.timeout(self.policy.timeout_s):
                return await self._fetch(url, headers)
        except TimeoutError as e:
            raise FetchBlocked(f"timed out after {self.policy.timeout_s}s") from e
        except (
            httpx.HTTPError
        ) as e:  # connection refused, TLS failure, protocol error: say what kind, not the details
            raise FetchBlocked(f"the request failed ({type(e).__name__})") from e

    async def _fetch(self, url: str, extra_headers: Mapping[str, str] | None) -> FetchResult:
        p = self.policy
        hops: list[str] = []
        timeout = httpx.Timeout(p.timeout_s, connect=p.connect_timeout_s)
        async with httpx.AsyncClient(
            transport=self._transport,
            follow_redirects=False,
            trust_env=False,
            timeout=timeout,
            verify=self._verify,
        ) as client:  # type: ignore[arg-type]
            current = url
            for _ in range(p.max_redirects + 1):
                connect, host, host_header = await self._target(current)
                hdrs = {
                    "user-agent": p.user_agent,
                    "accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.1",
                    **(extra_headers or {}),
                    "host": host_header,
                }
                ext = {"sni_hostname": host} if connect.scheme == "https" else {}
                async with client.stream("GET", connect, headers=hdrs, extensions=ext) as resp:
                    if resp.is_redirect and (loc := resp.headers.get("location")):
                        nxt = urljoin(current, loc)
                        if httpx.URL(current).scheme == "https" and httpx.URL(nxt).scheme == "http":
                            raise FetchBlocked("a redirect from https to http was refused")
                        hops.append(nxt)
                        current = nxt
                        continue
                    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    if (
                        resp.status_code < 400
                        and ctype
                        and not any(ctype.startswith(t) for t in p.content_types)
                    ):
                        raise FetchBlocked(f"content type {ctype!r} is not allowed")
                    body, truncated = bytearray(), False
                    async for chunk in resp.aiter_bytes():
                        body += chunk
                        if len(body) > p.max_bytes:
                            del body[p.max_bytes :]
                            truncated = True
                            break
                    charset = resp.charset_encoding or "utf-8"
                    try:
                        text = bytes(body).decode(charset, errors="replace")
                    except LookupError:
                        text = bytes(body).decode("utf-8", errors="replace")
                    if ctype in ("text/html", "application/xhtml+xml"):
                        from aaa.ingest.loaders import html_to_text

                        text = html_to_text(text)[0]
                    return FetchResult(current, resp.status_code, ctype, text, truncated, hops)
        raise FetchBlocked(f"more than {p.max_redirects} redirects")
