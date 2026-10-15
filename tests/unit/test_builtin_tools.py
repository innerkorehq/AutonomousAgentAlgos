"""http_fetch (SSRF table, redirects, pinning, limits), read_file confinement, web_search providers."""

import asyncio
import gzip
import ipaddress
import json
import socket
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from aaa.tools.builtin import (
    BraveSearch,
    FetchBlocked,
    FetchPolicy,
    FileAccess,
    SafeFetcher,
    SearXNGSearch,
    http_fetch_tool,
    list_files_tool,
    read_file_tool,
    web_search_tool,
)
from aaa.tools.builtin.net import blocked_reason, system_resolver
from aaa.tools.tool import ToolError

PUBLIC = "93.184.216.34"


def resolver(mapping: dict[str, list[str]], calls: list | None = None):
    async def r(host: str, port: int) -> list[str]:
        if calls is not None:
            calls.append(host)
        if host not in mapping:
            raise OSError("no such host")
        return mapping[host]

    return r


def ok_transport(
    seen: list | None = None,
    body="<html><title>T</title><body><p>hello</p><script>x()</script></body></html>",
    ctype="text/html; charset=utf-8",
    status=200,
):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(
            status,
            content=body.encode() if isinstance(body, str) else body,
            headers={"content-type": ctype},
        )

    return httpx.MockTransport(handler)


# -- which addresses count as public ------------------------------------------------------------
@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1", "127.1.2.3", "0.0.0.0", "10.0.0.5", "172.16.0.1", "172.31.255.255", "192.168.1.1", "169.254.169.254", "100.64.0.1", "224.0.0.1", "240.0.0.1", "255.255.255.255",
        "192.0.0.8", "198.18.0.1", "::1", "::", "fe80::1", "fc00::1", "fd12:3456::1", "ff02::1", "::ffff:127.0.0.1", "::ffff:169.254.169.254", "::ffff:a9fe:a9fe",
        "64:ff9b::7f00:1", "64:ff9b::a00:1", "2002:7f00:1::", "2002:a9fe:a9fe::1", "2001::1", "2001:db8::1",
    ],
)  # fmt: skip
def test_non_public_addresses_are_blocked(ip):
    assert blocked_reason(ipaddress.ip_address(ip)) is not None


@pytest.mark.parametrize(
    "ip",
    [
        "93.184.216.34",
        "8.8.8.8",
        "1.1.1.1",
        "2606:4700:4700::1111",
        "::ffff:8.8.8.8",
        "64:ff9b::808:808",
    ],
)
def test_public_addresses_pass(ip):
    assert blocked_reason(ipaddress.ip_address(ip)) is None


# -- the fetch policy, end to end through the resolver ------------------------------------------
BLOCKED_URLS = [
    ("http://localhost/", {"localhost": ["127.0.0.1", "::1"]}),
    ("http://127.0.0.1/", {"127.0.0.1": ["127.0.0.1"]}),
    ("http://[::1]/", {"::1": ["::1"]}),
    (
        "http://metadata.google.internal/computeMetadata/v1/",
        {"metadata.google.internal": ["169.254.169.254"]},
    ),
    ("http://169.254.169.254/latest/meta-data/", {"169.254.169.254": ["169.254.169.254"]}),
    ("http://intranet.corp/", {"intranet.corp": ["10.1.2.3"]}),
    (
        "http://mixed.example/",
        {"mixed.example": [PUBLIC, "10.0.0.1"]},
    ),  # one private record is enough to refuse
    ("http://mapped.example/", {"mapped.example": ["::ffff:7f00:1"]}),
    ("http://localhost./", {"localhost": ["127.0.0.1"]}),  # trailing dot
    ("http://LOCALHOST/", {"localhost": ["127.0.0.1"]}),
]


@pytest.mark.parametrize(("url", "dns"), BLOCKED_URLS)
async def test_internal_destinations_are_refused_and_nothing_is_sent(url, dns):
    seen: list = []
    f = SafeFetcher(resolver=resolver(dns), transport=ok_transport(seen))
    with pytest.raises(FetchBlocked):
        await f.fetch(url)
    assert seen == []


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com/",
        "javascript:alert(1)",
        "data:text/plain,hi",
        "ws://example.com/",
        "//example.com/x",
        "not a url",
        "http:///nohost",
        "",
    ],
)
async def test_other_schemes_and_malformed_urls_are_refused(url):
    seen: list = []
    f = SafeFetcher(resolver=resolver({"example.com": [PUBLIC]}), transport=ok_transport(seen))
    with pytest.raises(FetchBlocked):
        await f.fetch(url)
    assert seen == []


async def test_credentials_in_the_url_and_odd_ports_are_refused():
    f = SafeFetcher(
        resolver=resolver({"example.com": [PUBLIC], "good.com": [PUBLIC]}), transport=ok_transport()
    )
    for url in (
        "http://good.com@127.0.0.1/",
        "http://user:pw@example.com/",
        "http://example.com:22/",
        "http://example.com:6379/",
        "https://example.com:8443/",
    ):
        with pytest.raises(FetchBlocked):
            await f.fetch(url)
    assert (
        await SafeFetcher(
            FetchPolicy(allowed_ports=(443, 8443)),
            resolver=resolver({"example.com": [PUBLIC]}),
            transport=ok_transport(),
        ).fetch("https://example.com:8443/")
    ).status == 200


@pytest.mark.parametrize("literal", ["2130706433", "0x7f.1", "017700000001", "127.1", "0177.0.0.1"])
async def test_alternative_ipv4_spellings_cannot_reach_loopback(literal):
    """Run through the REAL system resolver: whatever this OS makes of the spelling is checked as an address, not as text."""
    seen: list = []
    f = SafeFetcher(resolver=system_resolver, transport=ok_transport(seen))
    with pytest.raises(FetchBlocked):
        await f.fetch(f"http://{literal}/")
    assert seen == []


async def test_allow_and_deny_lists():
    dns = {
        h: [PUBLIC] for h in ("docs.example.com", "example.com", "evil.example.com", "other.org")
    }
    pol = FetchPolicy(allowed_hosts=("*.example.com",), denied_hosts=("evil.example.com",))
    f = SafeFetcher(pol, resolver=resolver(dns), transport=ok_transport())
    assert (await f.fetch("https://docs.example.com/")).status == 200 and (
        await f.fetch("https://example.com/")
    ).status == 200
    for blocked in ("https://evil.example.com/", "https://other.org/"):
        with pytest.raises(FetchBlocked, match="not allowed"):
            await f.fetch(blocked)
    strict = SafeFetcher(
        FetchPolicy.strict(["example.com"]), resolver=resolver(dns), transport=ok_transport()
    )
    with pytest.raises(FetchBlocked, match="scheme"):
        await strict.fetch("http://example.com/")


async def test_refusal_messages_do_not_leak_addresses():
    f = SafeFetcher(resolver=resolver({"intranet.corp": ["10.1.2.3"]}), transport=ok_transport())
    with pytest.raises(FetchBlocked) as ei:
        await f.fetch("http://intranet.corp/")
    assert "10.1.2.3" not in str(ei.value)


# -- pinning and redirects ----------------------------------------------------------------------
async def test_the_connection_goes_to_the_validated_address_and_the_name_is_resolved_once_per_hop():
    seen: list = []
    calls: list = []
    f = SafeFetcher(
        resolver=resolver({"example.com": [PUBLIC]}, calls), transport=ok_transport(seen)
    )
    await f.fetch("https://example.com/path?q=1")
    assert calls == ["example.com"]  # one lookup: there is no second one to rebind
    req = seen[0]
    assert (
        req.url.host == PUBLIC
        and req.headers["host"] == "example.com"
        and req.url.path == "/path"
        and req.url.query == b"q=1"
    )
    assert req.extensions["sni_hostname"] == "example.com"


async def test_a_rebinding_resolver_cannot_redirect_the_connection():
    answers = iter([[PUBLIC], ["127.0.0.1"]])  # public when checked, loopback if asked again
    seen: list = []

    async def r(host, port):
        return next(answers)

    f = SafeFetcher(resolver=r, transport=ok_transport(seen))
    await f.fetch("https://rebind.example/")
    assert [req.url.host for req in seen] == [PUBLIC]


def redirect_transport(chain: dict[str, str], seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.headers["host"]
        seen.append(host)
        if host in chain and request.url.path in ("/", "/next"):
            return httpx.Response(302, headers={"location": chain[host]})
        return httpx.Response(200, content=b"final", headers={"content-type": "text/plain"})

    return httpx.MockTransport(handler)


async def test_every_redirect_hop_is_checked_again():
    dns = {"a.example": [PUBLIC], "b.example": [PUBLIC], "inner.example": ["10.0.0.9"]}
    seen: list = []
    f = SafeFetcher(
        resolver=resolver(dns),
        transport=redirect_transport(
            {"a.example": "http://b.example/next", "b.example": "http://inner.example/secret"}, seen
        ),
    )
    with pytest.raises(FetchBlocked, match="non-public"):
        await f.fetch("http://a.example/")
    assert seen == ["a.example", "b.example"]  # the internal host was never contacted
    ok = SafeFetcher(
        resolver=resolver(dns), transport=redirect_transport({"a.example": "/relative"}, [])
    )
    r = await ok.fetch("http://a.example/")
    assert r.text == "final" and r.redirects == ["http://a.example/relative"]


async def test_redirects_to_metadata_to_other_schemes_and_downgrades_are_refused():
    dns = {"a.example": [PUBLIC], "169.254.169.254": ["169.254.169.254"]}
    for target, why in (
        ("http://169.254.169.254/latest/meta-data/", "non-public"),
        ("file:///etc/passwd", "scheme"),
        ("gopher://a.example/", "scheme"),
    ):
        f = SafeFetcher(
            resolver=resolver(dns), transport=redirect_transport({"a.example": target}, [])
        )
        with pytest.raises(FetchBlocked, match=why):
            await f.fetch("http://a.example/")
    f = SafeFetcher(
        resolver=resolver(dns),
        transport=redirect_transport({"a.example": "http://a.example/plain"}, []),
    )
    with pytest.raises(FetchBlocked, match="https to http"):
        await f.fetch("https://a.example/")


async def test_redirect_loops_stop():
    f = SafeFetcher(
        FetchPolicy(max_redirects=3),
        resolver=resolver({"a.example": [PUBLIC]}),
        transport=redirect_transport({"a.example": "http://a.example/"}, []),
    )
    with pytest.raises(FetchBlocked, match="redirects"):
        await f.fetch("http://a.example/")


# -- limits and content -------------------------------------------------------------------------
async def test_content_types_and_html_to_text():
    dns = {"example.com": [PUBLIC]}
    r = await SafeFetcher(resolver=resolver(dns), transport=ok_transport()).fetch(
        "http://example.com/"
    )
    assert r.text == "hello" and r.content_type == "text/html"
    j = await SafeFetcher(
        resolver=resolver(dns), transport=ok_transport(body='{"a": 1}', ctype="application/json")
    ).fetch("http://example.com/")
    assert json.loads(j.text) == {"a": 1}
    with pytest.raises(FetchBlocked, match="content type"):
        await SafeFetcher(
            resolver=resolver(dns), transport=ok_transport(body=b"\x89PNG", ctype="image/png")
        ).fetch("http://example.com/")
    with pytest.raises(FetchBlocked, match="content type"):
        await SafeFetcher(
            resolver=resolver(dns),
            transport=ok_transport(body=b"MZ", ctype="application/octet-stream"),
        ).fetch("http://example.com/")


async def test_the_body_is_cut_while_streaming_not_after_downloading_it():
    sent = [0]

    async def body():
        for _ in range(1000):
            sent[0] += 1
            yield b"x" * 10_000

    def handler(request):
        return httpx.Response(200, content=body(), headers={"content-type": "text/plain"})

    f = SafeFetcher(
        FetchPolicy(max_bytes=50_000),
        resolver=resolver({"example.com": [PUBLIC]}),
        transport=httpx.MockTransport(handler),
    )
    r = await f.fetch("http://example.com/")
    assert (
        r.truncated and len(r.text) == 50_000 and sent[0] < 20
    )  # stopped after a handful of chunks of a 10 MB body


async def test_a_compression_bomb_is_capped_after_decoding():
    bomb = gzip.compress(b"\x00" * 50_000_000)  # ~50 KB on the wire, 50 MB decoded

    def handler(request):
        return httpx.Response(
            200, content=bomb, headers={"content-type": "text/plain", "content-encoding": "gzip"}
        )

    f = SafeFetcher(
        FetchPolicy(max_bytes=10_000),
        resolver=resolver({"example.com": [PUBLIC]}),
        transport=httpx.MockTransport(handler),
    )
    r = await f.fetch("http://example.com/")
    assert r.truncated and len(r.text) == 10_000


async def test_resolution_failures_and_timeouts_are_clean_refusals():
    with pytest.raises(FetchBlocked, match="resolved"):
        await SafeFetcher(resolver=resolver({}), transport=ok_transport()).fetch(
            "http://nowhere.example/"
        )

    async def slow_handler_body():
        await asyncio.sleep(5)
        yield b"late"

    def handler(request):
        return httpx.Response(
            200, content=slow_handler_body(), headers={"content-type": "text/plain"}
        )

    f = SafeFetcher(
        FetchPolicy(timeout_s=0.2),
        resolver=resolver({"example.com": [PUBLIC]}),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(FetchBlocked, match="timed out"):
        await f.fetch("http://example.com/")


async def test_the_tool_wrapper_returns_text_and_caps_it():
    t = http_fetch_tool(
        fetcher=SafeFetcher(
            resolver=resolver({"example.com": [PUBLIC]}),
            transport=ok_transport(body="y" * 500, ctype="text/plain"),
        ),
        max_chars=100,
    )
    out = await t.invoke({"url": "http://example.com/"}, None)  # type: ignore[arg-type]
    assert out["status"] == 200 and len(out["text"]) == 100 and out["truncated"] is True
    assert t.spec().name == "http_fetch"


# -- real sockets: the pin must hold on an actual connection, including TLS --------------------------
class _H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect-private":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()
            return
        body = f"host={self.headers['Host']}".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def local_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


async def test_a_real_local_server_is_unreachable_by_default_and_reachable_when_you_say_so(
    local_server,
):
    port = local_server.server_address[1]
    url = f"http://pinned.test:{port}/"
    dns = resolver({"pinned.test": ["127.0.0.1"]})
    with pytest.raises(FetchBlocked, match="non-public"):
        await SafeFetcher(FetchPolicy(allowed_ports=(port,)), resolver=dns).fetch(url)
    r = await SafeFetcher(
        FetchPolicy(allowed_ports=(port,), allow_private_networks=True), resolver=dns
    ).fetch(url)
    assert (
        r.text == f"host=pinned.test:{port}"
    )  # connected to 127.0.0.1, spoke as the original name
    with pytest.raises(
        FetchBlocked, match="non-public"
    ):  # the redirect target is checked even when the first hop was allowed to be private?
        await SafeFetcher(
            FetchPolicy(allowed_ports=(port, 80)),
            resolver=resolver(
                {"pinned.test": ["127.0.0.1"], "169.254.169.254": ["169.254.169.254"]}
            ),
        ).fetch(f"http://pinned.test:{port}/redirect-private")


def _make_cert(tmp_path, name):
    key, crt = tmp_path / f"{name}.key", tmp_path / f"{name}.crt"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(crt),
            "-subj",
            f"/CN={name}",
            "-addext",
            f"subjectAltName=DNS:{name}",
            "-days",
            "1",
        ],
        check=True,
        capture_output=True,
    )
    return key, crt


@pytest.fixture
def tls_server(tmp_path):
    try:
        key, crt = _make_cert(tmp_path, "pinned.test")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("openssl not available")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(crt), str(key))
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, crt
    srv.shutdown()


async def test_tls_is_verified_against_the_original_name_even_though_we_connect_to_the_pinned_address(
    tls_server,
):
    srv, crt = tls_server
    port = srv.server_address[1]
    trust = ssl.create_default_context(cafile=str(crt))
    pol = FetchPolicy(allowed_ports=(port,), allow_private_networks=True)
    ok = await SafeFetcher(
        pol, resolver=resolver({"pinned.test": ["127.0.0.1"]}), verify=trust
    ).fetch(f"https://pinned.test:{port}/")
    assert ok.status == 200 and ok.text == f"host=pinned.test:{port}"
    # the certificate is for pinned.test: a different name pointed at the same address must fail verification (SNI and hostname check use the NAME, not the IP)
    with pytest.raises(FetchBlocked, match="request failed"):
        await SafeFetcher(
            pol, resolver=resolver({"other.test": ["127.0.0.1"]}), verify=trust
        ).fetch(f"https://other.test:{port}/")
    with pytest.raises(
        FetchBlocked, match="request failed"
    ):  # and with the default trust store the self-signed certificate is rejected
        await SafeFetcher(pol, resolver=resolver({"pinned.test": ["127.0.0.1"]})).fetch(
            f"https://pinned.test:{port}/"
        )


# -- read_file / list_files ---------------------------------------------------------------------
@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "project"
    (root / "sub").mkdir(parents=True)
    (root / "readme.md").write_text("line1\nline2\nline3\nline4\n")
    (root / "sub" / "a.txt").write_text("alpha")
    (root / ".env").write_text("SECRET=1")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("x")
    (root / "server.pem").write_text("KEY")
    (root / "bin.dat").write_bytes(b"ab\x00cd")
    (root / "big.txt").write_text("z" * 5000)
    outside = tmp_path / "outside.txt"
    outside.write_text("top secret")
    (root / "link.txt").symlink_to(outside)
    (root / "dirlink").symlink_to(tmp_path)
    return root, tmp_path


def test_files_are_confined_to_the_root(tree):
    root, tmp = tree
    fa = FileAccess(root)
    assert (
        fa.read("readme.md")["text"].startswith("line1") and fa.read("sub/a.txt")["text"] == "alpha"
    )
    for bad in (
        "../outside.txt",
        "/etc/passwd",
        str(tmp / "outside.txt"),
        "link.txt",
        "dirlink/outside.txt",
        "sub/../../outside.txt",
        ".env",
        ".git/config",
        "server.pem",
        "~/.ssh/id_rsa",
        "bad\x00name",
    ):
        with pytest.raises(ToolError):
            fa.read(bad)


def test_reads_are_bounded_and_ranged(tree):
    root, _ = tree
    fa = FileAccess(root, max_bytes=100)
    big = fa.read("big.txt")
    assert big["truncated"] and len(big["text"]) == 100
    with pytest.raises(ToolError, match="binary"):
        FileAccess(root).read("bin.dat")
    part = FileAccess(root).read("readme.md", offset=1, limit=2)
    assert part["text"] == "line2\nline3" and part["total_lines"] == 4 and part["truncated"] is True
    with pytest.raises(ToolError, match="not a file"):
        FileAccess(root).read("sub")


def test_listing_hides_what_reading_refuses(tree):
    root, _ = tree
    names = FileAccess(root).list()["entries"]
    assert "readme.md" in names and "sub/" in names
    assert not {".env", ".git/", "server.pem", "link.txt", "dirlink/"} & set(names)
    assert FileAccess(root).list("sub", pattern="*.txt")["entries"] == ["a.txt"]
    with pytest.raises(ToolError):
        FileAccess(root).list("..")


async def test_file_tools_wrap_access(tree):
    root, _ = tree
    assert (read_file_tool(root).spec().name, list_files_tool(root).spec().name) == (
        "read_file",
        "list_files",
    )
    out = read_file_tool(root).fn("sub/a.txt")  # type: ignore[attr-defined]
    assert out["text"] == "alpha"


# -- search providers ---------------------------------------------------------------------------
async def test_searxng_provider_parses_results_and_drops_unsafe_urls():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "A", "url": "https://a.example/", "content": "x" * 900},
                    {"title": "evil", "url": "javascript:alert(1)", "content": "c"},
                    {"title": "B", "url": "http://b.example/", "content": "snippet"},
                ]
            },
        )

    p = SearXNGSearch(
        "http://searx.internal/", categories="general", transport=httpx.MockTransport(handler)
    )
    got = await p.search("hello world", k=5)
    assert [r.url for r in got] == ["https://a.example/", "http://b.example/"] and len(
        got[0].snippet
    ) == 500
    q = dict(seen[0].url.params)
    assert q["q"] == "hello world" and q["format"] == "json" and q["categories"] == "general"
    with pytest.raises(ToolError, match="format=json"):
        await SearXNGSearch(
            "http://x", transport=httpx.MockTransport(lambda r: httpx.Response(403))
        ).search("q")


async def test_brave_provider_request_and_response():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "T",
                            "url": "https://t.example/",
                            "description": "D",
                            "extra_snippets": ["e"],
                        }
                    ]
                },
                "query": {"original": "q"},
            },
        )

    p = BraveSearch("brave-key", transport=httpx.MockTransport(handler))
    got = await p.search("python", k=3)
    assert [(r.title, r.url, r.snippet) for r in got] == [("T", "https://t.example/", "D")]
    req = seen[0]
    assert req.headers["x-subscription-token"] == "brave-key" and str(req.url).startswith(
        "https://api.search.brave.com/res/v1/web/search?"
    )
    assert dict(req.url.params) == {"q": "python", "count": "3", "safesearch": "moderate"}
    for status, text in ((401, "API key"), (429, "rate limit")):
        with pytest.raises(ToolError, match=text):
            await BraveSearch(
                "k", transport=httpx.MockTransport(lambda r, s=status: httpx.Response(s))
            ).search("q")


async def test_the_search_tool_never_exposes_the_provider_key():
    p = BraveSearch(
        "super-secret-key",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"web": {"results": []}})),
    )
    t = web_search_tool(p)
    assert "super-secret-key" not in json.dumps(t.spec().model_dump())
    assert await t.invoke({"query": "x"}, None) == []  # type: ignore[arg-type]


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


async def test_searxng_provider_against_a_real_instance():
    """Needs a SearXNG with `json` in search.formats (CI starts one in Docker); AAA_TEST_SEARXNG_URL points at it."""
    import os

    url = os.environ.get("AAA_TEST_SEARXNG_URL")
    if not url:
        pytest.skip("set AAA_TEST_SEARXNG_URL to a SearXNG instance")
    p = SearXNGSearch(url)
    got = await p.search("python programming language", k=3)
    assert 1 <= len(got) <= 3 and all(
        r.url.startswith(("http://", "https://")) and r.title for r in got
    )
    await p.aclose()
    # an instance that does not enable JSON answers 403, and the provider says how to fix it
    with pytest.raises(ToolError):
        await SearXNGSearch("http://127.0.0.1:9").search("x")
