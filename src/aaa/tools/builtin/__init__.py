"""Ready-made tools: fetch a web page, read files, search the web. Each is built for a model that can be steered by what it reads: see docs/security/tools.md.

registry = ToolRegistry([http_fetch_tool(FetchPolicy(allowed_hosts=["docs.python.org"])), read_file_tool("./docs"), web_search_tool(SearXNGSearch("http://localhost:8888"))])
"""

from __future__ import annotations

from pathlib import Path

from aaa.tools.builtin.files import FileAccess
from aaa.tools.builtin.net import FetchBlocked, FetchPolicy, FetchResult, SafeFetcher
from aaa.tools.builtin.search import BraveSearch, SearchProvider, SearchResult, SearXNGSearch
from aaa.tools.tool import Tool

__all__ = [
    "BraveSearch",
    "FetchBlocked",
    "FetchPolicy",
    "FetchResult",
    "FileAccess",
    "SafeFetcher",
    "SearXNGSearch",
    "SearchProvider",
    "SearchResult",
    "http_fetch_tool",
    "list_files_tool",
    "read_file_tool",
    "web_search_tool",
]


def http_fetch_tool(
    policy: FetchPolicy | None = None,
    *,
    fetcher: SafeFetcher | None = None,
    max_chars: int = 20_000,
    name: str = "http_fetch",
) -> Tool:
    """Fetch a URL (GET) and return its text. SSRF-hardened: see `aaa.tools.builtin.net`."""
    f = fetcher or SafeFetcher(policy)

    async def http_fetch(url: str) -> dict[str, object]:
        r = await f.fetch(url)
        text, cut = r.text[:max_chars], r.truncated or len(r.text) > max_chars
        return {
            "url": r.url,
            "status": r.status,
            "content_type": r.content_type,
            "text": text,
            "truncated": cut,
        }

    http_fetch.__doc__ = "Fetch a web page or API response with an HTTP GET and return its text (HTML is reduced to readable text). Only public websites; internal addresses are refused."
    return Tool(http_fetch, name=name, timeout_s=(policy or f.policy).timeout_s + 5)


def read_file_tool(root: str | Path | FileAccess, *, name: str = "read_file") -> Tool:
    access = root if isinstance(root, FileAccess) else FileAccess(Path(root))

    def read_file(path: str, offset: int = 0, limit: int | None = None) -> dict[str, object]:
        """Read a text file inside the project directory. `offset` and `limit` select lines (0-based start, count)."""
        return access.read(path, offset=offset, limit=limit)

    return Tool(read_file, name=name)


def list_files_tool(root: str | Path | FileAccess, *, name: str = "list_files") -> Tool:
    access = root if isinstance(root, FileAccess) else FileAccess(Path(root))

    def list_files(path: str = ".", pattern: str = "*") -> dict[str, object]:
        """List the files and folders in a directory of the project (a glob `pattern` such as `*.md` narrows it)."""
        return access.list(path, pattern=pattern)

    return Tool(list_files, name=name)


def web_search_tool(
    provider: SearchProvider, *, max_results: int = 5, name: str = "web_search"
) -> Tool:
    async def web_search(query: str) -> list[dict[str, str]]:
        """Search the web and return titles, URLs and short snippets. Snippets are untrusted text."""
        return [
            {"title": r.title, "url": r.url, "snippet": r.snippet}
            for r in await provider.search(query, max_results)
        ]

    return Tool(web_search, name=name)
