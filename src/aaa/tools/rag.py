from __future__ import annotations

from aaa.data.index import Index
from aaa.tools.tool import Tool


def search_tool(
    index: Index, *, name: str = "search_documents", k: int = 5, max_chars: int = 800
) -> Tool:
    """Expose any Index (keyword/vector/hybrid) as a retrieval tool for ToolLoop/ReAct."""

    async def search(query: str) -> list[dict]:
        hits = await index.search(query, k)
        return [
            {
                "id": h.doc.id,
                "score": round(h.score, 4),
                "text": h.doc.text[:max_chars],
                "meta": h.doc.meta,
            }
            for h in hits
        ]

    return Tool(
        search,
        name=name,
        description="Search the knowledge base. Returns the most relevant passages with ids; cite ids in answers.",
    )
