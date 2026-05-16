import re
import json
from langchain.tools import tool
from rag.retriever import search as rag_search
from db.rag import get_documents as db_list_documents
from db.rag import get_image_by_id
from logic.prompts import RAG_SEARCH_RESULT_HEADER
import logging

logger = logging.getLogger("rag-tools")

_IMG_PLACEHOLDER = re.compile(r'\{"_img"\s*:\s*\{[^}]*"i"\s*:\s*(\d+)[^}]*"d"\s*:\s*"([^"]*)"\s*\}\s*\}')


async def _resolve_content(text: str) -> str:
    """Resolve {"_img":{...}} placeholders back to markdown images."""
    matches = list(_IMG_PLACEHOLDER.finditer(text))
    if not matches:
        return text

    result = text
    for m in reversed(matches):
        img_id = int(m.group(1))
        desc = m.group(2)
        img = await get_image_by_id(img_id)
        if img:
            alt = img.get("alt", "")
            url = img.get("url", "")
            replacement = f"![{alt}]({url})"
            if desc:
                replacement += f"\n> {desc}"
        else:
            replacement = f"[图片描述: {desc}]"
        result = result[:m.start()] + replacement + result[m.end():]

    return result


@tool
async def search_knowledge(query: str, top_k: int = 5) -> str:
    """Search the user's personal knowledge base for documents relevant to the query.
    Use this when the user asks about information that might be in their uploaded documents.
    Returns the most relevant text chunks with source filenames and images."""
    from tools.rag import _current_user_id
    user_id = _current_user_id.get()
    if not user_id:
        return "Error: No user context available for knowledge search."

    try:
        results = rag_search(user_id, query, top_k)
    except Exception as e:
        logger.warning("search_knowledge error: %s", e)
        return f"Knowledge search failed: {e}"

    if not results:
        return "No relevant documents found in your knowledge base."

    lines = []
    for i, r in enumerate(results, 1):
        heading = f" (## {r['heading']})" if r.get("heading") else ""
        content = await _resolve_content(r["content"])
        block = f"[{i}] Source: {r['filename']}{heading} (relevance: {r['score']})\n{content}"
        lines.append(block)

    results_text = "\n\n---\n\n".join(lines)
    return RAG_SEARCH_RESULT_HEADER.format(results=results_text)


@tool
async def list_knowledge(query: str = "") -> str:
    """List all documents in the user's personal knowledge base.
    Use this to show the user what documents they have uploaded."""
    from tools.rag import _current_user_id
    user_id = _current_user_id.get()
    if not user_id:
        return "Error: No user context available."

    try:
        docs = await db_list_documents(user_id)
    except Exception as e:
        logger.warning("list_knowledge error: %s", e)
        return f"Failed to list documents: {e}"

    if not docs:
        return "Your knowledge base is empty. You can upload documents to add knowledge."

    lines = [f"Your knowledge base has {len(docs)} document(s):"]
    for d in docs:
        lines.append(f"- {d['title'] or d['filename']} ({d['file_type']}, {d['chunk_count']} chunks, added {d['created_at'][:10]})")
    return "\n".join(lines)


# Context variable for passing user_id to tools
import contextvars
_current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar("rag_user_id", default="")


rag_tools = [search_knowledge, list_knowledge]
