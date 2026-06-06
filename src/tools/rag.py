import re
import json
from langchain.tools import ToolRuntime, tool
from rag.retriever import search as rag_search
from db.rag import get_documents as db_list_documents
from db.rag import get_image_by_id, get_document_by_id
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
async def search_knowledge(runtime: ToolRuntime, query: str, top_k: int = 5) -> str:
    """Search the user's personal knowledge base for documents relevant to the query.
    Use this when the user asks about information that might be in their uploaded documents.
    Returns the most relevant text chunks with source filenames and images."""
    user_id = _get_rag_user_id(runtime)
    if not user_id:
        return "Error: No user context available for knowledge search."

    try:
        results = rag_search(user_id, query, top_k)
    except Exception as e:
        logger.warning("search_knowledge error: %s", e)
        return f"Knowledge search failed: {e}"

    if not results:
        return "No relevant documents found in your knowledge base."

    # Cache document lookups for date info
    doc_cache: dict[str, str] = {}

    lines = []
    for i, r in enumerate(results, 1):
        doc_id = r.get("document_id", "")
        if doc_id and doc_id not in doc_cache:
            doc = await get_document_by_id(doc_id)
            if doc and doc.get("created_at"):
                doc_cache[doc_id] = doc["created_at"][:10]  # YYYY-MM-DD
            else:
                doc_cache[doc_id] = ""

        heading = f" (## {r['heading']})" if r.get("heading") else ""
        ingested = f" [摄入: {doc_cache[doc_id]}]" if doc_cache.get(doc_id) else ""
        content = await _resolve_content(r["content"])
        block = f"[{i}] Source: {r['filename']}{heading}{ingested} (relevance: {r['score']})\n{content}"
        lines.append(block)

    results_text = "\n\n---\n\n".join(lines)
    return RAG_SEARCH_RESULT_HEADER.format(results=results_text)


@tool
async def list_knowledge(runtime: ToolRuntime, query: str = "") -> str:
    """List all documents in the user's personal knowledge base.
    Use this to show the user what documents they have uploaded."""
    user_id = _get_rag_user_id(runtime)
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


def _get_rag_user_id(runtime: ToolRuntime) -> str:
    """Get user_id from LangGraph auth/runtime context."""
    if hasattr(runtime, "config") and runtime.config:
        conf = runtime.config.get("configurable", {})
        auth_user = conf.get("langgraph_auth_user", {})
        if isinstance(auth_user, dict) and "identity" in auth_user:
            return auth_user["identity"]

        metadata = runtime.config.get("metadata", {})
        if "owner" in metadata:
            return metadata["owner"]
        if "owner" in conf:
            return conf["owner"]
        if "thread_owner" in conf:
            return conf["thread_owner"]

    if hasattr(runtime, "state") and runtime.state:
        if isinstance(runtime.state, dict):
            return runtime.state.get("user_id", "")
        return getattr(runtime.state, "user_id", "")
    return ""


rag_tools = [search_knowledge, list_knowledge]
