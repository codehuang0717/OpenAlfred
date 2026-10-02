import re
import json
from langchain.tools import ToolRuntime, tool
from rag.retriever import search as rag_search
from db.rag import get_documents as db_list_documents
from db.rag import get_image_by_id, get_document_by_id
from logic.prompts import RAG_SEARCH_RESULT_HEADER
import logging
from utils.auth_utils import require_explicit_user_id, require_runtime_user_id
from services.tool_observations import observed, failed, fields, action, clipped

logger = logging.getLogger("rag-tools")

_IMG_PLACEHOLDER = re.compile(r'\{"_img"\s*:\s*\{[^}]*"i"\s*:\s*(\d+)[^}]*"d"\s*:\s*"([^"]*)"\s*\}\s*\}')


async def _resolve_content(text: str, user_id: str) -> str:
    """Resolve {"_img":{...}} placeholders back to markdown images."""
    user_id = require_explicit_user_id(user_id)
    matches = list(_IMG_PLACEHOLDER.finditer(text))
    if not matches:
        return text

    result = text
    for m in reversed(matches):
        img_id = int(m.group(1))
        desc = m.group(2)
        img = await get_image_by_id(img_id, user_id=user_id)
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

    try:
        results = rag_search(user_id, query, top_k)
    except Exception as e:
        logger.warning("search_knowledge error: %s", e)
        return failed(f"Knowledge search failed: {e}", "知识库检索失败")

    if not results:
        return observed(
            "No relevant documents found in your knowledge base.",
            "未找到相关知识片段",
            outcome="empty",
            actions=[action("panel", "查看知识库", "knowledge")],
        )

    # Cache document lookups for date info
    doc_cache: dict[str, str] = {}

    lines = []
    for i, r in enumerate(results, 1):
        doc_id = r.get("document_id", "")
        if doc_id and doc_id not in doc_cache:
            doc = await get_document_by_id(doc_id, user_id=user_id)
            if doc and doc.get("created_at"):
                doc_cache[doc_id] = doc["created_at"][:10]  # YYYY-MM-DD
            else:
                doc_cache[doc_id] = ""

        heading = f" (## {r['heading']})" if r.get("heading") else ""
        ingested = f" [摄入: {doc_cache[doc_id]}]" if doc_cache.get(doc_id) else ""
        content = await _resolve_content(r["content"], user_id)
        block = f"[{i}] Source: {r['filename']}{heading}{ingested} (relevance: {r['score']})\n{content}"
        lines.append(block)

    results_text = "\n\n---\n\n".join(lines)
    return observed(
        RAG_SEARCH_RESULT_HEADER.format(results=results_text),
        f"找到 {len(results)} 个片段，来自 {len({r.get('document_id') or r['filename'] for r in results})} 份文档",
        details=[
            {
                "label": f"来源 {i}",
                "value": f"{r['filename']} · {r.get('heading') or '无章节'}\n{clipped(r['content'], 2000)}",
            }
            for i, r in enumerate(results[:20], 1)
        ],
        actions=[action("panel", "查看知识库", "knowledge")],
    )


@tool
async def list_knowledge(runtime: ToolRuntime, query: str = "") -> str:
    """List all documents in the user's personal knowledge base.
    Use this to show the user what documents they have uploaded."""
    user_id = _get_rag_user_id(runtime)

    try:
        docs = await db_list_documents(user_id)
    except Exception as e:
        logger.warning("list_knowledge error: %s", e)
        return failed(f"Failed to list documents: {e}", "读取知识库文档列表失败")

    if not docs:
        return observed(
            "Your knowledge base is empty. You can upload documents to add knowledge.",
            "知识库暂无文档",
            outcome="empty",
            actions=[action("panel", "打开知识库", "knowledge")],
        )

    lines = [f"Your knowledge base has {len(docs)} document(s):"]
    for d in docs:
        lines.append(f"- {d['title'] or d['filename']} ({d['file_type']}, {d['chunk_count']} chunks, added {d['created_at'][:10]})")
    return observed(
        "\n".join(lines),
        f"知识库共有 {len(docs)} 份文档",
        details=fields(
            文档="\n".join(
                f"{d['title'] or d['filename']} · {d['file_type']} · {d['created_at'][:10]}"
                for d in docs
            )
        ),
        actions=[action("panel", "查看知识库", "knowledge")],
    )


def _get_rag_user_id(runtime: ToolRuntime) -> str:
    """Get a verified user_id from LangGraph request metadata."""
    return require_runtime_user_id(runtime)


rag_tools = [search_knowledge, list_knowledge]
