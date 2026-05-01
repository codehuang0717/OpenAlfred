import logging
import httpx
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel

from config import config
from context_manager import ContextManager
ctx_manager = ContextManager()
from routers.auth import get_current_user, security

router = APIRouter(prefix="/api/threads", tags=["threads"])
logger = logging.getLogger("threads-router")

class ThreadRenameRequest(BaseModel):
    title: str

def _lg_headers(token: str) -> dict:
    """Build headers for proxied requests to LangGraph Server."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

@router.get("")
async def list_threads(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user: dict = Depends(get_current_user),
):
    """List all conversation threads owned by the current user."""
    headers = _lg_headers(credentials.credentials)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{config.LANGGRAPH_API_URL}/threads/search",
            headers=headers,
            json={
                "metadata": {"owner": user["id"]},
                "limit": 100,
            },
            timeout=10.0,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Failed to fetch threads")
        threads = resp.json()

    # Transform to a simplified format for the frontend
    result = []
    for t in threads:
        metadata = t.get("metadata", {})
        if metadata.get("type") == "call":
            continue  # Hide calls from regular chat list
        result.append({
            "thread_id": t["thread_id"],
            "title": metadata.get("title", "新对话"),
            "updated_at": t.get("updated_at", t.get("created_at", "")),
            "created_at": t.get("created_at", ""),
        })

    # Sort by updated_at descending
    result.sort(key=lambda x: x["updated_at"], reverse=True)
    return result

@router.post("")
async def create_thread(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user: dict = Depends(get_current_user),
):
    """Create a new conversation thread."""
    headers = _lg_headers(credentials.credentials)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{config.LANGGRAPH_API_URL}/threads",
            headers=headers,
            json={
                "metadata": {
                    "owner": user["id"],
                    "title": "新对话",
                },
            },
            timeout=10.0,
        )
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=resp.status_code, detail="Failed to create thread")
        thread = resp.json()

    return {
        "thread_id": thread["thread_id"],
        "title": "新对话",
        "created_at": thread.get("created_at", ""),
    }

@router.patch("/{thread_id}")
async def rename_thread(
    thread_id: str,
    req: ThreadRenameRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user: dict = Depends(get_current_user),
):
    """Rename a conversation thread."""
    headers = _lg_headers(credentials.credentials)
    async with httpx.AsyncClient() as client:
        resp = await client.patch(
            f"{config.LANGGRAPH_API_URL}/threads/{thread_id}",
            headers=headers,
            json={
                "metadata": {
                    "owner": user["id"],
                    "title": req.title,
                },
            },
            timeout=10.0,
        )
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="Thread not found")
        if resp.status_code not in (200, 204):
            raise HTTPException(status_code=resp.status_code, detail="Failed to rename thread")

    return {"status": "updated", "title": req.title}

@router.delete("/{thread_id}")
async def delete_thread(
    thread_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user: dict = Depends(get_current_user),
):
    """Delete a conversation thread."""
    headers = _lg_headers(credentials.credentials)
    async with httpx.AsyncClient() as client:
        resp = await client.delete(
            f"{config.LANGGRAPH_API_URL}/threads/{thread_id}",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="Thread not found")
        if resp.status_code not in (200, 204):
            raise HTTPException(status_code=resp.status_code, detail="Failed to delete thread")

    return {"status": "deleted"}

@router.get("/{thread_id}/messages")
async def get_thread_messages(
    thread_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user: dict = Depends(get_current_user),
):
    """Get the message history for a specific thread."""
    headers = _lg_headers(credentials.credentials)
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{config.LANGGRAPH_API_URL}/threads/{thread_id}/state",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="Thread not found")
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Failed to get messages")

        state = resp.json()

    values = state.get("values", {})
    messages = values.get("messages", [])

    result = []
    current_ai_msg = None

    for msg in messages:
        msg_type = msg.get("type", "")

        if msg_type == "human":
            if current_ai_msg:
                result.append(current_ai_msg)
                current_ai_msg = None

            result.append({
                "id": msg.get("id", ""),
                "role": "user",
                "content": msg.get("content", ""),
            })

        elif msg_type == "ai":
            if not current_ai_msg:
                current_ai_msg = {
                    "id": msg.get("id", ""),
                    "role": "assistant",
                    "content": "",
                    "tools": [],
                }

            content = msg.get("content", "")
            if isinstance(content, list):
                text = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
                content = text

            if content and content.strip():
                if current_ai_msg["content"]:
                    current_ai_msg["content"] += "\n" + content
                else:
                    current_ai_msg["content"] = content

            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                name = tc.get("name", "")
                if name:
                    current_ai_msg["tools"].append({
                        "name": name,
                        "status": "done"
                    })

    if current_ai_msg:
        result.append(current_ai_msg)

    return result

@router.post("/{thread_id}/title")
async def generate_thread_title(
    thread_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user: dict = Depends(get_current_user),
):
    """Auto-generate a title for the thread based on the first user message."""
    headers = _lg_headers(credentials.credentials)

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{config.LANGGRAPH_API_URL}/threads/{thread_id}/state",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Failed to get thread state")

        state = resp.json()

    values = state.get("values", {})
    messages = values.get("messages", [])

    first_user_msg = None
    for msg in messages:
        if msg.get("type") == "human":
            first_user_msg = msg.get("content", "")
            break

    if not first_user_msg:
        return {"title": "新对话"}

    try:
        from llm import get_model
        from langchain_core.messages import HumanMessage

        title_prompt = ctx_manager.build_title_prompt(first_user_msg)
        llm = get_model("gpt-cloud")
        result = await llm.ainvoke([HumanMessage(content=title_prompt)])
        title = result.content.strip().strip('"\'')[:20]

        async with httpx.AsyncClient() as client:
            await client.patch(
                f"{config.LANGGRAPH_API_URL}/threads/{thread_id}",
                headers=headers,
                json={"metadata": {"owner": user["id"], "title": title}},
                timeout=10.0,
            )

        return {"title": title}
    except Exception as e:
        logger.error(f"Title generation failed: {e}")
        return {"title": "新对话"}
