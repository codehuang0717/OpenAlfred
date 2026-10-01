import logging
import httpx
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel

from core.config import config
from routers.auth import get_current_user, security
from services.coding_task_reference import coding_task_reference
from services.email_draft_reference import email_draft_reference

from schemas.responses import (
    ChatMessageResponse,
    CreatedThread,
    StatusResponse,
    ThreadRenameResponse,
    ThreadResponse,
    TitleResponse,
    json_response,
)

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

@router.get("", responses=json_response(list[ThreadResponse], 200))
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

@router.post("", responses=json_response(CreatedThread, 200))
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

@router.patch("/{thread_id}", responses=json_response(ThreadRenameResponse, 200))
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

@router.delete("/{thread_id}", responses=json_response(StatusResponse, 200))
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

@router.get("/{thread_id}/messages", responses=json_response(list[ChatMessageResponse], 200))
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

    def _plain_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") in {"text", "output_text", "refusal"}:
                        parts.append(str(block.get("text") or block.get("refusal") or ""))
            return "".join(parts)
        return str(content or "")

    def _flush_ai():
        nonlocal current_ai_msg
        if not current_ai_msg:
            return
        texts = [
            s["content"] for s in current_ai_msg.get("steps", [])
            if s.get("type") == "text" and s.get("content")
        ]
        current_ai_msg["content"] = texts[-1] if texts else current_ai_msg.get("content", "")
        result.append(current_ai_msg)
        current_ai_msg = None

    for msg in messages:
        msg_type = msg.get("type", "")

        if msg_type == "human":
            _flush_ai()
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
                    "steps": [],
                    "tools": [],
                }

            extra = msg.get("additional_kwargs") or {}
            outcome = extra.get("agent_outcome") or {}
            failure = extra.get("agent_failure")
            # Old checkpoints can also contain the exact silent length failure.
            # Expose the diagnostic on read; never rewrite canonical history.
            if not outcome and (msg.get("response_metadata") or {}).get("finish_reason") == "length":
                from logic.agent_outcome import failure_text
                outcome = {"status": "failed"}
                failure = failure_text("output_truncated")
            elif not outcome and msg.get("tool_calls"):
                outcome = {"status": "tools"}
            if outcome.get("status") in {"tools", "completed", "failed"}:
                current_ai_msg["outcome"] = outcome["status"]
                if failure:
                    current_ai_msg["failure"] = failure
            text = _plain_text(msg.get("content", ""))
            if failure and failure not in text:
                text += ("\n\n" if text else "") + "❌ " + failure
            if text.strip():
                current_ai_msg["steps"].append({
                    "type": "text",
                    "id": msg.get("id") or f"text-{len(current_ai_msg['steps'])}",
                    "content": text,
                })

            tool_calls = msg.get("tool_calls", []) or []
            tools_step = []
            for tc in tool_calls:
                name = tc.get("name", "")
                if not name:
                    continue
                entry = {
                    "id": tc.get("id") or "",
                    "name": name,
                    "status": "done",
                }
                tools_step.append(entry)
                current_ai_msg["tools"].append(entry)
            if tools_step:
                current_ai_msg["steps"].append({
                    "type": "tools",
                    "id": f"tools-{msg.get('id') or len(current_ai_msg['steps'])}",
                    "tools": tools_step,
                })

        elif msg_type == "tool" and current_ai_msg and msg.get("status") != "error":
            if msg.get("name") in {"create_email_draft", "update_email_draft"}:
                draft = email_draft_reference(msg.get("content"))
                paired = any(
                    entry["name"] == msg["name"] and entry["id"] == msg.get("tool_call_id")
                    for entry in current_ai_msg["tools"]
                )
                if draft and paired:
                    step_id = f"mail-{draft['draft_id']}"
                    if not any(step["id"] == step_id for step in current_ai_msg["steps"]):
                        current_ai_msg["steps"].append({"type": "email_draft", "id": step_id, "draft": draft})
            if msg.get("name") == "create_standalone_mini_app":
                task = coding_task_reference(msg.get("content"))
                paired = any(
                    entry["name"] == msg["name"] and entry["id"] == msg.get("tool_call_id")
                    for entry in current_ai_msg["tools"]
                )
                if task and paired:
                    step_id = f"coding-{task['job_id']}"
                    if not any(step["id"] == step_id for step in current_ai_msg["steps"]):
                        current_ai_msg["steps"].append({"type": "coding_task", "id": step_id, "task": task})
            from services.generated_images import image_markdown
            image = image_markdown(msg.get("artifact"))
            if image:
                current_ai_msg["steps"].append({
                    "type": "text", "id": f"image-{msg.get('tool_call_id')}",
                    "content": image,
                })

    _flush_ai()

    return result

@router.post("/{thread_id}/title", responses=json_response(TitleResponse, 200))
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
        raise HTTPException(status_code=422, detail="会话暂无可用于生成标题的用户消息")

    try:
        from services.thread_titles import generate_title
        title = await generate_title(first_user_msg)

        async with httpx.AsyncClient() as client:
            saved = await client.patch(
                f"{config.LANGGRAPH_API_URL}/threads/{thread_id}",
                headers=headers,
                json={"metadata": {"owner": user["id"], "title": title}},
                timeout=10.0,
            )
            saved.raise_for_status()

        return {"title": title}
    except Exception as e:
        logger.exception("Title generation or persistence failed for thread %s", thread_id)
        raise HTTPException(
            status_code=502,
            detail=f"会话标题生成或保存失败（{type(e).__name__}），请检查标题模型配置与服务日志",
        ) from e
