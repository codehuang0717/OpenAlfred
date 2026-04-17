"""
FastAPI application — Business API layer for OpenAlfred.

Provides:
- User authentication (register/login/refresh/me)
- Thread management (CRUD, proxied through LangGraph Server)
- Auto title generation for new conversations
- Model selection and Ollama status
- Todos and reminders CRUD
"""

import uuid
import logging
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
import httpx
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import asyncio
from pydantic import BaseModel
from typing import Optional

from database import (
    get_all_todos,
    update_todo as db_update_todo,
    delete_todo as db_delete_todo,
    init_db,
    get_all_reminders,
    update_reminder as db_update_reminder,
    delete_reminder as db_delete_reminder,
    get_setting,
    set_setting,
    create_user,
    get_user_by_username,
    get_user_by_id,
    update_user_last_login,
)
from tools.reminder import check_and_send_pending_reminders
from config import config
from context_manager import ContextManager

logger = logging.getLogger("api")
security = HTTPBearer()
ctx_manager = ContextManager()


# ─── Pydantic Models ──────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = ""

class LoginRequest(BaseModel):
    username: str
    password: str

class ThreadRenameRequest(BaseModel):
    title: str

class ModelSelectionRequest(BaseModel):
    model_selection: str = "gpt-cloud"

class TodoUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    emoji: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None
    expected_completion_at: Optional[str] = None

class ReminderUpdateRequest(BaseModel):
    scheduled_at: Optional[str] = None
    title: Optional[str] = None
    body: Optional[str] = None


# ─── JWT Helpers ───────────────────────────────────────────────────────────

def create_jwt_token(user_id: str, username: str) -> str:
    """Create a signed JWT token."""
    payload = {
        "sub": user_id,
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=config.JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def verify_jwt_token(token: str) -> dict:
    """Decode and verify a JWT token. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """FastAPI dependency: extract and verify the current user from JWT."""
    payload = verify_jwt_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ─── LangGraph Client Helper ──────────────────────────────────────────────

def _lg_headers(token: str) -> dict:
    """Build headers for proxied requests to LangGraph Server."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# ─── Lifespan ─────────────────────────────────────────────────────────────

async def run_scheduler():
    """Background task to check and send pending reminders."""
    print("Scheduler started!")
    while True:
        try:
            print("Checking pending reminders...")
            await check_and_send_pending_reminders()
        except Exception as e:
            print(f"Scheduler error: {e}")
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    asyncio.create_task(run_scheduler())
    print("Scheduler task created")

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    """Register a new user account."""
    if len(req.username) < 2:
        raise HTTPException(status_code=400, detail="Username must be at least 2 characters")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    existing = await get_user_by_username(req.username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")

    user_id = str(uuid.uuid4())
    password_hash = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    display_name = req.display_name or req.username

    user = await create_user(
        id=user_id,
        username=req.username,
        display_name=display_name,
        password_hash=password_hash,
    )

    token = create_jwt_token(user_id, req.username)

    return {
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
        },
    }


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """Authenticate and receive a JWT token."""
    user = await get_user_by_username(req.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if not bcrypt.checkpw(req.password.encode(), user["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    await update_user_last_login(user["id"])
    token = create_jwt_token(user["id"], user["username"])

    return {
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
        },
    }


@app.get("/api/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Retrieve the profile of the currently authenticated user."""
    return {
        "id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
        "created_at": user.get("created_at"),
    }


@app.post("/api/auth/refresh")
async def refresh_token(user: dict = Depends(get_current_user)):
    """Issue a fresh JWT token for the authenticated user."""
    token = create_jwt_token(user["id"], user["username"])
    return {"token": token}


# ═══════════════════════════════════════════════════════════════════════════
# THREAD MANAGEMENT (proxied to LangGraph Server)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/threads")
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

@app.get("/api/calls/threads")
async def list_calls(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user: dict = Depends(get_current_user),
):
    """List all voice call threads owned by the current user."""
    headers = _lg_headers(credentials.credentials)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{config.LANGGRAPH_API_URL}/threads/search",
            headers=headers,
            json={
                "metadata": {"owner": user["id"], "type": "call"},
                "limit": 100,
            },
            timeout=10.0,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Failed to fetch calls")
        threads = resp.json()

    result = []
    for t in threads:
        metadata = t.get("metadata", {})
        room_name = metadata.get("room_name", "")
        title = metadata.get("title", "")
        # Derive call direction from the original room name or title
        is_outbound = room_name.startswith("outbound-") or "外拨" in title
        direction = "outbound" if is_outbound else "inbound"
        result.append({
            "thread_id": t["thread_id"],
            "title": metadata.get("title", "语音通话记录"),
            "updated_at": t.get("updated_at", t.get("created_at", "")),
            "created_at": t.get("created_at", ""),
            "direction": direction,
            "room_name": room_name,
        })

    result.sort(key=lambda x: x["updated_at"], reverse=True)
    return result


@app.post("/api/threads")
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


@app.patch("/api/threads/{thread_id}")
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


@app.delete("/api/threads/{thread_id}")
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


@app.get("/api/threads/{thread_id}/messages")
async def get_thread_messages(
    thread_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user: dict = Depends(get_current_user),
):
    """Get the message history for a specific thread."""
    headers = _lg_headers(credentials.credentials)
    async with httpx.AsyncClient() as client:
        # Get thread state which contains messages
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

    # Extract messages from state values
    values = state.get("values", {})
    messages = values.get("messages", [])

    # Transform to simplified format, grouping tool calls and AI messages
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

            # Extract content
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

            # Extract tool calls
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                name = tc.get("name", "")
                if name:
                    current_ai_msg["tools"].append({
                        "name": name,
                        "status": "done" # Render as done since it's from history
                    })

    if current_ai_msg:
        result.append(current_ai_msg)

    return result


@app.post("/api/threads/{thread_id}/title")
async def generate_thread_title(
    thread_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user: dict = Depends(get_current_user),
):
    """Auto-generate a title for the thread based on the first user message."""
    headers = _lg_headers(credentials.credentials)

    # Get the first message
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

    # Find first user message
    first_user_msg = None
    for msg in messages:
        if msg.get("type") == "human":
            first_user_msg = msg.get("content", "")
            break

    if not first_user_msg:
        return {"title": "新对话"}

    # Generate title using LLM
    try:
        from llm import get_model
        from langchain_core.messages import HumanMessage

        title_prompt = ctx_manager.build_title_prompt(first_user_msg)
        llm = get_model("gpt-cloud")
        result = await llm.ainvoke([HumanMessage(content=title_prompt)])
        title = result.content.strip().strip('"\'')[:20]

        # Update thread metadata with the title
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


# ═══════════════════════════════════════════════════════════════════════════
# EXISTING ENDPOINTS (todos, reminders, models)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/todos")
async def get_todos(user: dict = Depends(get_current_user)):
    """Get all todos for the current user."""
    todos = await get_all_todos(user_id=user["id"])
    return todos


@app.patch("/api/todos/{todo_id}")
async def update_todo_api(
    todo_id: str,
    req: TodoUpdateRequest,
    user: dict = Depends(get_current_user),
):
    """Update a todo by ID."""
    await db_update_todo(
        id=todo_id,
        title=req.title,
        description=req.description,
        emoji=req.emoji,
        status=req.status,
        notes=req.notes,
        expected_completion_at=req.expected_completion_at,
    )
    return {"status": "updated"}


@app.delete("/api/todos/{todo_id}")
async def delete_todo_api(
    todo_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a todo by ID."""
    await db_delete_todo(todo_id)
    return {"status": "deleted"}


@app.get("/api/reminders")
async def get_reminders(user: dict = Depends(get_current_user)):
    """Get all reminders for the current user."""
    reminders = await get_all_reminders(user_id=user["id"])
    return reminders


@app.patch("/api/reminders/{reminder_id}")
async def update_reminder_api(
    reminder_id: str,
    req: ReminderUpdateRequest,
    user: dict = Depends(get_current_user),
):
    """Update a reminder by ID."""
    await db_update_reminder(
        id=reminder_id,
        scheduled_at=req.scheduled_at,
        title=req.title,
        body=req.body,
    )
    return {"status": "updated"}


@app.delete("/api/reminders/{reminder_id}")
async def delete_reminder_api(
    reminder_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a reminder by ID."""
    await db_delete_reminder(reminder_id)
    return {"status": "deleted"}


@app.post("/api/reminders/check")
async def check_reminders():
    """Manually trigger checking pending reminders."""
    await check_and_send_pending_reminders()
    return {"status": "checked"}



@app.get("/api/models")
async def get_available_models():
    """Return available model options."""
    return [
        {
            "id": "gpt-cloud",
            "name": "GPT-5.4 Nano",
            "provider": "openai",
            "icon": "cloud",
            "description": "OpenAI 云端模型，响应快速稳定",
        },
        {
            "id": "gemma-local",
            "name": "Gemma4 E2B",
            "provider": "ollama",
            "icon": "computer",
            "description": "本地 Ollama 部署，隐私安全，无网络延迟",
        },
    ]


@app.post("/api/model/check-ollama")
async def check_ollama_status():
    """Check if local Ollama is running and responsive."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                has_gemma = any("gemma4" in m for m in models)
                return {
                    "online": True,
                    "models": models,
                    "has_gemma4": has_gemma,
                }
    except Exception:
        pass
    return {"online": False, "models": [], "has_gemma4": False}


@app.get("/api/model/selection")
async def get_model_selection_api():
    """Get the globally selected model type."""
    selection = await get_setting("model_selection", "gpt-cloud")
    return {"model_selection": selection}


@app.post("/api/model/selection")
async def set_model_selection_api(data: ModelSelectionRequest):
    """Set the globally selected model type."""
    await set_setting("model_selection", data.model_selection)
    return {"status": "updated", "model_selection": data.model_selection}
class SupervisorConfigRequest(BaseModel):
    enabled: bool

@app.get("/api/supervisor/config")
async def get_supervisor_config_api(user: dict = Depends(get_current_user)):
    """Get the current supervisor enabled status."""
    enabled_str = await get_setting("supervisor_enabled", "true")
    return {"enabled": enabled_str.lower() == "true"}

@app.post("/api/supervisor/config")
async def set_supervisor_config_api(data: SupervisorConfigRequest, user: dict = Depends(get_current_user)):
    """Set the supervisor enabled status."""
    await set_setting("supervisor_enabled", str(data.enabled).lower())
    return {"status": "updated", "enabled": data.enabled}
