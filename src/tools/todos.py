import uuid
from typing import Optional, Literal
from datetime import datetime, timezone
from pydantic import BaseModel
from langchain.tools import ToolRuntime, tool
from langchain.messages import ToolMessage
from langgraph.types import Command
from schema import AgentState, TodoDict
from utils.time_utils import localize_to_utc
from database import (
    get_all_todos,
    get_active_user,
    add_todo as db_add_todo,
    update_todo as db_update_todo,
    delete_todo as db_delete_todo,
)


async def _get_user_id(runtime: ToolRuntime) -> str:
    """Extract user_id from RunnableConfig populated by LangGraph Auth or custom metadata."""
    if hasattr(runtime, "config") and runtime.config:
        conf = runtime.config.get("configurable", {})
        
        # 1. LangGraph Auth (Service JWT sub)
        auth_user = conf.get("langgraph_auth_user", {})
        if isinstance(auth_user, dict) and "identity" in auth_user:
            return auth_user["identity"]
            
        # 2. Direct configurable fields (Voice Agent Explicit Injection)
        if "user_id" in conf: return conf["user_id"]
        if "owner" in conf: return conf["owner"]
        if "thread_owner" in conf: return conf["thread_owner"]

        # 3. Request Metadata (Passed in runs/wait body)
        metadata = runtime.config.get("metadata", {})
        if "owner" in metadata:
            return metadata["owner"]

    # 4. Global Fallback: Query the currently active user from DB (Last Resort)
    try:
        active_user = await get_active_user()
        if active_user:
            return active_user["id"]
    except Exception:
        pass

    # Fallback to state payload
    if hasattr(runtime, "state") and runtime.state:
        if isinstance(runtime.state, dict):
            return runtime.state.get("user_id", "default")
        return getattr(runtime.state, "user_id", "default")
    return "default"


async def initialize_todos(state: AgentState) -> dict:
    """Initialize todos from database on agent startup."""
    user_id = state.user_id if hasattr(state, "user_id") else "default"
    todos = await get_all_todos(user_id=user_id)
    return {"todos": todos}


async def sync_todos_to_state(runtime: ToolRuntime):
    user_id = await _get_user_id(runtime)
    todos = await get_all_todos(user_id=user_id)
    return Command(update={"todos": todos})


@tool
async def get_todos(runtime: ToolRuntime) -> list[TodoDict]:
    """
    Get all current todos from the database.
    """
    user_id = await _get_user_id(runtime)
    todos = await get_all_todos(user_id=user_id)
    return todos


@tool
async def add_todo(
    runtime: ToolRuntime,
    title: str,
    description: str = "",
    emoji: str = "🎯",
    notes: str = "",
    expected_completion_at: Optional[str] = None,
    scheduled_start_at: Optional[str] = None,
) -> Command:
    """
    Add a new todo to the list.
    """
    user_id = await _get_user_id(runtime)
    id = str(uuid.uuid4())
    
    # Standardize time if provided
    formatted_time = expected_completion_at
    if expected_completion_at:
        try:
            formatted_time = localize_to_utc(expected_completion_at)
        except Exception as e:
            # Fallback to original or handle error - for high availability, we log and keep 
            # if LLM produced something truly weird, but our tool description should prevent this.
            pass

    formatted_start_time = scheduled_start_at
    if scheduled_start_at:
        try:
            formatted_start_time = localize_to_utc(scheduled_start_at)
        except:
            pass

    await db_add_todo(
        id=id,
        title=title,
        description=description,
        emoji=emoji,
        notes=notes,
        expected_completion_at=formatted_time,
        scheduled_start_at=formatted_start_time,
        user_id=user_id,
    )

    return Command(
        update={
            "todos": await get_all_todos(user_id=user_id),
            "messages": [
                ToolMessage(
                    content=f"Successfully added todo: {title}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
async def update_todo(
    runtime: ToolRuntime,
    id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    emoji: Optional[str] = None,
    status: Optional[Literal["pending", "completed"]] = None,
    notes: Optional[str] = None,
    expected_completion_at: Optional[str] = None,
    scheduled_start_at: Optional[str] = None,
) -> Command:
    """
    Update an existing todo by its ID.
    """
    user_id = await _get_user_id(runtime)
    
    # Standardize time if provided
    if expected_completion_at:
        try:
            expected_completion_at = localize_to_utc(expected_completion_at)
        except:
            pass

    if scheduled_start_at:
        try:
            scheduled_start_at = localize_to_utc(scheduled_start_at)
        except:
            pass

    await db_update_todo(
        id=id,
        title=title,
        description=description,
        emoji=emoji,
        status=status,
        notes=notes,
        expected_completion_at=expected_completion_at,
        scheduled_start_at=scheduled_start_at,
    )

    return Command(
        update={
            "todos": await get_all_todos(user_id=user_id),
            "messages": [
                ToolMessage(
                    content=f"Successfully updated todo",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
async def delete_todo(runtime: ToolRuntime, id: str) -> Command:
    """
    Delete a todo by its ID.
    """
    user_id = await _get_user_id(runtime)
    await db_delete_todo(id)

    return Command(
        update={
            "todos": await get_all_todos(user_id=user_id),
            "messages": [
                ToolMessage(
                    content=f"Successfully deleted todo",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


todo_tools = [
    get_todos,
    add_todo,
    update_todo,
    delete_todo,
]
