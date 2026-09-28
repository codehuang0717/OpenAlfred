import uuid
from typing import Optional, Literal
from langchain.tools import ToolRuntime, tool
from langchain.messages import ToolMessage
from langgraph.types import Command
from logic.schema import AgentState, TodoDict
from utils.time_utils import localize_to_utc
from utils.auth_utils import require_explicit_user_id, require_runtime_user_id
from core.database import (
    get_all_todos,
    add_todo as db_add_todo,
    update_todo as db_update_todo,
    delete_todo as db_delete_todo,
)


async def _get_user_id(runtime: ToolRuntime) -> str:
    """Extract a verified user_id from LangGraph request metadata."""
    return require_runtime_user_id(runtime)


async def initialize_todos(state: AgentState) -> dict:
    """Initialize todos from database on agent startup."""
    user_id = require_explicit_user_id(getattr(state, "user_id", ""))
    todos = await get_all_todos(user_id=user_id)
    return {"todos": todos}


async def sync_todos_to_state(runtime: ToolRuntime):
    user_id = await _get_user_id(runtime)
    todos = await get_all_todos(user_id=user_id)
    return Command(update={"todos": todos})


@tool
async def get_todos(
    runtime: ToolRuntime,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list[TodoDict]:
    """Get current todos, optionally filtered by date range.
    
    Args:
        date_from: Start of date range (local time, e.g. '2026-05-01T00:00:00'). Only todos with scheduled_start_at or expected_completion_at on or after this time are returned.
        date_to: End of date range (local time, e.g. '2026-05-01T23:59:59'). Only todos with scheduled_start_at or expected_completion_at on or before this time are returned.
    
    When the user asks for a specific day's todos (e.g. "tomorrow", "today"), you MUST pass both date_from and date_to to get accurate results.
    Example: For "tomorrow" (May 1), pass date_from='2026-05-01T00:00:00', date_to='2026-05-01T23:59:59'.
    """
    user_id = await _get_user_id(runtime)
    todos = await get_all_todos(user_id=user_id)
    
    # Apply date range filter if provided
    if date_from or date_to:
        from utils.time_utils import localize_to_utc, parse_to_aware_utc
        
        utc_from = None
        utc_to = None
        if date_from:
            utc_from = parse_to_aware_utc(localize_to_utc(date_from))
        if date_to:
            utc_to = parse_to_aware_utc(localize_to_utc(date_to))
        
        filtered = []
        for t in todos:
            # Check if any time field falls within the range
            time_fields = [t.get('scheduled_start_at'), t.get('expected_completion_at')]
            matched = False
            for tf in time_fields:
                if not tf:
                    continue
                t_dt = parse_to_aware_utc(tf)
                if utc_from and t_dt < utc_from:
                    continue
                if utc_to and t_dt > utc_to:
                    continue
                matched = True
                break
            # When date filtering is active, skip todos without any time field
            if matched:
                filtered.append(t)
        
        todos = filtered
    
    # Convert UTC timestamps to local time for LLM readability
    from utils.time_utils import utc_to_local
    for t in todos:
        if t.get('scheduled_start_at'):
            t['scheduled_start_at'] = utc_to_local(t['scheduled_start_at'])
        if t.get('expected_completion_at'):
            t['expected_completion_at'] = utc_to_local(t['expected_completion_at'])
    
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
    """Add a new todo item."""
    user_id = await _get_user_id(runtime)
    id = str(uuid.uuid4())
    
    # Standardize time if provided
    formatted_time = localize_to_utc(expected_completion_at) if expected_completion_at else None
    formatted_start_time = localize_to_utc(scheduled_start_at) if scheduled_start_at else None

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
    """Update an existing todo by ID."""
    user_id = await _get_user_id(runtime)
    
    # Standardize time if provided
    if expected_completion_at:
        expected_completion_at = localize_to_utc(expected_completion_at)

    if scheduled_start_at:
        scheduled_start_at = localize_to_utc(scheduled_start_at)

    await db_update_todo(
        id=id,
        user_id=user_id,
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
    """Delete a todo by ID."""
    user_id = await _get_user_id(runtime)
    await db_delete_todo(id, user_id=user_id)

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
