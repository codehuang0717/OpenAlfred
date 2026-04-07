from pydantic import BaseModel
from langchain.tools import ToolRuntime, tool
from langchain.messages import ToolMessage
from langgraph.types import Command
from typing import TypedDict, Literal, Optional
import uuid
from datetime import datetime, timezone

from database import (
    get_all_todos,
    add_todo as db_add_todo,
    update_todo as db_update_todo,
    delete_todo as db_delete_todo,
    TodoDict,
)


class AgentState(BaseModel):
    todos: list[TodoDict]
    mem0_user_id: str
    tts_text: Optional[str]
    jump_to: str
    structured_response: dict
    model_selection: Optional[str]
    chat_session_id: Optional[str]


async def initialize_todos(state: AgentState) -> dict:
    """Initialize todos from database on agent startup."""
    todos = await get_all_todos()
    return {"todos": todos}


async def sync_todos_to_state(runtime: ToolRuntime):
    todos = await get_all_todos()
    return Command(update={"todos": todos})


@tool
async def get_todos(runtime: ToolRuntime) -> list[TodoDict]:
    """
    Get all current todos from the database.
    """
    todos = await get_all_todos()
    return todos


@tool
async def add_todo(
    runtime: ToolRuntime,
    title: str,
    description: str = "",
    emoji: str = "🎯",
    notes: str = "",
    expected_completion_at: Optional[str] = None,
) -> Command:
    """
    Add a new todo to the list.
    """
    id = str(uuid.uuid4())
    await db_add_todo(
        id=id,
        title=title,
        description=description,
        emoji=emoji,
        notes=notes,
        expected_completion_at=expected_completion_at,
    )

    return Command(
        update={
            "todos": await get_all_todos(),
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
) -> Command:
    """
    Update an existing todo by its ID.
    """
    await db_update_todo(
        id=id,
        title=title,
        description=description,
        emoji=emoji,
        status=status,
        notes=notes,
        expected_completion_at=expected_completion_at,
    )

    return Command(
        update={
            "todos": await get_all_todos(),
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
    await db_delete_todo(id)

    return Command(
        update={
            "todos": await get_all_todos(),
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
