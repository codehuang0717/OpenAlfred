from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.database import (
    get_all_todos,
    update_todo as db_update_todo,
    delete_todo as db_delete_todo,
)
from routers.auth import get_current_user

router = APIRouter(prefix="/api/todos", tags=["todos"])

class TodoUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    emoji: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None
    expected_completion_at: Optional[str] = None
    scheduled_start_at: Optional[str] = None


@router.get("")
async def get_todos(user: dict = Depends(get_current_user)):
    """Get all todos for the current user."""
    todos = await get_all_todos(user_id=user["id"])
    return todos


@router.patch("/{todo_id}")
async def update_todo_api(
    todo_id: str,
    req: TodoUpdateRequest,
    user: dict = Depends(get_current_user),
):
    """Update a todo by ID."""
    updated = await db_update_todo(
        id=todo_id,
        user_id=user["id"],
        title=req.title,
        description=req.description,
        emoji=req.emoji,
        status=req.status,
        notes=req.notes,
        expected_completion_at=req.expected_completion_at,
        scheduled_start_at=req.scheduled_start_at,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Todo not found")
    return {"status": "updated"}


@router.delete("/{todo_id}")
async def delete_todo_api(
    todo_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a todo by ID."""
    deleted = await db_delete_todo(todo_id, user_id=user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Todo not found")
    return {"status": "deleted"}
