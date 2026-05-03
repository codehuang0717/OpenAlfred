from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from core.database import (
    get_all_reminders,
    update_reminder as db_update_reminder,
    delete_reminder as db_delete_reminder,
)
from routers.auth import get_current_user

router = APIRouter(prefix="/api/reminders", tags=["reminders"])

class ReminderUpdateRequest(BaseModel):
    scheduled_at: Optional[str] = None
    title: Optional[str] = None
    body: Optional[str] = None


@router.get("")
async def get_reminders(user: dict = Depends(get_current_user)):
    """Get all reminders for the current user."""
    reminders = await get_all_reminders(user_id=user["id"])
    return reminders


@router.patch("/{reminder_id}")
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


@router.delete("/{reminder_id}")
async def delete_reminder_api(
    reminder_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a reminder by ID."""
    await db_delete_reminder(reminder_id)
    return {"status": "deleted"}
