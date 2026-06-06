"""
Todo repository — CRUD operations for the todos table.
"""

from typing import Optional
from datetime import datetime, timezone
from logic.schema import TodoDict
from db.connection import get_db
from core.event_bus import event_bus, EventType


async def get_all_todos(user_id: str = "default") -> list[TodoDict]:
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM todos WHERE deleted = 0 AND user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def add_todo(
    id: str,
    title: str,
    description: str = "",
    emoji: str = "🎯",
    status: str = "pending",
    notes: str = "",
    expected_completion_at: Optional[str] = None,
    scheduled_start_at: Optional[str] = None,
    user_id: str = "default",
):
    created_at = datetime.now(timezone.utc).isoformat()
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO todos (id, title, description, emoji, status, created_at, completed_at, deleted, notes, expected_completion_at, scheduled_start_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
            """,
            (
                id,
                title,
                description,
                emoji,
                status,
                created_at,
                None,
                notes,
                expected_completion_at,
                scheduled_start_at,
                user_id,
            ),
        )
        await db.commit()

    await event_bus.publish(EventType.TODO_CREATED, {"id": id, "user_id": user_id})
    if scheduled_start_at:
        await event_bus.schedule(EventType.TODO_NOTIFICATION_DUE, {"id": id}, scheduled_start_at)



async def update_todo(
    id: str,
    user_id: str = "default",
    title: Optional[str] = None,
    description: Optional[str] = None,
    emoji: Optional[str] = None,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    expected_completion_at: Optional[str] = None,
    scheduled_start_at: Optional[str] = None,
):
    updates = []
    params = []

    if title is not None:
        updates.append("title = ?")
        params.append(title)
    if description is not None:
        updates.append("description = ?")
        params.append(description)
    if emoji is not None:
        updates.append("emoji = ?")
        params.append(emoji)
    if status is not None:
        updates.append("status = ?")
        params.append(status)
        if status == "completed":
            updates.append("completed_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())
        elif status == "pending":
            updates.append("completed_at = NULL")
    if notes is not None:
        updates.append("notes = ?")
        params.append(notes)
    if expected_completion_at is not None:
        updates.append("expected_completion_at = ?")
        params.append(expected_completion_at)
    if scheduled_start_at is not None:
        updates.append("scheduled_start_at = ?")
        params.append(scheduled_start_at)

    if not updates:
        return

    params.extend([id, user_id])
    async with get_db() as db:
        cursor = await db.execute(
            f"UPDATE todos SET {', '.join(updates)} WHERE id = ? AND user_id = ? AND deleted = 0",
            params,
        )
        await db.commit()
        updated = cursor.rowcount > 0

    if not updated:
        return False

    await event_bus.publish(EventType.TODO_UPDATED, {"id": id, "user_id": user_id})

    if scheduled_start_at is not None:
        await event_bus.unschedule(EventType.TODO_NOTIFICATION_DUE, {"id": id})
        if scheduled_start_at != "":  # Not clearing the schedule
            await event_bus.schedule(EventType.TODO_NOTIFICATION_DUE, {"id": id}, scheduled_start_at)
            
    if status == "completed":
        await event_bus.unschedule(EventType.TODO_NOTIFICATION_DUE, {"id": id})
    return True


async def delete_todo(id: str, user_id: str = "default") -> bool:
    async with get_db() as db:
        cursor = await db.execute(
            "UPDATE todos SET deleted = 1 WHERE id = ? AND user_id = ? AND deleted = 0",
            (id, user_id),
        )
        await db.commit()
        deleted = cursor.rowcount > 0

    if not deleted:
        return False
    
    await event_bus.publish(EventType.TODO_DELETED, {"id": id, "user_id": user_id})
    await event_bus.unschedule(EventType.TODO_NOTIFICATION_DUE, {"id": id})
    return True


async def get_todo_by_id(id: str, user_id: Optional[str] = None):
    sql = "SELECT * FROM todos WHERE id = ? AND deleted = 0"
    params = [id]
    if user_id is not None:
        sql += " AND user_id = ?"
        params.append(user_id)

    async with get_db() as db:
        async with db.execute(
            sql,
            params,
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def get_pending_todo_notifications():
    """Scan todos that are scheduled to start but haven't sent a notification."""
    now = datetime.now(timezone.utc).isoformat()
    async with get_db() as db:
        # Find todos where scheduled_start_at <= now and notification_sent = 0
        async with db.execute(
            "SELECT * FROM todos WHERE status = 'pending' AND deleted = 0 AND notification_sent = 0 AND scheduled_start_at IS NOT NULL AND scheduled_start_at <= ?",
            (now,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def mark_todo_notification_sent(id: str) -> bool:
    """Mark a todo notification as sent. Returns True if actually updated (idempotent)."""
    async with get_db() as db:
        cursor = await db.execute(
            "UPDATE todos SET notification_sent = 1 WHERE id = ? AND notification_sent = 0",
            (id,),
        )
        await db.commit()
        updated = cursor.rowcount > 0
        if updated:
            await event_bus.publish(EventType.TODO_UPDATED, {"id": id, "notification_sent": 1})
        return updated
