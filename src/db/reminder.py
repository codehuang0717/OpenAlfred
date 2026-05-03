"""
Reminder repository — CRUD operations for the reminders table.
"""

from typing import Optional
from datetime import datetime, timezone
from db.connection import get_db
from utils.logger import get_logger
from core.event_bus import event_bus, EventType

_logger = get_logger("db.reminder")


async def add_reminder(
    id: str,
    body: str,
    scheduled_at: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    level: str = "active",
    sound: Optional[str] = None,
    delivery_method: str = "push",
    audio_path: str = "",
    user_id: str = "default",
):
    created_at = datetime.now(timezone.utc).isoformat()
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO reminders (id, title, subtitle, body, scheduled_at, sent, level, sound, created_at, delivery_method, audio_path, user_id)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
            """,
            (
                id,
                title,
                subtitle,
                body,
                scheduled_at,
                level,
                sound,
                created_at,
                delivery_method,
                audio_path,
                user_id,
            ),
        )
        await db.commit()

    # Publish creation event for UI
    await event_bus.publish(EventType.REMINDER_CREATED, {"id": id, "user_id": user_id})
    # Schedule precise trigger in Redis delayed queue
    await event_bus.schedule(EventType.REMINDER_DUE, {"id": id}, scheduled_at)


async def get_pending_reminders():
    now = datetime.now(timezone.utc)

    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM reminders WHERE sent = 0 ORDER BY scheduled_at ASC"
        ) as cursor:
            rows = await cursor.fetchall()
            reminders = [dict(row) for row in rows]

    filtered = []
    for r in reminders:
        try:
            scheduled = datetime.fromisoformat(r["scheduled_at"].replace("Z", "+00:00"))
            if scheduled <= now:
                filtered.append(r)
        except (ValueError, TypeError) as e:
            _logger.warning(f"Skipping reminder {r['id']} with unparseable scheduled_at='{r['scheduled_at']}': {e}")

    return filtered


async def mark_reminder_sent(id: str) -> bool:
    """Mark a reminder as sent. Returns True if actually updated (idempotent)."""
    async with get_db() as db:
        cursor = await db.execute(
            "UPDATE reminders SET sent = 1 WHERE id = ? AND sent = 0",
            (id,),
        )
        await db.commit()
        updated = cursor.rowcount > 0
        if updated:
            await event_bus.publish(EventType.REMINDER_SENT, {"id": id})
        return updated


async def update_reminder(
    id: str,
    scheduled_at: Optional[str] = None,
    title: Optional[str] = None,
    body: Optional[str] = None,
):
    updates = []
    params = []

    if scheduled_at is not None:
        updates.append("scheduled_at = ?")
        params.append(scheduled_at)
    if title is not None:
        updates.append("title = ?")
        params.append(title)
    if body is not None:
        updates.append("body = ?")
        params.append(body)

    if not updates:
        return

    params.append(id)
    async with get_db() as db:
        await db.execute(
            f"UPDATE reminders SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        await db.commit()

    await event_bus.publish(EventType.REMINDER_UPDATED, {"id": id})
    if scheduled_at:
        # Re-schedule in delayed queue
        await event_bus.schedule(EventType.REMINDER_DUE, {"id": id}, scheduled_at)


async def get_all_reminders(user_id: str = "default"):
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM reminders WHERE user_id = ? ORDER BY scheduled_at DESC",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def delete_reminder(id: str):
    async with get_db() as db:
        await db.execute("DELETE FROM reminders WHERE id = ?", (id,))
        await db.commit()
    
    await event_bus.publish(EventType.REMINDER_DELETED, {"id": id})
    # Remove from delayed queue if present
    await event_bus.unschedule(EventType.REMINDER_DUE, {"id": id})


async def get_reminder_by_id(id: str):
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM reminders WHERE id = ?",
            (id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None
