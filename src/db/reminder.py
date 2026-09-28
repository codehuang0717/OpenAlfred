"""
Reminder repository — CRUD operations for the reminders table.
"""

from typing import Optional
from datetime import datetime, timezone
from db.connection import get_db
from utils.logger import get_logger
from core.event_bus import event_bus, EventType
from utils.auth_utils import require_explicit_user_id

_logger = get_logger("db.reminder")


async def add_reminder(
    id: str,
    body: str,
    scheduled_at: str,
    *,
    user_id: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    level: str = "active",
    sound: Optional[str] = None,
    delivery_method: str = "push",
    audio_path: str = "",
):
    user_id = require_explicit_user_id(user_id)
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
    await event_bus.schedule(
        EventType.REMINDER_DUE,
        {"id": id, "user_id": user_id},
        scheduled_at,
    )


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


async def mark_reminder_sent(id: str, *, user_id: str) -> bool:
    """Mark a reminder as sent. Returns True if actually updated (idempotent)."""
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        cursor = await db.execute(
            "UPDATE reminders SET sent = 1 WHERE id = ? AND user_id = ? AND sent = 0",
            (id, user_id),
        )
        await db.commit()
        updated = cursor.rowcount > 0
        if updated:
            await event_bus.publish(EventType.REMINDER_SENT, {"id": id, "user_id": user_id})
        return updated


async def update_reminder(
    id: str,
    *,
    user_id: str,
    scheduled_at: Optional[str] = None,
    title: Optional[str] = None,
    body: Optional[str] = None,
) -> bool:
    user_id = require_explicit_user_id(user_id)
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

    params.extend([id, user_id])
    async with get_db() as db:
        cursor = await db.execute(
            f"UPDATE reminders SET {', '.join(updates)} WHERE id = ? AND user_id = ?",
            params,
        )
        await db.commit()
        updated = cursor.rowcount > 0

    if not updated:
        return False

    await event_bus.publish(EventType.REMINDER_UPDATED, {"id": id, "user_id": user_id})
    if scheduled_at:
        # Re-schedule in delayed queue
        await event_bus.schedule(
            EventType.REMINDER_DUE,
            {"id": id, "user_id": user_id},
            scheduled_at,
        )
    return True


async def get_all_reminders(user_id: str):
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM reminders WHERE user_id = ? ORDER BY scheduled_at DESC",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def delete_reminder(id: str, *, user_id: str) -> bool:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        cursor = await db.execute("DELETE FROM reminders WHERE id = ? AND user_id = ?", (id, user_id))
        await db.commit()
        deleted = cursor.rowcount > 0

    if not deleted:
        return False
    
    await event_bus.publish(EventType.REMINDER_DELETED, {"id": id, "user_id": user_id})
    # Remove from delayed queue if present
    await event_bus.unschedule(EventType.REMINDER_DUE, {"id": id})
    return True


async def get_reminder_by_id(id: str, *, user_id: str):
    user_id = require_explicit_user_id(user_id)
    sql = "SELECT * FROM reminders WHERE id = ? AND user_id = ?"
    params = [id, user_id]

    async with get_db() as db:
        async with db.execute(
            sql,
            params,
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None
