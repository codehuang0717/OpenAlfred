from utils.logger import get_logger
from core.database import (
    get_pending_reminders,
    mark_reminder_sent,
    get_pending_todo_notifications,
    mark_todo_notification_sent,
    get_reminder_by_id,
    get_todo_by_id,
    get_user_bark_url,
)
from services.notification import notification_service
from utils.auth_utils import require_explicit_user_id

logger = get_logger("scheduler")

async def _send_bark_notification(
    body: str,
    title: str = None,
    subtitle: str = None,
    level: str = "active",
    sound: str = None,
    bark_url: str = None,
) -> bool:
    """Internal function to send Bark notification using the NotificationService."""
    success = await notification_service.send_bark_notification(
        body=body,
        title=title,
        subtitle=subtitle,
        level=level,
        sound=sound,
        group="OpenAlfred-Reminders",
        icon="https://cdn-icons-png.flaticon.com/512/3602/3602123.png",
        bark_url=bark_url,
    )
    return success


class ReminderDeliveryError(RuntimeError):
    pass


def _row_user_id(row: dict) -> str:
    return require_explicit_user_id(row.get("user_id"))


async def _deliver_reminder(reminder: dict) -> None:
    from tools.call_user import dial_user

    user_id = _row_user_id(reminder)
    if reminder.get("delivery_method") == "call":
        status = await dial_user(
            user_id=user_id,
            phone_number="",
            initial_speech=reminder["body"],
            reminder_id=reminder["id"],
        )
        logger.info("LiveKit SIP dialing status: %s", status)
        if not (
            status.startswith("Call answered")
            or "Bark fallback sent" in status
        ):
            raise ReminderDeliveryError(status)
    else:
        bark_url = await get_user_bark_url(user_id)
        if not bark_url:
            raise ReminderDeliveryError(f"User {user_id} has no Bark URL configured")
        delivered = await _send_bark_notification(
            body=reminder["body"],
            title=reminder.get("title"),
            subtitle=reminder.get("subtitle"),
            level=reminder.get("level", "active"),
            sound=reminder.get("sound"),
            bark_url=bark_url,
        )
        if not delivered:
            raise ReminderDeliveryError(
                f"Bark delivery failed for reminder {reminder['id']}"
            )

async def check_and_send_pending_reminders():
    """Scan and send reminders that are due. Supports Push and SIP calls."""
    pending = await get_pending_reminders()
    for r in pending:
        try:
            logger.info(f"Sending reminder: {r['body']} via {r['delivery_method']}")
            await _deliver_reminder(r)
            await mark_reminder_sent(r["id"], user_id=_row_user_id(r))
        except Exception as e:
            logger.error(
                "Reminder %s delivery failed: %s", r.get("id"), e, exc_info=True
            )

async def check_and_send_todo_notifications():
    """Scan and send notifications for scheduled Todos."""
    try:
        pending_todos = await get_pending_todo_notifications()
        for todo in pending_todos:
            logger.info(f"Triggered notification for Todo: {todo['title']}")
            # We don't send bark notifications for Todos anymore per user request, 
            # we just mark it as sent so supervisor wakes up and handles it.
            await mark_todo_notification_sent(todo['id'], user_id=_row_user_id(todo))
    except Exception as e:
        logger.error(f"Error in check_and_send_todo_notifications: {e}", exc_info=True)

async def send_single_reminder(reminder_id: str, user_id: str):
    """Process a single reminder by ID."""
    user_id = require_explicit_user_id(user_id)
    r = await get_reminder_by_id(reminder_id, user_id=user_id)
    if not r or r.get("sent"):
        return
    logger.info(f"Triggering individual reminder: {r['body']} via {r['delivery_method']}")
    await _deliver_reminder(r)
    await mark_reminder_sent(r["id"], user_id=user_id)

async def send_single_todo_notification(todo_id: str, user_id: str):
    """Process a single todo notification by ID."""
    user_id = require_explicit_user_id(user_id)
    todo = await get_todo_by_id(todo_id, user_id=user_id)
    if not todo or todo.get("notification_sent") or todo.get("status") == "completed":
        return
    logger.info(f"Triggering individual todo notification for: {todo['title']}")
    # We no longer send bark notifications for Todos per user request,
    # just mark it as sent so the supervisor is woken up and handles it.
    await mark_todo_notification_sent(todo['id'], user_id=user_id)

