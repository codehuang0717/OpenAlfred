import httpx
from utils.logger import get_logger
from typing import Optional, Literal
from core.config import config

logger = get_logger("notification_service")

class NotificationService:
    """
    A service to handle outgoing notifications with rich features.
    Currently supports Bark (iOS).
    """
    
    @staticmethod
    async def send_bark_notification(
        body: str,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        level: Literal["active", "timeSensitive", "passive", "critical"] = "active",
        sound: Optional[str] = None,
        icon: Optional[str] = None,
        group: Optional[str] = None,
        url: Optional[str] = None,
        is_archive: bool = True,
        copy_text: Optional[str] = None,
        badge: Optional[int] = None,
        bark_url: Optional[str] = None,
    ) -> bool:
        """
        Sends a rich notification via Bark.

        Args:
            bark_url: Per-user Bark device URL. Falls back to global config.BARK_URL.
        """
        target_url = bark_url or config.BARK_URL
        if not target_url:
            logger.warning("BARK_URL is not configured. Skipping notification.")
            return False

        # Construct the payload
        payload = {
            "body": body,
            "title": title,
            "subtitle": subtitle,
            "level": level,
            "sound": sound,
            "icon": icon,
            "group": group,
            "url": url,
            "isArchive": 1 if is_archive else 0,
            "copy": copy_text,
            "badge": badge
        }

        # Clean up None values
        payload = {k: v for k, v in payload.items() if v is not None}

        logger.info(f"[Bark] Attempting to send rich notification to {target_url}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(target_url, json=payload, timeout=10.0)

            if response.status_code == 200:
                logger.info(f"[Bark] SUCCESS: Notification '{title or body[:20]}' sent.")
                return True
            else:
                logger.error(f"[Bark] ERROR: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"[Bark] EXCEPTION: Failed to send notification: {str(e)}")
            return False

    @classmethod
    async def send_todo_reminder(cls, todo: dict, bark_url: Optional[str] = None):
        """Send a rich notification specifically for a Todo item."""
        title = f"🎯 任务提醒: {todo.get('title')}"
        body = todo.get('description') or "是时候开始这项任务了！"

        icon = "https://cdn-icons-png.flaticon.com/512/4697/4697260.png"
        group = "OpenAlfred-Todos"
        level = "timeSensitive"
        url = "http://localhost:3000"

        return await cls.send_bark_notification(
            body=body,
            title=title,
            icon=icon,
            group=group,
            level=level,
            url=url,
            sound="birdsong",
            bark_url=bark_url,
        )

notification_service = NotificationService()
