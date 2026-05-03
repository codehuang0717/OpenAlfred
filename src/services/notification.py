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
        badge: Optional[int] = None
    ) -> bool:
        """
        Sends a rich notification via Bark.
        
        Args:
            body: The main content of the notification.
            title: The title of the notification.
            subtitle: The subtitle.
            level: Interruption level (active, timeSensitive, passive, critical).
            sound: Name of the sound file to play.
            icon: URL to an icon image (iOS 15+).
            group: Group name for organizing notifications.
            url: URL to open when the notification is tapped.
            is_archive: Whether to archive the message in the Bark app.
            copy_text: Text to copy to clipboard when interacting.
            badge: Number to display on the app icon badge.
        """
        if not config.BARK_URL:
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
        
        logger.info(f"[Bark] Attempting to send rich notification to {config.BARK_URL}")
        
        try:
            async with httpx.AsyncClient() as client:
                # Bark supports POST to the device URL with a JSON body
                response = await client.post(config.BARK_URL, json=payload, timeout=10.0)
                
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
    async def send_todo_reminder(cls, todo: dict):
        """Send a rich notification specifically for a Todo item."""
        title = f"🎯 任务提醒: {todo.get('title')}"
        body = todo.get('description') or "是时候开始这项任务了！"
        
        # Use a premium icon for todos
        icon = "https://cdn-icons-png.flaticon.com/512/4697/4697260.png" # Checklist icon
        
        # Group by "OpenAlfred-Todos"
        group = "OpenAlfred-Todos"
        
        # Level: timeSensitive for todo reminders
        level = "timeSensitive"
        
        # Link back to the web app (assuming local dev for now, can be configured)
        # We can use a base URL from core.config if available
        url = "http://localhost:3000" # Placeholder, ideally deep link to the todo
        
        return await cls.send_bark_notification(
            body=body,
            title=title,
            icon=icon,
            group=group,
            level=level,
            url=url,
            sound="birdsong"
        )

notification_service = NotificationService()
