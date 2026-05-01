import asyncio
from utils.logger import get_logger
import httpx
from datetime import datetime, timezone
from config import config
from database import (
    get_pending_reminders,
    mark_reminder_sent,
    get_pending_todo_notifications,
    mark_todo_notification_sent,
    get_reminder_by_id,
    get_todo_by_id,
)
from notification_service import notification_service

logger = get_logger("scheduler")

async def _send_bark_notification(
    body: str,
    title: str = None,
    subtitle: str = None,
    level: str = "active",
    sound: str = None,
) -> str:
    """Internal function to send Bark notification using the NotificationService."""
    success = await notification_service.send_bark_notification(
        body=body,
        title=title,
        subtitle=subtitle,
        level=level,
        sound=sound,
        group="OpenAlfred-Reminders",
        icon="https://cdn-icons-png.flaticon.com/512/3602/3602123.png"
    )
    return "success" if success else "error"

async def check_and_send_pending_reminders():
    """Scan and send reminders that are due. Supports Push and SIP calls."""
    from tools.call_user import generate_sip_token, OUTBOUND_TRUNK_ID
    
    try:
        pending = await get_pending_reminders()
        for r in pending:
            logger.info(f"Sending reminder: {r['body']} via {r['delivery_method']}")
            
            if r.get("delivery_method") == "call":
                jwt_token = generate_sip_token()
                api_url = config.LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")
                if api_url.endswith("/"): api_url = api_url[:-1]
                url = f"{api_url}/twirp/livekit.SIP/CreateSIPParticipant"
                
                room_name = f"outbound-reminder-{r['id']}"
                async with httpx.AsyncClient() as client:
                    await client.post(
                        url,
                        headers={"Authorization": f"Bearer {jwt_token}"},
                        json={
                            "sipTrunkId": OUTBOUND_TRUNK_ID,
                            "sipCallTo": "100",
                            "roomName": room_name,
                        },
                        timeout=10.0,
                    )
            else:
                # Bark Push
                await _send_bark_notification(
                    body=r['body'],
                    title=r.get('title'),
                    subtitle=r.get('subtitle'),
                    level=r.get('level', 'active'),
                    sound=r.get('sound')
                )
            
            await mark_reminder_sent(r["id"])
    except Exception as e:
        logger.error(f"Error in check_and_send_pending_reminders: {e}", exc_info=True)

async def check_and_send_todo_notifications():
    """Scan and send notifications for scheduled Todos."""
    try:
        pending_todos = await get_pending_todo_notifications()
        for todo in pending_todos:
            logger.info(f"Sending notification for Todo: {todo['title']}")
            success = await notification_service.send_todo_reminder(todo)
            if success:
                await mark_todo_notification_sent(todo['id'])
    except Exception as e:
        logger.error(f"Error in check_and_send_todo_notifications: {e}", exc_info=True)

async def send_single_reminder(reminder_id: str):
    """Process a single reminder by ID."""
    from tools.call_user import generate_sip_token, OUTBOUND_TRUNK_ID
    
    try:
        r = await get_reminder_by_id(reminder_id)
        if not r or r.get("sent"):
            return

        logger.info(f"Triggering individual reminder: {r['body']} via {r['delivery_method']}")
        
        if r.get("delivery_method") == "call":
            jwt_token = generate_sip_token()
            api_url = config.LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")
            if api_url.endswith("/"): api_url = api_url[:-1]
            url = f"{api_url}/twirp/livekit.SIP/CreateSIPParticipant"
            
            room_name = f"outbound-reminder-{r['id']}"
            async with httpx.AsyncClient() as client:
                await client.post(
                    url,
                    headers={"Authorization": f"Bearer {jwt_token}"},
                    json={
                        "sipTrunkId": OUTBOUND_TRUNK_ID,
                        "sipCallTo": "100",
                        "roomName": room_name,
                    },
                    timeout=10.0,
                )
        else:
            await _send_bark_notification(
                body=r['body'],
                title=r.get('title'),
                subtitle=r.get('subtitle'),
                level=r.get('level', 'active'),
                sound=r.get('sound')
            )
        
        await mark_reminder_sent(r["id"])
    except Exception as e:
        logger.error(f"Error in send_single_reminder for {reminder_id}: {e}", exc_info=True)

async def send_single_todo_notification(todo_id: str):
    """Process a single todo notification by ID."""
    try:
        todo = await get_todo_by_id(todo_id)
        if not todo or todo.get("notification_sent") or todo.get("status") == "completed":
            return
            
        logger.info(f"Triggering individual todo notification for: {todo['title']}")
        success = await notification_service.send_todo_reminder(todo)
        if success:
            await mark_todo_notification_sent(todo['id'])
    except Exception as e:
        logger.error(f"Error in send_single_todo_notification for {todo_id}: {e}", exc_info=True)

