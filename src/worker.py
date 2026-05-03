import asyncio
import logging
import sys
import os

# Add src to python path if necessary
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.scheduler import (
    check_and_send_pending_reminders, 
    check_and_send_todo_notifications,
    send_single_reminder,
    send_single_todo_notification
)
from core.database import init_db
from core.event_bus import event_bus, EventType

from utils.logger import setup_logging, get_logger

# Initialize unified logging
setup_logging(log_file="worker.log")
logger = get_logger("worker")

async def main():
    logger.info("Starting OpenAlfred Background Worker...")
    
    # Ensure database is initialized
    await init_db()
    await event_bus.connect()
    logger.info("Database initialized and EventBus connected.")
    
    # 1. Initial cleanup: Run the scanning logic once to handle any events missed while worker was down
    logger.info("Running initial scan for pending items...")
    await check_and_send_pending_reminders()
    await check_and_send_todo_notifications()
    
    logger.info("Starting Redis event consumer loop. Resolution: 1s. Fallback scan: 60s.")
    
    iteration = 0
    while True:
        try:
            # 2. Check for due events in Redis (reminders, etc.)
            due_events = await event_bus.get_due_events()
            for event in due_events:
                etype = event.get("type")
                data = event.get("data", {})
                item_id = data.get("id")
                
                if not item_id:
                    continue

                if etype == EventType.REMINDER_DUE.value:
                    logger.info(f"Processing scheduled reminder event for {item_id}")
                    await send_single_reminder(item_id)
                elif etype == EventType.TODO_NOTIFICATION_DUE.value:
                    logger.info(f"Processing scheduled todo notification for {item_id}")
                    await send_single_todo_notification(item_id)
                    # Notify the supervisor to wake up immediately
                    await event_bus.publish(EventType.SUPERVISOR_WAKEUP)
            
            # 3. Fallback polling: Run the full scanning logic every 60 seconds
            # This catches any items missed if Redis was down or if events were lost
            iteration += 1
            if iteration >= 60:
                logger.debug("Running periodic fallback scan...")
                await check_and_send_pending_reminders()
                await check_and_send_todo_notifications()
                iteration = 0
                
        except Exception as e:
            logger.error(f"Error in worker event loop: {e}", exc_info=True)
        
        # Short sleep to prevent busy-waiting
        await asyncio.sleep(1)



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker stopped by user.")
