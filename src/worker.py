import asyncio
import logging
import sys
import os

# Add src to python path if necessary
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scheduler import check_and_send_pending_reminders, check_and_send_todo_notifications
from database import init_db

from utils.logger import setup_logging, get_logger

# Initialize unified logging
setup_logging(log_file="worker.log")
logger = get_logger("worker")

async def main():
    logger.info("Starting OpenAlfred Background Worker...")
    
    # Ensure database is initialized
    await init_db()
    logger.info("Database initialized.")
    
    logger.info("Reminder scheduler started. Scanning every 30 seconds.")
    
    while True:
        try:
            # Run the scanning logic
            await check_and_send_pending_reminders()
            await check_and_send_todo_notifications()
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}", exc_info=True)
        
        # Interval between scans
        await asyncio.sleep(30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker stopped by user.")
