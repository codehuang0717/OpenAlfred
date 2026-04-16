import asyncio
import logging
import sys
import os
import json
from datetime import datetime, timezone

# Add src to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import init_db, get_all_todos
from tools.eye import get_recent_ocr_text
from tools.call_user import dial_user
from llm import get_model
from prompts import SUPERVISOR_PROMPT
from langchain_core.messages import HumanMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("supervisor")

class ProactiveSupervisor:
    def __init__(self):
        self.consecutive_distractions = 0 # Track intervals of distraction (10 mins each)
        self.last_status = "NORMAL"

    async def run_cycle(self):
        logger.info("--- Starting Supervision Cycle ---")
        
        # 1. Fetch Context
        todos = await get_all_todos(user_id="default")
        pending_todos = [t for t in todos if t['status'] == 'pending']
        
        # 2. Fetch Vision
        ocr_context = await get_recent_ocr_text(minutes=10)
        
        if not pending_todos:
            logger.info("No pending todos. Enjoy your free time!")
            self.consecutive_distractions = 0
            return

        # 3. Analyze with LLM
        # We estimate distraction duration based on how many cycles we've seen distraction
        duration = self.consecutive_distractions * 10 
        
        tasks_str = "\n".join([f"- {t['title']}: {t['description']}" for t in pending_todos])
        focus_task = pending_todos[0]['title'] if pending_todos else "None"
        
        prompt = SUPERVISOR_PROMPT.format(
            tasks=tasks_str,
            focus_task=focus_task,
            ocr_context=ocr_context,
            distraction_duration=duration
        )
        
        llm = get_model("gpt-cloud")
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            # Simple JSON parsing (expecting raw JSON from prompt instructions)
            result = json.loads(response.content.strip().replace('```json', '').replace('```', ''))
            
            status = result.get("status", "NORMAL")
            reason = result.get("reason", "No reason provided.")
            greeting = result.get("call_greeting", "")
            
            logger.info(f"Supervisor Decision: {status} | Reason: {reason}")
            
            if status != "NORMAL":
                self.consecutive_distractions += 1
                
                # Check if we should call based on duration
                # User's logic: < 2 mins (ignore/gentle), > 10 mins (strict), etc.
                # Since our cycle is 10 mins, if status is not NORMAL, it's already 10 mins of distraction
                # unless it's just a brief switch. 
                # The LLM sees the OCR and decides.
                
                if status in ["STRICT_WARNING", "SEVERE_DISCIPLINE", "GENTLE_REMINDER"]:
                    logger.warning(f"触发提醒计划! 状态: {status}")
                    call_status = await dial_user(phone_number="100", initial_speech=greeting)
                    logger.info(f"Call Result: {call_status}")
            else:
                self.consecutive_distractions = 0
                
        except Exception as e:
            logger.error(f"Error in supervision logic: {e}")

    async def start(self):
        await init_db()
        logger.info("Supervisor Service Initialized. Interval: 10 minutes.")
        
        while True:
            await self.run_cycle()
            # Interval: 10 minutes
            await asyncio.sleep(600) 

if __name__ == "__main__":
    supervisor = ProactiveSupervisor()
    try:
        asyncio.run(supervisor.start())
    except KeyboardInterrupt:
        logger.info("Supervisor stopped by user.")
