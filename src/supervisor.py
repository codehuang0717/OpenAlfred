import asyncio
import logging
import sys
import os
import json
from datetime import datetime, timezone
from typing import Literal, Optional, List
from pydantic import BaseModel, Field

# Ensure UTF-8 output for Windows Console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import (
    init_db, 
    get_all_todos, 
    get_active_user, 
    get_supervisor_state, 
    update_supervisor_state, 
    reset_supervisor_state,
    get_setting
)
from tools.eye import get_recent_ocr_text
from tools.call_user import dial_user
from llm import get_model
from prompts import SUPERVISOR_PROMPT
from langchain_core.messages import HumanMessage
from config import config
from rich.console import Console
from rich.panel import Panel

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("supervisor")

class SupervisorDecision(BaseModel):
    """Schema for supervisor decision logic."""
    status: Literal["NORMAL", "GENTLE_REMINDER", "STRICT_WARNING", "SEVERE_DISCIPLINE"] = Field(
        description="The distraction status of the user."
    )
    reason: str = Field(description="A brief explanation of why this status was chosen.")
    call_greeting: str = Field(description="The opening speech for the phone call if status is not NORMAL.")

class ProactiveSupervisor:
    def __init__(self):
        self.model = get_model("gpt-cloud").with_structured_output(SupervisorDecision)

    async def run_cycle(self):
        # 0. Check if Supervision is enabled in settings
        is_enabled_str = await get_setting("supervisor_enabled", "true")
        if is_enabled_str.lower() != "true":
            logger.info("Supervision is currently DISABLED. Skipping cycle.")
            return

        logger.info("--- Starting Supervision Cycle ---")
        
        # 1. Fetch Active User
        active_user = await get_active_user()
        if not active_user:
            logger.warning("No active user found. Skipping cycle.")
            return
            
        user_id = active_user['id']
        username = active_user['username']
        
        logger.info(f"Targeting active user: {username} ({user_id})")
        
        # 2. Get state from DB
        state = await get_supervisor_state(user_id)
        if not state:
            # Initialize state if missing
            await update_supervisor_state(user_id, is_distracted=False)
            state = await get_supervisor_state(user_id)

        # 3. Fetch Context
        todos = await get_all_todos(user_id=user_id)
        pending_todos = [t for t in todos if t['status'] == 'pending']
        
        # Display Monitoring Context
        tasks_display = "\n".join([f"[blue]•[/blue] {t['title']}" for t in pending_todos]) if pending_todos else "None (IDLE)"
        console.print(Panel(
            f"[bold cyan]User:[/bold cyan] {username}\n"
            f"[bold cyan]Monitoring Tasks:[/bold cyan]\n{tasks_display}",
            title="[supervisor] Current Context",
            border_style="blue"
        ))

        if not pending_todos:
            logger.info(f"User {username} has no pending tasks. Monitoring status: Idle.")
            if state.get('is_distracted'):
                await reset_supervisor_state(user_id)
            return

        ocr_context = await get_recent_ocr_text(minutes=config.SUPERVISOR_OCR_WINDOW_MINS)
        
        # 4. Calculate Distraction Duration
        now = datetime.now(timezone.utc)
        distraction_duration = 0
        if state.get('is_distracted') and state.get('distraction_start_time'):
            start_time = datetime.fromisoformat(state['distraction_start_time'])
            distraction_duration = int((now - start_time).total_seconds() / 60)
        
        # 5. Analyze with LLM
        tasks_str = "\n".join([f"- {t['title']}: {t['description']}" for t in pending_todos])
        focus_task = pending_todos[0]['title'] 
        
        prompt = SUPERVISOR_PROMPT.format(
            tasks=tasks_str,
            focus_task=focus_task,
            ocr_context=ocr_context,
            distraction_duration=distraction_duration
        )
        
        try:
            decision: SupervisorDecision = await self.model.ainvoke([HumanMessage(content=prompt)])
            
            # Decision Dashboard
            status_color = "green" if decision.status == "NORMAL" else "bold red"
            console.print(Panel(
                f"[bold yellow]Status:[/bold yellow] {decision.status}\n"
                f"[bold magenta]AI Analysis:[/bold magenta]\n{decision.reason}\n"
                f"[bold cyan]Call Greeting:[/bold cyan]\n{decision.call_greeting if decision.call_greeting else 'N/A'}",
                title=f"[{status_color}]Supervisor Decision[/{status_color}]",
                border_style=status_color
            ))
            
            if decision.status != "NORMAL":
                # User is distracted
                new_start_time = state.get('distraction_start_time') or now.isoformat()
                current_consecutive = state.get('consecutive_distractions') or 0
                next_consecutive = current_consecutive + 1
                last_alert_time = state.get('last_alert_time')
                
                # Action logic: Trigger call for non-normal status
                if decision.status in ["GENTLE_REMINDER", "STRICT_WARNING", "SEVERE_DISCIPLINE"]:
                    console.print(f"[bold red]!! Action Required !![/bold red] Triggering alert for status: [white on red]{decision.status}[/white on red]")
                    call_status = await dial_user(
                        phone_number=config.SUPERVISOR_PHONE_NUMBER, 
                        initial_speech=decision.call_greeting,
                        user_id=user_id
                    )
                    last_alert_time = now.isoformat()
                    console.print(f"[bold green]Call Sent:[/bold green] {call_status}")

                # ATOMIC UPDATE: Save all state in one go
                await update_supervisor_state(
                    user_id=user_id,
                    is_distracted=True,
                    distraction_start_time=new_start_time,
                    last_alert_time=last_alert_time,
                    consecutive_distractions=next_consecutive,
                    last_decision=decision.model_dump_json()
                )
            else:
                # User is focused
                if state.get('is_distracted'):
                    console.print("[bold green]Success:[/bold green] User returned to focus. Resetting supervisor state.")
                    await reset_supervisor_state(user_id)
            
        except Exception as e:
            logger.error(f"Error in supervision logic: {e}")

    async def start(self):
        await init_db()
        logger.info(f"Supervisor Service Initialized. Interval: {config.SUPERVISOR_INTERVAL}s")
        
        while True:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"Critical error in supervisor cycle: {e}")
                
            await asyncio.sleep(config.SUPERVISOR_INTERVAL) 

if __name__ == "__main__":
    supervisor = ProactiveSupervisor()
    try:
        asyncio.run(supervisor.start())
    except KeyboardInterrupt:
        logger.info("Supervisor stopped by user.")
