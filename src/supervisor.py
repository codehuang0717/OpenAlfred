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
        now = datetime.now(timezone.utc)
        todos = await get_all_todos(user_id=user_id)
        pending_todos = [t for t in todos if t['status'] == 'pending']
        
        # Filter Active Tasks
        active_todos = []
        scheduled_todos = []
        
        for t in pending_todos:
            start_str = t.get('scheduled_start_at')
            if not start_str:
                active_todos.append(t)
                continue
            
            try:
                # Handle Z or +00:00 format
                start_dt = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                if start_dt <= now:
                    active_todos.append(t)
                else:
                    scheduled_todos.append(t)
            except Exception as e:
                logger.warning(f"Error parsing start time for todo {t['id']}: {e}")
                active_todos.append(t) # Default to active if parsing fails

        # Display Monitoring Context
        active_display = "\n".join([f"[blue]•[/blue] {t['title']}" for t in active_todos]) if active_todos else "[italic grey]None (Idle)[/italic grey]"
        scheduled_display = "\n".join([f"[grey]• {t['title']} (Starts: {t['scheduled_start_at']})[/grey]" for t in scheduled_todos])
        
        display_text = f"[bold cyan]User:[/bold cyan] {username}\n[bold cyan]Monitoring Tasks:[/bold cyan]\n{active_display}"
        if scheduled_todos:
            display_text += f"\n\n[bold yellow]Coming Up:[/bold yellow]\n{scheduled_display}"

        console.print(Panel(
            display_text,
            title="[supervisor] Current Context",
            border_style="blue"
        ))

        if not active_todos:
            logger.info(f"User {username} has no active tasks at this time. Monitoring status: Idle.")
            if state.get('is_distracted'):
                # Also reset distraction if user was distracted but now has no tasks to do
                await reset_supervisor_state(user_id)
            return

        ocr_context = await get_recent_ocr_text(minutes=config.SUPERVISOR_OCR_WINDOW_MINS)
        
        # 4. Calculate Distraction Duration
        # now is already defined above
        distraction_duration = 0
        if state.get('is_distracted') and state.get('distraction_start_time'):
            try:
                start_time = datetime.fromisoformat(state['distraction_start_time'].replace('Z', '+00:00'))
                distraction_duration = int((now - start_time).total_seconds() / 60)
            except:
                pass
        
        # 5. Analyze with LLM
        tasks_list = []
        for t in active_todos:
            t_str = f"- {t['title']}: {t['description']}"
            if t.get('scheduled_start_at'):
                t_str += f" (Scheduled Start: {t['scheduled_start_at']})"
            if t.get('expected_completion_at'):
                t_str += f" (Deadline: {t['expected_completion_at']})"
            tasks_list.append(t_str)
            
        tasks_str = "\n".join(tasks_list)
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
