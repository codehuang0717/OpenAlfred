import asyncio
import logging
import sys
import os
import time
import json
import psutil
import subprocess
from datetime import datetime, timezone
from typing import Literal, Optional, List
from pydantic import BaseModel, Field

# Ensure UTF-8 output for Windows Console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database import (
    init_db,
    get_all_todos,
    get_active_user,
    get_supervisor_state,
    update_supervisor_state,
    reset_supervisor_state,
    get_setting,
    set_setting,
    AUDIO_CACHE_DIR
)
from services.tts import save_tts_to_file
from tools.eye import get_recent_ocr_text
from utils.time_utils import utc_to_local, parse_to_aware_utc
from tools.call_user import dial_user
from services.llm import get_model
from logic.prompts import SUPERVISOR_PROMPT
from langchain_core.messages import HumanMessage
from core.config import config
from rich.console import Console
from rich.panel import Panel

console = Console()

from utils.logger import setup_logging, get_logger

# Initialize unified logging
setup_logging(log_file="supervisor.log")
logger = get_logger("supervisor")

def get_screenpipe_processes():
    procs = []
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] and proc.info['name'].lower() == 'screenpipe.exe':
                procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return procs

def get_screenpipe_process():
    procs = get_screenpipe_processes()
    return procs[0] if procs else None

def start_screenpipe() -> bool:
    if not get_screenpipe_process():
        logger.info("Starting screenpipe dynamically...")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "body", "windows_system", "eye", "setup_eye.ps1")
        subprocess.Popen(
            ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", script_path],
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return True # Indicates it was just started
    return False

def stop_screenpipe():
    procs = get_screenpipe_processes()
    if procs:
        logger.info(f"Stopping {len(procs)} screenpipe processes and their children dynamically to save resources...")
        for proc in procs:
            try:
                parent = proc.parent()
                for child in proc.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                proc.kill()
                if parent and parent.name().lower() == 'powershell.exe':
                    try:
                        parent.kill()
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                logger.error(f"Error stopping screenpipe: {e}")


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
        # 0. Check if Supervision/Recording is enabled in settings
        recording_str = await get_setting("recording_enabled", "true")
        smart_str = await get_setting("smart_supervision_enabled", "true")
        
        recording_enabled = (recording_str.lower() == "true")
        smart_supervision_enabled = (smart_str.lower() == "true")

        if not recording_enabled:
            logger.info("Recording is currently DISABLED. Skipping cycle and stopping Screenpipe.")
            stop_screenpipe()
            return
            
        # If recording is enabled, ensure screenpipe is running
        just_started = start_screenpipe()
        
        if not smart_supervision_enabled:
            logger.info("Smart Supervision is currently DISABLED. Screenpipe is running, but skipping LLM analysis.")
            return

        if just_started:
            logger.info("Screenpipe just started. Waiting 5 seconds for it to warm up before capturing context...")
            await asyncio.sleep(5)

        logger.info("--- Starting Smart Supervision Cycle ---")
        
        # 1. Fetch Active User
        active_user = await get_active_user()
        if not active_user:
            logger.warning("No active user found. Skipping cycle.")
            return
            
        user_id = active_user['id']
        username = active_user['username']
        
        logger.info(f"Targeting active user: {username} ({user_id}). Preparing to fetch context...")
        
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
                start_dt = parse_to_aware_utc(start_str)
                if start_dt <= now:
                    active_todos.append(t)
                else:
                    scheduled_todos.append(t)
            except Exception as e:
                logger.warning(f"Error parsing start time for todo {t['id']}: {e}")
                active_todos.append(t) # Default to active if parsing fails

        # Display Monitoring Context
        active_display = "\n".join([f"[blue]•[/blue] {t['title']}" for t in active_todos]) if active_todos else "[italic grey]None (Idle)[/italic grey]"
        scheduled_display = "\n".join([f"[grey]• {t['title']} (Starts: {utc_to_local(t['scheduled_start_at'])})[/grey]" for t in scheduled_todos])
        
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
                t_str += f" (Scheduled Start: {utc_to_local(t['scheduled_start_at'])})"
            if t.get('expected_completion_at'):
                t_str += f" (Deadline: {utc_to_local(t['expected_completion_at'])})"
            tasks_list.append(t_str)
            
        tasks_str = "\n".join(tasks_list)
        focus_task = pending_todos[0]['title'] 
        
        prompt = SUPERVISOR_PROMPT.format(
            tasks=tasks_str,
            focus_task=focus_task,
            ocr_context=ocr_context,
            distraction_duration=distraction_duration
        )
        
        logger.info("Context assembled. Requesting LLM analysis...")
        try:
            decision: SupervisorDecision = await self.model.ainvoke([HumanMessage(content=prompt)])
            
            logger.info(f"LLM analysis complete. Status: {decision.status}")
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
                    logger.info(f"Triggering alert for status: {decision.status}")
                    console.print(f"[bold red]!! Action Required !![/bold red] Triggering alert for status: [white on red]{decision.status}[/white on red]")
                    
                    # Pre-generate audio to reduce latency
                    supervisor_id = f"sup_{int(time.time())}"
                    wav_path = os.path.join(AUDIO_CACHE_DIR, f"supervisor_{supervisor_id}.wav")
                    logger.info(f"Pre-generating supervisor audio: {wav_path}")
                    await save_tts_to_file(decision.call_greeting, wav_path)
                    
                    call_status = await dial_user(
                        phone_number=config.SUPERVISOR_PHONE_NUMBER, 
                        initial_speech=decision.call_greeting,
                        user_id=user_id,
                        supervisor_id=supervisor_id
                    )
                    last_alert_time = now.isoformat()
                    logger.info(f"Call Sent: {call_status}")
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

    async def _listen_for_wakeups(self, wakeup_event: asyncio.Event):
        from core.event_bus import event_bus, EventType
        async for event in event_bus.subscribe(EventType.SUPERVISOR_WAKEUP.value):
            logger.info("Supervisor received wakeup event!")
            wakeup_event.set()

    async def start(self):
        await init_db()
        from core.event_bus import event_bus
        await event_bus.connect()
        logger.info(f"Supervisor Service Initialized. Interval: {config.SUPERVISOR_INTERVAL}s")
        
        wakeup_event = asyncio.Event()
        asyncio.create_task(self._listen_for_wakeups(wakeup_event))
        
        while True:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"Critical error in supervisor cycle: {e}")
                
            try:
                await asyncio.wait_for(wakeup_event.wait(), timeout=config.SUPERVISOR_INTERVAL)
                logger.info("Supervisor cycle triggered early by wakeup event.")
                wakeup_event.clear()
            except asyncio.TimeoutError:
                # Normal interval passed
                pass

if __name__ == "__main__":
    supervisor = ProactiveSupervisor()
    try:
        asyncio.run(supervisor.start())
    except KeyboardInterrupt:
        logger.info("Supervisor stopped by user.")
