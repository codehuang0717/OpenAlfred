import asyncio
import logging
import sys
import os
import time
from contextlib import suppress
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field

# Ensure UTF-8 output for Windows Console
if sys.platform == "win32" and __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database import (
    init_db,
    get_all_todos,
    get_user_by_id,
    get_supervisor_state,
    update_supervisor_state,
    reset_supervisor_state,
    AUDIO_CACHE_DIR
)
from services.tts import save_tts_to_file
from tools.eye import get_recent_ocr_text
from utils.time_utils import utc_to_local, parse_to_aware_utc
from services.user_time import get_user_timezone
from tools.call_user import dial_user
from services.llm import get_model
from logic.prompts import SUPERVISOR_PROMPT
from langchain_core.messages import HumanMessage
from core.config import config
from services.screen_monitor import ScreenRecorder, read_preferences, require_screen_owner, monitor_status, device_lock
from services.screen_binding import configured_owner
from rich.console import Console
from rich.panel import Panel

console = Console()

from utils.logger import setup_logging, get_logger

# Initialize unified logging
setup_logging(log_file="supervisor.log")
logger = get_logger("supervisor")

class SupervisorDecision(BaseModel):
    """Schema for supervisor decision logic."""
    status: Literal["NORMAL", "GENTLE_REMINDER", "STRICT_WARNING", "SEVERE_DISCIPLINE"] = Field(
        description="The distraction status of the user."
    )
    reason: str = Field(description="A brief explanation of why this status was chosen.")
    call_greeting: str = Field(description="The opening speech for the phone call if status is not NORMAL.")

class ProactiveSupervisor:
    def __init__(self):
        self.model = None
        self.user_id = configured_owner()

    async def run_cycle(self):
        prefs = await read_preferences(self.user_id)
        if not prefs["recording_enabled"] or not prefs["smart_supervision_enabled"]:
            return

        logger.info("--- Starting Smart Supervision Cycle ---")
        
        # 1. Fetch Active User
        active_user = await get_user_by_id(self.user_id)
        if not active_user:
            raise RuntimeError("绑定的屏幕监控账号不存在")
            
        user_id = active_user['id']
        user_timezone = await get_user_timezone(user_id)
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
                raise ValueError(f"Invalid scheduled_start_at for todo {t['id']}") from e

        # Display Monitoring Context
        active_display = "\n".join([f"[blue]•[/blue] {t['title']}" for t in active_todos]) if active_todos else "[italic grey]None (Idle)[/italic grey]"
        scheduled_display = "\n".join([f"[grey]• {t['title']} (Starts: {utc_to_local(t['scheduled_start_at'], user_timezone)})[/grey]" for t in scheduled_todos])
        
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

        ocr_context = await get_recent_ocr_text(user_id=self.user_id, minutes=config.SUPERVISOR_OCR_WINDOW_MINS)
        
        # 4. Calculate Distraction Duration
        # now is already defined above
        distraction_duration = 0
        if state.get('is_distracted') and state.get('distraction_start_time'):
            try:
                start_time = datetime.fromisoformat(state['distraction_start_time'].replace('Z', '+00:00'))
                distraction_duration = int((now - start_time).total_seconds() / 60)
            except ValueError as exc:
                raise ValueError("Invalid distraction_start_time") from exc
        
        # 5. Analyze with LLM
        tasks_list = []
        for t in active_todos:
            t_str = f"- {t['title']}: {t['description']}"
            if t.get('scheduled_start_at'):
                t_str += f" (Scheduled Start: {utc_to_local(t['scheduled_start_at'], user_timezone)})"
            if t.get('expected_completion_at'):
                t_str += f" (Deadline: {utc_to_local(t['expected_completion_at'], user_timezone)})"
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
            if self.model is None:
                self.model = get_model("gpt-cloud").with_structured_output(SupervisorDecision)
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
                prefs = await read_preferences(self.user_id)
                if not prefs["recording_enabled"] or not prefs["smart_supervision_enabled"]:
                    return
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
                    await save_tts_to_file(decision.call_greeting, wav_path, user_id=user_id)
                    prefs = await read_preferences(self.user_id)
                    if not prefs["recording_enabled"] or not prefs["smart_supervision_enabled"]:
                        return
                    
                    call_status = await dial_user(
                        phone_number="",
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
            raise

    async def start(self):
        await init_db()
        from services.screenpipe_models import model_directory, verify_models
        try:
            logger.info("Supervisor startup: pid=%s models=%s", os.getpid(), model_directory())
            await asyncio.to_thread(verify_models)
        except (RuntimeError, OSError) as exc:
            logger.warning("Screenpipe startup model check: %s", exc)
        while self.user_id is None:
            logger.info("本机尚未绑定账号，等待用户在主管设置中确认；不会采集屏幕")
            await asyncio.sleep(3)
            self.user_id = configured_owner()
        if not await get_user_by_id(self.user_id):
            raise RuntimeError("本机绑定的账号不存在")
        recorder = ScreenRecorder(self.user_id)
        analysis = None
        next_analysis = 0.0
        analysis_error = None
        recording_error = None
        try:
            while True:
                error = analysis_error
                try:
                    prefs = await read_preferences(self.user_id)
                    if not await get_user_by_id(self.user_id):
                        raise RuntimeError("绑定的屏幕监控账号已不存在")
                    if not prefs["recording_enabled"] or not prefs["smart_supervision_enabled"]:
                        if analysis:
                            analysis.cancel()
                            with suppress(asyncio.CancelledError, Exception):
                                await analysis
                            analysis = None
                        next_analysis = 0.0
                        analysis_error = error = None
                    if prefs["recording_enabled"]:
                        if recording_error is None:
                            await recorder.start()
                        else:
                            error = recording_error
                    else:
                        await recorder.stop()
                        recording_error = None
                    await recorder.heartbeat(error=error, analysis=analysis is not None and not analysis.done())
                    if analysis and analysis.done():
                        try:
                            analysis.result()
                            analysis_error = None
                        except Exception as exc:
                            analysis_error = str(exc) or type(exc).__name__
                            logger.exception("Screen supervision failed")
                        error = analysis_error
                        analysis = None
                    if recording_error is None and prefs["smart_supervision_enabled"] and analysis is None and time.monotonic() >= next_analysis:
                        status = await monitor_status(self.user_id)
                        if status["screenpipe_running"]:
                            analysis = asyncio.create_task(self.run_cycle())
                            next_analysis = time.monotonic() + config.SUPERVISOR_INTERVAL
                except Exception as exc:
                    error = str(exc) or type(exc).__name__
                    recording_error = error + "；排除故障后关闭再开启记录以重试"
                    error = recording_error
                    logger.exception("Screen monitor failed")
                    if analysis:
                        analysis.cancel()
                        with suppress(asyncio.CancelledError, Exception):
                            await analysis
                        analysis = None
                    await recorder.stop()
                await recorder.heartbeat(error=error, analysis=analysis is not None and not analysis.done())
                await asyncio.sleep(3)
        finally:
            if analysis:
                analysis.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await analysis
            await recorder.stop()
            await recorder.heartbeat(error="Supervisor 已停止", stopped=True)

if __name__ == "__main__":
    supervisor = ProactiveSupervisor()
    try:
        with device_lock():
            asyncio.run(supervisor.start())
    except KeyboardInterrupt:
        logger.info("Supervisor stopped by user.")
