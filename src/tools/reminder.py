from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
import httpx
from typing import Optional, Literal
import uuid
from datetime import datetime, timezone
import os
import wave
import asyncio
from zoneinfo import ZoneInfo
from core.config import config
from services.tts import save_tts_to_file

# Import DB and utils functions
from utils.time_utils import localize_to_utc
from core.database import (
    add_reminder as db_add_reminder,
    get_all_reminders,
    delete_reminder as db_delete_reminder,
    update_reminder as db_update_reminder,
    AUDIO_CACHE_DIR,
)

def save_wav_blocking(path: str, data: bytes):
    """保存原始 PCM 为 WAV"""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(data)

async def pre_render_tts(text: str, filename: str) -> str:
    """非阻塞预渲染 TTS，返回绝对路径"""
    try:
        print(f"[TTS] Generating TTS for: {text[:50]}...")
        out_path = os.path.join(AUDIO_CACHE_DIR, filename)
        
        await save_tts_to_file(text, out_path)
        
        if os.path.exists(out_path):
            file_size = os.path.getsize(out_path)
            print(f"[TTS] SUCCESS: Audio saved ({file_size} bytes): {out_path}")
            return out_path
        else:
            print("[TTS] ERROR: File was not created after save")
            return ""
    except Exception as e:
        import traceback
        print(f"[TTS] ERROR: 预渲染失败: {e}")
        traceback.print_exc()
        return ""




def _get_user_id(runtime: ToolRuntime) -> str:
    """Extract user_id from RunnableConfig populated by LangGraph Auth."""
    if hasattr(runtime, "config") and runtime.config:
        conf = runtime.config.get("configurable", {})
        auth_user = conf.get("langgraph_auth_user", {})
        if isinstance(auth_user, dict) and "identity" in auth_user:
            return auth_user["identity"]
            
        metadata = runtime.config.get("metadata", {})
        if "owner" in metadata:
            return metadata["owner"]
            
        if "thread_owner" in conf:
            return conf["thread_owner"]
    if hasattr(runtime, "state") and runtime.state:
        if isinstance(runtime.state, dict): return runtime.state.get("user_id", "default")
        return getattr(runtime.state, "user_id", "default")
    return "default"

@tool
async def add_reminder(
    runtime: ToolRuntime,
    body: str,
    scheduled_at: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    level: str = "active",
    sound: Optional[str] = None,
    delivery_method: Literal["push", "call"] = "push",
    call_greeting: Optional[str] = None,
) -> Command:
    """Set a timed reminder ONLY when the user explicitly requests to be notified at a specific time. DO NOT call this tool for general conversational tasks or follow-ups unless a time is mentioned."""
    try:
        user_id = _get_user_id(runtime)
        
        # Semantic Integrity Check: If the user didn't mention time-related words in the last message, 
        # but the LLM is trying to add a reminder, it's likely a hallucination.
        last_msg = ""
        if hasattr(runtime, "config") and "configurable" in runtime.config:
             # We can't easily access full history here without more plumbing, 
             # but we can at least check if 'scheduled_at' is too 'generic' (like exactly now or a fixed offset)
             pass

        reminder_id = str(uuid.uuid4())
        
        # 1. 严格使用统一的本地化逻辑解析时间
        try:
            final_time_utc = localize_to_utc(scheduled_at)
            if not final_time_utc:
                raise ValueError("Scheduled time cannot be empty")
        except Exception as e:
            return Command(update={"messages": [ToolMessage(content=f"ERROR: {str(e)}", tool_call_id=runtime.tool_call_id)]})

        audio_path = ""
        # 即使是普通提醒，我们也尝试预渲染，因为这样能确保调用 call_user 时的语音是“生成的”而不是默认音频文件
        if call_greeting:
            filename = f"reminder_{reminder_id}.wav"
            audio_path = await pre_render_tts(call_greeting, filename)
        elif delivery_method == "call":
            # 如果是电话提醒但没有特定话术，至少使用 body 作为话术
            filename = f"reminder_{reminder_id}.wav"
            audio_path = await pre_render_tts(body, filename)

        await db_add_reminder(
            id=reminder_id,
            body=body,
            scheduled_at=final_time_utc,
            title=title,
            subtitle=subtitle,
            level=level,
            sound=sound,
            delivery_method=delivery_method,
            audio_path=audio_path,
            user_id=user_id,
        )
        
        return Command(
            update={
                "reminders": await get_all_reminders(user_id=user_id),
                "messages": [
                    ToolMessage(
                        content=f"SUCCESS: Reminder set for {scheduled_at}. ID: {reminder_id[:8]}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    except Exception as e:
        return Command(update={"messages": [ToolMessage(content=f"ERROR: {str(e)}", tool_call_id=runtime.tool_call_id)]})

@tool
async def list_reminders(
    runtime: ToolRuntime,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    """List scheduled reminders, optionally filtered by date range.
    
    Args:
        date_from: Start of date range (local time, e.g. '2026-05-01T00:00:00'). Only reminders scheduled on or after this time are returned.
        date_to: End of date range (local time, e.g. '2026-05-01T23:59:59'). Only reminders scheduled on or before this time are returned.
    
    When the user asks for a specific day's reminders (e.g. "tomorrow"), pass both date_from and date_to.
    """
    try:
        from utils.time_utils import utc_to_local, parse_to_aware_utc
        
        user_id = _get_user_id(runtime)
        reminders = await get_all_reminders(user_id=user_id)
        
        # Apply date range filter
        if date_from or date_to:
            utc_from = None
            utc_to = None
            if date_from:
                try:
                    utc_from = parse_to_aware_utc(localize_to_utc(date_from))
                except Exception:
                    pass
            if date_to:
                try:
                    utc_to = parse_to_aware_utc(localize_to_utc(date_to))
                except Exception:
                    pass
            
            filtered = []
            for r in reminders:
                scheduled = r.get('scheduled_at', '')
                if not scheduled:
                    continue
                try:
                    r_dt = parse_to_aware_utc(scheduled)
                    if utc_from and r_dt < utc_from:
                        continue
                    if utc_to and r_dt > utc_to:
                        continue
                    filtered.append(r)
                except Exception:
                    continue  # Skip unparseable entries
            reminders = filtered
        
        if not reminders:
            return "当前没有任何提醒任务。"
        
        # Separate into upcoming (unsent) and past (sent)
        upcoming = []
        past = []
        for r in reminders:
            if r['sent']:
                past.append(r)
            else:
                upcoming.append(r)
        
        # Sort upcoming by scheduled_at ascending (nearest first)
        def parse_scheduled(r):
            try:
                return parse_to_aware_utc(r.get('scheduled_at', ''))
            except Exception:
                return datetime.max.replace(tzinfo=timezone.utc)
        
        upcoming.sort(key=parse_scheduled)
        
        res = ""
        if upcoming:
            res += f"📋 待触发提醒 ({len(upcoming)}条):\n"
            for i, r in enumerate(upcoming):
                method = "📞电话" if r['delivery_method'] == "call" else "📱推送"
                local_time = utc_to_local(r.get('scheduled_at', ''))
                marker = "👉 [下一个] " if i == 0 else ""
                res += f"{marker}🔔 [{r['id'][:8]}] {local_time}: {r['body']} ({method})\n"
        else:
            res += "当前没有待触发的提醒。\n"
        
        if past:
            res += f"\n✅ 已完成提醒 ({len(past)}条):\n"
            for r in past:
                method = "📞电话" if r['delivery_method'] == "call" else "📱推送"
                local_time = utc_to_local(r.get('scheduled_at', ''))
                res += f"✅ [{r['id'][:8]}] {local_time}: {r['body']} ({method})\n"
        
        return res
    except Exception as e:
        return f"获取列表失败: {str(e)}"

@tool
async def update_reminder(
    runtime: ToolRuntime,
    id: str,
    scheduled_at: Optional[str] = None,
    title: Optional[str] = None,
    body: Optional[str] = None,
) -> Command:
    """Update an existing reminder by ID (or first 8 chars of ID)."""
    try:
        user_id = _get_user_id(runtime)
        # 支持短 ID 匹配
        if len(id) == 8:
            all_r = await get_all_reminders(user_id=user_id)
            matches = [r for r in all_r if r['id'].startswith(id)]
            if not matches: 
                return Command(update={"messages": [ToolMessage(content="ERROR: 未找到匹配的提醒。", tool_call_id=runtime.tool_call_id)]})
            id = matches[0]['id']

        # 如果修改时间，需要解析
        final_time_utc = None
        if scheduled_at:
            final_time_utc = localize_to_utc(scheduled_at)

        await db_update_reminder(
            id=id,
            user_id=user_id,
            scheduled_at=final_time_utc,
            title=title,
            body=body
        )

        return Command(
            update={
                "reminders": await get_all_reminders(user_id=user_id),
                "messages": [
                    ToolMessage(
                        content=f"已成功更新提醒 [{id[:8]}]。",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    except Exception as e:
        return Command(update={"messages": [ToolMessage(content=f"更新失败: {str(e)}", tool_call_id=runtime.tool_call_id)]})

@tool
async def cancel_reminder(runtime: ToolRuntime, id: str) -> Command:
    """Cancel a pending reminder by ID (or first 8 chars of ID)."""
    try:
        user_id = _get_user_id(runtime)
        # 支持短 ID 匹配
        if len(id) == 8:
            all_r = await get_all_reminders(user_id=user_id)
            matches = [r for r in all_r if r['id'].startswith(id)]
            if not matches: 
                return Command(update={"messages": [ToolMessage(content="ERROR: 未找到匹配的提醒。", tool_call_id=runtime.tool_call_id)]})
            id = matches[0]['id']
            
        await db_delete_reminder(id, user_id=user_id)
        return Command(
            update={
                "reminders": await get_all_reminders(user_id=user_id),
                "messages": [
                    ToolMessage(
                        content=f"已成功取消提醒 [{id[:8]}]。",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    except Exception as e:
        return Command(update={"messages": [ToolMessage(content=f"取消失败: {str(e)}", tool_call_id=runtime.tool_call_id)]})



reminder_tools = [
    add_reminder,
    list_reminders,
    update_reminder,
    cancel_reminder,
]
