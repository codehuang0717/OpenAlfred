from langchain.tools import tool, ToolRuntime
import httpx
from typing import Optional, Literal
import uuid
from datetime import datetime, timezone
import os
import wave
import asyncio
from zoneinfo import ZoneInfo
from config import config
from tts import save_tts_to_file

# Import DB and utils functions
from utils.time_utils import localize_to_utc
from database import (
    add_reminder as db_add_reminder,
    get_pending_reminders,
    mark_reminder_sent,
    get_all_reminders,
    delete_reminder as db_delete_reminder,
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

async def _send_bark_notification(
    body: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    level: str = "active",
    sound: Optional[str] = None,
) -> str:
    """Internal function to send Bark notification."""
    if not config.BARK_URL:
        print("[Bark] ERROR: BARK_URL not set in config.")
        return "error: BARK_URL not set"
    
    try:
        import urllib.parse
        # Construct Bark URL: host/key/[title]/[subtitle]/body
        url_parts = [config.BARK_URL.rstrip("/")]
        if title: url_parts.append(urllib.parse.quote(title))
        if subtitle: url_parts.append(urllib.parse.quote(subtitle))
        url_parts.append(urllib.parse.quote(body))
        
        url = "/".join(url_parts)
        params = {"level": level}
        if sound:
            params["sound"] = sound

        print(f"[Bark] Sending request to: {url} with params {params}")
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)

        if response.status_code == 200:
            print("[Bark] SUCCESS: Notification sent.")
            return "success"
        else:
            print(f"[Bark] ERROR: {response.status_code} - {response.text}")
            return f"error: {response.text}"
    except Exception as e:
        print(f"[Bark] EXCEPTION: {e}")
        return f"error: {str(e)}"

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
) -> str:
    """
    设置一个定时提醒。
    
    Args:
        body: 提醒的内容 (必须)
        scheduled_at: 提醒的绝对时间字符串 (必须)。
                      基于系统提供的 "Current Time" 计算。
                      注意：务必提供用户所在的本地时间 (Europe/London)。
                      不要带 "Z" 或时区偏移。例如: "2026-04-08T15:30:00"
        title: 提醒标题 (可选)
        subtitle: 提醒副标题 (可选)
        level: 推送优先级 - "active" (默认), "critical", "timeSensitive", 或 "passive"
        sound: 推送声音名称 (可选)
        delivery_method: "push" (Bark推送) 或 "call" (电话呼叫通知)
        call_greeting: 如果是电话通知，播放的问候语内容
    """
    try:
        reminder_id = str(uuid.uuid4())
        
        # 1. 严格使用统一的本地化逻辑解析时间
        try:
            final_time_utc = localize_to_utc(scheduled_at)
            if not final_time_utc:
                raise ValueError("Scheduled time cannot be empty")
        except Exception as e:
            return f"ERROR: {str(e)}"

        audio_path = ""
        if delivery_method == "call" and call_greeting:
            filename = f"reminder_{reminder_id}.wav"
            audio_path = await pre_render_tts(call_greeting, filename)

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
            user_id=_get_user_id(runtime),
        )
        return f"SUCCESS: Reminder set for {scheduled_at} (UTC: {final_time_utc}). Method: {delivery_method}"
    except Exception as e:
        return f"ERROR: {str(e)}"

@tool
async def list_reminders(runtime: ToolRuntime) -> str:
    """列出所有已设置的提醒任务。"""
    try:
        reminders = await get_all_reminders(user_id=_get_user_id(runtime))
        if not reminders:
            return "当前没有任何提醒任务。"
        res = "📋 提醒任务列表:\n"
        for r in reminders:
            status = "🔔" if not r['sent'] else "✅"
            method = "📞电话" if r['delivery_method'] == "call" else "📱推送"
            res += f"{status} [{r['id'][:8]}] {r['scheduled_at']}: {r['body']} ({method})\n"
        return res
    except Exception as e:
        return f"获取列表失败: {str(e)}"

@tool
async def cancel_reminder(id: str) -> str:
    """取消一个还没发送的提醒任务。需要传入 ID (或 ID 的前8位)。"""
    try:
        # 支持短 ID 匹配
        if len(id) == 8:
            all_r = await get_all_reminders()
            matches = [r for r in all_r if r['id'].startswith(id)]
            if not matches: return "未找到匹配的提醒。"
            id = matches[0]['id']
            
        await db_delete_reminder(id)
        return f"已成功取消提醒 [{id[:8]}]。"
    except Exception as e:
        return f"取消失败: {str(e)}"

async def check_and_send_pending_reminders():
    """后台扫描并发送到期的提醒。支持推送和电话。"""
    from tools.call_user import generate_sip_token, OUTBOUND_TRUNK_ID
    
    try:
        pending = await get_pending_reminders()
        for r in pending:
            print(f"[Scheduler] Sending reminder: {r['body']} via {r['delivery_method']}")
            
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
                            "sipCallTo": "100", # 默认拨打 100
                            "roomName": room_name,
                        },
                        timeout=10.0,
                    )
            else:
                # Bark 推送
                await _send_bark_notification(
                    body=r['body'],
                    title=r.get('title'),
                    subtitle=r.get('subtitle'),
                    level=r.get('level', 'active'),
                    sound=r.get('sound')
                )
            
            await mark_reminder_sent(r["id"])
    except Exception as e:
        print(f"[Scheduler] ERROR: {e}")

reminder_tools = [
    add_reminder,
    list_reminders,
    cancel_reminder,
]
