from langchain.tools import tool
import httpx
from typing import Optional, Literal
import uuid
from datetime import datetime, timedelta, timezone
import os
import wave
import numpy as np
import asyncio
import re
import dotenv
from database import AUDIO_CACHE_DIR
from zoneinfo import ZoneInfo

dotenv.load_dotenv(override=True)


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
        import dashscope
        from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat

        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        voice_id = os.getenv("VOICE_ID")
        if not voice_id:
            print("[TTS] VOICE_ID not set")
            return ""

        print(f"[TTS] Generating TTS for: {text[:50]}...")
        print(f"[TTS] Using voice: {voice_id}")
        print(f"[TTS] Model: cosyvoice-v3.5-flash, Format: PCM_48000HZ_MONO_16BIT")

        synthesizer = SpeechSynthesizer(
            model="cosyvoice-v3.5-flash",
            voice=voice_id,
            format=AudioFormat.PCM_48000HZ_MONO_16BIT,
        )

        # Run blocking TTS call in separate thread
        audio_data = await asyncio.to_thread(synthesizer.call, text)

        print(f"[TTS] Response type: {type(audio_data)}")
        print(f"[TTS] Response is None: {audio_data is None}")
        if audio_data:
            print(
                f"[TTS] Response length: {len(audio_data) if hasattr(audio_data, '__len__') else 'N/A'}"
            )
            # Check if it's WAV format (starts with RIFF header)
            if len(audio_data) >= 4:
                print(f"[TTS] First 4 bytes (hex): {audio_data[:4].hex()}")

        if not audio_data:
            print("[TTS] ERROR: TTS returned empty audio")
            return ""

        out_path = os.path.join(AUDIO_CACHE_DIR, filename)
        print(f"[TTS] Saving to: {out_path}")

        # WAV format already has headers, write directly
        await asyncio.to_thread(save_wav_blocking, out_path, audio_data)

        if os.path.exists(out_path):
            file_size = os.path.getsize(out_path)
            print(f"[TTS] SUCCESS: Audio saved ({file_size} bytes): {out_path}")
            return out_path
        print("[TTS] ERROR: File was not created after save")
        return ""
    except Exception as e:
        import traceback

        print(f"[TTS] ERROR:预渲染失败: {e}")
        traceback.print_exc()
        return ""


def parse_relative_time_cn(time_str: str, base_dt: datetime) -> datetime:
    """解析中文相对时间字符串，返回基于base_dt的datetime对象"""
    # 处理半小时
    if time_str == "半小时":
        return base_dt + timedelta(minutes=30)
    # 处理一个小时、一小时
    if time_str in ["一个小时", "一小时"]:
        return base_dt + timedelta(hours=1)
    # 处理两个小时等
    hour_match = re.match(r"(\d+(?:\.\d+)?)个?小时", time_str)
    if hour_match:
        hours = float(hour_match.group(1))
        return base_dt + timedelta(hours=hours)
    # 处理分钟
    minute_match = re.match(r"(\d+(?:\.\d+)?)分钟", time_str)
    if minute_match:
        minutes = float(minute_match.group(1))
        return base_dt + timedelta(minutes=minutes)
    # 处理天
    day_match = re.match(r"(\d+(?:\.\d+)?)天", time_str)
    if day_match:
        days = float(day_match.group(1))
        return base_dt + timedelta(days=days)
    # 如果没有匹配，返回基础时间（不添加任何时间）
    return base_dt


def parse_absolute_time(time_str: str) -> Optional[datetime]:
    """尝试解析绝对时间字符串，返回datetime对象（无时区信息）"""
    # 尝试ISO格式
    try:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except ValueError:
        pass
    # 尝试YYYY-MM-DD HH:MM:SS
    try:
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass
    # 尝试YYYY-MM-DD HH:MM
    try:
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M")
    except ValueError:
        pass
    # 尝试HH:MM:SS
    try:
        return datetime.strptime(time_str, "%H:%M:%S")
    except ValueError:
        pass
    # 尝试HH:MM
    try:
        return datetime.strptime(time_str, "%H:%M")
    except ValueError:
        pass
    return None


@tool
async def add_reminder(
    body: str,
    scheduled_at: str,
    delivery_method: Literal["push", "call"] = "push",
    call_greeting: Optional[str] = None,
) -> str:
    """
    设置提醒。可直接传入相对时间（如"10秒"、"半小时"、"2小时"）或绝对时间（如"15:30"），会自动转换为英国时区的UTC时间。不需要先调用get_absolute_time_in_uk。
    Args:
        body: 提醒内容
        scheduled_at: 时间字符串，可直接传入相对时间（秒、分钟、小时后天等）或绝对时间（HH:MM、YYYY-MM-DDTHH:MM:SS），会自动解析
        delivery_method: "push" 推送通知 或 "call" 电话提醒
        call_greeting: 电话提醒时需要播放的问候语内容
    """
    from database import add_reminder as db_add_reminder

    try:
        reminder_id = str(uuid.uuid4())

        now_uk = datetime.now(ZoneInfo("Europe/London"))

        abs_dt = parse_absolute_time(scheduled_at)
        if abs_dt is not None:
            dt_uk = abs_dt.replace(tzinfo=ZoneInfo("Europe/London"))
        else:
            dt_uk = parse_relative_time_cn(scheduled_at, now_uk)
            if dt_uk.tzinfo is None:
                dt_uk = dt_uk.replace(tzinfo=ZoneInfo("Europe/London"))

        dt_utc = dt_uk.astimezone(ZoneInfo("UTC"))
        final_time_utc = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        audio_path = ""
        if delivery_method == "call" and call_greeting:
            print(
                f"[add_reminder] Generating audio for greeting: {call_greeting[:50]}..."
            )
            filename = f"reminder_{reminder_id}.wav"
            audio_path = await pre_render_tts(call_greeting, filename)
            print(f"[add_reminder] Audio path result: {audio_path}")

        await db_add_reminder(
            id=reminder_id,
            body=body,
            scheduled_at=final_time_utc,
            delivery_method=delivery_method,
            audio_path=audio_path,
        )
        return f"SUCCESS: Reminder set for {final_time_utc} (UTC). Audio: {'Created' if audio_path else 'None'}"
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
async def get_absolute_time_in_uk(time_str: str) -> str:
    """
    将相对或绝对时间字符串转换为英国时区的绝对时间字符串（带时区偏移）。
    Args:
        time_str: 时间字符串，可以是相对时间（如"10分钟"、"半小时"）或绝对时间（如"2026-03-12T15:30:00"、"15:30"）
    Returns:
        英国时区的绝对时间字符串，格式为ISO 8601带时区偏移（如"2026-03-12T15:30:00+00:00"或"2026-03-12T15:30:00+01:00"）
    """
    now_uk = datetime.now(ZoneInfo("Europe/London"))

    abs_dt = parse_absolute_time(time_str)
    if abs_dt is not None:
        dt_uk = abs_dt.replace(tzinfo=ZoneInfo("Europe/London"))
    else:
        dt_uk = parse_relative_time_cn(time_str, now_uk)
        if dt_uk.tzinfo is None:
            dt_uk = dt_uk.replace(tzinfo=ZoneInfo("Europe/London"))

    return dt_uk.isoformat()


@tool
async def list_reminders() -> str:
    """List all scheduled reminders."""
    from database import get_all_reminders

    try:
        reminders = await get_all_reminders()
        if not reminders:
            return "No reminders."
        res = "📋 Reminders:\n"
        for r in reminders:
            res += f"- {r['scheduled_at']}: {r['body']} [Method: {r['delivery_method']}] [Audio: {'Yes' if r['audio_path'] else 'No'}]\n"
        return res
    except Exception as e:
        return str(e)


@tool
async def cancel_reminder(id: str) -> str:
    """Cancel a reminder by ID."""
    from database import delete_reminder

    try:
        await delete_reminder(id)
        return "Cancelled."
    except Exception as e:
        return str(e)


async def check_and_send_pending_reminders():
    from database import get_pending_reminders, mark_reminder_sent
    from tools.call_user import generate_sip_token, OUTBOUND_TRUNK_ID, LIVEKIT_URL

    try:
        pending = await get_pending_reminders()
        for r in pending:
            if r.get("delivery_method") == "call":
                jwt_token = generate_sip_token()
                api_url = LIVEKIT_URL.replace("ws://", "http://").replace(
                    "wss://", "https://"
                )
                if api_url.endswith("/"):
                    api_url = api_url[:-1]
                url = f"{api_url}/twirp/livekit.SIP/CreateSIPParticipant"
                # 包含完整 UUID
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
                await mark_reminder_sent(r["id"])
            else:
                # 略过 Bark，只标记发送
                await mark_reminder_sent(r["id"])
    except Exception as e:
        pass


reminder_tools = [
    add_reminder,
    list_reminders,
    cancel_reminder,
    get_absolute_time_in_uk,
]
