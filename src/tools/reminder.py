from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from typing import Optional, Literal
import uuid
import os
import wave
from services.tts import save_tts_to_file

# Import DB and utils functions
from utils.time_utils import localize_to_utc
from services.user_time import runtime_timezone
from services.tool_observations import observed, failed, fields, action, changes
from utils.time_utils import utc_to_local
from utils.auth_utils import require_runtime_user_id
from core.database import (
    add_reminder as db_add_reminder,
    get_all_reminders,
    get_reminder_by_id,
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

async def pre_render_tts(text: str, filename: str, user_id: str) -> str:
    """非阻塞预渲染 TTS，返回绝对路径"""
    try:
        print(f"[TTS] Generating TTS for: {text[:50]}...")
        out_path = os.path.join(AUDIO_CACHE_DIR, filename)
        
        await save_tts_to_file(text, out_path, user_id=user_id)
        
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
    """Extract a verified user_id from LangGraph request metadata."""
    return require_runtime_user_id(runtime)

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
            final_time_utc = localize_to_utc(scheduled_at, runtime_timezone(runtime))
            if not final_time_utc:
                raise ValueError("Scheduled time cannot be empty")
        except Exception as e:
            failed(
                None,
                "提醒时间无法解析，未创建提醒",
                details=fields(处理建议="请指定有效日期、时间及用户时区"),
            )
            return Command(update={"messages": [ToolMessage(content=f"ERROR: {str(e)}", tool_call_id=runtime.tool_call_id)]})

        audio_path = ""
        # 即使是普通提醒，我们也尝试预渲染，因为这样能确保调用 call_user 时的语音是“生成的”而不是默认音频文件
        if call_greeting:
            filename = f"reminder_{reminder_id}.wav"
            audio_path = await pre_render_tts(call_greeting, filename, user_id)
        elif delivery_method == "call":
            # 如果是电话提醒但没有特定话术，至少使用 body 作为话术
            filename = f"reminder_{reminder_id}.wav"
            audio_path = await pre_render_tts(body, filename, user_id)

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

        observed(
            None,
            f"已设置提醒：{title or body}",
            outcome="partial"
            if (call_greeting or delivery_method == "call") and not audio_path
            else "completed",
            details=fields(
                内容=body,
                触发时间=utc_to_local(final_time_utc, runtime_timezone(runtime)),
                时区=runtime_timezone(runtime),
                通知方式="电话" if delivery_method == "call" else "推送",
                语音预渲染="已准备" if audio_path else "未准备",
                说明="到时触发；此时尚未发送通知",
            ),
            actions=[action("panel", "查看提醒", "reminders", reminder_id)],
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
        failed(None, "设置提醒失败，未确认创建成功")
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
        user_timezone = runtime_timezone(runtime)
        
        # Apply date range filter
        if date_from or date_to:
            utc_from = None
            utc_to = None
            if date_from:
                utc_from = parse_to_aware_utc(localize_to_utc(date_from, user_timezone))
            if date_to:
                utc_to = parse_to_aware_utc(localize_to_utc(date_to, user_timezone))
            
            filtered = []
            for r in reminders:
                scheduled = r.get('scheduled_at', '')
                if not scheduled:
                    continue
                r_dt = parse_to_aware_utc(scheduled)
                if utc_from and r_dt < utc_from:
                    continue
                if utc_to and r_dt > utc_to:
                    continue
                filtered.append(r)
            reminders = filtered
        
        if not reminders:
            return observed(
                "当前没有任何提醒任务。",
                "所选范围内没有提醒",
                outcome="empty",
                actions=[action("panel", "查看提醒", "reminders")],
            )

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
            return parse_to_aware_utc(r.get('scheduled_at', ''))
        
        upcoming.sort(key=parse_scheduled)
        
        res = ""
        if upcoming:
            res += f"📋 待触发提醒 ({len(upcoming)}条):\n"
            for i, r in enumerate(upcoming):
                method = "📞电话" if r['delivery_method'] == "call" else "📱推送"
                local_time = utc_to_local(r.get('scheduled_at', ''), user_timezone)
                marker = "👉 [下一个] " if i == 0 else ""
                res += f"{marker}🔔 [{r['id'][:8]}] {local_time}: {r['body']} ({method})\n"
        else:
            res += "当前没有待触发的提醒。\n"
        
        if past:
            res += f"\n✅ 已完成提醒 ({len(past)}条):\n"
            for r in past:
                method = "📞电话" if r['delivery_method'] == "call" else "📱推送"
                local_time = utc_to_local(r.get('scheduled_at', ''), user_timezone)
                res += f"✅ [{r['id'][:8]}] {local_time}: {r['body']} ({method})\n"

        return observed(
            res,
            f"{len(upcoming)} 条待触发，{len(past)} 条已处理",
            details=fields(
                时区=user_timezone,
                提醒="\n".join(
                    f"{r.get('title') or r['body']} · {utc_to_local(r['scheduled_at'], user_timezone)} · {'已处理' if r['sent'] else '待触发'}"
                    for r in upcoming + past
                ),
            ),
            actions=[action("panel", "查看提醒", "reminders")],
        )
    except Exception as e:
        return failed(f"获取列表失败: {str(e)}", "查询提醒失败")


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
            if len(matches) != 1:
                failed(
                    None,
                    "提醒不存在或短 ID 匹配多个对象，本次未执行",
                    details=fields(处理建议="刷新提醒列表，重新选择具体提醒"),
                )
                raise ValueError("提醒不存在或短 ID 匹配不唯一，请使用完整 ID")
            id = matches[0]['id']

        before = await get_reminder_by_id(id, user_id=user_id)
        if not before:
            failed(None, "提醒不存在或不属于当前用户，本次未修改")
            raise ValueError("提醒不存在或不属于当前用户")
        # 如果修改时间，需要解析
        final_time_utc = None
        if scheduled_at:
            final_time_utc = localize_to_utc(scheduled_at, runtime_timezone(runtime))

        updated = await db_update_reminder(
            id=id, user_id=user_id, scheduled_at=final_time_utc, title=title, body=body
        )
        if updated is False:
            raise ValueError("提醒不存在或不属于当前用户")
        after = await get_reminder_by_id(id, user_id=user_id)
        delta = changes(
            before,
            after,
            {"title": "标题变更", "body": "内容变更", "scheduled_at": "触发时间变更"},
            user_timezone=runtime_timezone(runtime),
        )
        observed(
            None,
            f"已修改提醒：{after.get('title') or after['body']}"
            if delta
            else "提醒内容没有变化",
            outcome="completed" if delta else "no_change",
            details=delta + fields(时区=runtime_timezone(runtime)),
            actions=[action("panel", "查看提醒", "reminders", id)],
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
        failed(None, "修改提醒失败，未确认变更成功")
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
            if len(matches) != 1:
                failed(
                    None,
                    "提醒不存在或短 ID 匹配多个对象，本次未执行",
                    details=fields(处理建议="刷新提醒列表，重新选择具体提醒"),
                )
                raise ValueError("提醒不存在或短 ID 匹配不唯一，请使用完整 ID")
            id = matches[0]['id']

        before = await get_reminder_by_id(id, user_id=user_id)
        if not before or not await db_delete_reminder(id, user_id=user_id):
            failed(None, "提醒不存在或不属于当前用户，本次未取消")
            raise ValueError("提醒不存在或不属于当前用户")
        observed(
            None,
            f"已取消提醒：{before.get('title') or before['body']}",
            details=fields(
                原内容=before["body"],
                原触发时间=utc_to_local(
                    before["scheduled_at"], runtime_timezone(runtime)
                ),
                时区=runtime_timezone(runtime),
            ),
            actions=[action("panel", "查看提醒", "reminders")],
        )
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
        failed(None, "取消提醒失败，未确认取消成功")
        return Command(update={"messages": [ToolMessage(content=f"取消失败: {str(e)}", tool_call_id=runtime.tool_call_id)]})



reminder_tools = [
    add_reminder,
    list_reminders,
    update_reminder,
    cancel_reminder,
]
