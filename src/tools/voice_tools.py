"""
Voice-specific tools for the hand-crafted voice Agent graph.

These use standard @tool with plain string returns (no ToolRuntime/Command),
compatible with LangGraph's built-in ToolNode for parallel execution.
"""

from langchain.tools import tool
from typing import Optional, Literal
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from database import (
    get_all_todos,
    add_todo as db_add_todo,
    add_reminder as db_add_reminder,
    get_all_reminders,
    delete_reminder,
)
from config import config


# ─── Todo Tools ───────────────────────────────────────────────────────────────

@tool
async def voice_get_todos() -> str:
    """获取所有当前任务列表。"""
    todos = await get_all_todos()
    if not todos:
        return "当前没有任务。"
    result = []
    for t in todos:
        status = "✅" if t["status"] == "completed" else "⏳"
        result.append(f"{t['emoji']} {t['title']} {status}")
    return "\n".join(result)


@tool
async def voice_add_todo(
    title: str,
    description: str = "",
    emoji: str = "🎯",
) -> str:
    """
    添加一个新任务。

    Args:
        title: 任务标题
        description: 任务描述
        emoji: 任务表情符号
    """
    todo_id = str(uuid.uuid4())
    await db_add_todo(
        id=todo_id,
        title=title,
        description=description,
        emoji=emoji,
    )
    return f"已添加任务: {title}"


# ─── Reminder Tools ───────────────────────────────────────────────────────────

@tool
async def voice_add_reminder(
    body: str,
    scheduled_at: str,
    delivery_method: Literal["push", "call"] = "push",
    call_greeting: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    level: str = "active",
    sound: Optional[str] = None,
) -> str:
    """
    设置提醒。
    
    Args:
        body: 提醒内容
        scheduled_at: 绝对 ISO 8601 时间字符串 (必须)。基于当前时间计算。
        delivery_method: "push" 推送通知 或 "call" 电话提醒
        call_greeting: 电话提醒时需要播放的问候语内容
        title: 提醒标题 (可选)
        subtitle: 提醒副标题 (可选)
        level: 推送优先级
        sound: 推送声音
    """
    from tools.reminder import pre_render_tts

    try:
        reminder_id = str(uuid.uuid4())
        
        # 解析 ISO 时间
        try:
            dt_with_tz = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
        except ValueError:
            dt_naive = datetime.fromisoformat(scheduled_at)
            dt_with_tz = dt_naive.replace(tzinfo=ZoneInfo("Europe/London"))

        dt_utc = dt_with_tz.astimezone(ZoneInfo("UTC"))
        final_time_utc = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        audio_path = ""
        if delivery_method == "call" and call_greeting:
            filename = f"reminder_{reminder_id}.wav"
            audio_path = await pre_render_tts(call_greeting, filename)

        await db_add_reminder(
            id=reminder_id,
            body=body,
            scheduled_at=final_time_utc,
            delivery_method=delivery_method,
            audio_path=audio_path,
            title=title,
            subtitle=subtitle,
            level=level,
            sound=sound,
        )
        return f"提醒已设置: {body}，时间 {final_time_utc}"
    except Exception as e:
        return f"设置提醒失败: {str(e)}"


@tool
async def voice_list_reminders() -> str:
    """列出所有已设置的提醒。"""
    try:
        reminders = await get_all_reminders()
        if not reminders:
            return "当前没有提醒。"
        result = []
        for r in reminders:
            method = "📞电话" if r["delivery_method"] == "call" else "📱推送"
            sent = "已发送" if r["sent"] else "待发送"
            result.append(f"- {r['scheduled_at']}: {r['body']} [{method}] [{sent}]")
        return "\n".join(result)
    except Exception as e:
        return f"获取提醒列表失败: {str(e)}"


@tool
async def voice_cancel_reminder(id: str) -> str:
    """
    取消一个提醒。

    Args:
        id: 提醒的ID
    """
    try:
        await delete_reminder(id)
        return "提醒已取消。"
    except Exception as e:
        return f"取消失败: {str(e)}"


# ─── Memory Tools ─────────────────────────────────────────────────────────────

@tool
def voice_search_memory(query: str) -> str:
    """
    搜索用户的记忆和偏好。

    Args:
        query: 搜索关键词
    """
    from mem0 import MemoryClient
    try:
        from mem0 import MemoryClient
        client = MemoryClient(api_key=config.MEM0_API_KEY)
        results = client.search(query=query, filters={"AND": [{"user_id": "default"}]})
        if results and isinstance(results, dict):
            results = results.get("results", results)
        if isinstance(results, list) and results:
            return "\n".join([r["memory"] for r in results])
        return "无相关记忆"
    except Exception as e:
        return f"搜索记忆失败: {str(e)}"


@tool
def voice_add_memory(content: str) -> str:
    """
    存储用户的信息到长期记忆。

    Args:
        content: 要记住的内容
    """
    from mem0 import MemoryClient

    try:
        client = MemoryClient(api_key=config.MEM0_API_KEY)
        messages = [{"role": "user", "content": content}]
        result = client.add(messages=messages, user_id="default")
        return f"已记住: {content}"
    except Exception as e:
        return f"存储记忆失败: {str(e)}"


# ─── Call Tools ───────────────────────────────────────────────────────────────

@tool
async def voice_make_outbound_call(phone_number: str = "100") -> str:
    """
    主动拨打电话给用户。当你需要紧急提醒用户时使用。

    Args:
        phone_number: 拨打的目标号码，默认为 "100"（分机号）。
    """
    from tools.call_user import generate_sip_token, OUTBOUND_TRUNK_ID, LIVEKIT_URL
    import httpx
    import time

    try:
        jwt_token = generate_sip_token()
        api_url = config.LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")
        if api_url.endswith("/"):
            api_url = api_url[:-1]
        url = f"{api_url}/twirp/livekit.SIP/CreateSIPParticipant"

        dial_data = {
            "sipTrunkId": OUTBOUND_TRUNK_ID,
            "sipCallTo": phone_number,
            "roomName": f"outbound-{int(time.time())}",
            "participantIdentity": "agent_caller",
            "participantName": "AI Assistant",
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers={"Authorization": f"Bearer {jwt_token}"}, json=dial_data, timeout=10.0)
            if resp.status_code == 200:
                return f"呼叫已发起，目标分机 {phone_number}。"
            else:
                return f"呼叫失败 (状态码: {resp.status_code})。"
    except Exception as e:
        return f"呼叫异常: {str(e)}"


# ─── All Voice Tools ─────────────────────────────────────────────────────────

voice_tool_list = [
    voice_get_todos,
    voice_add_todo,
    voice_add_reminder,
    voice_list_reminders,
    voice_cancel_reminder,
    voice_search_memory,
    voice_add_memory,
    voice_make_outbound_call,
]
