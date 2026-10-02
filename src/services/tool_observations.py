"""Explicit business observations, never infer success from human prose."""

import asyncio
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from schemas.tool_display import ToolDisplay, ToolStatus

TOOL_LABELS = {
    "get_todos": "查询待办",
    "add_todo": "创建待办",
    "update_todo": "修改待办",
    "delete_todo": "删除待办",
    "add_reminder": "设置提醒",
    "list_reminders": "查询提醒",
    "update_reminder": "修改提醒",
    "cancel_reminder": "取消提醒",
    "get_email_accounts": "获取可用邮箱",
    "get_recent_emails": "查询最近邮件",
    "read_email": "读取邮件",
    "create_email_draft": "保存邮件草稿",
    "get_email_draft": "读取邮件草稿",
    "update_email_draft": "提出邮件修改建议",
    "get_user_profile": "读取个人资料与偏好",
    "get_user_memory_category": "读取分类记忆",
    "update_user_memory": "保存长期记忆",
    "web_search": "搜索网页",
    "search_knowledge": "检索知识库",
    "list_knowledge": "列出知识库文档",
    "get_weather": "查询天气",
    "read_context_excerpt": "回查本次会话",
    "view_screen": "查询屏幕活动",
    "search_screen_time": "查询时间段活动",
    "take_screenshot": "分析当前屏幕",
    "generate_image": "生成图片",
    "create_todo_timeline_panel": "配置待办时间线",
    "create_standalone_mini_app": "提交小程序生成",
    "get_coding_task": "查看生成进度",
    "cancel_coding_task": "取消小程序生成",
    "make_outbound_call": "拨打电话",
    "request_end_call": "请求结束通话",
    "list_files": "查看任务文件",
    "read_file": "读取任务文件",
    "write_file": "编写任务文件",
    "apply_patch": "修改任务文件",
    "validate_app": "检查小程序代码",
    "finish_task": "交付待预览版本",
}
READ_TOOLS = {
    "get_todos",
    "list_reminders",
    "get_email_accounts",
    "get_recent_emails",
    "read_email",
    "get_email_draft",
    "get_user_profile",
    "get_user_memory_category",
    "web_search",
    "search_knowledge",
    "list_knowledge",
    "get_weather",
    "read_context_excerpt",
    "view_screen",
    "search_screen_time",
    "take_screenshot",
    "get_coding_task",
    "list_files",
    "read_file",
    "validate_app",
}
CATEGORIES = {
    "profile": "个人资料",
    "preferences": "个人偏好",
    "relationship": "关系记录",
    "patterns": "行为模式",
}
INPUT_FIELDS = {
    "title": "标题",
    "query": "查询内容",
    "prompt": "图片描述",
    "requirements": "需求",
    "date_from": "范围开始",
    "date_to": "范围结束",
    "date": "请求日期",
    "location": "请求地点",
    "account_filter": "邮箱筛选",
    "limit": "请求上限",
    "top_k": "片段上限",
    "size": "请求尺寸",
    "start_time": "范围开始",
    "end_time": "范围结束",
    "scheduled_at": "提醒时间",
    "delivery_method": "通知方式",
    "app_name": "应用筛选",
    "content_type": "内容类型",
    "path": "任务文件",
    "start_line": "起始行",
    "line_count": "行数上限",
}
_current: ContextVar[dict | None] = ContextVar("tool_observation", default=None)


def clipped(value: Any, limit: int = 5000) -> str:
    text = str(value if value is not None else "未设置")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def fields(**values: Any) -> list[dict]:
    return [
        {"label": clipped(label, 80), "value": clipped(value)}
        for label, value in values.items()
        if value is not None
    ]


def action(kind: str, label: str, target: str, secondary: str | None = None) -> dict:
    return {"kind": kind, "label": label, "target": target, "secondary": secondary}


def observed(
    value: Any,
    summary: str,
    *,
    status: ToolStatus = "succeeded",
    outcome: str = "completed",
    details: list[dict] | None = None,
    actions: list[dict] | None = None,
) -> Any:
    """Return the existing model-facing value while recording verified UI fields."""
    context = _current.get()
    if context is not None:
        context["result"] = {
            "status": status,
            "outcome": outcome,
            "summary": clipped(summary, 240),
            "fields": [
                {
                    "label": clipped(item.get("label", "详情"), 80),
                    "value": clipped(item.get("value")),
                }
                for item in (details or [])[:40]
            ],
            "actions": [
                {
                    **item,
                    "label": clipped(item.get("label", "打开"), 80),
                    "target": clipped(item.get("target", ""), 2048),
                }
                for item in (actions or [])[:10]
            ],
        }
    return value


def failed(
    value: Any,
    summary: str = "执行失败，请检查输入或相关设置",
    *,
    details: list[dict] | None = None,
) -> Any:
    context = _current.get()
    previous = context.get("result") if context else None
    if previous and previous["status"] == "succeeded":
        # A committed write followed by a failed refresh is not a failed write.
        return observed(
            value,
            previous["summary"] + "；后续步骤失败，请刷新确认",
            outcome="partial",
            details=previous["fields"],
            actions=previous["actions"],
        )
    if previous and previous["status"] == "failed":
        return value
    return observed(value, summary, status="failed", details=details)


def bounded_fields(items: list[dict]) -> list[dict]:
    remaining = 12000
    result = []
    for item in items[:39]:
        value = item["value"]
        if len(value) > remaining:
            if remaining > 0:
                result.append({**item, "value": clipped(value, remaining)})
            result += fields(
                展示范围="详情过长，仅展示部分内容；请打开业务入口查看完整记录"
            )
            break
        result.append(item)
        remaining -= len(value)
    if len(items) > 39 and len(result) == 39:
        result += fields(展示范围="还有更多详情，请打开业务入口查看")
    return result


def changes(
    before: dict,
    after: dict,
    labels: dict[str, str],
    *,
    user_timezone: str | None = None,
) -> list[dict]:
    before, after = dict(before), dict(after)
    if user_timezone:
        from utils.time_utils import utc_to_local

        for row in (before, after):
            for key in ("scheduled_at", "scheduled_start_at", "expected_completion_at"):
                if row.get(key):
                    row[key] = utc_to_local(row[key], user_timezone)
    for row in (before, after):
        if row.get("status") in {"pending", "completed"}:
            row["status"] = "待完成" if row["status"] == "pending" else "已完成"
    return [
        {
            "label": label,
            "value": clipped(
                f"{clipped(before.get(key), 2000)} → {clipped(after.get(key), 2000)}"
            ),
        }
        for key, label in labels.items()
        if before.get(key) != after.get(key)
    ]


def input_display(name: str, args: dict) -> dict:
    args = args if isinstance(args, dict) else {}
    category = args.get("category") if isinstance(args.get("category"), str) else None
    details = fields(
        **{
            label: args[key]
            for key, label in INPUT_FIELDS.items()
            if key in args and args[key] not in (None, "")
        }
    )
    if "category" in args:
        details += fields(记忆类别=CATEGORIES.get(category, "未知类别"))
    # These arguments are not applied by the current screen implementation.
    if name == "view_screen":
        ignored = (
            {"应用筛选"}
            if args.get("mode", "current") == "history"
            else {"应用筛选", "查询内容", "范围开始", "范围结束"}
            if args.get("mode", "current") == "current"
            else set()
        )
        details = [item for item in details if item["label"] not in ignored]
        if args.get("mode", "current") == "current":
            details += fields(实际范围="最近 5 分钟活动记录")
    if name == "list_knowledge":
        details = []  # The current tool does not apply query filtering.
    title = TOOL_LABELS.get(name, "执行工具")
    target = (
        args.get("title")
        or args.get("subject")
        or args.get("query")
        or args.get("location")
        or CATEGORIES.get(category)
        or (
            "任务只读数据"
            if str(args.get("path", "")).startswith("data/")
            else args.get("path")
        )
    )
    summary = clipped(target, 160) if target else "正在执行"
    if name == "view_screen" and args.get("mode", "current") == "current":
        summary = "最近 5 分钟活动记录"
    if name == "list_knowledge":
        summary = "全部已上传文档"
    return ToolDisplay(
        title=title, status="running", summary=summary, fields=details
    ).model_dump()


def display_from_artifact(artifact: Any) -> dict | None:
    if not isinstance(artifact, dict) or not isinstance(
        artifact.get("tool_display"), dict
    ):
        return None
    try:
        return ToolDisplay.model_validate(artifact["tool_display"]).model_dump()
    except ValueError:
        return None


async def observe_tool_call(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command:
    """Execute once, keep artifacts intact and emit the same observation we persist."""
    call = request.tool_call
    display = input_display(call["name"], call.get("args", {}))
    display["started_at"] = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    writer = getattr(getattr(request, "runtime", None), "stream_writer", None)

    def emit(value: dict) -> None:
        if writer:
            writer(
                {
                    "type": "tool_display",
                    "id": call["id"],
                    "name": call["name"],
                    "display": value,
                }
            )

    context: dict = {}
    token = _current.set(context)
    emit(display)
    try:
        result = await execute(request)
        messages = (
            result.update.get("messages", [])
            if isinstance(result, Command) and isinstance(result.update, dict)
            else [result]
        )
        paired = [
            msg
            for msg in messages
            if isinstance(msg, ToolMessage) and msg.tool_call_id == call["id"]
        ]
        recorded = context.get("result")
        if recorded:
            display.update(recorded)
            # Keep sanitized input detail alongside explicit business output.
            display["fields"] = (
                input_display(call["name"], call.get("args", {}))["fields"]
                + recorded["fields"]
            )[:40]
        elif any(msg.status == "error" for msg in paired):
            display.update(
                status="failed", summary="执行失败，没有可信结果；请查看输入或相关设置"
            )
        else:
            display.update(status="unknown", summary="调用已返回，缺少可核验的业务结果")
        if (
            recorded
            and recorded["status"] == "succeeded"
            and call["name"] not in READ_TOOLS
            and recorded["outcome"] not in {"blocked", "empty", "exists", "no_change"}
            and any(msg.status == "error" for msg in paired)
        ):
            display.update(
                outcome="partial",
                summary=clipped(display["summary"] + "；后续步骤失败，请刷新确认", 240),
            )
        display["fields"] = bounded_fields(display["fields"])
        display.update(
            finished_at=datetime.now(timezone.utc).isoformat(),
            elapsed_ms=round((time.monotonic() - start) * 1000),
        )
        display = ToolDisplay.model_validate(display).model_dump()
        for msg in paired:
            msg.artifact = {
                **(msg.artifact if isinstance(msg.artifact, dict) else {}),
                "tool_display": display,
            }
            if display["status"] == "failed":
                msg.status = "error"
        emit(display)
        return result
    except asyncio.CancelledError:
        display.update(
            status="interrupted" if call["name"] in READ_TOOLS else "unknown",
            summary="执行中断，未确认完成；不会自动重试",
        )
        emit(display)
        raise
    finally:
        _current.reset(token)
