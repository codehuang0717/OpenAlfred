"""Agent tool for creating a personalized right-side panel."""

import json
from typing import Literal

from langchain.tools import ToolRuntime, tool

from services.user_apps import TodoTimelineOptions, create_todo_timeline
from services.code_apps import CodeAppRequest, create_code_app
from db.coding_jobs import get_job, stop_job
from utils.auth_utils import require_runtime_user_id


@tool
async def create_todo_timeline_panel(
    runtime: ToolRuntime,
    title: str,
    description: str = "按时间查看待办",
    date_field: Literal["scheduled_start_at", "expected_completion_at"] = "scheduled_start_at",
    include_completed: bool = False,
    accent: Literal["emerald", "blue", "violet", "amber"] = "emerald",
) -> str:
    """Create or revise the user's personal todo timeline in the right-side toolbox.

    Use only when the user asks to create, customize, or change a todo timeline
    panel. Choose the title, explanatory text, date ordering, completed-item
    visibility, and accent from the user's request. This tool does not create todos.
    """
    user_id = require_runtime_user_id(runtime)
    options = TodoTimelineOptions(
        title=title.strip(),
        description=description.strip(),
        date_field=date_field,
        include_completed=include_completed,
        accent=accent,
    )
    app = await create_todo_timeline(user_id, options)
    return f"已在右侧工具箱创建「{app['title']}」时间线，可立即打开查看。"


@tool
async def create_standalone_mini_app(
    runtime: ToolRuntime, title: str, requirements: str,
) -> str:
    """Write a standalone mini-app for the user's right-side toolbox.

    Submit a background Coding subagent for calculators, dashboards, games,
    or custom visualizations. The coder can read this user's todos, emails,
    memory and knowledge through authenticated read-only tools when needed.
    Generated apps use data snapshots, never live account access or credentials.
    Returns a job ID immediately; do not poll in a loop or claim it is ready.
    For the existing standard live todo timeline, use create_todo_timeline_panel.
    """
    user_id = require_runtime_user_id(runtime)
    configurable = runtime.config.get("configurable", {})
    selection = configurable.get("model_selection")
    if not isinstance(selection, str) or not selection:
        raise ValueError("未指定小程序生成模型；请先在聊天界面选择模型")
    request = CodeAppRequest(title=title.strip(), prompt=requirements.strip())
    result = await create_code_app(user_id, request, selection, run_config=runtime.config)
    return json.dumps({"type": "coding_task", "title": request.title, **result,
                       "message": "已提交后台编码；进度和最终报告会更新，完成后请预览并手动发布。"}, ensure_ascii=False)


@tool
async def get_coding_task(runtime: ToolRuntime, job_id: str) -> str:
    """Check an owned coding task when the user asks about progress; never poll in a loop."""
    job = await get_job(require_runtime_user_id(runtime), job_id)
    if job is None:
        raise ValueError("编码任务不存在或不属于当前用户")
    return json.dumps({key: job[key] for key in ("status", "stage", "report", "error", "app_id")}, ensure_ascii=False)


@tool
async def cancel_coding_task(runtime: ToolRuntime, job_id: str) -> str:
    """Cancel an owned coding task only when the user requests cancellation."""
    owner = require_runtime_user_id(runtime)
    if await get_job(owner, job_id) is None:
        raise ValueError("编码任务不存在或不属于当前用户")
    if not await stop_job(owner, job_id):
        raise ValueError("编码任务已经结束，不能取消")
    return "已取消编码任务，没有发布代码。"


user_app_tools = [create_todo_timeline_panel, create_standalone_mini_app, get_coding_task, cancel_coding_task]
