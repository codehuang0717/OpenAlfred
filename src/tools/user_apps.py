"""Agent tool for creating a personalized right-side panel."""

from typing import Literal

from langchain.tools import ToolRuntime, tool

from services.user_apps import TodoTimelineOptions, create_todo_timeline
from services.code_apps import CodeAppRequest, create_code_app
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

    Use for calculators, timers, trackers, games, and other apps that do not
    need private OpenAlfred data. Generated code has no todo, mail, memory,
    account, network, or filesystem access. It needs user preview and publish.
    For a timeline of real personal todos, use create_todo_timeline_panel.
    """
    user_id = require_runtime_user_id(runtime)
    configurable = runtime.config.get("configurable", {})
    selection = configurable.get("model_selection")
    if not isinstance(selection, str) or not selection:
        raise ValueError("未指定小程序生成模型；请先在聊天界面选择模型")
    request = CodeAppRequest(title=title.strip(), prompt=requirements.strip())
    result = await create_code_app(user_id, request, selection)
    if result["status"] == "failed":
        return f"小程序生成失败：{result['error']}。没有发布任何代码。"
    return f"「{request.title}」代码草稿已生成。请在右侧工具箱预览，确认后点击发布。"


user_app_tools = [create_todo_timeline_panel, create_standalone_mini_app]
