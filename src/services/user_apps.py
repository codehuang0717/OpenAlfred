"""Validate and construct the allowed generated UI catalog entries."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.event_bus import EventType, event_bus
from db.user_apps import save_user_app
from utils.auth_utils import require_explicit_user_id


class TodoTimelineOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=40)
    description: str = Field(default="按时间查看待办", max_length=100)
    date_field: Literal["scheduled_start_at", "expected_completion_at"] = "scheduled_start_at"
    include_completed: bool = False
    accent: Literal["emerald", "blue", "violet", "amber"] = "emerald"


async def create_todo_timeline(user_id: str, options: TodoTimelineOptions) -> dict:
    user_id = require_explicit_user_id(user_id)
    spec = {
        "root": "timeline",
        "elements": {
            "timeline": {
                "type": "TodoTimeline",
                "props": options.model_dump(),
                "children": [],
            }
        },
    }
    app = await save_user_app(user_id, "todo_timeline", options.title, spec)
    await event_bus.publish(EventType.USER_APP_UPDATED, {"id": app["id"], "user_id": user_id})
    return app
