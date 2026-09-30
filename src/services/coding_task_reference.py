"""Small task references shared by live runs and persisted chat history."""

import json
import re

_UUID = re.compile(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", re.I)


def coding_task_reference(content: object) -> dict | None:
    """Parse display metadata only; the task API still enforces ownership."""
    if not isinstance(content, str) or len(content) > 3000:
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or value.get("type") != "coding_task":
        return None
    if not isinstance(value.get("title"), str):
        return None
    for field in ("job_id", "app_id"):
        if not isinstance(value.get(field), str) or not _UUID.fullmatch(value[field]):
            return None
    return {"type": "coding_task", "job_id": value["job_id"],
            "app_id": value["app_id"], "title": value["title"][:80]}
