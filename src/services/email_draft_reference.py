"""Public references contain no recipient addresses or message bodies."""
import json
import uuid


def email_draft_reference(content) -> dict | None:
    try:
        if isinstance(content, str) and len(content) > 4096:
            return None
        value = json.loads(content) if isinstance(content, str) else content
        if not isinstance(value, dict) or value.get("type") != "email_draft":
            return None
        identity = str(uuid.UUID(value["draft_id"]))
        return {"type": "email_draft", "draft_id": identity, "subject": str(value.get("subject") or "未命名草稿")[:998]}
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
