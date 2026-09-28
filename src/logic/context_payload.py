"""Canonical payloads and stable references into the untouched graph history."""

import hashlib
import json
from langchain_core.messages import AIMessage, ToolMessage


def payload(message) -> dict:
    value = {"role": message.type, "content": message.content}
    if isinstance(message, AIMessage) and message.tool_calls:
        value["tool_calls"] = message.tool_calls
    if isinstance(message, ToolMessage):
        value.update(tool_call_id=message.tool_call_id, status=message.status, name=message.name)
    return value


def serialize(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def reference(index: int, message) -> str:
    digest = hashlib.sha256(serialize(payload(message)).encode()).hexdigest()[:16]
    return f"ctx:{index}:{digest}"


def history_hash(messages: list, count: int) -> str:
    return hashlib.sha256(serialize([payload(m) for m in messages[:count]]).encode()).hexdigest()
