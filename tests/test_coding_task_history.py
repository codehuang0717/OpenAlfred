"""The HTTP history contract retains coding cards even after main-run errors."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.security import HTTPAuthorizationCredentials
from routers import threads
from services.coding_task_reference import coding_task_reference

TASK = {"type": "coding_task", "title": "计算器",
        "job_id": "12345678-1234-1234-1234-123456789abc",
        "app_id": "87654321-1234-1234-1234-123456789abc"}


class TestCodingTaskReference(unittest.TestCase):
    def test_reference_is_small_and_contains_no_source_or_credentials(self):
        self.assertEqual(coding_task_reference(json.dumps({**TASK, "html": "private"})), TASK)
        for content in [None, "[]", "null", '"text"', "{", "x" * 3001,
                        json.dumps({**TASK, "job_id": ["invalid"]}),
                        json.dumps({**TASK, "app_id": "../secret"})]:
            with self.subTest(content=str(content)[:50]):
                self.assertIsNone(coding_task_reference(content))


class TestCodingTaskHistory(unittest.IsolatedAsyncioTestCase):
    async def history(self, tool_message):
        messages = [
            {"type": "human", "id": "user", "content": "做个计算器"},
            {"type": "ai", "id": "plan", "content": "开始编码", "tool_calls": [
                {"name": "create_standalone_mini_app", "id": "create"}]},
            tool_message,
            {"type": "ai", "id": "failed", "content": "APIConnectionError",
             "additional_kwargs": {"agent_outcome": {"status": "failed"}, "agent_failure": "连接失败"}},
        ]
        original = json.dumps(messages)
        response = SimpleNamespace(status_code=200, json=lambda: {"values": {"messages": messages}})
        client = SimpleNamespace(get=AsyncMock(return_value=response))
        with patch.object(threads.httpx, "AsyncClient") as factory:
            factory.return_value.__aenter__.return_value = client
            result = await threads.get_thread_messages(
                "thread", HTTPAuthorizationCredentials(scheme="Bearer", credentials="synthetic"), {"id": "alice"})
        self.assertEqual(json.dumps(messages), original)
        return result[-1]

    async def test_card_survives_failed_final_reply_and_history_reload(self):
        message = {"type": "tool", "name": "create_standalone_mini_app", "tool_call_id": "create",
                   "content": json.dumps({**TASK, "status": "queued"})}
        result = await self.history(message)
        cards = [step for step in result["steps"] if step["type"] == "coding_task"]
        self.assertEqual(cards, [{"type": "coding_task", "id": f"coding-{TASK['job_id']}", "task": TASK}])
        self.assertEqual(result["outcome"], "failed")
        self.assertEqual(result["failure"], "连接失败")
        self.assertEqual(await self.history(message), result)

    async def test_failed_unpaired_or_private_data_results_cannot_create_cards(self):
        message = {"type": "tool", "name": "create_standalone_mini_app", "tool_call_id": "create",
                   "content": json.dumps(TASK)}
        for change in [{"status": "error"}, {"tool_call_id": "wrong"}, {"name": "read_email"}]:
            result = await self.history({**message, **change})
            self.assertFalse(any(step["type"] == "coding_task" for step in result["steps"]))
