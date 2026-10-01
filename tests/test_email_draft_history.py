"""Draft references survive graph failures and remain owner-scoped on refresh."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.security import HTTPAuthorizationCredentials
from routers import events, threads
from services.email_draft_reference import email_draft_reference

DRAFT = {"type": "email_draft", "draft_id": "12345678-1234-1234-1234-123456789abc", "subject": "会议安排"}


class TestEmailDraftReference(unittest.TestCase):
    def test_reference_excludes_content_and_rejects_invalid_ids(self):
        self.assertEqual(email_draft_reference(json.dumps({**DRAFT, "body": "私密正文", "to_address": "secret@example.com"})), DRAFT)
        for content in [None, "[]", "null", "{", "x" * 4097, {**DRAFT, "draft_id": [DRAFT["draft_id"]]}, {**DRAFT, "draft_id": "../secret"}]:
            self.assertIsNone(email_draft_reference(content))


class TestEmailDraftHistory(unittest.IsolatedAsyncioTestCase):
    async def history(self, tool_message):
        messages = [
            {"type": "human", "id": "user", "content": "写封邮件"},
            {"type": "ai", "id": "plan", "content": "正在保存草稿", "tool_calls": [{"name": "create_email_draft", "id": "create"}]},
            tool_message, tool_message,
            {"type": "ai", "id": "failed", "content": "连接中断", "additional_kwargs": {"agent_outcome": {"status": "failed"}}},
        ]
        original = json.dumps(messages)
        response = SimpleNamespace(status_code=200, json=lambda: {"values": {"messages": messages}})
        client = SimpleNamespace(get=AsyncMock(return_value=response))
        with patch.object(threads.httpx, "AsyncClient") as factory:
            factory.return_value.__aenter__.return_value = client
            result = await threads.get_thread_messages("thread", HTTPAuthorizationCredentials(scheme="Bearer", credentials="synthetic"), {"id": "alice"})
        self.assertEqual(json.dumps(messages), original)
        return result[-1]

    async def test_saved_draft_survives_failed_final_reply_once(self):
        message = {"type": "tool", "name": "create_email_draft", "tool_call_id": "create", "content": json.dumps(DRAFT)}
        result = await self.history(message)
        self.assertEqual([step for step in result["steps"] if step["type"] == "email_draft"], [{"type": "email_draft", "id": f"mail-{DRAFT['draft_id']}", "draft": DRAFT}])
        self.assertEqual(result["outcome"], "failed")
        self.assertEqual(await self.history(message), result)

    async def test_failed_or_unpaired_tools_do_not_create_cards(self):
        message = {"type": "tool", "name": "create_email_draft", "tool_call_id": "create", "content": json.dumps(DRAFT)}
        for change in [{"status": "error"}, {"tool_call_id": "wrong"}, {"name": "read_email"}]:
            result = await self.history({**message, **change})
            self.assertFalse(any(step["type"] == "email_draft" for step in result["steps"]))

    async def test_mail_event_subscription_only_exposes_the_current_owner(self):
        async def subscribe(*patterns):
            self.assertIn("email.*", patterns)
            for owner in [None, "bob", "alice"]:
                yield {"type": "email.updated", "data": {"user_id": owner, "draft_id": DRAFT["draft_id"]}}
        with patch.object(events.event_bus, "subscribe", side_effect=subscribe):
            response = await events.event_stream({"id": "alice"})
            chunks = [chunk async for chunk in response.body_iterator]
        self.assertEqual(len(chunks), 2)
        self.assertEqual(json.loads(chunks[-1][6:])["data"]["user_id"], "alice")
