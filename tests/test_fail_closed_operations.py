"""Tests that invalid context and delivery failures remain visible."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from services import scheduler  # noqa: E402
from services.notification import notification_service  # noqa: E402
from routers.auth import create_jwt_token  # noqa: E402
from tools.call_user import _resolve_phone_number  # noqa: E402
from tools.todos import add_todo  # noqa: E402
from utils.auth_utils import MissingUserContextError  # noqa: E402
from utils.time_utils import utc_to_local  # noqa: E402


class TestFailClosedScheduling(unittest.IsolatedAsyncioTestCase):
    async def test_failed_delivery_is_not_marked_sent(self):
        reminder = {
            "id": "reminder-1",
            "user_id": "user-a",
            "body": "test",
            "delivery_method": "push",
            "sent": 0,
        }
        with (
            patch.object(
                scheduler, "get_reminder_by_id", new=AsyncMock(return_value=reminder)
            ),
            patch.object(
                scheduler,
                "_deliver_reminder",
                new=AsyncMock(side_effect=scheduler.ReminderDeliveryError("failed")),
            ),
            patch.object(
                scheduler, "mark_reminder_sent", new_callable=AsyncMock
            ) as mark_sent,
        ):
            with self.assertRaises(scheduler.ReminderDeliveryError):
                await scheduler.send_single_reminder("reminder-1", "user-a")
            mark_sent.assert_not_awaited()

    async def test_successful_delivery_is_marked_for_same_user(self):
        reminder = {
            "id": "reminder-1",
            "user_id": "user-a",
            "body": "test",
            "delivery_method": "push",
            "sent": 0,
        }
        with (
            patch.object(
                scheduler, "get_reminder_by_id", new=AsyncMock(return_value=reminder)
            ) as lookup,
            patch.object(scheduler, "_deliver_reminder", new_callable=AsyncMock),
            patch.object(
                scheduler, "mark_reminder_sent", new_callable=AsyncMock
            ) as mark_sent,
        ):
            await scheduler.send_single_reminder("reminder-1", "user-a")
            lookup.assert_awaited_once_with("reminder-1", user_id="user-a")
            mark_sent.assert_awaited_once_with("reminder-1", user_id="user-a")

    async def test_missing_user_never_delivers(self):
        with self.assertRaises(MissingUserContextError):
            await scheduler._deliver_reminder(
                {"id": "reminder-1", "body": "test", "delivery_method": "push"}
            )


class TestStrictConfigurationAndInput(unittest.IsolatedAsyncioTestCase):
    def test_jwt_rejects_reserved_default_user(self):
        with self.assertRaises(MissingUserContextError):
            create_jwt_token("default", "legacy")

    async def test_notification_requires_per_user_url(self):
        with self.assertRaisesRegex(RuntimeError, "user Bark URL"):
            await notification_service.send_bark_notification(
                body="test", bark_url=""
            )

    async def test_phone_resolution_requires_user_extension(self):
        # _resolve_phone_number imports the repository function locally.
        with patch(
            "core.database.get_user_by_id",
            new=AsyncMock(return_value={"id": "user-a", "sip_extension": ""}),
        ):
            with self.assertRaisesRegex(ValueError, "no SIP extension"):
                await _resolve_phone_number("user-a")

    async def test_invalid_todo_time_does_not_reach_database(self):
        runtime = SimpleNamespace(
            config={
                "configurable": {
                    "langgraph_auth_user": {"identity": "user-a"}
                }
            }
        )
        with patch("tools.todos.db_add_todo", new_callable=AsyncMock) as db_add:
            with self.assertRaises(ValueError):
                await add_todo.coroutine(
                    runtime=runtime,
                    title="bad time",
                    expected_completion_at="not-a-date",
                )
            db_add.assert_not_awaited()

    def test_invalid_utc_display_time_raises(self):
        with self.assertRaises(ValueError):
            utc_to_local("not-a-date", "Asia/Shanghai")


if __name__ == "__main__":
    unittest.main()
