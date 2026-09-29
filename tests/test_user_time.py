import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import user_time
from utils.time_utils import localize_to_utc, utc_to_local
from tools import todos
from tools import eye


class TestTimeConversion(unittest.TestCase):
    def test_voice_tools_use_frozen_run_timezone_and_reject_missing_snapshot(self):
        runtime = SimpleNamespace(config={}, state={"user_timezone": "Asia/Shanghai"})
        self.assertEqual(user_time.runtime_timezone(runtime), "Asia/Shanghai")
        runtime.state = {}
        with self.assertRaises(user_time.MissingTimezoneError):
            user_time.runtime_timezone(runtime)

    def test_afternoon_in_shanghai_is_not_london_afternoon(self):
        self.assertEqual(localize_to_utc("2026-09-28T14:00:00", "Asia/Shanghai"), "2026-09-28T06:00:00Z")
        self.assertIn("14:00", utc_to_local("2026-09-28T06:00:00Z", "Asia/Shanghai"))
        self.assertEqual(localize_to_utc("2026-09-28T14:00:00", "Europe/London"), "2026-09-28T13:00:00Z")

    def test_explicit_offset_is_authoritative_and_normalization_is_idempotent(self):
        utc = localize_to_utc("2026-09-28T14:00:00-04:00", "Asia/Shanghai")
        self.assertEqual(utc, "2026-09-28T18:00:00Z")
        self.assertEqual(localize_to_utc(utc), utc)

    def test_naive_time_without_timezone_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "时区"):
            localize_to_utc("2026-09-28T14:00:00")

    def test_dst_gap_and_fold_require_clarification(self):
        for value in ("2026-03-29T01:30:00", "2026-10-25T01:30:00"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "夏令时"):
                localize_to_utc(value, "Europe/London")
        self.assertEqual(localize_to_utc("2026-10-25T01:30:00+01:00"), "2026-10-25T00:30:00Z")


class TestUserTimezone(unittest.IsolatedAsyncioTestCase):
    async def test_run_timezone_does_not_change_when_another_session_updates_profile(self):
        with patch.object(user_time, "get_setting", new=AsyncMock(return_value="Europe/London")):
            self.assertEqual(await user_time.get_user_timezone("user-a", {"configurable": {"timezone": "Asia/Shanghai"}}), "Asia/Shanghai")
            self.assertEqual(await user_time.get_user_timezone("user-a"), "Europe/London")

    async def test_settings_are_scoped_and_invalid_input_never_falls_back(self):
        with patch.object(user_time, "get_setting", new=AsyncMock(return_value=None)) as read:
            with self.assertRaises(user_time.MissingTimezoneError):
                await user_time.get_user_timezone("user-a")
            read.assert_awaited_once_with("user_timezone:v1:user-a")
        with self.assertRaises(ValueError):
            await user_time.get_user_timezone("user-a", {"configurable": {"timezone": "Bogus/Zone"}})

    async def test_todo_tool_stores_browser_local_afternoon_as_utc(self):
        runtime = SimpleNamespace(config={"configurable": {"langgraph_auth_user": {"identity": "user-a"}, "timezone": "Asia/Shanghai"}}, tool_call_id="call-1")
        with patch.object(todos, "db_add_todo", new_callable=AsyncMock) as add, patch.object(todos, "get_all_todos", new=AsyncMock(return_value=[])):
            await todos.add_todo.coroutine(runtime=runtime, title="测评", scheduled_start_at="2026-09-28T14:00:00", expected_completion_at="2026-09-28T15:00:00")
        self.assertEqual(add.await_args.kwargs["scheduled_start_at"], "2026-09-28T06:00:00Z")
        self.assertEqual(add.await_args.kwargs["user_id"], "user-a")

    async def test_screen_time_search_uses_same_timezone_converter(self):
        runtime = SimpleNamespace(config={"configurable": {"langgraph_auth_user": {"identity": "user-a"}, "timezone": "Asia/Shanghai"}})
        with patch.object(eye, "_search_screenpipe", new=AsyncMock(return_value={"data": []})) as search:
            await eye.search_screen_time.coroutine(runtime=runtime, start_time="2026-09-28T14:00:00", end_time="2026-09-28T15:00:00")
        self.assertEqual(search.await_args.kwargs["start_time"], "2026-09-28T06:00:00Z")
        self.assertEqual(search.await_args.kwargs["end_time"], "2026-09-28T07:00:00Z")
