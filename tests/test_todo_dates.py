"""End-of-day timestamps and invalid todo inputs are handled explicitly."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI
import httpx

from routers import todos as todo_routes
from tools import todos as todo_tools
from utils.time_utils import localize_to_utc, parse_to_aware_utc, utc_to_local


class TestTodoDates(unittest.TestCase):
    def test_exact_24_midnight_is_next_day_not_an_invalid_hour(self):
        self.assertEqual(
            parse_to_aware_utc("2026-09-23T24:00:00+08:00").isoformat(),
            "2026-09-23T16:00:00+00:00",
        )
        self.assertEqual(
            localize_to_utc("2026-09-23T24:00:00+08:00"),
            "2026-09-23T16:00:00Z",
        )
        self.assertTrue(utc_to_local("2026-09-23T24:00:00+08:00"))

    def test_non_midnight_24_hour_and_other_bad_dates_still_raise(self):
        for value in ("2026-09-23T24:01:00+08:00", "2026-09-23T24:00:01+08:00", "2026-09-23T25:00:00+08:00"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_to_aware_utc(value)


class TestTodoDateIngress(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_end_of_day_row_can_be_read_by_agent(self):
        runtime = SimpleNamespace(config={"configurable": {"langgraph_auth_user": {"identity": "user-a"}}})
        row = {"id": "todo-a", "expected_completion_at": "2026-09-23T24:00:00+08:00", "scheduled_start_at": None}
        with patch.object(todo_tools, "get_all_todos", new=AsyncMock(return_value=[row])):
            result = await todo_tools.get_todos.coroutine(runtime=runtime)
        self.assertEqual(result[0]["id"], "todo-a")
        self.assertNotIn("24:00", result[0]["expected_completion_at"])

    async def test_http_patch_rejects_invalid_date_before_database_write(self):
        app = FastAPI()
        app.include_router(todo_routes.router)
        app.dependency_overrides[todo_routes.get_current_user] = lambda: {"id": "user-a"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with patch.object(todo_routes, "db_update_todo", new_callable=AsyncMock) as update:
                response = await client.patch("/api/todos/todo-a", json={"expected_completion_at": "2026-09-23T24:01:00+08:00"})
            self.assertEqual(response.status_code, 422)
            update.assert_not_awaited()

    async def test_http_patch_normalizes_exact_end_of_day(self):
        app = FastAPI()
        app.include_router(todo_routes.router)
        app.dependency_overrides[todo_routes.get_current_user] = lambda: {"id": "user-a"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with patch.object(todo_routes, "db_update_todo", new=AsyncMock(return_value=True)) as update:
                response = await client.patch("/api/todos/todo-a", json={"expected_completion_at": "2026-09-23T24:00:00+08:00"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(update.await_args.kwargs["expected_completion_at"], "2026-09-23T16:00:00Z")


if __name__ == "__main__":
    unittest.main()
