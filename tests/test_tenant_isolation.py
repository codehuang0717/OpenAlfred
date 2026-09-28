"""Database-level tests for strict tenant ownership."""

import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from db import connection  # noqa: E402
from db import rag as rag_repo  # noqa: E402
from db import reminder as reminder_repo  # noqa: E402
from db import todo as todo_repo  # noqa: E402
from utils.auth_utils import MissingUserContextError  # noqa: E402


class TenantDatabaseTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "test.db")
        self.path_patch = patch.object(connection, "DATABASE_PATH", self.db_path)
        self.path_patch.start()
        await connection.init_db()

    async def asyncTearDown(self):
        self.path_patch.stop()
        self.temp_dir.cleanup()

    async def test_fresh_schema_has_no_default_user_and_rejects_it(self):
        with closing(sqlite3.connect(self.db_path)) as db:
            for table in ("todos", "reminders", "documents", "image_lookup"):
                columns = {
                    row[1]: row for row in db.execute(f"PRAGMA table_info({table})")
                }
                self.assertEqual(columns["user_id"][3], 1, table)
                self.assertIsNone(columns["user_id"][4], table)

            with self.assertRaises(sqlite3.IntegrityError):
                db.execute(
                    "INSERT INTO todos (id, title, created_at, user_id) VALUES (?, ?, ?, ?)",
                    ("bad", "bad", "2026-01-01T00:00:00Z", "default"),
                )

    async def test_todo_lookup_is_scoped_and_user_is_required(self):
        with (
            patch.object(todo_repo.event_bus, "publish", new_callable=AsyncMock),
            patch.object(
                todo_repo.event_bus, "schedule", new_callable=AsyncMock
            ) as schedule,
        ):
            await todo_repo.add_todo(
                id="todo-a",
                title="A",
                scheduled_start_at="2026-01-01T00:00:00Z",
                user_id="user-a",
            )

        self.assertEqual(schedule.await_args.args[1], {"id": "todo-a", "user_id": "user-a"})

        self.assertEqual(
            (await todo_repo.get_todo_by_id("todo-a", user_id="user-a"))["title"],
            "A",
        )
        self.assertIsNone(
            await todo_repo.get_todo_by_id("todo-a", user_id="user-b")
        )
        with self.assertRaises(TypeError):
            todo_repo.get_todo_by_id("todo-a")
        with self.assertRaises(MissingUserContextError):
            await todo_repo.get_all_todos("default")

    async def test_reminder_lookup_is_scoped_and_user_is_required(self):
        with (
            patch.object(reminder_repo.event_bus, "publish", new_callable=AsyncMock),
            patch.object(
                reminder_repo.event_bus, "schedule", new_callable=AsyncMock
            ) as schedule,
        ):
            await reminder_repo.add_reminder(
                id="reminder-a",
                body="A",
                scheduled_at="2026-01-01T00:00:00Z",
                user_id="user-a",
            )

        self.assertEqual(
            schedule.await_args.args[1],
            {"id": "reminder-a", "user_id": "user-a"},
        )

        self.assertIsNotNone(
            await reminder_repo.get_reminder_by_id(
                "reminder-a", user_id="user-a"
            )
        )
        self.assertIsNone(
            await reminder_repo.get_reminder_by_id(
                "reminder-a", user_id="user-b"
            )
        )
        with self.assertRaises(TypeError):
            reminder_repo.get_reminder_by_id("reminder-a")

    async def test_rag_documents_and_images_are_scoped(self):
        document = await rag_repo.add_document(
            user_id="user-a", filename="a.md", title="A"
        )
        image_id = await rag_repo.add_image_lookup(
            document["id"],
            user_id="user-a",
            url="/api/images/a.png",
            alt="A",
            filename="a.png",
        )

        self.assertIsNotNone(
            await rag_repo.get_document_by_id(
                document["id"], user_id="user-a"
            )
        )
        self.assertIsNone(
            await rag_repo.get_document_by_id(
                document["id"], user_id="user-b"
            )
        )
        self.assertIsNotNone(
            await rag_repo.get_image_by_id(image_id, user_id="user-a")
        )
        self.assertIsNone(
            await rag_repo.get_image_by_id(image_id, user_id="user-b")
        )
        with self.assertRaises(MissingUserContextError):
            await rag_repo.get_documents("default")

    async def test_migration_errors_are_not_swallowed(self):
        other_path = str(Path(self.temp_dir.name) / "migration-error.db")
        with (
            patch.object(connection, "DATABASE_PATH", other_path),
            patch.object(
                connection,
                "_ensure_column",
                new=AsyncMock(side_effect=sqlite3.OperationalError("disk failure")),
            ),
        ):
            with self.assertRaisesRegex(sqlite3.OperationalError, "disk failure"):
                await connection.init_db()


if __name__ == "__main__":
    unittest.main()
