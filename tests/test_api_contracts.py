"""Validate generated contracts against actual tenant-scoped persistence output."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import TypeAdapter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from db import connection
from db.coding_jobs import get_job
from db.user_apps import create_code_app_job, get_user_app, list_user_apps, save_user_app
from routers.auth import get_me
from routers.user_apps import public_job
from schemas.responses import (
    CodingJobResponse, ProfileResponse, UserAppDetailsResponse, UserAppResponse,
)


class TestApiContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path_patch = patch.object(connection, "DATABASE_PATH", str(Path(self.directory.name) / "contracts.db"))
        self.path_patch.start()
        await connection.init_db()

    async def asyncTearDown(self):
        self.path_patch.stop()
        self.directory.cleanup()

    async def test_app_list_details_and_job_contracts_match_storage(self):
        await save_user_app("alice", "todo_timeline", "Timeline", {"root": "timeline", "elements": {}})
        job = await create_code_app_job("alice", "Calculator", "Create it", "test-model", queued=True)
        listed = await list_user_apps("alice")
        TypeAdapter(list[UserAppResponse]).validate_python(listed)
        for app in listed:
            UserAppDetailsResponse.model_validate(await get_user_app("alice", app["id"]))
        actual_job = await get_job("alice", job["job_id"])
        CodingJobResponse.model_validate(public_job(actual_job))
        self.assertEqual(await list_user_apps("bob"), [])
        self.assertIsNone(await get_job("bob", job["job_id"]))

    async def test_profile_contract_accepts_missing_sip_and_creation_date(self):
        profile = await get_me({"id": "alice", "username": "alice", "display_name": "Alice"})
        validated = ProfileResponse.model_validate(profile)
        self.assertIsNone(validated.sip_extension)
        self.assertIsNone(validated.created_at)
