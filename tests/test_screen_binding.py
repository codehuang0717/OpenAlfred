"""Local binding never displays a real dialog or changes the real device owner."""

import asyncio
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from services import screen_binding as binding
from services import screen_monitor as monitor


class TestScreenBinding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(patch.object(binding.config, "PROJECT_ROOT", Path(self.tmp.name)))
        self.enterContext(patch.object(binding.config, "SCREEN_MONITOR_USER_ID", ""))

    def test_unbound_lookup_has_no_writes(self):
        self.assertIsNone(binding.configured_owner())
        self.assertFalse(binding.binding_path().exists())

    def test_native_confirmation_saves_owner_visible_without_restart(self):
        with patch.object(binding, "confirm_on_desktop", return_value=True) as confirm:
            binding.bind_on_desktop("user-a", "Alice")
        confirm.assert_called_once_with("Alice", "user-a")
        self.assertEqual(binding.configured_owner(), "user-a")
        self.assertEqual(monitor.require_screen_owner("user-a"), "user-a")
        with self.assertRaises(PermissionError):
            monitor.require_screen_owner("user-b")

    def test_cancel_leaves_device_unbound(self):
        with patch.object(binding, "confirm_on_desktop", return_value=False):
            with self.assertRaisesRegex(PermissionError, "取消"):
                binding.bind_on_desktop("user-a", "Alice")
        self.assertIsNone(binding.configured_owner())
        self.assertFalse(binding.binding_path().exists())

    def test_confirmation_failure_does_not_bind(self):
        with patch.object(binding, "confirm_on_desktop", side_effect=RuntimeError("no desktop")):
            with self.assertRaises(RuntimeError):
                binding.bind_on_desktop("user-a", "Alice")
        self.assertIsNone(binding.configured_owner())

    def test_duplicate_is_idempotent_other_user_cannot_take_over(self):
        with patch.object(binding, "confirm_on_desktop", return_value=True) as confirm:
            binding.bind_on_desktop("user-a", "Alice")
            binding.bind_on_desktop("user-a", "Alice")
            with self.assertRaises(PermissionError):
                binding.bind_on_desktop("user-b", "Bob")
        confirm.assert_called_once()
        self.assertEqual(binding.configured_owner(), "user-a")

    def test_existing_env_binding_preserved_without_dialog(self):
        with patch.object(binding.config, "SCREEN_MONITOR_USER_ID", "user-a"), patch.object(binding, "confirm_on_desktop") as confirm:
            binding.bind_on_desktop("user-a", "Alice")
            with self.assertRaises(PermissionError):
                binding.bind_on_desktop("user-b", "Bob")
        confirm.assert_not_called()

    def test_env_file_conflict_is_an_error_not_fallback(self):
        with patch.object(binding, "confirm_on_desktop", return_value=True):
            binding.bind_on_desktop("user-a", "Alice")
        with patch.object(binding.config, "SCREEN_MONITOR_USER_ID", "user-b"):
            with self.assertRaisesRegex(ValueError, "冲突"):
                binding.configured_owner()

    def test_simultaneous_confirmation_rejected(self):
        with binding._confirmation_lock:
            with self.assertRaisesRegex(ValueError, "正在进行"):
                binding.bind_on_desktop("user-a", "Alice")

    def test_cross_worker_race_does_not_overwrite_first_owner(self):
        def other_worker_wins(*_):
            path = binding.binding_path()
            path.parent.mkdir(parents=True)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("CREATE TABLE device_binding (id INTEGER PRIMARY KEY, user_id TEXT NOT NULL)")
                db.execute("INSERT INTO device_binding VALUES (1, 'user-b')")
            return True
        with patch.object(binding, "confirm_on_desktop", side_effect=other_worker_wins):
            with self.assertRaises(PermissionError):
                binding.bind_on_desktop("user-a", "Alice")
        self.assertEqual(binding.configured_owner(), "user-b")

    def test_remote_peer_or_proxy_headers_denied(self):
        for peer, headers in [
            ("192.168.1.20", {"origin": "http://localhost:3000"}),
            ("127.0.0.1", {"origin": "https://remote.example"}),
            ("127.0.0.1", {"origin": "http://localhost:3000", "x-forwarded-for": "8.8.8.8, 127.0.0.1"}),
            ("127.0.0.1", {"origin": "http://localhost:3000", "cf-connecting-ip": "8.8.8.8"}),
            ("127.0.0.1", {"origin": "http://localhost:3000", "forwarded": "for=127.0.0.1"}),
            ("127.0.0.1", {}),
        ]:
            with self.subTest(peer=peer, headers=headers):
                with self.assertRaises(PermissionError):
                    binding.require_local_binding_request(peer, headers)

    def test_local_page_and_local_next_proxy_allowed_to_request_dialog(self):
        binding.require_local_binding_request("127.0.0.1", {"origin": "http://localhost:3000", "x-forwarded-for": "::1"})
        binding.require_local_binding_request("::1", {"origin": "http://127.0.0.1:3000"})
        binding.require_local_binding_request("::ffff:127.0.0.1", {"origin": "http://localhost:3000", "x-forwarded-for": "::ffff:127.0.0.1"})

    def test_windows_dialog_defaults_no_and_names_account(self):
        message_box = MagicMock(return_value=6)
        with patch.object(binding.ctypes.windll.user32, "MessageBoxW", message_box):
            self.assertTrue(binding.confirm_on_desktop("Alice", "user-a"))
        args = message_box.call_args.args
        self.assertIn("Alice", args[1])
        self.assertIn("user-a", args[1])
        self.assertTrue(args[3] & 0x100)  # MB_DEFBUTTON2: No

    def test_windows_dialog_no_or_creation_failure_never_confirms(self):
        with patch.object(binding.ctypes.windll.user32, "MessageBoxW", MagicMock(return_value=7)):
            self.assertFalse(binding.confirm_on_desktop("Alice", "user-a"))
        with patch.object(binding.ctypes.windll.user32, "MessageBoxW", MagicMock(return_value=0)):
            with self.assertRaises(RuntimeError):
                binding.confirm_on_desktop("Alice", "user-a")


class TestBindingRoutes(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        from fastapi import FastAPI
        from routers import settings
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(patch.object(binding.config, "PROJECT_ROOT", Path(self.tmp.name)))
        self.enterContext(patch.object(binding.config, "SCREEN_MONITOR_USER_ID", ""))
        self.settings = settings
        self.app = FastAPI()
        self.app.include_router(settings.router)
        self.app.dependency_overrides[settings.get_current_user] = lambda: {"id": "user-a", "username": "Alice"}
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app, client=("127.0.0.1", 1000)), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)

    async def test_unbound_get_is_actionable_state_not_403(self):
        result = await self.client.get("/api/supervisor/config")
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.json()["binding_required"])
        self.assertFalse(result.json()["recording_enabled"])

    async def test_bind_uses_authenticated_identity_not_client_body(self):
        with patch.object(self.settings, "bind_on_desktop") as bind, patch.object(self.settings, "monitor_status", new_callable=AsyncMock, return_value={}):
            result = await self.client.post("/api/supervisor/bind", headers={"origin": "http://localhost:3000"}, json={"user_id": "attacker"})
        self.assertEqual(result.status_code, 200)
        bind.assert_called_once_with("user-a", "Alice")

    async def test_remote_origin_does_not_show_dialog(self):
        with patch.object(self.settings, "bind_on_desktop") as bind:
            result = await self.client.post("/api/supervisor/bind", headers={"origin": "https://remote.example"})
        self.assertEqual(result.status_code, 403)
        bind.assert_not_called()

    async def test_authentication_required(self):
        self.app.dependency_overrides.clear()
        with patch.object(self.settings, "bind_on_desktop") as bind:
            result = await self.client.post("/api/supervisor/bind", headers={"origin": "http://localhost:3000"})
        self.assertIn(result.status_code, (401, 403))
        bind.assert_not_called()

    async def test_binding_does_not_enable_recording(self):
        from db import connection
        self.enterContext(patch.object(connection, "DATABASE_PATH", str(Path(self.tmp.name) / "test.db")))
        await connection.init_db()
        with patch.object(binding, "confirm_on_desktop", return_value=True):
            result = await self.client.post("/api/supervisor/bind", headers={"origin": "http://localhost:3000"})
        self.assertEqual(result.status_code, 200)
        self.assertFalse(result.json()["binding_required"])
        self.assertFalse(result.json()["recording_enabled"])
        self.assertFalse(result.json()["smart_supervision_enabled"])

    async def test_supervisor_waits_then_detects_binding_without_restart(self):
        import supervisor
        async def confirm_while_waiting(_):
            binding.bind_on_desktop("user-a", "Alice")
        with patch.object(supervisor, "init_db", new_callable=AsyncMock), patch.object(supervisor, "get_user_by_id", new_callable=AsyncMock, return_value=None), patch.object(supervisor.asyncio, "sleep", side_effect=confirm_while_waiting), patch.object(binding, "confirm_on_desktop", return_value=True):
            service = supervisor.ProactiveSupervisor()
            self.assertIsNone(service.user_id)
            with self.assertRaisesRegex(RuntimeError, "账号不存在"):
                await service.start()
            self.assertEqual(service.user_id, "user-a")
