"""Screen monitoring regressions: no real capture, network or model calls."""

import asyncio
import json
from pathlib import Path
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db import connection
from db.settings import set_setting
from services import screen_monitor as monitor
from tools import eye, screenshot
from utils.auth_utils import MissingUserContextError


class TestScreenOwnership(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(patch.object(connection, "DATABASE_PATH", str(Path(self.tmp.name) / "test.db")))
        self.enterContext(patch.object(monitor.config, "PROJECT_ROOT", Path(self.tmp.name)))
        self.enterContext(patch.object(monitor.config, "SCREEN_MONITOR_USER_ID", "owner-a"))
        await connection.init_db()

    async def test_default_off_ignores_legacy_global_flags(self):
        await set_setting("recording_enabled", "true")
        await set_setting("smart_supervision_enabled", "true")
        self.assertEqual(await monitor.read_preferences("owner-a"), {
            "recording_enabled": False, "smart_supervision_enabled": False,
        })

    async def test_atomic_preferences_roundtrip(self):
        await monitor.write_preferences("owner-a", True, True)
        self.assertEqual(await monitor.read_preferences("owner-a"), {
            "recording_enabled": True, "smart_supervision_enabled": True,
        })

    async def test_other_account_cannot_read_or_write(self):
        with self.assertRaises(PermissionError):
            await monitor.read_preferences("owner-b")
        with self.assertRaises(PermissionError):
            await monitor.write_preferences("owner-b", True, True)

    async def test_unbound_and_default_identity_are_rejected(self):
        with patch.object(monitor.config, "SCREEN_MONITOR_USER_ID", ""):
            with self.assertRaisesRegex(PermissionError, "绑定当前账号"):
                await monitor.read_preferences("owner-a")
        with self.assertRaises(MissingUserContextError):
            await monitor.read_preferences("default")

    async def test_invalid_preferences_rejected_not_coerced(self):
        with self.assertRaises(ValueError):
            await monitor.write_preferences("owner-a", False, True)
        await set_setting("screen_monitor:v1:owner-a", '{"recording_enabled":"true"}')
        with self.assertRaises(ValueError):
            await monitor.read_preferences("owner-a")

    async def test_directory_is_owner_scoped_and_not_legacy(self):
        path_a = monitor.data_directory("owner-a")
        self.assertEqual(len(path_a.name), 64)
        self.assertEqual(path_a.parent.name, "users")
        with patch.object(monitor.config, "SCREEN_MONITOR_USER_ID", "owner-b"):
            self.assertNotEqual(path_a, monitor.data_directory("owner-b"))
            self.assertFalse((await monitor.read_preferences("owner-b"))["recording_enabled"])

    async def save_runtime(self, **kwargs):
        await set_setting("screen_monitor:runtime:owner-a", json.dumps({
            "heartbeat": time.time(), "screenpipe_pid": 123, **kwargs,
        }))

    async def test_stale_heartbeat_never_claims_recording(self):
        await self.save_runtime(heartbeat=time.time() - 60)
        with patch.object(monitor, "healthy", new_callable=AsyncMock) as health:
            status = await monitor.monitor_status("owner-a")
        self.assertFalse(status["supervisor_running"])
        self.assertFalse(status["screenpipe_running"])
        health.assert_not_awaited()

    async def test_foreign_process_never_trusted_or_queried(self):
        await self.save_runtime()
        with patch.object(monitor, "process_status", return_value="mismatch"), patch.object(monitor, "healthy", new_callable=AsyncMock) as health:
            status = await monitor.monitor_status("owner-a")
        self.assertFalse(status["screenpipe_running"])
        self.assertIn("不匹配", status["error"])
        health.assert_not_awaited()

    async def test_ocr_health_required_for_running_status(self):
        await self.save_runtime()
        with patch.object(monitor, "process_status", return_value="ready"), patch.object(monitor, "healthy", new_callable=AsyncMock, return_value=False):
            status = await monitor.monitor_status("owner-a")
        self.assertFalse(status["screenpipe_running"])
        self.assertIn("OCR", status["error"])

    async def test_health_connection_error_is_visible(self):
        import httpx
        await self.save_runtime()
        with patch.object(monitor, "process_status", return_value="ready"), patch.object(monitor, "healthy", new_callable=AsyncMock, side_effect=httpx.ConnectError("down")):
            status = await monitor.monitor_status("owner-a")
        self.assertFalse(status["screenpipe_running"])
        self.assertIn("ConnectError", status["error"])

    async def test_healthy_owned_service_reports_recording(self):
        await self.save_runtime(analysis_running=True)
        await monitor.write_preferences("owner-a", True, True)
        with patch.object(monitor, "process_status", return_value="ready"), patch.object(monitor, "healthy", new_callable=AsyncMock, return_value=True):
            status = await monitor.monitor_status("owner-a")
            await monitor.require_screen_access("owner-a")
        self.assertTrue(status["screenpipe_running"])
        self.assertTrue(status["analysis_running"])

    async def test_process_starting_is_not_reported_as_account_mismatch(self):
        await self.save_runtime()
        with patch.object(monitor, "process_status", return_value="starting"), patch.object(monitor, "healthy", new_callable=AsyncMock) as health:
            status = await monitor.monitor_status("owner-a")
        self.assertFalse(status["screenpipe_running"])
        self.assertIn("正在启动", status["error"])
        self.assertNotIn("不匹配", status["error"])
        health.assert_not_awaited()

    async def test_disabled_recording_denies_queries_even_while_stopping(self):
        with patch.object(monitor, "monitor_status", new_callable=AsyncMock, return_value={
            "recording_enabled": False, "screenpipe_running": True, "error": None,
        }):
            with self.assertRaises(PermissionError):
                await monitor.require_screen_access("owner-a")

    async def test_foreign_screen_query_denied_before_http(self):
        with patch.object(eye.httpx, "AsyncClient") as client:
            with self.assertRaises(PermissionError):
                await eye._search_screenpipe(user_id="owner-b")
            client.assert_not_called()

    async def test_no_ocr_and_search_failure_never_become_llm_context(self):
        for response in ({"error": "connection refused"}, {"data": []}):
            with patch.object(eye, "_search_screenpipe", new_callable=AsyncMock, return_value=response):
                with self.assertRaises(RuntimeError):
                    await eye.get_recent_ocr_text("owner-a")

    async def test_screenshot_rejects_other_user_before_capture(self):
        runtime = SimpleNamespace(config={"configurable": {"langgraph_auth_user": {"identity": "owner-b"}}})
        with patch.object(screenshot.ImageGrab, "grab") as grab:
            result = await screenshot.take_screenshot.coroutine(query="screen", runtime=runtime)
        self.assertIn("未绑定", result)
        grab.assert_not_called()

    async def test_tool_schemas_do_not_allow_model_to_choose_owner(self):
        for tool in (eye.view_screen, eye.search_screen_time, screenshot.take_screenshot):
            properties = tool.tool_call_schema.model_json_schema()["properties"]
            self.assertNotIn("runtime", properties)
            self.assertNotIn("user_id", properties)


class TestRecorderLifecycle(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(patch.object(monitor.config, "PROJECT_ROOT", Path(self.tmp.name)))
        self.enterContext(patch.object(monitor.config, "SCREEN_MONITOR_USER_ID", "owner-a"))
        self.enterContext(patch.object(monitor.config, "SCREENPIPE_URL", "http://localhost:3030"))
        self.recorder = monitor.ScreenRecorder("owner-a")

    async def test_remote_server_rejected(self):
        for url in ("https://example.com:3030", "http://192.168.1.10:3030", "http://localhost", "http://localhost:3030/search"):
            with patch.object(monitor.config, "SCREENPIPE_URL", url):
                with self.assertRaises(ValueError):
                    monitor.server_address()

    async def test_occupied_port_not_adopted(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with patch.object(monitor.config, "SCREENPIPE_URL", f"http://localhost:{port}"), patch.object(monitor.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "拒绝接管"):
                    await self.recorder.start()
        spawn.assert_not_called()

    async def test_missing_binary_explicit_error(self):
        with patch.object(monitor, "require_free_port"), patch.object(monitor.config, "SCREENPIPE_EXE", str(Path(self.tmp.name) / "missing.exe")):
            with self.assertRaises(FileNotFoundError):
                await self.recorder.start()

    async def test_spawn_is_owned_local_ocr_without_audio_and_stops_only_child(self):
        process = MagicMock(pid=456)
        process.poll.return_value = None
        with patch.object(monitor, "require_free_port"), patch.object(monitor.subprocess, "Popen", return_value=process) as spawn:
            await self.recorder.start()
            await self.recorder.start()
            args = spawn.call_args.args[0]
            self.assertIn("--disable-audio", args)
            self.assertIn("--auto-destruct-pid", args)
            self.assertIn("windows-native", args)
            self.assertIn(str(monitor.data_directory("owner-a")), args)
            spawn.assert_called_once()
            await self.recorder.stop()
        process.terminate.assert_called_once()
        self.assertIsNone(self.recorder.log_file)

    async def test_free_port_starts_even_when_connect_probe_would_timeout(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        process = MagicMock(pid=456)
        process.poll.return_value = None
        with patch.object(monitor.config, "SCREENPIPE_URL", f"http://localhost:{port}"), patch.object(monitor.asyncio, "open_connection", new_callable=AsyncMock, side_effect=TimeoutError) as connect, patch.object(monitor.subprocess, "Popen", return_value=process) as spawn:
            await self.recorder.start()
            await self.recorder.stop()
        connect.assert_not_awaited()
        spawn.assert_called_once()

    async def test_crash_is_not_silently_restarted(self):
        self.recorder.process = MagicMock(returncode=3)
        self.recorder.process.poll.return_value = 3
        with self.assertRaisesRegex(RuntimeError, "退出"):
            await self.recorder.start()
        await self.recorder.stop()

    async def test_bundled_binary_rejects_bad_models_before_capture(self):
        exe = Path(self.tmp.name) / "src/body/windows_system/eye/screenpipe-0.3.6-x86_64-pc-windows-msvc/bin/screenpipe.exe"
        exe.parent.mkdir(parents=True)
        exe.touch()
        with patch.object(monitor.config, "SCREENPIPE_EXE", str(exe)), patch.object(monitor, "require_free_port"), patch.object(monitor, "verify_models", side_effect=RuntimeError("模型校验失败")), patch.object(monitor.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "模型校验失败"):
                await self.recorder.start()
        spawn.assert_not_called()

    async def test_duplicate_supervisor_lock_is_rejected(self):
        with monitor.device_lock():
            with self.assertRaisesRegex(RuntimeError, "已有 Supervisor"):
                with monitor.device_lock():
                    self.fail("Second supervisor acquired lock")
        with monitor.device_lock():
            pass

    async def test_process_verification_checks_directory_executable_and_port(self):
        process = MagicMock()
        process.exe.return_value = monitor.config.SCREENPIPE_EXE
        process.cmdline.return_value = ["screenpipe.exe", "--data-dir", str(monitor.data_directory("owner-a"))]
        process.net_connections.return_value = [SimpleNamespace(status=monitor.psutil.CONN_LISTEN, laddr=SimpleNamespace(port=3030))]
        with patch.object(monitor.psutil, "Process", return_value=process):
            self.assertTrue(monitor.verified_process(123, "owner-a"))
            self.assertEqual(monitor.process_status(123, "owner-a"), "ready")
            process.cmdline.return_value = ["screenpipe.exe", "--data-dir", str(Path(self.tmp.name) / "legacy")]
            self.assertFalse(monitor.verified_process(123, "owner-a"))
            process.cmdline.return_value = ["screenpipe.exe", "--data-dir", str(monitor.data_directory("owner-a"))]
            process.exe.return_value = str(Path(self.tmp.name) / "unknown.exe")
            self.assertFalse(monitor.verified_process(123, "owner-a"))
            process.exe.return_value = monitor.config.SCREENPIPE_EXE
            process.net_connections.return_value = []
            self.assertFalse(monitor.verified_process(123, "owner-a"))
            self.assertEqual(monitor.process_status(123, "owner-a"), "starting")


class TestSupervisorLoop(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import supervisor
        self.supervisor = supervisor
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(patch.object(monitor.config, "PROJECT_ROOT", Path(self.tmp.name)))
        self.enterContext(patch.object(monitor.config, "SCREEN_MONITOR_USER_ID", "owner-a"))
        self.enterContext(patch.object(supervisor, "init_db", new_callable=AsyncMock))
        self.lookup = self.enterContext(patch.object(supervisor, "get_user_by_id", new_callable=AsyncMock, return_value={"id": "owner-a", "username": "Owner"}))
        self.recorder = MagicMock(start=AsyncMock(), stop=AsyncMock(), heartbeat=AsyncMock())
        self.enterContext(patch.object(supervisor, "ScreenRecorder", return_value=self.recorder))
        self.model = self.enterContext(patch.object(supervisor, "get_model"))

    async def test_disabled_monitor_does_not_start_recorder_or_model(self):
        with patch.object(self.supervisor, "read_preferences", new_callable=AsyncMock, return_value={"recording_enabled": False, "smart_supervision_enabled": False}), patch.object(self.supervisor.asyncio, "sleep", new_callable=AsyncMock, side_effect=asyncio.CancelledError):
            service = self.supervisor.ProactiveSupervisor()
            with self.assertRaises(asyncio.CancelledError):
                await service.start()
        self.recorder.start.assert_not_awaited()
        self.model.assert_not_called()
        self.recorder.heartbeat.assert_any_await(error="Supervisor 已停止", stopped=True)
        self.lookup.assert_awaited_with("owner-a")

    async def test_recording_only_never_constructs_model(self):
        with patch.object(self.supervisor, "read_preferences", new_callable=AsyncMock, return_value={"recording_enabled": True, "smart_supervision_enabled": False}), patch.object(self.supervisor.asyncio, "sleep", new_callable=AsyncMock, side_effect=asyncio.CancelledError):
            service = self.supervisor.ProactiveSupervisor()
            with self.assertRaises(asyncio.CancelledError):
                await service.start()
        self.recorder.start.assert_awaited_once()
        self.recorder.stop.assert_awaited_once()
        self.model.assert_not_called()

    async def test_start_failure_latches_until_recording_disabled(self):
        self.recorder.start.side_effect = RuntimeError("binary missing")
        with patch.object(self.supervisor, "read_preferences", new_callable=AsyncMock, return_value={"recording_enabled": True, "smart_supervision_enabled": False}), patch.object(self.supervisor.asyncio, "sleep", new_callable=AsyncMock, side_effect=[None, asyncio.CancelledError]):
            service = self.supervisor.ProactiveSupervisor()
            with self.assertRaises(asyncio.CancelledError):
                await service.start()
        self.recorder.start.assert_awaited_once()
        self.assertTrue(any("binary missing" in str(call) for call in self.recorder.heartbeat.await_args_list))

    async def test_empty_exception_still_reports_failure_type(self):
        self.recorder.start.side_effect = TimeoutError()
        with patch.object(self.supervisor, "read_preferences", new_callable=AsyncMock, return_value={"recording_enabled": True, "smart_supervision_enabled": False}), patch.object(self.supervisor.asyncio, "sleep", new_callable=AsyncMock, side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await self.supervisor.ProactiveSupervisor().start()
        self.assertTrue(any("TimeoutError；排除故障" in str(call) for call in self.recorder.heartbeat.await_args_list))

    async def test_turning_recording_off_stops_managed_process(self):
        with patch.object(self.supervisor, "read_preferences", new_callable=AsyncMock, side_effect=[
            {"recording_enabled": True, "smart_supervision_enabled": False},
            {"recording_enabled": False, "smart_supervision_enabled": False},
        ]), patch.object(self.supervisor.asyncio, "sleep", new_callable=AsyncMock, side_effect=[None, asyncio.CancelledError]):
            service = self.supervisor.ProactiveSupervisor()
            with self.assertRaises(asyncio.CancelledError):
                await service.start()
        self.recorder.start.assert_awaited_once()
        self.assertEqual(self.recorder.stop.await_count, 2)

    async def test_unknown_bound_user_rejected_before_capture(self):
        self.lookup.return_value = None
        with self.assertRaisesRegex(RuntimeError, "账号不存在"):
            await self.supervisor.ProactiveSupervisor().start()
        self.recorder.start.assert_not_awaited()

    async def test_disabling_supervision_cancels_inflight_analysis(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        async def analyze():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        real_sleep = asyncio.sleep
        sleeps = 0
        async def tick(_):
            nonlocal sleeps
            sleeps += 1
            if sleeps == 1:
                await real_sleep(0)
            else:
                raise asyncio.CancelledError
        with patch.object(self.supervisor, "read_preferences", new_callable=AsyncMock, side_effect=[
            {"recording_enabled": True, "smart_supervision_enabled": True},
            {"recording_enabled": False, "smart_supervision_enabled": False},
        ]), patch.object(self.supervisor, "monitor_status", new_callable=AsyncMock, return_value={"screenpipe_running": True}), patch.object(self.supervisor.asyncio, "sleep", side_effect=tick):
            service = self.supervisor.ProactiveSupervisor()
            service.run_cycle = analyze
            with self.assertRaises(asyncio.CancelledError):
                await service.start()
        self.assertTrue(started.is_set())
        self.assertTrue(cancelled.is_set())


class TestSupervisorRoutes(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from fastapi import FastAPI
        from routers import settings
        import httpx
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(patch.object(monitor.config, "PROJECT_ROOT", Path(self.tmp.name)))
        self.settings = settings
        app = FastAPI()
        app.include_router(settings.router)
        self.user = {"id": "owner-b"}
        app.dependency_overrides[settings.get_current_user] = lambda: self.user
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)

    async def test_get_wrong_owner_returns_403(self):
        with patch.object(monitor.config, "SCREEN_MONITOR_USER_ID", "owner-a"):
            response = await self.client.get("/api/supervisor/config")
        self.assertEqual(response.status_code, 403)

    async def test_post_wrong_owner_returns_403_without_write(self):
        with patch.object(monitor.config, "SCREEN_MONITOR_USER_ID", "owner-a"), patch.object(monitor, "set_setting", new_callable=AsyncMock) as write:
            response = await self.client.post("/api/supervisor/config", json={"recording_enabled": True, "smart_supervision_enabled": False})
        self.assertEqual(response.status_code, 403)
        write.assert_not_awaited()

    async def test_post_invalid_combination_returns_422(self):
        self.user = {"id": "owner-a"}
        with patch.object(monitor.config, "SCREEN_MONITOR_USER_ID", "owner-a"):
            response = await self.client.post("/api/supervisor/config", json={"recording_enabled": False, "smart_supervision_enabled": True})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
