"""Generated toolbox panel ownership and catalog regression tests."""

import sys
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import aiosqlite
import httpx
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from db import connection  # noqa: E402
from db.user_apps import (  # noqa: E402
    create_code_app_job, create_code_app_revision_job, delete_user_app,
    finish_code_app_job, get_user_app, list_user_apps, publish_user_app_revision,
)
from db.coding_jobs import claim_job, expire_jobs, progress
from services import user_apps as app_service  # noqa: E402
from services import code_apps  # noqa: E402
from routers.auth import get_current_user  # noqa: E402
from routers.user_apps import router as user_app_router  # noqa: E402
from tools.user_apps import create_standalone_mini_app, create_todo_timeline_panel  # noqa: E402
from utils.auth_utils import MissingUserContextError  # noqa: E402


class TestUserApps(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path_patch = patch.object(
            connection, "DATABASE_PATH", str(Path(self.directory.name) / "apps.db")
        )
        self.path_patch.start()
        await connection.init_db()

    async def asyncTearDown(self):
        self.path_patch.stop()
        self.directory.cleanup()

    async def test_timeline_is_private_and_revision_keeps_same_id(self):
        with patch.object(app_service.event_bus, "publish", new_callable=AsyncMock):
            first = await app_service.create_todo_timeline(
                "alice", app_service.TodoTimelineOptions(title="我的计划")
            )
            second = await app_service.create_todo_timeline(
                "alice",
                app_service.TodoTimelineOptions(
                    title="截止日程", date_field="expected_completion_at", accent="violet"
                ),
            )

        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(await list_user_apps("alice")), 1)
        self.assertEqual(await list_user_apps("bob"), [])
        self.assertEqual(second["spec"]["elements"]["timeline"]["props"]["accent"], "violet")
        revisions = (await get_user_app("alice", first["id"]))["revisions"]
        self.assertEqual([item["revision_number"] for item in revisions], [2, 1])
        self.assertEqual(second["published_revision_id"], revisions[0]["id"])
        async with connection.get_db() as db:
            await db.execute(
                "UPDATE user_apps SET spec_json = ? WHERE id = ?",
                ('{"stale":true}', first["id"]),
            )
            await db.commit()
        self.assertEqual(
            (await list_user_apps("alice"))[0]["spec"]["elements"]["timeline"]["props"]["accent"],
            "violet",
        )
        self.assertFalse(await delete_user_app("bob", first["id"]))
        self.assertEqual(len(await list_user_apps("alice")), 1)
        self.assertTrue(await delete_user_app("alice", first["id"]))
        self.assertEqual(await list_user_apps("alice"), [])

    async def test_tool_requires_authenticated_runtime_identity(self):
        runtime = SimpleNamespace(config={}, state={"user_id": "alice"})
        with self.assertRaises(MissingUserContextError):
            await create_todo_timeline_panel.coroutine(runtime=runtime, title="我的计划")

    async def test_tool_creates_panel_for_authenticated_user_only(self):
        runtime = SimpleNamespace(
            config={"configurable": {"langgraph_auth_user": {"identity": "alice"}}}
        )
        with patch.object(app_service.event_bus, "publish", new_callable=AsyncMock):
            result = await create_todo_timeline_panel.coroutine(
                runtime=runtime, title="每周计划", accent="blue"
            )
        self.assertIn("每周计划", result)
        self.assertEqual(len(await list_user_apps("alice")), 1)
        self.assertEqual(await list_user_apps("bob"), [])

    async def test_code_tool_uses_authenticated_identity_and_requires_publish(self):
        runtime = SimpleNamespace(
            config={"configurable": {
                "langgraph_auth_user": {"identity": "alice"},
                "model_selection": "deepseek",
            }}
        )
        with patch("tools.user_apps.create_code_app", new_callable=AsyncMock) as create:
            create.return_value = {"status": "queued", "app_id": "app", "job_id": "job"}
            result = await create_standalone_mini_app.coroutine(
                runtime=runtime, title="时钟", requirements="创建一个简单的桌面时钟小程序"
            )
        self.assertEqual(create.await_args.args[0], "alice")
        self.assertEqual(create.await_args.args[2], "deepseek")
        self.assertIn("预览", result)
        self.assertIn("发布", result)

    async def test_code_tool_rejects_missing_model_selection(self):
        runtime = SimpleNamespace(
            config={"configurable": {"langgraph_auth_user": {"identity": "alice"}}}
        )
        with patch("tools.user_apps.create_code_app", new_callable=AsyncMock) as create:
            with self.assertRaisesRegex(ValueError, "未指定小程序生成模型"):
                await create_standalone_mini_app.coroutine(
                    runtime=runtime, title="时钟", requirements="创建一个简单的桌面时钟小程序"
                )
        create.assert_not_awaited()

    def test_catalog_rejects_unapproved_options(self):
        with self.assertRaises(ValueError):
            app_service.TodoTimelineOptions(title="X", accent="red")

    async def test_code_candidate_requires_owner_publish_and_never_auto_publishes(self):
        job = await create_code_app_job("alice", "倒计时", "创建一个倒计时小程序", "test-model")
        app_id = job["app_id"]
        self.assertIsNone(await get_user_app("bob", app_id))
        self.assertIsNone((await get_user_app("alice", app_id))["published_revision_id"])
        source = {"html": "<p>Ready</p>", "css": "", "javascript": "const x = 1;"}
        revision_id = await finish_code_app_job(
            "alice", job["job_id"], source=source,
            validation={"javascript_syntax": "passed", "data_access": "none"},
        )
        self.assertEqual((await get_user_app("alice", app_id))["status"], "ready")
        self.assertEqual((await list_user_apps("alice"))[0]["has_unpublished_revision"], 1)
        self.assertIsNone(await publish_user_app_revision("bob", app_id, revision_id))
        self.assertIsNone(await publish_user_app_revision("alice", app_id, "wrong-revision"))
        self.assertIsNone((await get_user_app("alice", app_id))["published_revision_id"])
        published = await publish_user_app_revision("alice", app_id, revision_id)
        self.assertEqual(published["published_revision_id"], revision_id)
        self.assertEqual(published["status"], "published")
        self.assertTrue(await delete_user_app("alice", app_id))
        self.assertIsNone(await get_user_app("alice", app_id))

    async def test_failed_generation_has_no_revision(self):
        job = await create_code_app_job("alice", "时钟", "创建一个桌面时钟小程序", "test-model")
        await finish_code_app_job("alice", job["job_id"], error="Syntax error")
        app = await get_user_app("alice", job["app_id"])
        self.assertEqual(app["status"], "failed")
        self.assertEqual(app["revisions"], [])
        self.assertEqual(app["latest_job"]["error"], "Syntax error")

    async def test_responsive_revision_is_private_and_keeps_published_version(self):
        first = await create_code_app_job("alice", "时钟", "创建一个桌面时钟小程序", "gpt-cloud")
        source = {"html": "<main>Clock</main>", "css": ".container{max-width:320px}", "javascript": ""}
        first_revision = await finish_code_app_job(
            "alice", first["job_id"], source=source,
            validation={"javascript_syntax": "passed", "data_access": "none"},
        )
        await publish_user_app_revision("alice", first["app_id"], first_revision)
        self.assertIsNone(await create_code_app_revision_job("bob", first["app_id"]))
        job = await create_code_app_revision_job("alice", first["app_id"], queued=True)
        self.assertEqual(job["model"], "gpt-cloud")
        self.assertEqual(job["previous_source"], source)
        with self.assertRaisesRegex(ValueError, "正在进行"):
            await create_code_app_revision_job("alice", first["app_id"])
        claimed = await claim_job("test-worker", 2)
        await progress(claimed, "validation", "正在验证")
        generating = await get_user_app("alice", first["app_id"])
        self.assertEqual(generating["latest_job"]["stage"], "validation")
        self.assertEqual(generating["published_revision_id"], first_revision)
        await finish_code_app_job("alice", job["job_id"], error="模型超时")
        failed = await get_user_app("alice", first["app_id"])
        self.assertEqual(failed["status"], "published")
        self.assertEqual(failed["published_revision_id"], first_revision)

    async def test_interrupted_web_revision_is_recoverable(self):
        first = await create_code_app_job("alice", "时钟", "创建一个桌面时钟小程序", "gpt-cloud")
        revision_id = await finish_code_app_job(
            "alice", first["job_id"],
            source={"html": "<main>Clock</main>", "css": "", "javascript": ""},
            validation={"javascript_syntax": "passed", "data_access": "none"},
        )
        await publish_user_app_revision("alice", first["app_id"], revision_id)
        await create_code_app_revision_job("alice", first["app_id"])
        self.assertEqual(await expire_jobs(), 1)
        app = await get_user_app("alice", first["app_id"])
        self.assertEqual(app["status"], "published")
        self.assertEqual(app["published_revision_id"], revision_id)
        self.assertEqual(app["latest_job"]["status"], "interrupted")
        self.assertIn("服务中断", app["latest_job"]["error"])
        self.assertIsNotNone(await create_code_app_revision_job("alice", first["app_id"]))

    async def test_responsive_revision_saves_new_draft_without_publishing(self):
        first = await create_code_app_job("alice", "时钟", "创建一个桌面时钟小程序", "gpt-cloud")
        old_source = {"html": "<main>Clock</main>", "css": ".container{max-width:320px}", "javascript": ""}
        first_revision = await finish_code_app_job(
            "alice", first["job_id"], source=old_source,
            validation={"javascript_syntax": "passed", "data_access": "none"},
        )
        await publish_user_app_revision("alice", first["app_id"], first_revision)
        job = await create_code_app_revision_job("alice", first["app_id"])
        new_source = code_apps.CodeAppSource(
            html="<main>Clock</main>", css="main{width:100%;display:grid}", javascript="",
        )
        self.assertEqual(job["previous_source"], old_source)
        await finish_code_app_job("alice", job["job_id"], source=new_source.model_dump(),
                                  validation=code_apps.validate_code_source(new_source))
        app = await get_user_app("alice", first["app_id"])
        self.assertEqual(app["status"], "published")
        self.assertEqual(app["published_revision_id"], first_revision)
        self.assertEqual([r["revision_number"] for r in app["revisions"]], [2, 1])
        self.assertEqual(app["revisions"][0]["status"], "ready")
        self.assertEqual((await list_user_apps("alice"))[0]["has_unpublished_revision"], 1)

    async def test_unvalidated_source_cannot_become_a_candidate(self):
        job = await create_code_app_job("alice", "计时器", "创建一个简洁的计时器小程序", "test-model")
        with self.assertRaises(ValueError):
            await finish_code_app_job(
                "alice", job["job_id"],
                source={"html": "<p>Start</p>", "css": "", "javascript": ""},
                validation={"javascript_syntax": "passed"},
            )
        self.assertEqual((await get_user_app("alice", job["app_id"]))["revisions"], [])

    async def test_http_detail_and_publish_enforce_authenticated_owner(self):
        job = await create_code_app_job("alice", "时钟", "创建一个桌面时钟小程序", "test-model")
        revision_id = await finish_code_app_job(
            "alice", job["job_id"],
            source={"html": "<p>Now</p>", "css": "", "javascript": ""},
            validation={"javascript_syntax": "passed", "data_access": "none"},
        )
        app = FastAPI()
        app.include_router(user_app_router)
        app.dependency_overrides[get_current_user] = lambda: {"id": "bob"}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            detail = await client.get(f"/api/user-apps/{job['app_id']}")
            publish = await client.post(
                f"/api/user-apps/{job['app_id']}/publish",
                json={"revision_id": revision_id},
            )
        self.assertEqual(detail.status_code, 404)
        self.assertEqual(publish.status_code, 404)
        self.assertIsNone((await get_user_app("alice", job["app_id"]))["published_revision_id"])

    async def test_http_delete_failed_app_is_owner_scoped_and_cascades(self):
        job = await create_code_app_job("alice", "时钟", "创建一个桌面时钟小程序", "deepseek")
        await finish_code_app_job("alice", job["job_id"], error="连接失败")
        app = FastAPI()
        app.include_router(user_app_router)
        app.dependency_overrides[get_current_user] = lambda: {"id": "bob"}
        with patch("routers.user_apps.event_bus.publish", new_callable=AsyncMock):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                denied = await client.delete(f"/api/user-apps/{job['app_id']}")
                self.assertEqual(denied.status_code, 404)
                self.assertIsNotNone(await get_user_app("alice", job["app_id"]))
                app.dependency_overrides[get_current_user] = lambda: {"id": "alice"}
                deleted = await client.delete(f"/api/user-apps/{job['app_id']}")
        self.assertEqual(deleted.status_code, 200)
        self.assertIsNone(await get_user_app("alice", job["app_id"]))
        async with connection.get_db() as db:
            async with db.execute("SELECT COUNT(*) FROM user_app_jobs WHERE app_id = ?", (job["app_id"],)) as cursor:
                self.assertEqual((await cursor.fetchone())[0], 0)

    async def test_http_responsive_revision_requires_owner_and_queues_once(self):
        first = await create_code_app_job("alice", "时钟", "创建一个桌面时钟小程序", "gpt-cloud")
        await finish_code_app_job(
            "alice", first["job_id"],
            source={"html": "<main>Clock</main>", "css": "", "javascript": ""},
            validation={"javascript_syntax": "passed", "data_access": "none"},
        )
        app = FastAPI()
        app.include_router(user_app_router)
        app.dependency_overrides[get_current_user] = lambda: {"id": "bob"}
        with patch("routers.user_apps.coding_context", new_callable=AsyncMock, return_value={"timezone": "Asia/Shanghai", "model_name": "fake-model"}), \
             patch("routers.user_apps.event_bus.publish", new_callable=AsyncMock):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                denied = await client.post(f"/api/user-apps/{first['app_id']}/responsive-revision")
                self.assertEqual(denied.status_code, 404)
                app.dependency_overrides[get_current_user] = lambda: {"id": "alice"}
                accepted = await client.post(f"/api/user-apps/{first['app_id']}/responsive-revision")
                duplicate = await client.post(f"/api/user-apps/{first['app_id']}/responsive-revision")
        self.assertEqual(accepted.status_code, 202)
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(accepted.json()["status"], "queued")
        self.assertEqual((await get_user_app("alice", first["app_id"]))["latest_job"]["stage"], "model")

    async def test_legacy_html_injection_is_not_republished_or_silently_rewritten(self):
        first = await create_code_app_job("alice", "Legacy", "保留已有面板功能并修订安全接口", "gpt-cloud")
        source = {"html": "<main></main>", "css": "", "javascript": "document.body.innerHTML = '<p>old</p>';"}
        revision_id = await finish_code_app_job("alice", first["job_id"], source=source,
            validation={"javascript_syntax": "passed", "data_access": "none"})
        app = FastAPI()
        app.include_router(user_app_router)
        app.dependency_overrides[get_current_user] = lambda: {"id": "alice"}
        with patch("routers.user_apps.coding_context", new_callable=AsyncMock,
                   return_value={"timezone": "Asia/Shanghai", "model_name": "fake-model"}), \
                patch("routers.user_apps.event_bus.publish", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                blocked = await client.post(f"/api/user-apps/{first['app_id']}/publish", json={"revision_id": revision_id})
                self.assertEqual(blocked.status_code, 422)
                self.assertIn("生成兼容草稿", blocked.json()["detail"])
                queued = await client.post(f"/api/user-apps/{first['app_id']}/responsive-revision")
                self.assertEqual(queued.status_code, 202)
        after = await get_user_app("alice", first["app_id"])
        self.assertIsNone(after["published_revision_id"])
        self.assertEqual(after["revisions"][0]["source"], source)
        async with connection.get_db() as db:
            async with db.execute("SELECT prompt FROM user_app_jobs WHERE id = ?", (queued.json()["job_id"],)) as cursor:
                self.assertIn("innerHTML", (await cursor.fetchone())[0])

    async def test_publish_revalidates_safe_source_and_reports_validator_unavailability(self):
        first = await create_code_app_job("alice", "Safe", "创建一个安全的简单面板小程序", "gpt-cloud")
        revision_id = await finish_code_app_job("alice", first["job_id"],
            source={"html": "<main>Ready</main>", "css": "", "javascript": ""},
            validation={"javascript_syntax": "passed", "data_access": "none"})
        app = FastAPI()
        app.include_router(user_app_router)
        app.dependency_overrides[get_current_user] = lambda: {"id": "alice"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with patch("routers.user_apps.validate_code_source", side_effect=RuntimeError("Node unavailable")):
                unavailable = await client.post(f"/api/user-apps/{first['app_id']}/publish", json={"revision_id": revision_id})
            self.assertEqual(unavailable.status_code, 503)
            self.assertIsNone((await get_user_app("alice", first["app_id"]))["published_revision_id"])
            with patch("routers.user_apps.event_bus.publish", new_callable=AsyncMock):
                published = await client.post(f"/api/user-apps/{first['app_id']}/publish", json={"revision_id": revision_id})
        self.assertEqual(published.status_code, 200)
        self.assertEqual(published.json()["published_revision_id"], revision_id)

    def test_validation_distinguishes_function_callbacks_from_dynamic_constructor(self):
        for javascript in ["(function(){ const n = 1; })();", "setTimeout(function () {}, 100);",
                           "document.querySelector('button').addEventListener('click', function(event) {});"]:
            result = code_apps.validate_code_source(code_apps.CodeAppSource(
                html="<button>Run</button>", css="", javascript=javascript))
            self.assertEqual(result["javascript_syntax"], "passed")
        for javascript in ["Function('return 1')();", "new Function('return 1')();", "window.Function('return 1')();"]:
            with self.assertRaisesRegex(ValueError, "dynamic execution"):
                code_apps.validate_code_source(code_apps.CodeAppSource(html="<button/>", css="", javascript=javascript))

    def test_validation_parses_javascript_without_executing_it(self):
        with self.assertRaises(ValueError):
            code_apps.validate_code_source(code_apps.CodeAppSource(
                html="<script>alert(1)</script>", css="", javascript=""
            ))
        with self.assertRaises(ValueError):
            code_apps.validate_code_source(code_apps.CodeAppSource(
                html="<p>Hello</p>", css="", javascript="function {"
            ))
        result = code_apps.validate_code_source(code_apps.CodeAppSource(
            html="<p>Hello</p>", css="", javascript="const ready = true;"
        ))
        self.assertEqual(result["javascript_syntax"], "passed")

    def test_validation_rejects_narrow_root_but_allows_small_components(self):
        with self.assertRaisesRegex(ValueError, "narrow width"):
            code_apps.validate_code_source(code_apps.CodeAppSource(
                html="<div class='container'>Hi</div>",
                css=".container { max-width: 320px; margin: auto; }", javascript="",
            ))
        result = code_apps.validate_code_source(code_apps.CodeAppSource(
            html="<main>Hi</main>",
            css="main { width: 100%; max-width: 960px; } .card { max-width: 320px; }",
            javascript="",
        ))
        self.assertEqual(result["javascript_syntax"], "passed")

    def test_codegen_prompt_requires_native_responsive_layout(self):
        self.assertIn("不要用 320px 的固定宽度", code_apps.SYSTEM_PROMPT)
        self.assertIn("devicePixelRatio", code_apps.SYSTEM_PROMPT)

    def test_codegen_model_selection_never_falls_back(self):
        with patch.object(code_apps, "get_strict_model", return_value="selected") as strict:
            self.assertEqual(code_apps.create_codegen_model("deepseek"), "selected")
            strict.assert_called_once_with("deepseek", isolated_http_clients=True)
        from core.config import config
        with patch.object(config, "DEEPSEEK_API_KEY", ""):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                code_apps.create_codegen_model("deepseek")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            code_apps.create_codegen_model("unknown")

    async def test_codegen_service_queues_without_publishing(self):
        with patch.object(code_apps, "coding_context", new_callable=AsyncMock, return_value={"timezone": "Asia/Shanghai", "model_name": "fake-model"}), \
             patch.object(code_apps.event_bus, "publish", new_callable=AsyncMock):
            result = await code_apps.create_code_app(
                "alice", code_apps.CodeAppRequest(
                    title="计时器", prompt="创建一个可以开始暂停的计时器小程序"
                ), "gpt-cloud",
            )
        self.assertEqual(result["status"], "queued")
        app = await get_user_app("alice", result["app_id"])
        self.assertIsNone(app["published_revision_id"])
        self.assertEqual(app["revisions"], [])

    async def test_codegen_invalid_identity_never_queues(self):
        with self.assertRaises(MissingUserContextError):
            await code_apps.create_code_app("default", code_apps.CodeAppRequest(
                title="计时器", prompt="创建一个可以开始暂停的计时器小程序"), "deepseek")

    async def test_old_timeline_schema_migrates_without_changing_app_id(self):
        legacy_path = str(Path(self.directory.name) / "legacy.db")
        legacy_spec = {
            "root": "timeline",
            "elements": {"timeline": {"type": "TodoTimeline", "props": {}, "children": []}},
        }
        async with aiosqlite.connect(legacy_path) as db:
            await db.execute(
                """
                CREATE TABLE user_apps (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL,
                    title TEXT NOT NULL, spec_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    UNIQUE(user_id, kind)
                )
                """
            )
            await db.execute(
                "INSERT INTO user_apps VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("legacy-app", "alice", "todo_timeline", "旧时间线",
                 json.dumps(legacy_spec), "2026-01-01", "2026-01-01"),
            )
            await db.commit()
        with patch.object(connection, "DATABASE_PATH", legacy_path):
            await connection.init_db()
            migrated = await get_user_app("alice", "legacy-app")
            self.assertEqual(migrated["id"], "legacy-app")
            self.assertEqual(migrated["published_revision_id"], "legacy-app:v1")
            self.assertEqual(migrated["revisions"][0]["source"], legacy_spec)
            await connection.init_db()
            self.assertEqual(len((await get_user_app("alice", "legacy-app"))["revisions"]), 1)


if __name__ == "__main__":
    unittest.main()
