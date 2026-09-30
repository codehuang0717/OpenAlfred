"""Coding loop, tenant reads, checkpoint recovery and job lifecycle regressions."""

import asyncio
import hashlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import aiosqlite
import httpx
from fastapi import FastAPI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents.middleware import ModelRequest
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import APIConnectionError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.config import config
from db import connection
from db.coding_jobs import app_jobs, claim_job, expire_jobs, get_job, job_events, progress, requeue_job, stop_job
from db.user_apps import create_code_app_job, delete_user_app, finish_code_app_job, get_user_app
from logic import coding_agent as coder
from routers.auth import get_current_user
from routers.user_apps import router
from services import code_apps
from services.coding_worker import CodingWorker, checkpoint_config
from tools.user_apps import create_standalone_mini_app


def call(name: str, args: dict, identity: str = "call") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": identity}],
                     response_metadata={"finish_reason": "tool_calls"})


class ScriptedModel(BaseChatModel):
    responses: list
    model_name: str = "fake-model"

    @property
    def _llm_type(self):
        return "scripted-coder"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses.pop(0)
        if callable(response):
            response = response(messages)
        if isinstance(response, Exception):
            raise response
        return ChatResult(generations=[ChatGeneration(message=response)])


def working_script() -> list:
    return [call("write_file", {"path": "index.html", "content": "<main>Ready</main>"}, "html"),
            call("write_file", {"path": "styles.css", "content": "main{width:100%}"}, "css"),
            call("write_file", {"path": "app.js", "content": "const ok = true;"}, "js"),
            call("validate_app", {}, "validate"),
            call("finish_task", {"summary": "已生成响应式草稿，等待预览。"}, "finish")]


class TestCodingAgent(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path_patch = patch.object(connection, "DATABASE_PATH", str(Path(self.directory.name) / "apps.db"))
        self.path_patch.start()
        await connection.init_db()
        self.notifications = patch("services.coding_worker.event_bus.publish", new_callable=AsyncMock)
        self.notifications.start()

    async def asyncTearDown(self):
        self.notifications.stop()
        self.path_patch.stop()
        self.directory.cleanup()

    async def job(self, owner="alice"):
        result = await create_code_app_job(owner, "我的面板", "创建一个显示待办事项的面板", "gpt-cloud", queued=True,
            context={"timezone": "Asia/Shanghai", "model_name": "fake-model", "source_thread_id": "chat-1"})
        return await get_job(owner, result["job_id"])

    async def invoke(self, responses, state=None):
        stages = []

        async def on_progress(stage, message, metrics):
            stages.append(stage)

        job = await self.job()
        graph, harness = coder.build_coding_agent(ScriptedModel(responses=responses), "gpt-cloud", InMemorySaver(), on_progress)
        result = await graph.ainvoke(state or coder.initial_coding_state(job), config=checkpoint_config(job))
        return result, harness, stages

    async def test_real_framework_tool_loop_finishes_only_validated_source(self):
        state, harness, stages = await self.invoke(working_script())
        self.assertEqual(state["summary"], "已生成响应式草稿，等待预览。")
        self.assertEqual(state["validation"]["source_hash"], coder.source_hash(state["files"]))
        self.assertEqual(harness.model_calls, 5)
        self.assertIn("validation", stages)

    async def test_failed_validation_is_observed_and_repaired(self):
        script = working_script()
        script[2] = call("write_file", {"path": "app.js", "content": "function {"}, "broken")
        script[4:4] = [call("write_file", {"path": "app.js", "content": "const ok = true;"}, "repair"),
                        call("validate_app", {}, "revalidate")]
        state, _, stages = await self.invoke(script)
        self.assertEqual(state["validation_failures"], 1)
        self.assertIn("repairing", stages)
        self.assertEqual(state["validation"]["javascript_syntax"], "passed")

    async def test_private_read_tools_use_bound_owner_and_timezone(self):
        from core.database import add_todo
        await add_todo(id="alice-todo", title="Alice only", user_id="alice")
        await add_todo(id="bob-todo", title="Bob secret", user_id="bob")
        state, _, _ = await self.invoke([call("get_todos", {}, "read-todos"), *working_script()])
        observations = "\n".join(state["private_results"].values())
        self.assertIn("Alice only", observations)
        self.assertNotIn("Bob secret", observations)
        self.assertEqual(state["read_tools"], ["get_todos"])
        self.assertEqual(state["validation"]["private_data_tools"], ["get_todos"])
        names = {tool.name for tool in coder.private_read_tools()}
        self.assertTrue({"get_todos", "read_email", "search_knowledge", "get_user_profile"} <= names)
        self.assertFalse({"add_todo", "delete_todo", "update_user_memory", "call_user"} & names)

    async def test_paths_cannot_escape_workspace_and_patch_checks_hash(self):
        runtime = SimpleNamespace(config={"configurable": {"owner": "alice"}}, state={"files": {"app.js": "let x=1;"}}, tool_call_id="patch")
        for path in ["../.env", "D:/secret", "\\\\server\\share", "data/private.txt"]:
            with self.assertRaises(ValueError):
                coder.write_file.func(runtime=runtime, path=path, content="secret")
        with self.assertRaisesRegex(ValueError, "已变化"):
            coder.apply_patch.func(runtime=runtime, path="app.js", old="x=1", new="x=2", base_hash="bad")
        result = coder.apply_patch.func(runtime=runtime, path="app.js", old="x=1", new="x=2", base_hash=hashlib.sha256(b"let x=1;").hexdigest())
        self.assertEqual(result.update["files"]["app.js"], "let x=2;")
        self.assertEqual(result.update["validation"], {})

    async def test_bare_final_answer_truncation_and_empty_response_are_failures(self):
        for response in [AIMessage(content="完成", response_metadata={"finish_reason": "stop"}),
                         AIMessage(content="", response_metadata={"finish_reason": "stop"}),
                         AIMessage(content="半截", response_metadata={"finish_reason": "length"})]:
            with self.subTest(response=response):
                with self.assertRaises(RuntimeError):
                    await self.invoke([response])

    async def test_modified_source_invalidates_validation_and_can_be_revalidated(self):
        script = working_script()
        script.insert(4, call("write_file", {"path": "app.js", "content": "const changed=true;"}, "change"))
        script.extend([call("validate_app", {}, "revalidate"),
                       call("finish_task", {"summary": "重新验证后交付草稿"}, "finish-again")])
        state, _, _ = await self.invoke(script)
        premature = next(m for m in state["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "finish")
        self.assertEqual(premature.status, "error")
        self.assertEqual(state["summary"], "重新验证后交付草稿")
        self.assertEqual(state["validation"]["source_hash"], coder.source_hash(state["files"]))

    async def test_call_budget_is_hard_and_does_not_retry(self):
        with patch.object(config, "CODING_MAX_MODEL_CALLS", 1):
            with self.assertRaisesRegex(RuntimeError, "调用次数"):
                await self.invoke(working_script())

    async def test_claim_limits_per_user_and_global_and_records_events(self):
        a = await self.job("alice")
        await self.job("alice")
        await self.job("bob")
        first = await claim_job("worker", 2)
        second = await claim_job("worker", 2)
        self.assertNotEqual(first["user_id"], second["user_id"])
        self.assertIsNone(await claim_job("other-worker", 2))
        self.assertTrue(await job_events(first["user_id"], first["id"]))
        self.assertIsNone(await job_events("eve", a["id"]))

    async def test_cancel_and_delete_fence_late_completion(self):
        queued = await self.job()
        job = await claim_job("worker", 2)
        self.assertFalse(await stop_job("bob", job["id"]))
        self.assertTrue(await stop_job("alice", job["id"]))
        with self.assertRaises(ValueError):
            await finish_code_app_job("alice", job["id"], epoch=job["epoch"], error="late result")
        self.assertTrue(await delete_user_app("alice", queued["app_id"]))
        with self.assertRaises(ValueError):
            await progress(job, "coding", "late progress")
        self.assertIsNone(await get_job("alice", job["id"]))

    async def test_expired_lease_is_interrupted_without_automatic_retry(self):
        await self.job()
        job = await claim_job("worker", 2)
        async with connection.get_db() as db:
            await db.execute("UPDATE user_app_jobs SET lease_until = ? WHERE id = ?", (time.time() - 1, job["id"]))
            await db.commit()
        self.assertEqual(await expire_jobs(), 1)
        self.assertEqual((await get_job("alice", job["id"]))["status"], "interrupted")
        self.assertIsNone(await claim_job("worker", 2))
        self.assertIsNone(await requeue_job("bob", job["id"], resume=True))
        await requeue_job("alice", job["id"], resume=False)
        newer = await claim_job("worker", 2)
        self.assertGreater(newer["epoch"], job["epoch"])
        self.assertNotEqual(checkpoint_config(newer)["configurable"]["thread_id"], checkpoint_config(job)["configurable"]["thread_id"])

    async def test_worker_persists_ready_draft_and_report_without_publication(self):
        await self.job()
        job = await claim_job("worker", 2)
        async with aiosqlite.connect(str(Path(self.directory.name) / "checkpoints.db")) as conn:
            worker = CodingWorker()
            worker.saver = AsyncSqliteSaver(conn)
            await worker.saver.setup()
            await worker.run_job(job, model=ScriptedModel(responses=working_script()))
        app = await get_user_app("alice", job["app_id"])
        self.assertEqual(app["latest_job"]["status"], "ready")
        self.assertTrue(app["latest_job"]["report"])
        self.assertIsNone(app["published_revision_id"])
        self.assertEqual(app["revisions"][0]["status"], "ready")

    async def test_worker_model_error_is_visible_without_fallback(self):
        await self.job()
        job = await claim_job("worker", 2)
        worker = CodingWorker()
        with patch("services.coding_worker.create_codegen_model", side_effect=ValueError("wrong model")) as factory:
            await worker.run_job(job)
        current = await get_job("alice", job["id"])
        self.assertEqual(current["status"], "failed")
        self.assertIn("ValueError", current["error"])
        self.assertIn("未切换模型", current["error"])
        factory.assert_called_once_with("gpt-cloud")

    async def test_api_connection_error_retains_type_and_never_publishes(self):
        await self.job()
        job = await claim_job("worker", 2)
        worker = CodingWorker()
        failure = APIConnectionError(request=httpx.Request("POST", "https://test.invalid"))
        await worker.run_job(job, model=ScriptedModel(responses=[failure]))
        current = await get_job("alice", job["id"])
        self.assertEqual(current["status"], "failed")
        self.assertIn("APIConnectionError", current["error"])
        self.assertEqual(current["metrics"]["model_calls"], 1)
        self.assertGreater(current["metrics"]["token_upper_bound"], 0)
        self.assertIsNone(current["revision_id"])

    async def test_disk_checkpoint_resumes_after_restart_without_rewriting_files(self):
        await self.job()
        job = await claim_job("worker", 2)
        checkpoint_path = str(Path(self.directory.name) / "resume.db")
        script = working_script()
        async with aiosqlite.connect(checkpoint_path) as conn:
            worker = CodingWorker()
            worker.saver = AsyncSqliteSaver(conn)
            await worker.saver.setup()
            await worker.run_job(job, model=ScriptedModel(responses=[*script[:2], RuntimeError("connection interrupted")]))
        before = await get_job("alice", job["id"])
        self.assertEqual(before["metrics"]["model_calls"], 3)
        await requeue_job("alice", job["id"], resume=True)
        resumed = await claim_job("restarted-worker", 2)
        async with aiosqlite.connect(checkpoint_path) as conn:
            worker = CodingWorker()
            worker.saver = AsyncSqliteSaver(conn)
            await worker.run_job(resumed, model=ScriptedModel(responses=script[2:]))
        after = await get_job("alice", job["id"])
        self.assertEqual(after["status"], "ready")
        self.assertEqual(after["metrics"]["model_calls"], 6)
        self.assertGreater(after["metrics"]["token_upper_bound"], before["metrics"]["token_upper_bound"])
        app = await get_user_app("alice", job["app_id"])
        self.assertEqual(app["revisions"][0]["source"]["html"], "<main>Ready</main>")

    async def test_resume_without_checkpoint_fails_explicitly(self):
        await self.job()
        job = await claim_job("worker", 2)
        await stop_job("alice", job["id"])
        await requeue_job("alice", job["id"], resume=True)
        resumed = await claim_job("worker", 2)
        async with aiosqlite.connect(":memory:") as conn:
            worker = CodingWorker()
            worker.saver = AsyncSqliteSaver(conn)
            await worker.saver.setup()
            model = ScriptedModel(responses=working_script())
            await worker.run_job(resumed, model=model)
        current = await get_job("alice", job["id"])
        self.assertEqual(current["status"], "failed")
        self.assertIn("没有可恢复", current["error"])
        self.assertEqual(len(model.responses), 5)

    async def test_retry_starts_fresh_budget_but_resume_keeps_failed_call_reservation(self):
        await self.job()
        job = await claim_job("worker", 2)
        await progress(job, "coding", "request reserved", metrics={"model_calls": 24, "token_upper_bound": 300000})
        await stop_job("alice", job["id"])
        await requeue_job("alice", job["id"], resume=True)
        resumed = await claim_job("worker", 2)
        self.assertEqual(resumed["metrics"]["model_calls"], 24)
        await stop_job("alice", job["id"])
        await requeue_job("alice", job["id"], resume=False)
        retried = await claim_job("worker", 2)
        self.assertEqual(retried["metrics"], {})

    async def test_delete_api_erases_private_checkpoints_and_keeps_other_users(self):
        from core.database import add_todo
        await add_todo(id="private", title="Sensitive task", user_id="alice")
        alice = await self.job()
        bob = await self.job("bob")
        claimed = await claim_job("worker", 2)
        async with aiosqlite.connect(":memory:") as conn:
            worker = CodingWorker()
            worker.saver = AsyncSqliteSaver(conn)
            await worker.saver.setup()
            await worker.run_job(claimed, model=ScriptedModel(responses=[call("get_todos", {}), *working_script()]))
            before = await worker.saver.aget_tuple(checkpoint_config(claimed))
            self.assertTrue(before.checkpoint["channel_values"]["private_results"])
            app = FastAPI()
            app.include_router(router)
            app.state.coding_worker = worker
            app.dependency_overrides[get_current_user] = lambda: {"id": "alice"}
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                self.assertEqual((await client.delete(f"/api/user-apps/{bob['app_id']}")).status_code, 404)
                self.assertEqual((await client.delete(f"/api/user-apps/{alice['app_id']}")).status_code, 200)
            self.assertIsNone(await worker.saver.aget_tuple(checkpoint_config(claimed)))
        self.assertIsNotNone(await get_job("bob", bob["id"]))
        self.assertEqual(await app_jobs("alice", alice["app_id"]), [])

    async def test_cancelled_child_does_not_kill_worker_pool(self):
        alice = await self.job()
        bob = await self.job("bob")
        entered = asyncio.Event()
        drained = asyncio.Event()

        async def controlled_run(job):
            if job["user_id"] == "alice":
                entered.set()
                await asyncio.Event().wait()
            else:
                await finish_code_app_job("bob", job["id"], error="deliberate test stop", epoch=job["epoch"])
                drained.set()

        worker = CodingWorker(Path(self.directory.name) / "pool.db")
        with patch.object(config, "CODING_CONCURRENCY", 1), patch.object(worker, "run_job", side_effect=controlled_run):
            await worker.start()
            try:
                await asyncio.wait_for(entered.wait(), 3)
                await stop_job("alice", alice["id"])
                worker.running[alice["id"]].cancel()
                await asyncio.wait_for(drained.wait(), 3)
                self.assertFalse(worker.tasks[0].done())
            finally:
                await worker.close()
        self.assertIsNone(worker.conn)
        self.assertEqual((await get_job("bob", bob["id"]))["status"], "failed")

    async def test_input_and_total_budget_fail_before_dispatch(self):
        for field in ("CODING_INPUT_TOKENS", "CODING_TOTAL_TOKENS"):
            with self.subTest(field=field), patch.object(config, field, 1):
                with self.assertRaisesRegex(RuntimeError, "预算"):
                    await self.invoke(working_script())

    async def test_zero_disables_optional_token_caps_not_call_limit(self):
        with patch.object(config, "CODING_INPUT_TOKENS", 0), patch.object(config, "CODING_TOTAL_TOKENS", 0):
            result, _, _ = await self.invoke(working_script())
            self.assertTrue(result["summary"])
            with patch.object(config, "CODING_MAX_MODEL_CALLS", 1):
                with self.assertRaisesRegex(RuntimeError, "调用次数"):
                    await self.invoke(working_script())

    async def test_utf8_bytes_and_duplicate_sdk_metadata_do_not_inflate_tokens(self):
        script = working_script()
        script[0].tool_calls[0]["args"]["content"] = "<main>" + "待办事项" * 3500 + "</main>"
        script[0].additional_kwargs["tool_calls"] = [
            {"id": "html", "type": "function", "function": {"name": "write_file",
             "arguments": json.dumps(script[0].tool_calls[0]["args"], ensure_ascii=False)}}]
        script[0].additional_kwargs["reasoning_content"] = "reasoning " * 1000
        self.assertGreater(len(json.dumps(script[0].model_dump(mode="json"), ensure_ascii=False).encode()), 48000)
        with patch.object(config, "CODING_INPUT_TOKENS", 48000), patch.object(config, "CODING_TOTAL_TOKENS", 0):
            result, harness, _ = await self.invoke(script)
        self.assertTrue(result["summary"])
        self.assertEqual(harness.model_calls, 5)
        self.assertEqual(harness.metrics()["token_count_method"], "cl100k_payload_estimate")

    async def test_budget_counts_reasoning_and_full_tool_schema(self):
        harness = coder.CodingHarness("gpt-cloud", AsyncMock())
        message = call("list_files", {})
        plain = harness.token_counter.message_tokens([message])
        message.additional_kwargs["reasoning_content"] = "bounded reasoning " * 100
        self.assertGreater(harness.token_counter.message_tokens([message]), plain)
        tools = [coder.read_file]
        schema = harness.token_counter.tool_tokens(tools)
        text = HumanMessage(content="read a file")
        request = ModelRequest(model=ScriptedModel(responses=[]), messages=[text], tools=tools, state={})
        handler = AsyncMock()
        with patch.object(config, "CODING_INPUT_TOKENS", harness.token_counter.message_tokens([text]) + schema - 1):
            with self.assertRaisesRegex(RuntimeError, "估算"):
                await harness.awrap_model_call(request, handler)
        handler.assert_not_awaited()

    async def test_private_email_does_not_choose_first_account(self):
        state, _, _ = await self.invoke([
            call("read_email", {"email_id": "1", "account_id": ""}, "email"), *working_script(),
        ])
        response = next(m for m in state["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "email")
        self.assertEqual(response.status, "error")
        self.assertIn("不能回退", response.content)
        self.assertEqual(state["read_tools"], [])

    async def test_large_private_result_is_paginated_not_repeated_in_model_history(self):
        from core.database import add_todo
        for number in range(35):
            await add_todo(id=f"todo-{number}", title=f"Task {number}: " + "x" * 140, user_id="alice")
        state, _, _ = await self.invoke([call("get_todos", {}, "read"), *working_script()])
        path, body = next(iter(state["private_results"].items()))
        message = next(m for m in state["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "read")
        self.assertLess(len(message.content), 2400)
        self.assertIn(path, message.content)
        self.assertGreater(len(body.splitlines()), 100)
        runtime = SimpleNamespace(config={"configurable": {"owner": "alice"}}, state=state)
        result = coder.read_file.func(runtime=runtime, path=path, start_line=101, line_count=20)
        self.assertEqual(result["total_lines"], len(body.splitlines()))
        self.assertTrue(result["content"])

    async def test_long_single_line_artifact_can_be_read_in_full(self):
        body = "z" * 25001
        runtime = SimpleNamespace(config={"configurable": {"owner": "alice"}},
            state={"private_results": {"data/long.txt": body}})
        offset = 0
        parts = []
        while True:
            result = coder.read_file.func(runtime=runtime, path="data/long.txt", start_char=offset)
            parts.append(result["content"])
            if result["next_char"] is None:
                break
            offset = result["next_char"]
        self.assertEqual("".join(parts), body)

    async def test_relative_time_uses_submission_date_in_bound_timezone(self):
        job = await self.job()
        job["created_at"] = "2026-09-29T20:00:00+00:00"
        initial = coder.initial_coding_state(job)
        self.assertIn("2026-09-30T04:00:00+08:00", initial["messages"][0].content)

    async def test_startup_purges_checkpoint_orphaned_by_crash_during_delete(self):
        await self.job()
        job = await claim_job("worker", 2)
        checkpoint_path = Path(self.directory.name) / "orphans.db"
        async with aiosqlite.connect(str(checkpoint_path)) as conn:
            worker = CodingWorker()
            worker.saver = AsyncSqliteSaver(conn)
            await worker.saver.setup()
            await worker.run_job(job, model=ScriptedModel(responses=working_script()))
            self.assertIsNotNone(await worker.saver.aget_tuple(checkpoint_config(job)))
        await delete_user_app("alice", job["app_id"])
        worker = CodingWorker(checkpoint_path)
        await worker.start()
        try:
            self.assertIsNone(await worker.saver.aget_tuple(checkpoint_config(job)))
        finally:
            await worker.close()

    def test_snapshot_validation_rejects_navigation_and_html_execution(self):
        for html in ['<a href="https://test.invalid">link</a>', '<a href="&#104;ttps://test.invalid">link</a>',
                     '<form>submit</form>', '<button onclick="leak()">go</button>']:
            with self.subTest(html=html), self.assertRaises(ValueError):
                code_apps.validate_code_source(code_apps.CodeAppSource(html=html, css="", javascript=""))
        for javascript in ['fetch("https://test.invalid")', 'location.href="https://test.invalid"',
                           'window.open("https://test.invalid")', 'document.body.innerHTML=mail', 'eval(code)',
                           'new Function(code)', 'document.cookie', 'new WebSocket("wss://test.invalid")']:
            with self.subTest(javascript=javascript), self.assertRaises(ValueError):
                code_apps.validate_code_source(code_apps.CodeAppSource(html="<p>Snapshot</p>", css="", javascript=javascript))

    async def test_worker_startup_failure_is_visible_and_closes_connection(self):
        worker = CodingWorker(Path(self.directory.name) / "bad.db")
        with patch.object(worker, "_purge_orphans", side_effect=RuntimeError("bad checkpoint")):
            with self.assertRaisesRegex(RuntimeError, "bad checkpoint"):
                await worker.start()
        self.assertIsNone(worker.conn)
        self.assertEqual(worker.tasks, [])

    async def test_cancel_during_database_initialization_still_closes_connection(self):
        db = SimpleNamespace(execute=AsyncMock(side_effect=asyncio.CancelledError()), close=AsyncMock())
        with patch.object(connection.aiosqlite, "connect", new_callable=AsyncMock, return_value=db):
            with self.assertRaises(asyncio.CancelledError):
                async with connection.get_db():
                    self.fail("Initialization should have been cancelled")
        db.close.assert_awaited_once()

    async def test_late_cancellation_cannot_stop_explicitly_resumed_job(self):
        await self.job()
        job = await claim_job("old-worker", 2)
        entered = asyncio.Event()

        async def hold_progress(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        worker = CodingWorker()
        with patch("services.coding_worker.progress", side_effect=hold_progress):
            task = asyncio.create_task(worker.run_job(job, model=ScriptedModel(responses=working_script())))
            try:
                await asyncio.wait_for(entered.wait(), 3)
                await stop_job("alice", job["id"])
                await requeue_job("alice", job["id"], resume=True)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                current = await get_job("alice", job["id"])
                self.assertEqual(current["status"], "queued")
                self.assertGreater(current["epoch"], job["epoch"])
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_private_reader_text_errors_are_not_saved_as_success(self):
        harness = coder.CodingHarness("gpt-cloud", AsyncMock())
        for name, content in (("search_knowledge", "Knowledge search failed: backend unavailable"),
                              ("list_knowledge", "Failed to list documents: timeout"),
                              ("get_user_memory_category", "无效类别 'other'")):
            message = call(name, {}, "read-error")
            request = SimpleNamespace(tool_call=message.tool_calls[0], state={"messages": [message]})
            result = await harness.awrap_tool_call(request, AsyncMock(return_value=ToolMessage(content=content, tool_call_id="read-error")))
            self.assertIsInstance(result, ToolMessage)
            self.assertEqual(result.status, "error")

    async def test_private_reader_limits_are_rejected_without_silent_clamping(self):
        harness = coder.CodingHarness("gpt-cloud", AsyncMock())
        for name, args in (("get_recent_emails", {"limit": 500}), ("search_knowledge", {"query": "x", "top_k": 100})):
            message = call(name, args, "read-limit")
            handler = AsyncMock()
            result = await harness.awrap_tool_call(SimpleNamespace(tool_call=message.tool_calls[0], state={"messages": [message]}), handler)
            self.assertEqual(result.status, "error")
            handler.assert_not_awaited()

    async def test_legacy_job_schema_migration_preserves_owner_and_existing_error(self):
        legacy_path = str(Path(self.directory.name) / "legacy-jobs.db")
        async with aiosqlite.connect(legacy_path) as db:
            await db.execute("""CREATE TABLE user_apps (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL,
                title TEXT NOT NULL, spec_json TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(user_id, kind))""")
            await db.execute("INSERT INTO user_apps VALUES ('legacy-app', 'alice', 'code_app:legacy-app', 'old', '{}', '2026-01-01', '2026-01-01')")
            await db.execute("""CREATE TABLE user_app_jobs (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, app_id TEXT NOT NULL,
                prompt TEXT NOT NULL, model TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued', 'generating', 'ready', 'failed')),
                error TEXT, revision_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                FOREIGN KEY(app_id) REFERENCES user_apps(id) ON DELETE CASCADE)""")
            await db.execute("INSERT INTO user_app_jobs VALUES ('legacy-job', 'alice', 'legacy-app', 'old prompt', 'gpt-cloud', 'failed', 'original failure', NULL, '2026-01-01', '2026-01-01')")
            await db.commit()
        with patch.object(connection, "DATABASE_PATH", legacy_path):
            await connection.init_db()
            await connection.init_db()
            job = await get_job("alice", "legacy-job")
            self.assertEqual(job["error"], "original failure")
            self.assertEqual(job["model"], "gpt-cloud")
            self.assertEqual(job["status"], "failed")
            self.assertIsNone(await get_job("bob", "legacy-job"))
            self.assertEqual(job["metrics"], {})
            await requeue_job("alice", "legacy-job", resume=False)
            self.assertTrue(await stop_job("alice", "legacy-job"))

    async def test_submission_returns_immediately_with_private_tool_scope(self):
        runtime = SimpleNamespace(config={"configurable": {"owner": "alice", "thread_id": "chat-1", "timezone": "Asia/Shanghai", "model_selection": "gpt-cloud"}})
        with patch.object(code_apps, "create_codegen_model", return_value=SimpleNamespace(model_name="fake-model")):
            result = json.loads(await create_standalone_mini_app.coroutine(runtime=runtime, title="面板", requirements="请把我的待办整理为一个时间线面板"))
        self.assertEqual(result["type"], "coding_task")
        self.assertEqual(result["status"], "queued")
        job = await get_job("alice", result["job_id"])
        self.assertEqual(job["context"]["source_thread_id"], "chat-1")
        self.assertIsNone(await get_job("bob", result["job_id"]))

    async def test_job_http_apis_are_owner_scoped_and_do_not_expose_context(self):
        job = await self.job()
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: {"id": "bob"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for suffix, method in [("", "get"), ("/events", "get"), ("/stream", "get"), ("/cancel", "post"), ("/restart", "post")]:
                kwargs = {"json": {"resume": True}} if suffix == "/restart" else {}
                response = await getattr(client, method)(f"/api/user-apps/jobs/{job['id']}{suffix}", **kwargs)
                self.assertEqual(response.status_code, 404)
            app.dependency_overrides[get_current_user] = lambda: {"id": "alice"}
            response = await client.get(f"/api/user-apps/jobs/{job['id']}/events")
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("context", response.json()["job"])
            self.assertEqual((await client.post(f"/api/user-apps/jobs/{job['id']}/cancel")).status_code, 200)
