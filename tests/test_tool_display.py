"""Exercise actual ToolNode execution, persistence, parallel isolation and replay."""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

import db.connection  # noqa: F401 — initialize packages in application startup order.
from services.tool_observations import (
    TOOL_LABELS,
    display_from_artifact,
    failed,
    fields,
    input_display,
    observe_tool_call,
    observed,
)


def graph_for(tools):
    graph = StateGraph(MessagesState)
    graph.add_node(
        "tools",
        ToolNode(tools, awrap_tool_call=observe_tool_call, handle_tool_errors=True),
    )
    graph.set_entry_point("tools")
    graph.add_edge("tools", END)
    return graph.compile()


class TestToolDisplay(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_sync_async_command_and_image_keep_independent_observations(
        self,
    ):
        @tool
        def sync_read(value: str) -> str:
            """Read deterministic fixture data."""
            return observed(value, "已读取 " + value, details=fields(结果=value))

        @tool
        async def write_command(runtime: ToolRuntime) -> Command:
            """Write once with a Command result."""
            await asyncio.sleep(0)
            observed(None, "已保存草稿", details=fields(状态="尚未发送"))
            return Command(
                update={
                    "messages": [
                        ToolMessage(content="saved", tool_call_id=runtime.tool_call_id)
                    ]
                }
            )

        @tool(response_format="content_and_artifact")
        async def image() -> tuple[str, dict]:
            """Return an existing image artifact."""
            return observed(
                (
                    "image ready",
                    {
                        "type": "generated_image",
                        "url": "/api/generated-images/" + "a" * 32,
                    },
                ),
                "图片已生成",
            )

        graph = graph_for([sync_read, write_command, image])
        calls = [
            {"id": "read", "name": "sync_read", "args": {"value": "alpha"}},
            {"id": "write", "name": "write_command", "args": {}},
            {"id": "image", "name": "image", "args": {}},
        ]
        events = [
            event
            async for event in graph.astream(
                {"messages": [AIMessage(content="", tool_calls=calls)]},
                stream_mode=["custom", "values"],
            )
        ]
        final = [value for mode, value in events if mode == "values"][-1]
        results = {
            message.tool_call_id: message
            for message in final["messages"]
            if isinstance(message, ToolMessage)
        }
        self.assertEqual(
            display_from_artifact(results["read"].artifact)["summary"], "已读取 alpha"
        )
        self.assertEqual(results["write"].content, "saved")
        self.assertEqual(
            display_from_artifact(results["write"].artifact)["summary"], "已保存草稿"
        )
        self.assertEqual(results["image"].artifact["type"], "generated_image")
        displays = [
            value
            for mode, value in events
            if mode == "custom" and value.get("type") == "tool_display"
        ]
        self.assertEqual(len(displays), 6)
        for key, message in results.items():
            terminal = next(
                event
                for event in displays
                if event["id"] == key and event["display"]["status"] != "running"
            )
            self.assertEqual(terminal["display"], message.artifact["tool_display"])
            self.assertGreaterEqual(terminal["display"]["elapsed_ms"], 0)

    async def test_returned_failure_exception_and_unobserved_prose_never_become_success(
        self,
    ):
        executed = []

        @tool
        async def returned_failure() -> str:
            """Return a known business failure."""
            executed.append("failed")
            return failed("ERROR: rejected", "业务操作失败")

        @tool
        async def unobserved() -> str:
            """Return prose without verified business evidence."""
            executed.append("unknown")
            return "SUCCESS"  # Human text is never evidence.

        @tool
        async def thrown() -> str:
            """Raise a service failure."""
            executed.append("exception")
            raise ValueError("bad input")

        calls = [
            {"id": name, "name": name, "args": {}}
            for name in ["returned_failure", "unobserved", "thrown"]
        ]
        final = await graph_for([returned_failure, unobserved, thrown]).ainvoke(
            {"messages": [AIMessage(content="", tool_calls=calls)]}
        )
        messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(
            [m.artifact["tool_display"]["status"] for m in messages],
            ["failed", "unknown", "failed"],
        )
        self.assertEqual(messages[0].status, "error")
        self.assertCountEqual(executed, ["failed", "unknown", "exception"])

    async def test_cancellation_distinguishes_read_from_unconfirmed_side_effect(self):
        for name, expected in [("get_todos", "interrupted"), ("add_todo", "unknown")]:
            events = []
            request = SimpleNamespace(
                tool_call={"id": "cancel", "name": name, "args": {}},
                runtime=SimpleNamespace(stream_writer=events.append),
            )

            async def execute(_):
                raise asyncio.CancelledError()

            with self.assertRaises(asyncio.CancelledError):
                await observe_tool_call(request, execute)
            self.assertEqual(events[-1]["display"]["status"], expected)

    async def test_image_transport_failure_is_unknown_but_invalid_input_is_failed(self):
        import httpx
        from openai import APIConnectionError
        from tools import image_generation

        graph = graph_for([image_generation.generate_image])
        call = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "image",
                    "name": "generate_image",
                    "args": {"prompt": "Fixture image"},
                }
            ],
        )
        config = {"configurable": {"langgraph_auth_user": {"identity": "alice"}}}
        for error, expected in [
            (
                APIConnectionError(
                    request=httpx.Request("POST", "https://example.com")
                ),
                "unknown",
            ),
            (ValueError("invalid input"), "failed"),
        ]:
            with patch.object(
                image_generation,
                "generate_image_for_user",
                AsyncMock(side_effect=error),
            ) as generate:
                result = await graph.ainvoke({"messages": [call]}, config=config)
            self.assertEqual(
                result["messages"][-1].artifact["tool_display"]["status"], expected
            )
            generate.assert_awaited_once()

    async def test_write_followed_by_refresh_failure_preserves_committed_result(self):
        @tool
        async def write_then_refresh() -> str:
            """Commit the fixture write before a refresh fails."""
            observed(None, "已保存草稿")
            raise ValueError("refresh failed")

        graph = graph_for([write_then_refresh])
        call = AIMessage(
            content="",
            tool_calls=[{"id": "write", "name": "write_then_refresh", "args": {}}],
        )
        result = await graph.ainvoke({"messages": [call]})
        display = result["messages"][-1].artifact["tool_display"]
        self.assertEqual(display["status"], "succeeded")
        self.assertEqual(display["outcome"], "partial")
        self.assertIn("已保存草稿", display["summary"])

    def test_all_registered_tools_have_labels_and_inputs_exclude_secrets_and_ignored_filters(
        self,
    ):
        from tools import ALL_TOOLS
        from logic.coding_agent import (
            list_files,
            read_file,
            write_file,
            apply_patch,
            validate_app,
            finish_task,
        )

        names = {
            t.name
            for t in [
                *ALL_TOOLS,
                list_files,
                read_file,
                write_file,
                apply_patch,
                validate_app,
                finish_task,
            ]
        }
        self.assertEqual(names, set(TOOL_LABELS))
        self.assertEqual(len(names), 38)
        screen = input_display(
            "view_screen",
            {
                "mode": "current",
                "query": "ignored",
                "app_name": "ignored",
                "password": "SECRET",
            },
        )
        self.assertNotIn("SECRET", str(screen))
        self.assertNotIn("ignored", str(screen["fields"]))
        self.assertEqual(
            input_display("list_knowledge", {"query": "ignored"})["fields"], []
        )
        self.assertIsNone(display_from_artifact({"tool_display": {"version": 99}}))

    async def test_missing_and_ambiguous_reminders_never_claim_update_or_cancel(self):
        from tools import reminder

        runtime = SimpleNamespace(
            config={
                "configurable": {
                    "langgraph_auth_user": {"identity": "alice"},
                    "timezone": "Asia/Shanghai",
                }
            },
            tool_call_id="reminder",
        )
        with (
            patch.object(
                reminder,
                "get_all_reminders",
                AsyncMock(return_value=[{"id": "12345678-a"}, {"id": "12345678-b"}]),
            ),
            patch.object(reminder, "db_delete_reminder", AsyncMock()) as delete,
        ):
            result = await reminder.cancel_reminder.coroutine(runtime, "12345678")
            self.assertIn("不唯一", result.update["messages"][0].content)
            delete.assert_not_awaited()
        with (
            patch.object(reminder, "get_reminder_by_id", AsyncMock(return_value=None)),
            patch.object(reminder, "db_update_reminder", AsyncMock()) as update,
        ):
            result = await reminder.update_reminder.coroutine(
                runtime, "missing-id", title="New"
            )
            self.assertIn("不存在", result.update["messages"][0].content)
            update.assert_not_awaited()

    async def test_partial_mail_failure_is_visible_and_all_account_failure_is_not_empty(
        self,
    ):
        from services import email

        accounts = [
            {
                "account_id": key,
                "email_address": key + "@example.com",
                "encrypted_password": "fixture",
            }
            for key in ["a", "b"]
        ]
        with (
            patch.object(
                email, "get_email_credentials", AsyncMock(return_value=accounts)
            ),
            patch.object(email, "decrypt_password", return_value="fixture"),
            patch.object(
                email,
                "_fetch_recent_for_account",
                AsyncMock(
                    side_effect=[
                        [{"subject": "Hello", "date": "2026-10-01"}],
                        email.EmailServiceException("offline"),
                    ]
                ),
            ),
        ):
            batch = await email.get_recent_emails("alice")
        self.assertEqual(len(batch), 1)
        self.assertEqual([row["succeeded"] for row in batch.coverage], [True, False])
        with (
            patch.object(
                email, "get_email_credentials", AsyncMock(return_value=accounts)
            ),
            patch.object(email, "decrypt_password", return_value="fixture"),
            patch.object(
                email,
                "_fetch_recent_for_account",
                AsyncMock(side_effect=email.EmailServiceException("offline")),
            ),
        ):
            with self.assertRaises(email.EmailServiceException):
                await email.get_recent_emails("alice")

    async def test_thread_history_keeps_exact_failure_and_success_per_invocation(self):
        from routers import threads
        from fastapi.security import HTTPAuthorizationCredentials

        display = {
            "version": 1,
            "title": "修改待办",
            "status": "succeeded",
            "summary": "已修改：会议",
            "fields": [{"label": "标题", "value": "旧 → 新"}],
            "actions": [],
        }
        messages = [
            {
                "type": "ai",
                "id": "plan",
                "content": "",
                "tool_calls": [
                    {"id": "a", "name": "update_todo"},
                    {"id": "b", "name": "update_todo"},
                    {"id": "c", "name": "update_todo"},
                ],
            },
            {
                "type": "tool",
                "tool_call_id": "a",
                "artifact": {"tool_display": display},
            },
            {
                "type": "tool",
                "tool_call_id": "b",
                "status": "error",
                "artifact": {
                    "tool_display": {
                        **display,
                        "status": "failed",
                        "summary": "目标不存在",
                    }
                },
            },
            {"type": "tool", "tool_call_id": "c", "content": "SUCCESS"},
        ]
        response = SimpleNamespace(
            status_code=200, json=lambda: {"values": {"messages": messages}}
        )
        with patch.object(threads.httpx, "AsyncClient") as factory:
            factory.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=response
            )
            history = await threads.get_thread_messages(
                "fixture",
                HTTPAuthorizationCredentials(scheme="Bearer", credentials="fixture"),
                {"id": "alice"},
            )
        self.assertEqual(
            [tool["status"] for tool in history[0]["tools"]],
            ["succeeded", "failed", "unknown"],
        )
        self.assertEqual(history[0]["steps"][0]["tools"], history[0]["tools"])
        self.assertEqual(history[0]["tools"][0]["display"]["fields"], display["fields"])
