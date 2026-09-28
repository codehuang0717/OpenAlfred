"""Image generation, streaming, history and owner isolation regression tests."""

import base64
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI
from fastapi.security import HTTPAuthorizationCredentials
import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from routers import generated_images as routes, threads
from services import generated_images as images
from tools.image_generation import generate_image
from services import llm
from logic.context_manager import ContextManager
from tools import screenshot

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")


class TestGeneratedImages(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(patch.object(images.config, "PROJECT_ROOT", Path(self.tmp.name)))
        self.enterContext(patch.object(images.config, "OPENAI_API_KEY", "test-key"))

    def test_gpt_uses_responses_and_preserves_tool_results_and_output_budget(self):
        model = llm._create_gpt_model()
        messages = [
            AIMessage(content="", tool_calls=[{"name": "generate_image", "id": "call-image", "args": {"prompt": "A circle"}}]),
            ToolMessage(content="Image generated", tool_call_id="call-image"),
        ]
        payload = model._get_request_payload(messages, **llm.output_limit_kwargs("gpt-cloud", 4096))
        self.assertTrue(model.use_responses_api)
        self.assertEqual(payload["max_output_tokens"], 4096)
        self.assertNotIn("messages", payload)
        self.assertTrue(any(item.get("type") == "function_call_output" and item.get("call_id") == "call-image" for item in payload["input"]))

    def test_responses_reasoning_and_function_calls_survive_context_budgeting(self):
        blocks = [
            {"type": "reasoning", "id": "rs_test", "summary": []},
            {"type": "function_call", "id": "fc_test", "call_id": "call-image", "name": "generate_image", "arguments": '{"prompt":"A circle"}', "status": "completed"},
        ]
        message = AIMessage(content=blocks, tool_calls=[{"name": "generate_image", "id": "call-image", "args": {"prompt": "A circle"}}])
        manager = ContextManager()
        result = ToolMessage(content="Image generated", tool_call_id="call-image")
        self.assertGreater(manager.message_tokens([message, result]), 0)
        self.assertEqual(manager.count_payload(message)[0]["content"], blocks)
        self.assertEqual(manager.units([message, result]), [(0, 2)])

    async def test_gpt_screen_analysis_still_returns_only_visible_text(self):
        from PIL import Image
        response = AIMessage(content=[{"type": "reasoning", "summary": []}, {"type": "text", "text": "A test screen"}])
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=response))
        runtime = SimpleNamespace(config={"configurable": {"langgraph_auth_user": {"identity": "owner-a"}}})
        with patch.object(screenshot, "require_screen_owner"), patch.object(screenshot.ImageGrab, "grab", return_value=Image.new("RGB", (1, 1))), patch.object(screenshot, "get_model", return_value=model):
            result = await screenshot.take_screenshot.coroutine("Describe this", runtime)
        self.assertEqual(result, "A test screen")

    async def test_generate_saves_owner_scoped_png_and_small_artifact(self):
        client = AsyncMock()
        client.images.generate.return_value = SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(PNG).decode())])
        factory = MagicMock()
        factory.return_value.__aenter__ = AsyncMock(return_value=client)
        factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch.object(images, "AsyncOpenAI", factory):
            artifact = await images.generate_image_for_user("owner-a", "A blue circle", "auto")
        image_id = artifact["url"].rsplit("/", 1)[-1]
        self.assertEqual(images.image_path("owner-a", image_id).read_bytes(), PNG)
        self.assertFalse(images.image_path("owner-b", image_id).exists())
        self.assertNotIn("base64", str(artifact))
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
        self.assertEqual(client.images.generate.await_args.kwargs["model"], images.config.IMAGE_GENERATION_MODEL)

    async def test_failed_api_call_is_not_retried_or_saved(self):
        client = AsyncMock()
        client.images.generate.side_effect = TimeoutError("generation timed out")
        factory = MagicMock()
        factory.return_value.__aenter__ = AsyncMock(return_value=client)
        factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch.object(images, "AsyncOpenAI", factory), self.assertRaises(TimeoutError):
            await images.generate_image_for_user("owner-a", "A circle", "auto")
        client.images.generate.assert_awaited_once()
        self.assertEqual(list(Path(self.tmp.name).rglob("*.png")), [])

    async def test_only_owner_can_fetch_image_and_auth_is_required(self):
        artifact = images._save_image("owner-a", base64.b64encode(PNG).decode())
        app = FastAPI()
        app.include_router(routes.router)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            denied = await client.get(artifact["url"])
            self.assertIn(denied.status_code, (401, 403))
            app.dependency_overrides[routes.get_current_user] = lambda: {"id": "owner-a"}
            response = await client.get(artifact["url"])
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, PNG)
            self.assertEqual(response.headers["cache-control"], "private, no-store")
            app.dependency_overrides[routes.get_current_user] = lambda: {"id": "owner-b"}
            self.assertEqual((await client.get(artifact["url"])).status_code, 404)

    def test_invalid_image_and_artifact_are_rejected(self):
        with self.assertRaises(ValueError):
            images._save_image("owner-a", base64.b64encode(b"not a PNG").decode())
        with self.assertRaises(ValueError):
            images.image_path("owner-a", "../other")
        self.assertEqual(images.image_markdown({"type": "generated_image", "url": "https://example.com/tracker"}), "")

    async def test_real_langgraph_message_stream_preserves_tool_artifact(self):
        artifact = {"type": "generated_image", "url": "/api/generated-images/" + "a" * 32}
        graph = StateGraph(MessagesState)
        graph.add_node("tools", ToolNode([generate_image]))
        graph.set_entry_point("tools")
        graph.add_edge("tools", END)
        request = AIMessage(content="", tool_calls=[{"name": "generate_image", "id": "call-image", "args": {"prompt": "A circle"}}])
        with patch("tools.image_generation.generate_image_for_user", new=AsyncMock(return_value=artifact)):
            events = [event async for event in graph.compile().astream(
                {"messages": [request]},
                {"configurable": {"langgraph_auth_user": {"identity": "owner-a"}}},
                stream_mode="messages",
            )]
        results = [message for message, _ in events if isinstance(message, ToolMessage)]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].artifact, artifact)
        self.assertEqual(results[0].tool_call_id, "call-image")

    async def test_history_keeps_generated_image_between_tool_and_final_text(self):
        artifact = {"type": "generated_image", "url": "/api/generated-images/" + "a" * 32}
        response = MagicMock(status_code=200)
        response.json.return_value = {"values": {"messages": [
            {"type": "human", "content": "Draw a circle"},
            {"type": "ai", "id": "ai1", "content": "", "tool_calls": [{"name": "generate_image", "id": "call-image"}]},
            {"type": "tool", "name": "generate_image", "tool_call_id": "call-image", "artifact": artifact},
            {"type": "ai", "id": "ai2", "content": "Here it is."},
        ]}}
        client = AsyncMock()
        client.get.return_value = response
        with patch.object(threads.httpx, "AsyncClient") as factory:
            factory.return_value.__aenter__.return_value = client
            messages = await threads.get_thread_messages("thread-a", HTTPAuthorizationCredentials(scheme="Bearer", credentials="test"), {"id": "owner-a"})
        steps = messages[-1]["steps"]
        self.assertEqual([step["type"] for step in steps], ["tools", "text", "text"])
        self.assertIn(artifact["url"], steps[1]["content"])
        self.assertEqual(steps[2]["content"], "Here it is.")


if __name__ == "__main__":
    unittest.main()
