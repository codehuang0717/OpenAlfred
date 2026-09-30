"""Title failures must be visible and must not silently rename a thread."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import HTTPException

from core.config import config
from routers.threads import generate_thread_title
from services.llm import get_strict_model
from services.thread_titles import generate_title


class TestTitleModel(unittest.IsolatedAsyncioTestCase):
    def test_title_uses_configured_flash_model_without_provider_fallback(self):
        with patch.object(config, "MIMO_API_KEY", "synthetic-key"), \
                patch.object(config, "MIMO_TITLE_MODEL", "mimo-v2.6-flash"):
            model = get_strict_model("mimo-title")
        self.assertEqual(model.model_name, "mimo-v2.6-flash")
        self.assertEqual(model.root_async_client.max_retries, 0)
        with patch.object(config, "MIMO_API_KEY", ""), \
                patch("services.llm._create_gpt_model") as fallback:
            with self.assertRaises(ValueError):
                get_strict_model("mimo-title")
        fallback.assert_not_called()

    async def test_accepts_text_in_multimodal_first_message(self):
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=SimpleNamespace(content='"面试日程安排"'))
        with patch("services.thread_titles.get_strict_model", return_value=model):
            title = await generate_title([
                {"type": "text", "text": "帮我安排明天下午的面试"},
                {"type": "image_url", "image_url": {"url": "private-image"}},
            ])
        self.assertEqual(title, "面试日程安排")
        self.assertNotIn("private-image", model.ainvoke.await_args.args[0][0].content)

    async def test_empty_model_result_is_an_error(self):
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=SimpleNamespace(content=""))
        with patch("services.thread_titles.get_strict_model", return_value=model):
            with self.assertRaises(ValueError):
                await generate_title("hello")


class TestTitleRoute(unittest.IsolatedAsyncioTestCase):
    async def call_route(self, save_status=200, model_error=None, messages=None):
        self.saved = []

        async def respond(request):
            if request.method == "GET":
                return httpx.Response(200, json={"values": {"messages": messages if messages is not None else [
                    {"type": "human", "content": "明天下午面试"},
                ]}})
            self.saved.append(request)
            return httpx.Response(save_status, json={})

        client_type = httpx.AsyncClient

        def client():
            return client_type(transport=httpx.MockTransport(respond))

        with patch("routers.threads.httpx.AsyncClient", side_effect=client), \
                patch("services.thread_titles.generate_title", new_callable=AsyncMock,
                      return_value="面试安排", side_effect=model_error):
            return await generate_thread_title(
                "thread-1", SimpleNamespace(credentials="user-token"), {"id": "user-1"},
            )

    async def test_valid_title_is_saved_under_owner(self):
        self.assertEqual(await self.call_route(), {"title": "面试安排"})
        self.assertEqual(json.loads(self.saved[0].content)["metadata"],
                         {"owner": "user-1", "title": "面试安排"})

    async def test_provider_failure_is_not_a_successful_default_title(self):
        with self.assertRaises(HTTPException) as error:
            await self.call_route(model_error=ValueError("Unsupported model"))
        self.assertEqual(error.exception.status_code, 502)
        self.assertIn("ValueError", error.exception.detail)
        self.assertFalse(self.saved)

    async def test_save_failure_is_reported_even_when_generation_succeeds(self):
        with self.assertRaises(HTTPException) as error:
            await self.call_route(save_status=403)
        self.assertEqual(error.exception.status_code, 502)

    async def test_missing_first_message_reports_not_ready(self):
        with self.assertRaises(HTTPException) as error:
            await self.call_route(messages=[])
        self.assertEqual(error.exception.status_code, 422)
        self.assertFalse(self.saved)
