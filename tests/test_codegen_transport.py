"""Closing preflight/job clients must not close the chat provider's HTTP pool."""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import base as openai_adapter

from services import llm
from services.code_apps import coding_context, create_codegen_model


class TestCodegenTransport(unittest.IsolatedAsyncioTestCase):
    async def test_preflight_and_finished_jobs_do_not_poison_cached_chat_transport(self):
        requests = []

        def respond(request):
            requests.append(request)
            chunk = {"id": "synthetic", "model": "mimo-v2.6-pro", "created": 0,
                     "object": "chat.completion.chunk", "choices": [
                         {"index": 0, "delta": {"role": "assistant", "content": "ready"}, "finish_reason": "stop"}]}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as shared_async:
            with httpx.Client(transport=httpx.MockTransport(respond)) as shared_sync:
                with patch.object(llm.config, "MIMO_API_KEY", "synthetic"), \
                        patch.object(openai_adapter, "_get_default_async_httpx_client", return_value=shared_async), \
                        patch.object(openai_adapter, "_get_default_httpx_client", return_value=shared_sync), \
                        patch("services.code_apps.get_user_timezone", new_callable=AsyncMock, return_value="Asia/Shanghai"):
                    main = llm.get_strict_model("mimo")
                    # Executes the real construction + teardown used by the tool.
                    await coding_context("alice", "mimo")
                    self.assertFalse(shared_async.is_closed)
                    self.assertFalse(shared_sync.is_closed)
                    self.assertEqual((await main.ainvoke([HumanMessage(content="after preflight")])).content, "ready")
                    first = create_codegen_model("mimo")
                    second = create_codegen_model("mimo")
                    try:
                        self.assertIsNot(first.root_async_client._client, second.root_async_client._client)
                        self.assertIsNot(first.root_async_client._client, shared_async)
                        self.assertIsNot(first.root_client._client, shared_sync)
                        await first.root_async_client.close()
                        first.root_client.close()
                        self.assertFalse(second.root_async_client._client.is_closed)
                        self.assertFalse(shared_async.is_closed)
                        self.assertEqual((await main.ainvoke([HumanMessage(content="after job teardown")])).content, "ready")
                    finally:
                        await first.root_async_client.close()
                        first.root_client.close()
                        await second.root_async_client.close()
                        second.root_client.close()
        self.assertEqual(len(requests), 2)

    async def test_factory_failure_closes_only_owned_transports(self):
        sync_client = MagicMock()
        async_client = MagicMock()
        async_client.aclose = AsyncMock()
        factory = MagicMock(side_effect=ValueError("synthetic factory failure"))
        with patch.object(llm.config, "MIMO_API_KEY", "synthetic"), \
                patch.object(llm, "_factories", {"mimo": factory}), \
                patch.object(llm, "DefaultHttpxClient", return_value=sync_client), \
                patch.object(llm, "DefaultAsyncHttpxClient", return_value=async_client):
            with self.assertRaisesRegex(ValueError, "factory failure"):
                create_codegen_model("mimo")
            await asyncio.sleep(0)
        sync_client.close.assert_called_once()
        async_client.aclose.assert_awaited_once()

    def test_missing_provider_config_allocates_no_clients_and_never_falls_back(self):
        with patch.object(llm.config, "MIMO_API_KEY", ""), \
                patch.object(llm, "DefaultHttpxClient") as sync_client, \
                patch.object(llm, "DefaultAsyncHttpxClient") as async_client, \
                patch.object(llm, "_create_gpt_model") as fallback:
            with self.assertRaisesRegex(ValueError, "MIMO_API_KEY"):
                create_codegen_model("mimo")
        sync_client.assert_not_called()
        async_client.assert_not_called()
        fallback.assert_not_called()
