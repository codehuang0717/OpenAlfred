"""Probe safety tests: fake model only, never enable --run here."""

from pathlib import Path
import sys
import json
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_context_api import Meter, cases, failure_details
from langchain_core.messages import HumanMessage, AIMessageChunk
from logic.context_manager import RollingSummary
from logic.context_manager import ContextManager, ContextBudgetError
from unittest.mock import patch


class TestContextApiProbe(unittest.IsolatedAsyncioTestCase):
    def test_failure_report_keeps_local_cause_but_not_provider_error_body(self):
        error = ContextBudgetError("摘要生成超过 60 秒，未重试或保存摘要")
        error.__cause__ = TimeoutError()
        details = failure_details(error)
        self.assertEqual(details["cause_type"], "TimeoutError")
        self.assertIn("60 秒", details["context_error"])
        self.assertNotIn("secret", json.dumps(failure_details(RuntimeError("secret provider body"))))

    async def test_strict_model_really_sends_only_once_on_server_error(self):
        import httpx
        from services import llm
        bodies = []

        async def handle(request):
            bodies.append(request.method)
            return httpx.Response(500, json={"error": {"message": "synthetic server failure"}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            # Initialize the SDK with its normal defaults, then use the strict
            # factory to verify both LangChain and its existing SDK are updated.
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model="mimo-v2.6-pro", api_key="synthetic-key", base_url="https://test.invalid/v1", http_async_client=client)
            with patch.object(llm.config, "MIMO_API_KEY", "synthetic-key"), patch.dict(llm._factories, {"mimo": lambda: model}):
                strict = llm.get_strict_model("mimo")
            from openai import InternalServerError
            with self.assertRaises(InternalServerError):
                await strict.ainvoke("synthetic")
        self.assertEqual(len(bodies), 1)

    async def test_production_summary_sends_4096_cap_but_requests_1600_final(self):
        import httpx
        from langchain_openai import ChatOpenAI
        bodies = []

        async def handle(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "synthetic", "object": "chat.completion", "created": 1, "model": "mimo-v2.6-pro",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant",
                    "content": RollingSummary(current_goals=["synthetic"]).model_dump_json()}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            model = ChatOpenAI(model="mimo-v2.6-pro", api_key="synthetic-key", base_url="https://test.invalid/v1", http_async_client=client, max_retries=0)
            with patch("services.llm.get_strict_model", return_value=model):
                await ContextManager().merge_summary("", [{"evidence": "synthetic"}])
        self.assertEqual(bodies[0]["max_completion_tokens"], 4096)
        self.assertIn('"token_limit":1600', bodies[0]["messages"][1]["content"])

    async def test_raw_summary_instrumentation_preserves_http_output_limit(self):
        import httpx
        from langchain_openai import ChatOpenAI
        bodies = []

        async def handle(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "synthetic", "object": "chat.completion", "created": 1, "model": "mimo-v2.6-pro",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant",
                    "content": RollingSummary(current_goals=["synthetic"]).model_dump_json()}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            model = ChatOpenAI(model="mimo-v2.6-pro", api_key="synthetic-key", base_url="https://test.invalid/v1", http_async_client=client, max_retries=0)
            meter = Meter(model)
            result = await meter.with_structured_output(RollingSummary).ainvoke([HumanMessage(content="test")], max_tokens=1600)
        self.assertEqual(bodies[0]["max_completion_tokens"], 1600)
        self.assertEqual(result.current_goals, ["synthetic"])

    def test_request_cap_is_checked_before_sending(self):
        meter = Meter(None)
        meter.requests = 20
        with self.assertRaises(RuntimeError):
            meter.claim([HumanMessage(content="synthetic")])
        self.assertEqual(meter.requests, 20)

    def test_input_cap_is_checked_before_sending(self):
        meter = Meter(None)
        meter.estimated_input = 250000
        with self.assertRaises(RuntimeError):
            meter.claim([HumanMessage(content="synthetic")])
        self.assertEqual(meter.requests, 0)

    async def test_stream_captures_real_usage_fields_without_executing_tools(self):
        class Fake:
            def bind_tools(self, tools, tool_choice):
                assert tool_choice == "none"
                return self

            async def astream(self, messages, max_tokens, config):
                assert max_tokens == 1024
                yield AIMessageChunk(content="OK")
                yield AIMessageChunk(content="", usage_metadata={"input_tokens": 100, "output_tokens": 1,
                    "total_tokens": 101, "input_token_details": {"cache_read": 80}})

        meter = Meter(Fake())
        row = await meter.answer([HumanMessage(content="test")])
        self.assertEqual(row["cache_read_tokens"], 80)
        self.assertEqual(row["answer"], "OK")
        self.assertIsNotNone(row["first_text_ms"])

    def test_three_cases_are_synthetic_and_budgeted(self):
        self.assertEqual(len(cases()), 3)
        for _, messages, expected, budget in cases():
            self.assertTrue(messages)
            self.assertTrue(expected)
            self.assertLessEqual(budget, 32768)
