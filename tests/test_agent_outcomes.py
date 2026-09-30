"""Model completion is a validated outcome, not merely an HTTP 200 / no tools."""

import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx
from fastapi.security import HTTPAuthorizationCredentials
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from logic import nodes
from logic.agent import should_continue
from logic.agent_outcome import AgentRunError, classify_response
from logic.context_manager import ContextManager
from logic.context_metrics import model_usage_metrics
from logic.context_payload import payload
from logic.schema import AgentState
from services import llm
from services.mimo_chat import MiMoChatOpenAI
from routers import threads


def answer(content="done", reason="stop", **kwargs):
    return AIMessage(content=content, response_metadata={"finish_reason": reason}, **kwargs)


class TestCompletionContract(unittest.TestCase):
    def test_truncated_thinking_only_matches_observed_failure(self):
        response = answer("", "length", usage_metadata={
            "input_tokens": 32440, "output_tokens": 4096, "total_tokens": 36536,
            "output_token_details": {"reasoning": 4095},
        })
        outcome = classify_response(response, set())
        self.assertEqual((outcome.status, outcome.code), ("failed", "output_truncated"))
        metrics = model_usage_metrics(response, "mimo", 87000)
        self.assertEqual(metrics["reasoning_tokens"], 4095)
        self.assertEqual(metrics["finish_reason"], "length")

    def test_truncation_takes_priority_over_parsed_tool_calls(self):
        response = answer("partial", "length", tool_calls=[{"id": "x", "name": "write", "args": {}}])
        self.assertEqual(classify_response(response, {"write"}).code, "output_truncated")

    def test_empty_whitespace_reasoning_only_and_no_signal_are_not_answers(self):
        for content in ["", "  ", [{"type": "reasoning", "text": "private"}]]:
            self.assertEqual(classify_response(answer(content), set()).code, "empty_answer")
        self.assertEqual(classify_response(AIMessage(content="looks finished"), set()).code,
                         "missing_completion_signal")

    def test_completed_visible_text_refusal_and_responses_api(self):
        for response in [answer(), answer([{ "type": "text", "text": "done"}]),
                         answer("", additional_kwargs={"refusal": "Cannot assist"}),
                         AIMessage(content="done", response_metadata={"status": "completed"})]:
            self.assertEqual(classify_response(response, set()).status, "completed")

    def test_incomplete_filtered_and_unknown_finish_are_failures(self):
        for metadata, expected in [
            ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}, "provider_incomplete"),
            ({"status": "failed"}, "provider_incomplete"),
            ({"finish_reason": "content_filter"}, "content_blocked"),
            ({"finish_reason": "unexpected"}, "missing_completion_signal"),
        ]:
            response = AIMessage(content="partial", response_metadata=metadata)
            self.assertEqual(classify_response(response, set()).code, expected)

    def test_tool_contract_ids_names_and_parsing(self):
        call = {"id": "x", "name": "write", "args": {}}
        self.assertEqual(classify_response(answer("", "tool_calls", tool_calls=[call]), {"write"}).status, "tools")
        cases = [answer("", "tool_calls"),
                 answer("", "tool_calls", tool_calls=[call]),
                 answer("", "tool_calls", tool_calls=[call, call]),
                 answer("", "tool_calls", tool_calls=[{**call, "id": None}]),
                 answer("", "tool_calls", invalid_tool_calls=[{"name": "write", "id": "x", "args": "{", "error": "invalid"}])]
        for index, response in enumerate(cases):
            with self.subTest(index=index):
                allowed = set() if index == 1 else {"write"}
                self.assertEqual(classify_response(response, allowed).code, "invalid_tool_calls")

    def test_routing_requires_explicit_outcome(self):
        for status, route in [("failed", "fail_run"), ("tools", "tools"), ("completed", "extract_knowledge")]:
            self.assertEqual(should_continue(AgentState(agent_outcome={"status": status})), route)
        with self.assertRaises(RuntimeError):
            should_continue(AgentState(messages=[answer()]))


class TestProviderProtocol(unittest.IsolatedAsyncioTestCase):
    def test_nonstream_reasoning_roundtrip_budget_and_no_visible_leak(self):
        model = MiMoChatOpenAI(api_key="synthetic", model="mimo-v2.6-pro")
        result = model._create_chat_result({"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "reasoning_content": "private rationale",
            "tool_calls": [{"id": "call", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        }}]})
        message = result.generations[0].message
        self.assertEqual(message.content, "")
        self.assertEqual(message.additional_kwargs["reasoning_content"], "private rationale")
        wire = model._get_request_payload([message, ToolMessage(content="result", tool_call_id="call")])
        self.assertEqual(wire["messages"][0]["reasoning_content"], "private rationale")
        manager = ContextManager()
        self.assertGreater(manager.message_tokens([message]), manager.message_tokens([message.model_copy(update={"additional_kwargs": {}})]))
        self.assertNotIn("reasoning_content", payload(message))

    async def test_real_sdk_stream_accumulates_reasoning_and_resends_after_tool(self):
        requests = []
        async def handle(request):
            requests.append(json.loads(request.content))
            deltas = ([{"role": "assistant", "reasoning_content": "private "},
                       {"reasoning_content": "rationale"},
                       {"tool_calls": [{"index": 0, "id": "call", "type": "function", "function": {"name": "read", "arguments": "{}"}}]}]
                      if len(requests) == 1 else [{"role": "assistant", "content": "done"}])
            chunks = [{"id": "test", "model": "mimo-v2.6-pro", "object": "chat.completion.chunk", "created": 0,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": None}]} for delta in deltas]
            chunks.append({"id": "test", "model": "mimo-v2.6-pro", "object": "chat.completion.chunk", "created": 0,
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if len(requests) == 1 else "stop"}]})
            sse = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            model = MiMoChatOpenAI(api_key="synthetic", base_url="https://example.test/v1", model="mimo-v2.6-pro",
                                   streaming=True, http_async_client=client, extra_body={"thinking": {"type": "enabled"}}, max_retries=0)
            first = await model.ainvoke([HumanMessage(content="read")], max_tokens=16384)
            self.assertEqual(first.additional_kwargs["reasoning_content"], "private rationale")
            second = await model.ainvoke([HumanMessage(content="read"), first, ToolMessage(content="result", tool_call_id="call")], max_tokens=16384)
        self.assertEqual(second.content, "done")
        self.assertEqual(requests[1]["messages"][1]["reasoning_content"], "private rationale")
        self.assertEqual(requests[1]["max_completion_tokens"], 16384)
        self.assertEqual(requests[1]["thinking"], {"type": "enabled"})

    def test_main_binding_requires_selected_provider_and_disables_sdk_retries(self):
        with patch.object(llm, "_bound_cache", {}), patch.object(llm.config, "MIMO_API_KEY", ""):
            with self.assertRaisesRegex(ValueError, "MIMO_API_KEY"):
                llm.get_bound_model("mimo", frozenset(), [])
        with patch.object(llm.config, "MIMO_API_KEY", "synthetic"):
            model = llm.get_strict_model("mimo")
        self.assertIsInstance(model, MiMoChatOpenAI)
        self.assertEqual(model.max_retries, 0)
        self.assertEqual(model.root_async_client.max_retries, 0)


class TestRunLifecycle(unittest.IsolatedAsyncioTestCase):
    def build_graph(self, tools=()):
        async def prepare(state):
            return {"prepared_messages": state.messages}
        self.extract = AsyncMock(return_value={})
        graph = StateGraph(AgentState)
        graph.add_node("agent", nodes.agent_node)
        graph.add_node("prepare", prepare)
        async def extract(state):
            return await self.extract(state)
        graph.add_node("extract_knowledge", extract)
        graph.add_node("fail_run", nodes.fail_run_node)
        graph.add_node("tools", ToolNode(list(tools)))
        graph.set_entry_point("prepare")
        graph.add_edge("prepare", "agent")
        graph.add_conditional_edges("agent", should_continue)
        graph.add_edge("tools", "prepare")
        graph.add_edge("extract_knowledge", END)
        return graph.compile(checkpointer=InMemorySaver())

    async def test_failure_is_checkpointed_and_raised_not_extracted_or_retried(self):
        graph = self.build_graph()
        cfg = {"configurable": {"thread_id": "isolated", "model_selection": "mimo"}}
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=answer("", "length")))
        with patch.object(nodes, "get_bound_model", return_value=model):
            with self.assertRaises(AgentRunError):
                await graph.ainvoke({"messages": [HumanMessage(content="question")]}, cfg)
        model.ainvoke.assert_awaited_once()
        self.extract.assert_not_awaited()
        state = (await graph.aget_state(cfg)).values
        self.assertEqual(state["agent_outcome"]["status"], "failed")
        self.assertIn("output_truncated", state["messages"][-1].content)
        self.assertFalse(state["messages"][-1].tool_calls)
        # Explicit new input starts a fresh run, not a sticky failure state.
        model.ainvoke.return_value = answer()
        with patch.object(nodes, "get_bound_model", return_value=model):
            result = await graph.ainvoke({"messages": [HumanMessage(content="next")]}, cfg)
        self.assertEqual(result["agent_outcome"]["status"], "completed")
        self.extract.assert_awaited_once()

    async def test_completed_side_effect_is_not_replayed_after_model_failure(self):
        executions = []
        @tool
        async def write() -> str:
            """A synthetic side effect."""
            executions.append("once")
            return "saved"
        graph = self.build_graph([write])
        model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
            answer("", "tool_calls", tool_calls=[{"id": "call", "name": "write", "args": {}}]),
            answer("", "length"),
        ]))
        with patch.object(nodes, "selected_context_tools", return_value=[write]), patch.object(nodes, "get_bound_model", return_value=model):
            with self.assertRaises(AgentRunError):
                await graph.ainvoke({"messages": [HumanMessage(content="write")]}, {"configurable": {"thread_id": "side-effect"}})
        self.assertEqual(executions, ["once"])
        self.assertEqual(model.ainvoke.await_count, 2)
        self.extract.assert_not_awaited()

    async def test_partial_tool_call_is_never_dispatched_and_failure_stream_is_visible(self):
        executions = []
        @tool
        async def write() -> str:
            """A synthetic side effect."""
            executions.append("bad")
            return "saved"
        graph = self.build_graph([write])
        response = answer("", "length", tool_calls=[{"id": "call", "name": "write", "args": {}}])
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=response))
        events = []
        with patch.object(nodes, "selected_context_tools", return_value=[write]), patch.object(nodes, "get_bound_model", return_value=model):
            with self.assertRaises(AgentRunError):
                async for event in graph.astream({"messages": [HumanMessage(content="write")]},
                                                 {"configurable": {"thread_id": "partial"}}, stream_mode="messages"):
                    events.append(event)
        self.assertEqual(executions, [])
        self.assertTrue(any("output_truncated" in str(message.content) for message, _ in events))

    async def test_api_failure_is_visible_and_never_retried(self):
        model = SimpleNamespace(ainvoke=AsyncMock(side_effect=ConnectionError("connection failed")))
        with patch.object(nodes, "get_bound_model", return_value=model):
            result = await nodes.agent_node(AgentState(prepared_messages=[HumanMessage(content="hello")]), {}, lambda _: None)
        self.assertEqual(result["agent_outcome"]["code"], "model_request_failed")
        self.assertIn("ConnectionError", result["messages"][0].content)
        model.ainvoke.assert_awaited_once()


class TestOutcomeHistory(unittest.IsolatedAsyncioTestCase):
    async def history(self, messages):
        response = SimpleNamespace(status_code=200, json=lambda: {"values": {"messages": messages}})
        client = SimpleNamespace(get=AsyncMock(return_value=response))
        with patch.object(threads.httpx, "AsyncClient") as factory:
            factory.return_value.__aenter__.return_value = client
            return await threads.get_thread_messages(
                "synthetic-thread", HTTPAuthorizationCredentials(scheme="Bearer", credentials="synthetic"), {"id": "synthetic-owner"})

    async def test_failed_answer_survives_history_without_reasoning_or_orphan_tools(self):
        result = await self.history([
            {"type": "human", "id": "user", "content": "question"},
            {"type": "ai", "id": "plan", "content": "searching", "additional_kwargs": {"agent_outcome": {"status": "tools"}}},
            {"type": "ai", "id": "failed", "content": "output_truncated", "additional_kwargs": {
                "reasoning_content": "private", "agent_outcome": {"status": "failed"}, "agent_failure": "output_truncated"}},
        ])
        self.assertEqual(result[-1]["outcome"], "failed")
        self.assertEqual(result[-1]["failure"], "output_truncated")
        self.assertNotIn("private", str(result))

    async def test_legacy_empty_length_response_is_diagnosed_on_read_only(self):
        messages = [{"type": "human", "id": "user", "content": "question"},
                    {"type": "ai", "id": "empty", "content": "", "response_metadata": {"finish_reason": "length"}}]
        original = json.dumps(messages)
        result = await self.history(messages)
        self.assertEqual(result[-1]["outcome"], "failed")
        self.assertIn("output_truncated", result[-1]["content"])
        self.assertEqual(json.dumps(messages), original)

    async def test_tool_plan_is_not_marked_final_and_reasoning_blocks_are_hidden(self):
        result = await self.history([{"type": "ai", "id": "plan", "content": [
            {"type": "reasoning", "text": "private"}, {"type": "text", "text": "searching"}],
            "additional_kwargs": {"agent_outcome": {"status": "tools"}}}])
        self.assertEqual(result[-1]["outcome"], "tools")
        self.assertEqual(result[-1]["content"], "searching")
        self.assertNotIn("private", str(result))


if __name__ == "__main__":
    unittest.main()
