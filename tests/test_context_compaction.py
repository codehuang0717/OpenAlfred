"""Synthetic histories only; no real users, network, or model inference."""

import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from db import connection
from db.context_compactions import load_compaction, save_compaction
from logic.context_manager import ContextManager, ContextBudgetError, RollingSummary
from logic.context_payload import reference, history_hash, serialize, payload
from logic.schema import AgentState
from logic import nodes
from tools.context_history import read_context_excerpt


def pair(identifier="call", content="result"):
    return [AIMessage(content="", tool_calls=[{"name": "probe", "args": {}, "id": identifier}]),
            ToolMessage(content=content, tool_call_id=identifier)]


def summary(label="current goal"):
    return RollingSummary(current_goals=[label], pending_actions=["continue"] ).model_dump_json()


class TestContextGroups(unittest.TestCase):
    def setUp(self):
        self.manager = ContextManager()

    def test_parallel_calls_are_one_indivisible_group(self):
        messages = [HumanMessage(content="task"), AIMessage(content="", tool_calls=[
            {"name": "probe", "args": {}, "id": "a"}, {"name": "probe", "args": {}, "id": "b"},
        ]), ToolMessage(content="B", tool_call_id="b"), ToolMessage(content="A", tool_call_id="a")]
        self.assertEqual(self.manager.units(messages), [(0, 1), (1, 4)])

    def test_orphan_missing_duplicate_and_wrong_results_rejected(self):
        bad = [[ToolMessage(content="x", tool_call_id="a")], pair()[:1],
               [pair()[0], ToolMessage(content="x", tool_call_id="other")],
               [AIMessage(content="", tool_calls=[{"name":"x","args":{},"id":"a"}]*2)]]
        for messages in bad:
            with self.subTest(messages=messages), self.assertRaises(ContextBudgetError):
                self.manager.units(messages)

    def test_large_result_preview_keeps_original_and_error_status(self):
        original = ToolMessage(content="prefix " + "large output "*8000 + " final error", tool_call_id="a", status="error")
        messages, count = self.manager.compact_tools([original])
        self.assertEqual(count, 1)
        self.assertEqual(messages[0].tool_call_id, "a")
        self.assertEqual(messages[0].status, "error")
        preview = json.loads(messages[0].content)
        self.assertTrue(preview["truncated"])
        self.assertEqual(preview["original_ref"], reference(0, original))
        self.assertIn("final error", preview["tail"])
        self.assertGreater(len(original.content), len(messages[0].content))

    def test_images_have_cost_without_counting_base64_as_text(self):
        message = HumanMessage(content=[{"type":"image_url", "image_url":{"url":"data:image/png;base64," + "x"*100000}}])
        self.assertGreaterEqual(self.manager.message_tokens([message]), nodes.app_config.CONTEXT_IMAGE_TOKENS)
        self.assertLess(self.manager.message_tokens([message]), nodes.app_config.CONTEXT_IMAGE_TOKENS+100)

    def test_unsupported_multimodal_content_fails_explicitly(self):
        with self.assertRaises(ContextBudgetError):
            self.manager.message_tokens([HumanMessage(content=[{"type":"audio","data":"bytes"}])])

    def test_reference_changes_when_original_is_edited(self):
        self.assertNotEqual(reference(0, HumanMessage(content="a")), reference(0, HumanMessage(content="b")))

    def test_token_count_cache_is_bounded_and_content_addressed(self):
        self.assertNotEqual(self.manager.tokens("a"), self.manager.tokens("a " * 20))
        for i in range(2100):
            self.manager.tokens(f"test {i}")
        self.assertEqual(len(self.manager._token_counts), 2048)
        self.assertTrue(all(isinstance(k, bytes) for k in self.manager._token_counts))

    def test_summary_fragments_include_escaped_envelope_in_budget(self):
        value = {"ref": "ctx:0:abc", "content": '\\"\n路径😀' * 2000}
        fragments = self.manager.split_record(value, 500)
        self.assertGreater(len(fragments), 1)
        self.assertEqual("".join(f["fragment"] for f in fragments), serialize(value))
        for fragment in fragments:
            self.assertLessEqual(self.manager.tokens(serialize(fragment)), 500)

    def test_turns_include_all_parallel_results_loops_and_final_reply(self):
        messages = [HumanMessage(content="first"), *pair("a"), *pair("b"), AIMessage(content="done"),
                    HumanMessage(content="second"), *pair("c"), AIMessage(content="done"),
                    HumanMessage(content="current"), *pair("d")]
        self.manager.units(messages)
        self.assertEqual(self.manager.turns(messages), [(0, 6), (6, 10), (10, 13)])

    def test_only_old_moderate_results_are_replaced_with_referenced_previews(self):
        content = "prefix " * 1400 + "MIDDLE_EVIDENCE" + " tail" * 1400
        messages = [HumanMessage(content="old"), *pair("a", content), AIMessage(content="done"),
                    HumanMessage(content="previous"), *pair("b", content),
                    HumanMessage(content="current"), *pair("c", content)]
        projected, count = self.manager.compact_tools(messages, protected_start=4)
        self.assertEqual(count, 1)
        preview = json.loads(projected[2].content)
        self.assertEqual(preview["original_ref"], reference(2, messages[2]))
        self.assertEqual(projected[6].content, content)
        self.assertEqual(projected[9].content, content)
        self.assertEqual(messages[2].content, content)
        self.assertLessEqual(self.manager.message_tokens([projected[2]]), self.manager.tool_result_tokens)

    def test_fitting_summary_records_do_not_get_double_encoded(self):
        records = [{"ref": f"ctx:{i}:hash", "role": "tool", "status": "error", "content": '路径\\"' * 300}
                   for i in range(3)]
        batches = self.manager.summary_batches(records, 24038)
        self.assertEqual(batches, [records])
        self.assertNotIn("fragment", batches[0][0])

    def test_large_summary_records_split_without_losing_unicode_or_paths(self):
        value = {"ref": "ctx:0:hash", "content": '\\"\n路径😀' * 2000}
        batches = self.manager.summary_batches([value], 500)
        fragments = [record for batch in batches for record in batch]
        self.assertEqual("".join(record["fragment"] for record in fragments), serialize(value))
        self.assertTrue(all(self.manager.tokens(serialize(batch)) <= 500 for batch in batches))

    def test_invalid_pruning_and_turn_limits_are_rejected(self):
        for kwargs in [{"keep_recent_turns": 0}, {"old_tool_inline_tokens": 100},
                       {"old_tool_inline_tokens": 9000}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ContextManager(**kwargs)


class TestContextPlanner(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(patch.object(connection, "DATABASE_PATH", str(Path(self.tmp.name)/"test.db")))
        await connection.init_db()
        self.manager = ContextManager(max_context_tokens=7000, output_reserve=500, safety_margin=256, summary_tokens=500, summary_input_tokens=8000)
        self.merge = self.enterContext(patch.object(self.manager, "merge_summary", new_callable=AsyncMock, return_value=summary()))

    async def test_64k_budget_keeps_reserves_and_expected_thresholds(self):
        manager = ContextManager(max_context_tokens=65536)
        result = await manager.prepare([HumanMessage(content="hello")], "system", [], "a", "t")
        self.assertEqual(result.metrics["budget"], 65536)
        self.assertEqual(result.metrics["output_reserve"], nodes.app_config.CONTEXT_OUTPUT_RESERVE)
        self.assertEqual(result.metrics["safety_margin"], 2048)
        input_budget = 65536 - nodes.app_config.CONTEXT_OUTPUT_RESERVE - 2048
        self.assertEqual(result.metrics["trigger_tokens"], int(input_budget * .90))
        self.assertEqual(result.metrics["target_tokens"], int(input_budget * .65))
        self.merge.assert_not_awaited()

    async def test_24_message_tool_loop_keeps_goal_and_pairs(self):
        messages = [HumanMessage(content="original current task")]
        for i in range(11):
            messages.extend(pair(str(i)))
        messages.append(AIMessage(content="last analysis"))
        result = await self.manager.prepare(messages,"system",[],"a","t")
        self.assertTrue(any(isinstance(m,HumanMessage) and m.content==messages[0].content for m in result.messages))
        self.manager.units(result.messages)
        self.assertEqual(result.covered_count,0)
        self.merge.assert_not_awaited()
        self.assertEqual(len(messages),24)
        self.assertLessEqual(result.metrics["input_tokens_after"]+500+256,7000)

    async def test_current_huge_user_request_not_silently_cut(self):
        with self.assertRaises(ContextBudgetError):
            await self.manager.prepare([HumanMessage(content="huge "*15000)],"system",[],"a","t")
        self.merge.assert_not_awaited()
        self.assertIsNone(await load_compaction("a","t"))

    async def test_oversize_latest_request_does_not_waste_calls_summarizing_old_history(self):
        history = [HumanMessage(content=f"old {i}") for i in range(25)]
        history.append(HumanMessage(content="huge " * 15000))
        with self.assertRaises(ContextBudgetError):
            await self.manager.prepare(history, "system", [], "a", "t")
        self.merge.assert_not_awaited()

    async def test_tool_schema_is_included_in_budget(self):
        @tool
        def probe(value: str) -> str:
            """A tool with a substantial schema description."""
            return value
        small = ContextManager(max_context_tokens=800,output_reserve=200,safety_margin=100)
        with self.assertRaises(ContextBudgetError):
            await small.prepare([HumanMessage(content="hello")],"system "*800,[probe],"a","t")
        with patch.object(self.manager,"tool_tokens",return_value=6800):
            with self.assertRaises(ContextBudgetError):
                await self.manager.prepare([HumanMessage(content="hello")],"system",[probe],"a","t")

    async def test_big_tool_result_is_compacted_without_dropping_user(self):
        messages=[HumanMessage(content="question")]+pair(content="long result "*12000)
        result=await self.manager.prepare(messages,"system",[],"a","t")
        self.merge.assert_not_awaited()
        self.assertEqual(result.metrics["tool_results_compacted"],1)
        self.assertGreater(result.metrics["input_tokens_before"],result.metrics["input_tokens_after"])
        self.manager.units(result.messages)

    async def test_rolling_summary_replaces_old_summary_and_does_not_repeat_covered_messages(self):
        messages=[HumanMessage(content=f"message {i} " + "detail " * 300) for i in range(25)]
        await save_compaction("a","t",summary("obsolete goal"),2,history_hash(messages,2),0)
        result=await self.manager.prepare(messages,"system",[],"a","t")
        self.assertNotIn("obsolete goal",result.summary)
        self.assertIn("obsolete goal",self.merge.await_args_list[0].args[0])
        self.assertNotIn('message 0',serialize([payload(m) for m in result.messages]))
        row=await load_compaction("a","t")
        self.assertEqual(row["summary"],result.summary)
        self.assertEqual(row["revision"],2)

    async def test_progress_arrives_before_summary_finishes(self):
        messages = [HumanMessage(content=f"message {i} " + "detail " * 300) for i in range(25)]
        started = asyncio.Event()
        gate = asyncio.Event()
        events = []

        async def merge(previous, batch):
            started.set()
            await gate.wait()
            return summary()

        self.merge.side_effect = merge
        task = asyncio.create_task(self.manager.prepare(messages, "system", [], "a", "t", on_progress=events.append))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            self.assertEqual(events[0], {"type": "context_compaction", "status": "running", "batch": 1})
            self.assertFalse(task.done())
        finally:
            gate.set()
            await task
        self.assertEqual(events[-1]["status"], "completed")

    async def test_all_batches_share_one_deadline_and_failure_preserves_summary(self):
        messages = [HumanMessage(content=f"message {i} " + "detail " * 300) for i in range(25)]
        await save_compaction("a", "t", summary("old"), 2, history_hash(messages, 2), 0)
        self.manager.compaction_timeout = 0.08
        waits = []
        original_wait_for = asyncio.wait_for

        async def track_wait(awaitable, timeout):
            waits.append(timeout)
            return await original_wait_for(awaitable, timeout=timeout)

        async def slow_merge(previous, batch):
            await asyncio.sleep(0.05)
            return summary()

        self.merge.side_effect = slow_merge
        with patch("logic.context_manager.asyncio.wait_for", side_effect=track_wait):
            with self.assertRaisesRegex(ContextBudgetError, "总等待"):
                await self.manager.prepare(messages, "system", [], "a", "t")
        self.assertGreaterEqual(len(waits), 2)
        self.assertLess(waits[1], waits[0])
        row = await load_compaction("a", "t")
        self.assertEqual(row["summary"], summary("old"))
        self.assertEqual(row["revision"], 1)

    async def test_cancelled_compaction_does_not_commit_or_emit_completion(self):
        messages = [HumanMessage(content=f"message {i} " + "detail " * 300) for i in range(25)]
        started = asyncio.Event()
        events = []

        async def merge(previous, batch):
            started.set()
            await asyncio.Event().wait()

        self.merge.side_effect = merge
        task = asyncio.create_task(self.manager.prepare(messages, "system", [], "a", "t", on_progress=events.append))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(await load_compaction("a", "t"))
        self.assertFalse(any(event["status"] == "completed" for event in events))

    async def test_summary_failure_keeps_previous_row_unchanged(self):
        messages=[HumanMessage(content=f"message {i} " + "detail " * 300) for i in range(25)]
        await save_compaction("a","t",summary("old"),2,history_hash(messages,2),0)
        self.merge.side_effect=ConnectionError("offline")
        with self.assertRaises(ConnectionError):
            await self.manager.prepare(messages,"system",[],"a","t")
        self.assertEqual((await load_compaction("a","t"))["summary"],summary("old"))

    async def test_history_edit_rebuilds_instead_of_applying_stale_cursor(self):
        original=[HumanMessage(content="old text"),AIMessage(content="old answer")]
        await save_compaction("a","t",summary("old summary"),2,history_hash(original,2),0)
        result=await self.manager.prepare([HumanMessage(content="new branch")],"system",[],"a","t")
        self.assertTrue(result.metrics["history_rebuilt"])
        self.assertEqual(result.summary,"")
        self.assertEqual((await load_compaction("a","t"))["covered_count"],0)

    async def test_user_and_thread_isolation(self):
        await save_compaction("a","t",summary(),0,history_hash([],0),0)
        self.assertIsNone(await load_compaction("b","t"))
        self.assertIsNone(await load_compaction("a","other"))

    async def test_concurrent_update_is_not_overwritten(self):
        await save_compaction("a","t",summary(),0,history_hash([],0),0)
        await save_compaction("a","t",summary("new"),0,history_hash([],0),1)
        with self.assertRaisesRegex(RuntimeError,"并发"):
            await save_compaction("a","t",summary("stale"),0,history_hash([],0),1)
        self.assertEqual((await load_compaction("a","t"))["summary"],summary("new"))

    async def test_summary_sees_tool_evidence_and_exact_refs(self):
        messages=[HumanMessage(content="send"),*pair(content='{"status":"failed","id":"job-42"}'),
                  AIMessage(content="old analysis " + "detail " * 6500),
                  HumanMessage(content="next"),AIMessage(content="answer"),HumanMessage(content="now")]
        await self.manager.prepare(messages,"system",[],"a","t")
        evidence=serialize([call.args[1] for call in self.merge.await_args_list])
        self.assertIn("job-42",evidence)
        self.assertIn("failed",evidence)
        self.assertIn(reference(2,messages[2]),evidence)

    async def test_old_tool_pruning_avoids_summary_api_calls(self):
        manager = ContextManager()
        messages = []
        for i in range(18):
            messages.extend([HumanMessage(content=f"task {i}"), *pair(str(i), "evidence " * 4000),
                             AIMessage(content="finished")])
        with patch.object(manager, "merge_summary", new_callable=AsyncMock) as merge:
            result = await manager.prepare(messages, "system", [], "a", "t")
        merge.assert_not_awaited()
        self.assertEqual(result.metrics["tool_results_compacted"], 16)
        self.assertGreater(result.metrics["input_tokens_before"], result.metrics["trigger_tokens"])
        self.assertLess(result.metrics["input_tokens_after"], result.metrics["trigger_tokens"])
        self.assertEqual(result.covered_count, 0)
        self.assertEqual(result.messages[-2].content, messages[-2].content)
        self.assertEqual(messages[2].content, "evidence " * 4000)
        runtime = SimpleNamespace(config={"configurable": {"owner": "a", "thread_id": "t"}},
                                  state={"user_id": "a", "messages": messages})
        recovered = json.loads(read_context_excerpt.func(reference(2, messages[2]), runtime, offset=15000))
        self.assertEqual(recovered["excerpt"], serialize(payload(messages[2]))[15000:recovered["next_offset"]])

    async def test_recent_turns_remain_whole_through_compaction(self):
        messages = [HumanMessage(content=f"old {i} " + "detail " * 300) for i in range(25)]
        protected = [HumanMessage(content="previous request"), *pair("previous-a"), *pair("previous-b"),
                     AIMessage(content="previous reply"), HumanMessage(content="current request"),
                     *pair("current-a"), *pair("current-b")]
        messages.extend(protected)
        result = await self.manager.prepare(messages, "system", [], "a", "t")
        self.assertEqual(result.messages[-len(protected):], protected)
        self.assertLessEqual(result.covered_count, len(messages) - len(protected))
        self.assertIn(result.covered_count, [start for start, _ in self.manager.turns(messages)])
        self.manager.units(result.messages)
        evidence = serialize([call.args[1] for call in self.merge.await_args_list])
        self.assertNotIn("previous request", evidence)
        self.assertNotIn("current request", evidence)

    async def test_pruned_prefix_stays_stable_within_the_same_tool_loop(self):
        messages = [HumanMessage(content="old"), *pair("old", "evidence " * 3000), AIMessage(content="done"),
                    HumanMessage(content="previous"), AIMessage(content="done"), HumanMessage(content="current")]
        first = await self.manager.prepare(messages, "stable system", [], "a", "t")
        second = await self.manager.prepare(messages + pair("current", "new evidence"), "stable system", [], "a", "t")
        self.assertEqual(second.messages[:len(first.messages)], first.messages)
        self.assertEqual(second.metrics["protected_start"], first.metrics["protected_start"])
        self.merge.assert_not_awaited()

    async def test_oversize_previous_protected_turn_is_not_summarized_or_dropped(self):
        messages = [HumanMessage(content="old"), AIMessage(content="old reply"),
                    HumanMessage(content="previous " * 8000), AIMessage(content="previous reply"),
                    HumanMessage(content="current")]
        with self.assertRaisesRegex(ContextBudgetError, "最近完整对话"):
            await self.manager.prepare(messages, "system", [], "a", "t")
        self.merge.assert_not_awaited()
        self.assertIsNone(await load_compaction("a", "t"))

    async def test_legacy_cursor_inside_a_turn_rebuilds_with_full_recent_history(self):
        messages = [HumanMessage(content="previous request"), *pair(), AIMessage(content="previous reply"),
                    HumanMessage(content="current request")]
        await save_compaction("a", "t", summary("legacy"), 3, history_hash(messages, 3), 0)
        result = await self.manager.prepare(messages, "system", [], "a", "t")
        self.assertTrue(result.metrics["history_rebuilt"])
        self.assertEqual(result.messages[1:], messages)
        self.assertEqual(result.covered_count, 0)
        self.merge.assert_not_awaited()

    async def test_32k_summary_window_reduces_batches_without_dropping_evidence(self):
        messages = []
        for i in range(9):
            messages.extend([HumanMessage(content=f"request {i}"), AIMessage(content="detail " * 6000)])
        rows = []
        for window, thread in [(12000, "small"), (32768, "large")]:
            manager = ContextManager(summary_input_tokens=window)
            with patch.object(manager, "merge_summary", new_callable=AsyncMock, return_value=summary()) as merge:
                result = await manager.prepare(messages, "system", [], "a", thread)
            rows.append((result, merge))
            evidence = [record for call in merge.await_args_list for record in call.args[1]]
            for i in range(result.covered_count):
                self.assertTrue(any(record["ref"] == reference(i, messages[i]) for record in evidence))
            self.assertEqual(result.messages[-4:], messages[-4:])
            for call in merge.await_args_list:
                self.assertLessEqual(manager.tokens(serialize(call.args[1])), result.metrics["summary_evidence_budget"])
        self.assertEqual(rows[0][0].covered_count, rows[1][0].covered_count)
        self.assertGreater(rows[0][1].await_count, rows[1][1].await_count)
        self.assertLessEqual(rows[1][1].await_count, 2)

    async def test_moderate_result_keeps_middle_evidence(self):
        content = "prefix " * 900 + "MIDDLE_EVIDENCE" + " tail" * 900
        messages = [HumanMessage(content="question"), *pair(content=content)]
        result = await self.manager.prepare(messages, "system", [], "a", "t")
        self.assertEqual(result.metrics["tool_results_compacted"], 0)
        self.assertEqual(result.messages[-1].content, content)
        self.merge.assert_not_awaited()

    async def test_compaction_leaves_headroom_and_next_turn_does_not_compact(self):
        messages = [HumanMessage(content=f"old {i} " + "detail " * 300) for i in range(25)]
        first = await self.manager.prepare(messages, "system", [], "a", "t")
        self.assertLessEqual(first.metrics["input_tokens_after"], first.metrics["target_tokens"])
        self.merge.reset_mock()
        second = await self.manager.prepare(messages + [HumanMessage(content="next")], "system", [], "a", "t")
        self.merge.assert_not_awaited()
        self.assertEqual(first.summary, second.summary)

    async def test_runtime_context_at_tail_and_in_budget(self):
        messages = [HumanMessage(content="hello")]
        result = await self.manager.prepare(messages, "stable", [], "a", "t", runtime_context="time: now")
        self.assertEqual(result.messages[0].content, "stable")
        self.assertEqual(result.messages[-1].content, "time: now")
        self.assertEqual(result.metrics["input_tokens_after"], self.manager.message_tokens(result.messages) + self.manager.tool_tokens([]))
        with self.assertRaises(ContextBudgetError):
            await self.manager.prepare(messages, "stable", [], "a", "t", runtime_context="time " * 8000)


class TestSummaryValidation(unittest.IsolatedAsyncioTestCase):
    async def test_generation_allowance_is_separate_from_stored_summary_limit(self):
        manager = ContextManager(summary_tokens=1600, summary_generation_tokens=4096)
        model = MagicMock()
        invoke = AsyncMock(return_value=RollingSummary(current_goals=["keep goal"]))
        model.with_structured_output.return_value.ainvoke = invoke
        with patch("services.llm.get_strict_model", return_value=model):
            result = await manager.merge_summary("", [{"evidence": "test"}])
        self.assertEqual(invoke.await_args.kwargs["max_tokens"], 4096)
        self.assertIn('"token_limit":1600', invoke.await_args.args[0][1].content)
        self.assertLessEqual(manager.tokens(result), 1600)

    async def test_generation_allowance_is_counted_before_sending(self):
        manager = ContextManager(summary_input_tokens=6000, summary_generation_tokens=4096)
        model = MagicMock()
        with patch("services.llm.get_strict_model", return_value=model):
            with self.assertRaisesRegex(ContextBudgetError, "请求超预算"):
                await manager.merge_summary("", [{"evidence": "test"}])
        model.with_structured_output.assert_not_called()

    async def test_length_and_timeout_fail_once_without_retry(self):
        from openai import LengthFinishReasonError
        from openai.types.chat import ChatCompletion
        truncated = LengthFinishReasonError(completion=ChatCompletion(id="test", created=0, model="test", object="chat.completion", choices=[]))
        for error, expected in [(truncated, "长度限制截断"), (TimeoutError(), "配置时限")]:
            model = MagicMock()
            invoke = AsyncMock(side_effect=error)
            model.with_structured_output.return_value.ainvoke = invoke
            with patch("services.llm.get_strict_model", return_value=model):
                with self.assertRaisesRegex(ContextBudgetError, expected):
                    await ContextManager().merge_summary("", [{"evidence": "test"}])
            invoke.assert_awaited_once()

    def test_generation_allowance_cannot_be_less_than_final_limit(self):
        with self.assertRaises(ValueError):
            ContextManager(summary_tokens=1600, summary_generation_tokens=1000)

    async def test_oversize_or_empty_summary_fails(self):
        manager=ContextManager(summary_tokens=100)
        invoke=AsyncMock(return_value=RollingSummary(current_goals=["content "*60]*3))
        model=MagicMock()
        model.with_structured_output.return_value.ainvoke=invoke
        with patch("services.llm.get_strict_model",return_value=model):
            with self.assertRaises(ContextBudgetError):
                await manager.merge_summary("",[{"evidence":"test"}])
            invoke.return_value=RollingSummary()
            with self.assertRaises(ContextBudgetError):
                await manager.merge_summary("",[{"evidence":"test"}])

    async def test_summary_budget_checked_before_api_request(self):
        manager=ContextManager(summary_input_tokens=1000)
        model=MagicMock()
        with patch("services.llm.get_strict_model",return_value=model):
            with self.assertRaises(ContextBudgetError):
                await manager.merge_summary("old "*3000,[{"evidence":"test"}])
        model.with_structured_output.assert_not_called()

    async def test_quote_heavy_batches_fit_final_request_including_previous_summary(self):
        manager = ContextManager(summary_input_tokens=32768)
        previous = RollingSummary(key_facts=['"\\' * 100] * 3).model_dump_json()
        self.assertLessEqual(manager.tokens(previous), manager.summary_tokens)
        records = [{"ref": f"ctx:{i}:hash", "content": '"D:\\data\\file"\n' * 1000} for i in range(5)]
        batches = manager.summary_batches(records, manager.summary_evidence_budget())
        model = MagicMock()
        invoke = AsyncMock(return_value=RollingSummary(current_goals=["synthetic"]))
        model.with_structured_output.return_value.ainvoke = invoke
        with patch("services.llm.get_strict_model", return_value=model):
            for batch in batches:
                await manager.merge_summary(previous, batch)
        self.assertEqual(invoke.await_count, len(batches))
        self.assertEqual([record["ref"] for batch in batches for record in batch],
                         [record["ref"] for record in records])


class TestHistoryExcerpt(unittest.TestCase):
    def setUp(self):
        self.message=ToolMessage(content="文字😀"*4000,tool_call_id="a")
        self.runtime=SimpleNamespace(config={"configurable":{"langgraph_auth_user":{"identity":"a"},"thread_id":"t"}},state={"user_id":"a","messages":[self.message]})
        self.ref=reference(0,self.message)

    def test_paginated_read_has_stable_reference_and_small_payload(self):
        result=json.loads(read_context_excerpt.func(self.ref,self.runtime,limit=1200))
        self.assertEqual(result["excerpt"],serialize(payload(self.message))[:result["next_offset"]])
        self.assertEqual(result["ref"],self.ref)
        self.assertGreater(result["next_offset"],0)
        self.assertLess(result["next_offset"],result["total_chars"])
        projected = ToolMessage(content=serialize(result), tool_call_id="read", name="read_context_excerpt")
        self.assertLessEqual(ContextManager().message_tokens([projected]), nodes.app_config.CONTEXT_TOOL_RESULT_TOKENS)

    def test_bad_ref_and_wrong_user_rejected(self):
        with self.assertRaises(ValueError):
            read_context_excerpt.func("ctx:0:wrong",self.runtime)
        self.runtime.state["user_id"]="b"
        with self.assertRaises(PermissionError):
            read_context_excerpt.func(self.ref,self.runtime)


class TestContextGraphIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_load_context_separates_fresh_time_from_stable_instructions(self):
        cfg = {"configurable": {"langgraph_auth_user": {"identity": "a"}, "thread_id": "t", "timezone": "Asia/Shanghai"}}
        with patch.object(nodes.memory_manager, "build_injection_text", return_value="stable user facts"), \
             patch.object(nodes, "save_user_timezone", new_callable=AsyncMock), \
             patch.object(nodes, "get_weather_summary", new_callable=AsyncMock, return_value={}), \
             patch.object(nodes, "format_weather_prompt_context", return_value="current weather"):
            result = await nodes.load_context_node(AgentState(), cfg)
        self.assertIn("stable user facts", result["system_instruction"])
        self.assertNotIn("Current Time:", result["system_instruction"])
        self.assertIn("Current Time:", result["runtime_context"])
        self.assertIn("current weather", result["runtime_context"])

    async def test_main_model_uses_prepared_messages_and_output_cap(self):
        cfg = {"configurable": {"model_selection": "mimo"}}
        prepared = [SystemMessage(content="system"), HumanMessage(content="current")]
        state = AgentState(messages=[HumanMessage(content="raw excluded")], prepared_messages=prepared)
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="done", response_metadata={"finish_reason": "stop"})))
        with patch.object(nodes, "get_bound_model", return_value=model):
            await nodes.agent_node(state, cfg, lambda _event: None)
        model.ainvoke.assert_awaited_once_with(prepared, cfg, max_tokens=nodes.ctx_manager.output_reserve)

    async def test_agent_streams_tool_start_without_arguments(self):
        cfg = {"configurable": {"model_selection": "mimo"}}
        state = AgentState(prepared_messages=[HumanMessage(content="create an app")])
        response = AIMessage(content="", tool_calls=[{
            "id": "call-app", "name": "create_standalone_mini_app",
            "args": {"requirements": "private user request"},
        }], response_metadata={"finish_reason": "tool_calls"})
        events = []
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=response))
        with patch.object(nodes, "get_bound_model", return_value=model):
            await nodes.agent_node(state, cfg, events.append)
        self.assertEqual(events, [{
            "type": "tool_calls",
            "tools": [{"id": "call-app", "name": "create_standalone_mini_app"}],
        }])

    async def test_graph_custom_stream_announces_tool_before_execution(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", nodes.agent_node)
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)
        graph = workflow.compile()
        response = AIMessage(content="", tool_calls=[{
            "id": "call-app", "name": "create_standalone_mini_app", "args": {},
        }], response_metadata={"finish_reason": "tool_calls"})
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=response))
        events = []
        with patch.object(nodes, "get_bound_model", return_value=model):
            async for mode, data in graph.astream(
                {"prepared_messages": [HumanMessage(content="make an app")]},
                config={"configurable": {"model_selection": "mimo"}},
                stream_mode=["messages", "custom"],
            ):
                if mode == "custom":
                    events.append(data)
        self.assertEqual(events, [{
            "type": "tool_calls",
            "tools": [{"id": "call-app", "name": "create_standalone_mini_app"}],
        }])

    async def test_voice_does_not_skip_budget_planner(self):
        cfg={"configurable":{"langgraph_auth_user":{"identity":"a"},"thread_id":"t","channel":"voice"}}
        fake=SimpleNamespace(messages=[HumanMessage(content="ok")],summary="",covered_count=0,metrics={})
        with patch.object(nodes.ctx_manager,"prepare",new_callable=AsyncMock,return_value=fake) as prepare:
            result=await nodes.prepare_context_node(AgentState(messages=[HumanMessage(content="ok")]),cfg,lambda _event: None)
        prepare.assert_awaited_once()
        self.assertEqual(result["context_error"],"")

    async def test_graph_streams_compaction_status_before_main_model(self):
        async def prepare(*args, on_progress, **kwargs):
            on_progress({"type": "context_compaction", "status": "running", "batch": 1})
            await asyncio.sleep(0)
            on_progress({"type": "context_compaction", "status": "completed", "batches": 1})
            return SimpleNamespace(messages=[HumanMessage(content="ok")], summary="", covered_count=0, metrics={})

        graph = StateGraph(AgentState)
        graph.add_node("prepare_context", nodes.prepare_context_node)
        graph.add_edge(START, "prepare_context")
        graph.add_edge("prepare_context", END)
        with patch.object(nodes.ctx_manager, "prepare", side_effect=prepare):
            events = [data async for data in graph.compile().astream(
                AgentState(messages=[HumanMessage(content="ok")]),
                config={"configurable": {"owner": "a", "thread_id": "t"}}, stream_mode="custom",
            )]
        self.assertEqual([event["status"] for event in events], ["running", "completed"])

    async def test_preparation_failure_never_invokes_main_model(self):
        cfg={"configurable":{"langgraph_auth_user":{"identity":"a"},"thread_id":"t"}}
        with patch.object(nodes.ctx_manager,"prepare",new_callable=AsyncMock,side_effect=ContextBudgetError("too big")):
            update=await nodes.prepare_context_node(AgentState(),cfg,lambda _event: None)
        with patch.object(nodes,"get_bound_model") as model:
            result=await nodes.agent_node(AgentState(**update),cfg,lambda _event: None)
        model.assert_not_called()
        self.assertIn("上下文准备失败",result["messages"][0].content)

    def test_graph_routes_every_tool_result_back_through_budget_check(self):
        from logic.agent import workflow
        self.assertIn(("tools","prepare_context"),workflow.edges)
        self.assertIn(("prepare_context","agent"),workflow.edges)
        self.assertNotIn(("tools","agent"),workflow.edges)


class TestCompactionModelSelection(unittest.TestCase):
    def test_vision_uses_configured_flash_model(self):
        from core.config import config
        from rag import image_describer
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "synthetic.png"
            image.write_bytes(b"synthetic-image-not-sent-to-network")
            with patch.object(config, "MIMO_API_KEY", "synthetic-key"), \
                    patch.object(config, "MIMO_VISION_MODEL", "mimo-v2.6-flash"), \
                    patch("langchain_openai.ChatOpenAI") as factory, \
                    patch.object(image_describer, "_read_cache", return_value=None), \
                    patch.object(image_describer, "_write_cache"):
                factory.return_value.invoke.return_value = SimpleNamespace(content="合成图片描述")
                self.assertEqual(image_describer.describe_image(str(image)), "合成图片描述")
        self.assertEqual(factory.call_args.kwargs["model"], "mimo-v2.6-flash")
        message = factory.return_value.invoke.call_args.args[0][0]
        self.assertEqual(message.content[0]["type"], "image_url")

    def test_summary_flash_is_independent_from_chat(self):
        from services import llm
        with patch.object(llm.config, "MIMO_API_KEY", "synthetic-key"), \
                patch.object(llm.config, "MIMO_SUMMARY_MODEL", "mimo-v2.6-flash"), \
                patch.object(llm.config, "MIMO_CHAT_MODEL", "mimo-v2.6-pro"):
            model = llm.get_strict_model("mimo-summary")
            chat = llm.get_strict_model("mimo")
        self.assertEqual(model.model_name, "mimo-v2.6-flash")
        self.assertEqual(chat.model_name, "mimo-v2.6-pro")
        self.assertEqual(model.root_async_client.max_retries, 0)
        self.assertEqual(llm.output_limit_kwargs("mimo-summary", 4096), {"max_tokens": 4096})

    def test_missing_flash_summary_key_does_not_fall_back(self):
        from services import llm
        with patch.object(llm.config, "MIMO_API_KEY", ""), \
                patch.object(llm, "_create_gpt_model") as fallback:
            with self.assertRaisesRegex(ValueError, "MIMO_API_KEY"):
                llm.get_strict_model("mimo-summary")
        fallback.assert_not_called()

    def test_strict_provider_disables_sdk_retries(self):
        from services import llm
        with patch.object(llm.config, "MIMO_API_KEY", "synthetic-key"):
            model = llm.get_strict_model("mimo")
        self.assertEqual(model.max_retries, 0)
        self.assertEqual(model.root_client.max_retries, 0)
        self.assertEqual(model.root_async_client.max_retries, 0)

    def test_usage_missing_cache_is_unknown_not_zero(self):
        from logic.context_metrics import model_usage_metrics
        result = model_usage_metrics(AIMessage(content="x"), "mimo", 20)
        self.assertIsNone(result["cache_read_tokens"])
        response = AIMessage(content="x", usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "input_token_details": {"cache_read": 80}})
        result = model_usage_metrics(response, "mimo", 20)
        self.assertEqual(result["cache_read_tokens"], 80)
        self.assertEqual(result["input_tokens"], 100)

    def test_deepseek_reported_cache_usage_is_preserved(self):
        from logic.context_metrics import model_usage_metrics
        response = AIMessage(content="x", response_metadata={"token_usage": {"prompt_tokens": 100, "completion_tokens": 12, "prompt_cache_hit_tokens": 0}})
        self.assertEqual(model_usage_metrics(response, "deepseek", 5)["cache_read_tokens"], 0)

    def test_missing_summary_provider_key_never_falls_back(self):
        from services import llm
        with patch.object(llm.config, "MIMO_API_KEY", ""), patch.object(llm, "_create_gpt_model") as fallback:
            with self.assertRaisesRegex(ValueError, "MIMO_API_KEY"):
                llm.get_strict_model("mimo")
        fallback.assert_not_called()

    def test_unknown_summary_model_is_rejected(self):
        from services.llm import get_strict_model
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            get_strict_model("misspelled-model")
