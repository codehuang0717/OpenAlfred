"""Offline paired replay: uv run python tests/context_benchmark.py.

Synthetic histories, temporary database, deterministic fake summaries. No network,
personal memories or provider cache hits. Prefix overlap is only a proxy.
"""

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from db import connection
from logic.context_manager import ContextManager, RollingSummary, SUMMARY_INSTRUCTION
from logic.context_payload import payload, serialize
from logic.prompts import AGENT_SYSTEM_PROMPT
from tools import ALL_TOOLS

spec = importlib.util.spec_from_file_location("frozen_context_baseline", Path(__file__).parent / "fixtures/context_manager_baseline.py")
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)


def scenarios():
    cases = {}
    for name, turns, words in [("short_chat", 30, 8), ("long_chat", 40, 650)]:
        history, requests = [], []
        for i in range(turns):
            history.append(HumanMessage(content=f"任务 {i}：" + "detail " * words))
            requests.append(list(history))
            history.append(AIMessage(content=f"回答 {i}：" + "result " * (words // 3)))
        cases[name] = requests
    for name, words in [("tool_chain", 30), ("medium_results", 900), ("huge_results", 10000)]:
        history = [HumanMessage(content="检查任务证据，失败不得宣称成功。")]
        requests = [list(history)]
        for i in range(12):
            history.extend([
                AIMessage(content="", tool_calls=[{"name": "read_email", "args": {"email_id": str(i)}, "id": f"c{i}"}]),
                ToolMessage(content="begin " * words + f" MIDDLE_EVIDENCE_{i} status=failed " + "end " * words, tool_call_id=f"c{i}", status="error"),
            ])
            requests.append(list(history))
        cases[name] = requests
    return cases


async def replay(manager_class, name, requests, tuned):
    # Explicit budgets make this reproducible independently of local .env values.
    manager = manager_class(max_context_tokens=32768, output_reserve=4096, safety_margin=2048,
                            summary_tokens=1600, summary_input_tokens=12000, tool_result_tokens=1200, max_messages=20,
                            **({"tool_inline_tokens": 8192, "compact_trigger": .90, "compact_target": .65} if tuned else {}))
    summary_inputs = summary_outputs = merges = 0

    async def fake_summary(previous, records):
        nonlocal summary_inputs, summary_outputs, merges
        prompt = [SystemMessage(content=SUMMARY_INSTRUCTION), HumanMessage(content=serialize({
            "token_limit": manager.summary_tokens, "previous_summary": previous, "new_evidence": records,
        }))]
        request_tokens = manager.message_tokens(prompt) + manager.tokens(serialize(RollingSummary.model_json_schema()))
        generation_limit = manager.summary_generation_tokens if tuned else manager.summary_tokens
        assert request_tokens + generation_limit + manager.safety_margin <= manager.summary_input_tokens
        summary_inputs += request_tokens
        # Same deterministic summarizer for both policies; NOT a quality evaluation.
        digest = hashlib.sha256(serialize(records).encode()).hexdigest()[:16]
        result = RollingSummary(current_goals=["完成当前任务"], active_constraints=["失败不得宣称成功"],
                                key_facts=[f"batch={digest}"], evidence_refs=[r["ref"] for r in records[-4:]]).model_dump_json()
        summary_outputs += manager.tokens(result)
        merges += 1
        return result

    manager.merge_summary = fake_summary
    previous = []
    main_inputs = prefix_tokens = serialized_tokens = previews = evidence_visible = evidence_total = 0
    compacting_requests = 0
    latencies = []
    for i, history in enumerate(requests):
        # Tools within one user turn share time; distinct user turns change it.
        clock = i if name.endswith("chat") else 0
        runtime = f"[系统信息]\nCurrent Time: 2026-09-27 12:{clock:02d}:00.\n天气：晴，27°C。"
        memories = "[用户基本资料]\n使用中文，关注任务的正确执行。"
        stable = AGENT_SYSTEM_PROMPT + "\n\n" + memories
        system = stable if tuned else AGENT_SYSTEM_PROMPT + "\n\n" + runtime + "\n\n" + memories
        start = time.perf_counter()
        kwargs = {"runtime_context": runtime} if tuned else {}
        result = await manager.prepare(history, system, ALL_TOOLS, "benchmark-user", name, **kwargs)
        latencies.append((time.perf_counter() - start) * 1000)
        manager.units(result.messages)
        assert result.metrics["input_tokens_after"] + 4096 + 2048 <= 32768
        latest = next(m for m in reversed(history) if isinstance(m, HumanMessage))
        assert any(isinstance(m, HumanMessage) and m.content == latest.content for m in result.messages)
        main_inputs += result.metrics["input_tokens_after"]
        previews += result.metrics["tool_results_compacted"]
        compacting_requests += int(result.metrics["summary_merges"] > 0)
        # Canonical serialized prefix, NOT the provider's chat template or KV cache.
        rendered = serialize([convert_to_openai_tool(t) for t in ALL_TOOLS]) + serialize([payload(m) for m in result.messages])
        encoded = manager.encoding.encode(rendered, disallowed_special=())
        common = 0
        for left, right in zip(previous, encoded):
            if left != right:
                break
            common += 1
        prefix_tokens += common
        serialized_tokens += len(encoded)
        previous = encoded
        if isinstance(history[-1], ToolMessage):
            evidence_total += 1
            marker = f"MIDDLE_EVIDENCE_{i - 1}"
            evidence_visible += int(any(marker in str(m.content) for m in result.messages))
    return {
        "requests": len(requests), "main_input_tokens_est": main_inputs,
        "summary_input_tokens_est": summary_inputs, "summary_output_tokens_stub": summary_outputs,
        "total_input_plus_stub_summary_output": main_inputs + summary_inputs + summary_outputs,
        "summary_calls": merges, "requests_with_summary": compacting_requests,
        "prefix_overlap_pct_proxy": round(100 * prefix_tokens / serialized_tokens, 1),
        "latest_tool_middle_visible": f"{evidence_visible}/{evidence_total}",
        "tool_preview_occurrences": previews,
        "planner_ms_median": round(statistics.median(latencies), 2),
        "planner_ms_p95": round(sorted(latencies)[max(0, int(len(latencies) * .95) - 1)], 2),
    }


async def run():
    report = {"method": "offline synthetic replay; no billed tokens, no real cache hits, no model latency; main output excluded (fixed replay); 3 repeats with alternating policy order, timing medians across repeats", "cases": {}}
    # Warm tokenizer before measuring either planner.
    ContextManager().tokens("warmup")
    for name, requests in scenarios().items():
        report["cases"][name] = {}
        samples = {"baseline": [], "tuned": []}
        policies = [("baseline", baseline.ContextManager, False), ("tuned", ContextManager, True)]
        for repetition in range(3):
            for label, cls, tuned in policies if repetition % 2 == 0 else list(reversed(policies)):
                with tempfile.TemporaryDirectory() as tmp, patch.object(connection, "DATABASE_PATH", str(Path(tmp) / "bench.db")):
                    await connection.init_db()
                    samples[label].append(await replay(cls, name, requests, tuned))
        for label, rows in samples.items():
            combined = dict(rows[0])
            for key in combined:
                if key.startswith("planner_ms_"):
                    combined[key] = round(statistics.median(row[key] for row in rows), 2)
                else:
                    assert all(row[key] == combined[key] for row in rows), (name, label, key)
            report["cases"][name][label] = combined
    return report


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), ensure_ascii=False, indent=2))
