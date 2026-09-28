"""Opt-in real API A/B probe. Run: uv run python probe_context_api.py --run.

Synthetic data only. No tools executed. At most 20 requests, no retries, same
configured MiMo provider for both policies and real structured summaries.
Results print to stdout; do not run as part of automated unit tests.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import time
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "tests"))
from context_benchmark import baseline, ContextManager, connection, ALL_TOOLS
from logic.context_manager import ContextBudgetError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import tracing_context
from core.config import config
from logic.context_metrics import model_usage_metrics
from logic.context_payload import payload, reference
from services import llm


def failure_details(exc):
    """Safe diagnostic types; do not serialize provider errors or headers."""
    result = {"stopped_error_type": type(exc).__name__,
              "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None}
    if isinstance(exc, ContextBudgetError):
        # ContextBudgetError messages are authored locally, not provider bodies.
        result["context_error"] = str(exc)
    return result


def cases():
    short = []
    for i in range(12):
        short.extend([HumanMessage(content=f"合成记录 {i}：项目仅在本地测试，无真实邮件。"), AIMessage(content="已记录。")])
    short.append(HumanMessage(content="本轮测试代码是 SHORT-427。只回复这个代码。"))
    medium = [HumanMessage(content="读取结果中的 DECISION_CODE，只回复代码；看不到则回复 UNKNOWN，不猜测。"),
              AIMessage(content="", tool_calls=[{"name": "read_email", "args": {"email_id": "synthetic"}, "id": "probe-call"}]),
              ToolMessage(content="日志条目：无关记录。\n" * 180 + "\nDECISION_CODE=MIDDLE-731\n" + "日志条目：无关记录。\n" * 180, tool_call_id="probe-call")]
    pressure = []
    for i in range(12):
        pressure.extend([HumanMessage(content=f"记录 {i}：" + "synthetic event detail " * 170), AIMessage(content="仅是合成材料，未执行外部操作。")])
    pressure.append(HumanMessage(content="本轮代码 PRESSURE-925，只回复这个代码。"))
    return [("short", short, "SHORT-427", 32768), ("medium", medium, "MIDDLE-731", 32768),
            ("pressure", pressure, "PRESSURE-925", 12000)]


class Meter:
    def __init__(self, model, max_requests=20):
        self.model = model
        self.rows = []
        self.requests = 0
        self.phase = ""
        self.estimated_input = 0
        self.counter = ContextManager()
        self.max_requests = max_requests

    def claim(self, messages):
        next_input = self.estimated_input + self.counter.message_tokens(messages) + 4000
        if self.requests >= self.max_requests or next_input > 250000:
            raise RuntimeError("Probe request/input spending cap reached")
        self.requests += 1
        self.estimated_input = next_input

    def with_structured_output(self, schema):
        meter = self

        class Structured:
            async def ainvoke(self, messages, config=None, **kwargs):
                meter.claim(messages)
                start = time.perf_counter()
                # include_raw builds a RunnableMap which drops invocation kwargs
                # in this SDK version. Set the cap on the model before wrapping.
                limit = kwargs.pop("max_tokens")
                bounded = meter.model.model_copy(update={"max_tokens": limit})
                result = await bounded.with_structured_output(schema, include_raw=True).ainvoke(messages, config=config, **kwargs)
                row = model_usage_metrics(result["raw"], "mimo", round((time.perf_counter() - start) * 1000))
                row.update(phase=meter.phase, kind="summary")
                meter.rows.append(row)
                print(json.dumps({"event": "call", **row}, ensure_ascii=False), flush=True)
                if result.get("parsing_error"):
                    raise result["parsing_error"]
                return result["parsed"]

        return Structured()

    async def answer(self, messages):
        self.claim(messages)
        start = time.perf_counter()
        first_text = None
        combined = None
        # Schemas are sent for representative overhead; no tool may execute.
        bound = self.model.bind_tools(ALL_TOOLS, tool_choice="none")
        async for chunk in bound.astream(messages, max_tokens=1024, config={"callbacks": []}):
            if chunk.content and first_text is None:
                first_text = round((time.perf_counter() - start) * 1000)
            combined = chunk if combined is None else combined + chunk
        if combined is None:
            raise RuntimeError("Empty model stream")
        row = model_usage_metrics(combined, "mimo", round((time.perf_counter() - start) * 1000))
        row.update(phase=self.phase, kind="main", first_text_ms=first_text,
                   answer=combined.content, finish_reason=combined.response_metadata.get("finish_reason"),
                   tool_calls_returned=len(combined.tool_calls))
        self.rows.append(row)
        print(json.dumps({"event": "call", **row}, ensure_ascii=False), flush=True)
        return row


async def run(selected_case="all", repeats=2, max_requests=20, selected_policy="all"):
    model = llm.get_strict_model("mimo")
    model.max_retries = 0
    model.request_timeout = 50
    model.root_client.timeout = 50
    model.root_async_client.timeout = 50
    model.stream_usage = True
    meter = Meter(model, max_requests=max_requests)
    report = {"provider": "mimo", "model": config.MIMO_CHAT_MODEL, "cases": [], "calls": meter.rows,
              "limits": {"api_calls": max_requests, "estimated_input_with_schema_allowance": 250000, "main_output_per_call": 1024,
                         "summary_final_tokens": 1600, "summary_generation_tokens_tuned": 4096, "summary_generation_tokens_baseline": 1600},
              "probe_version": 3, "repeats": repeats,
              "method": "synthetic paired replay, two identical main requests per policy; real summaries; tools disabled; no automatic retries; temporary DB; not a production task-success evaluation"}
    nonce = uuid.uuid4().hex
    try:
        # Disable external tracing so only the selected provider receives fixtures.
        with tracing_context(enabled=False), patch.object(llm, "get_strict_model", return_value=meter):
            for case_index, (name, history, expected, budget) in enumerate(cases()):
                if selected_case != "all" and name != selected_case:
                    continue
                policies = [("baseline", baseline.ContextManager), ("tuned", ContextManager)]
                if case_index % 2:
                    policies.reverse()
                for label, cls in policies:
                    if selected_policy != "all" and label != selected_policy:
                        continue
                    # Equal-length distinct early prefixes prevent A from warming B.
                    system = f"probe_namespace={nonce}-{case_index}-{'A' if label == 'baseline' else 'B'}\n仅分析合成测试材料，不执行工具。回答严格遵循最新用户请求，只输出代码或 UNKNOWN。"
                    runtime = "当前测试时间：2026-09-27 10:00:00。"
                    tuned = label == "tuned"
                    manager = cls(max_context_tokens=budget, output_reserve=1024, safety_margin=2048,
                                  summary_tokens=1600, summary_input_tokens=12000, tool_result_tokens=1200, max_messages=20,
                                  **({"tool_inline_tokens": 8192, "compact_trigger": .90, "compact_target": .65, "summary_generation_tokens": 4096} if tuned else {}))
                    with tempfile.TemporaryDirectory() as tmp, patch.object(connection, "DATABASE_PATH", str(Path(tmp) / "probe.db")):
                        await connection.init_db()
                        meter.phase = f"{name}/{label}"
                        start = time.perf_counter()
                        prepared = await manager.prepare(history, system if tuned else system + "\n" + runtime,
                                                         ALL_TOOLS, "synthetic-probe", name,
                                                         **({"runtime_context": runtime} if tuned else {}))
                        prep_ms = round((time.perf_counter() - start) * 1000)
                        for attempt in range(repeats):
                            meter.phase = f"{name}/{label}/{attempt + 1}"
                            row = await meter.answer(prepared.messages)
                            row["correct"] = isinstance(row["answer"], str) and row["answer"].strip().strip('"') == expected
                            report["cases"].append({"scenario": name, "policy": label, "repeat": attempt + 1,
                                                    "prepare_ms": prep_ms if attempt == 0 else 0,
                                                    "summary_calls": prepared.metrics["summary_merges"] if attempt == 0 else 0,
                                                    **row})
    except Exception as exc:
        # Never print credentials, request headers, or unfiltered provider errors.
        report.update(failure_details(exc))
        report["stopped_phase"] = meter.phase
    report["requests_attempted"] = meter.requests
    print("PROBE_RESULT=" + json.dumps(report, ensure_ascii=False), flush=True)


async def run_summary_replay(max_requests=4):
    """Replay the two small incremental summary stages that previously truncated.

    Same synthetic history shape, now using the production merge method and
    separate generation allowance. Not a byte-identical deterministic model run.
    """
    model = llm.get_strict_model("mimo")
    model.request_timeout = 50
    model.root_client.timeout = 50
    model.root_async_client.timeout = 50
    model.stream_usage = True
    meter = Meter(model, max_requests=max_requests)
    manager = ContextManager()
    _, history, expected, _ = cases()[0]
    report = {"probe_version": 3, "scenario": "incremental_summary_replay", "model": config.MIMO_CHAT_MODEL,
              "calls": meter.rows, "summary_final_limit": manager.summary_tokens,
              "summary_generation_limit": manager.summary_generation_tokens, "stored_summary_tokens": []}
    previous = ""
    try:
        with tracing_context(enabled=False), patch.object(llm, "get_strict_model", return_value=meter):
            for stage, indices in enumerate([range(5), range(5, 6)]):
                records = []
                for index in indices:
                    value = payload(history[index])
                    value["ref"] = reference(index, history[index])
                    records.extend(manager.split_record(value, 3000))
                meter.phase = f"summary_replay/stage{stage + 1}"
                previous = await manager.merge_summary(previous, records)
                report["stored_summary_tokens"].append(manager.tokens(previous))
            messages = manager.render("仅分析合成测试，不执行工具，只回答本轮代码。", previous, history, 6, len(history) - 1)
            for repeat in range(2):
                meter.phase = f"summary_replay/main{repeat + 1}"
                row = await meter.answer(messages)
                row["correct"] = row["answer"].strip().strip('"') == expected
    except Exception as exc:
        report.update(failure_details(exc))
        report["stopped_phase"] = meter.phase
    report["requests_attempted"] = meter.requests
    print("PROBE_RESULT=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Authorize real billable model requests")
    parser.add_argument("--case", choices=["all", "short", "medium", "pressure", "summary_replay"], default="all")
    parser.add_argument("--repeats", type=int, choices=[1, 2], default=2)
    parser.add_argument("--max-requests", type=int, choices=range(1, 21), default=20)
    parser.add_argument("--policy", choices=["all", "baseline", "tuned"], default="all")
    args = parser.parse_args()
    if not args.run:
        parser.error("Pass --run to perform billable API calls")
    if args.case == "summary_replay":
        asyncio.run(run_summary_replay(args.max_requests))
    else:
        asyncio.run(run(args.case, args.repeats, args.max_requests, args.policy))
