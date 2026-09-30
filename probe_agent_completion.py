"""Manual real-API probe, synthetic messages only; no account data or side effects.

Run: uv run python probe_agent_completion.py
Performs two MiMo requests to validate reasoning/tool/result round-trip.
"""

import asyncio
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from core.config import config
from logic.agent_outcome import classify_response
from logic.context_metrics import model_usage_metrics
from logic.context_payload import serialize
from services.llm import get_strict_model, output_limit_kwargs


@tool
def read_test_value() -> str:
    """Read a synthetic test value. No side effects or account access."""
    return "42"


async def main() -> None:
    model = get_strict_model("mimo")
    messages = [SystemMessage(content="Call read_test_value exactly once, then answer with its value only. This is a synthetic protocol test."),
                HumanMessage(content="What is the test value?")]
    bound = model.bind_tools([read_test_value], tool_choice="read_test_value")
    started = time.monotonic()
    async with asyncio.timeout(120):
        first = await bound.ainvoke(messages, **output_limit_kwargs("mimo", config.CONTEXT_OUTPUT_RESERVE))
        first_outcome = classify_response(first, {"read_test_value"})
        print(serialize({**model_usage_metrics(first, "mimo", round((time.monotonic() - started) * 1000)),
                         "phase": "tool_request", "outcome": first_outcome.as_dict(),
                         "reasoning_preserved": isinstance(first.additional_kwargs.get("reasoning_content"), str)}), flush=True)
        if first_outcome.status != "tools" or len(first.tool_calls) != 1:
            raise RuntimeError("Expected one complete tool call")
        call = first.tool_calls[0]
        messages += [first, ToolMessage(content=read_test_value.invoke(call["args"]), tool_call_id=call["id"])]
        started = time.monotonic()
        second = await model.bind_tools([read_test_value], tool_choice="none").ainvoke(
            messages, **output_limit_kwargs("mimo", config.CONTEXT_OUTPUT_RESERVE))
        outcome = classify_response(second, {"read_test_value"})
        print(serialize({**model_usage_metrics(second, "mimo", round((time.monotonic() - started) * 1000)),
                         "phase": "final_answer", "outcome": outcome.as_dict(),
                         "answer_matches": second.content.strip() == "42"}), flush=True)
        if outcome.status != "completed" or second.content.strip() != "42":
            raise RuntimeError("Expected a completed visible answer")


if __name__ == "__main__":
    asyncio.run(main())
