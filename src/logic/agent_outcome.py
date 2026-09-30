"""Completion contract at the model/tool boundary, separate from HTTP success."""

from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import AIMessage


class AgentRunError(RuntimeError):
    """Raised after the failure observation has been checkpointed."""


@dataclass(frozen=True)
class ModelOutcome:
    status: Literal["tools", "completed", "failed"]
    code: str = ""
    finish_reason: str = ""

    def as_dict(self) -> dict:
        return {"status": self.status, "code": self.code, "finish_reason": self.finish_reason}


def visible_text(response: AIMessage) -> str:
    """Reasoning and function-call blocks are not user-visible answers."""
    if isinstance(response.content, str):
        text = response.content
    else:
        parts = []
        for block in response.content:
            if isinstance(block, str):
                parts.append(block)
            elif block.get("type") in {"text", "output_text"}:
                parts.append(str(block.get("text") or ""))
            elif block.get("type") == "refusal":
                parts.append(str(block.get("refusal") or ""))
        text = "".join(parts)
    refusal = response.additional_kwargs.get("refusal")
    return text + (refusal if isinstance(refusal, str) else "")


def classify_response(response: AIMessage, allowed_tools: set[str]) -> ModelOutcome:
    metadata = response.response_metadata
    reason = str(metadata.get("finish_reason") or metadata.get("done_reason") or "").lower()
    status = metadata.get("status")
    if reason in {"length", "max_tokens", "max_output_tokens", "repetition_truncation"}:
        return ModelOutcome("failed", "output_truncated", reason)
    if status in {"incomplete", "failed", "cancelled"} or metadata.get("incomplete_details") or metadata.get("error"):
        return ModelOutcome("failed", "provider_incomplete", reason)
    if reason in {"content_filter", "safety", "recitation", "blocklist", "prohibited_content"}:
        return ModelOutcome("failed", "content_blocked", reason)
    if reason not in {"stop", "end_turn", "tool_calls", "tool_use", "function_call"} and status != "completed":
        return ModelOutcome("failed", "missing_completion_signal", reason)
    if response.invalid_tool_calls:
        return ModelOutcome("failed", "invalid_tool_calls", reason)
    if response.tool_calls:
        ids = set()
        for call in response.tool_calls:
            call_id = call.get("id")
            if (not call_id or call_id in ids or call.get("name") not in allowed_tools
                    or not isinstance(call.get("args"), dict)):
                return ModelOutcome("failed", "invalid_tool_calls", reason)
            ids.add(call_id)
        return ModelOutcome("tools", finish_reason=reason)
    if reason in {"tool_calls", "tool_use", "function_call"}:
        return ModelOutcome("failed", "invalid_tool_calls", reason)
    if not visible_text(response).strip():
        return ModelOutcome("failed", "empty_answer", reason)
    return ModelOutcome("completed", finish_reason=reason)


def failure_text(code: str) -> str:
    descriptions = {
        "output_truncated": "模型生成达到上限，思考与正文共享生成预算，本次回答被截断",
        "provider_incomplete": "模型服务返回了未完成或失败的响应",
        "content_blocked": "模型服务拦截了本次响应",
        "missing_completion_signal": "模型响应缺少有效的完成信号，无法确认是否完整",
        "invalid_tool_calls": "模型返回的工具调用不完整或无效，本轮工具未执行",
        "empty_answer": "模型结束生成，但没有返回可展示的回答",
    }
    return (f"本次运行未完成（{code}）：{descriptions[code]}。"
            "未自动重试或切换模型；此前已完成的工具操作不会回滚，请勿盲目重复提交。")
