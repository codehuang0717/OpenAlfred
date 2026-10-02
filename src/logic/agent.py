"""
This is the main entry point for the text-based LangGraph agent.
It defines the workflow graph using custom nodes and edges.
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError
from utils.auth_utils import MissingThreadContextError, MissingUserContextError, UserContextMismatchError

from logic.schema import AgentState
from logic.nodes import load_context_node, prepare_context_node, agent_node, extract_knowledge_node, fail_run_node

logger = logging.getLogger("chat-agent")

# ─── Graph Logic ──────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["tools", "extract_knowledge", "fail_run"]:
    """Only validated model outcomes can dispatch tools or complete a turn."""
    status = state.agent_outcome.get("status")
    if status == "failed":
        return "fail_run"
    if status == "tools":
        return "tools"
    if status == "completed":
        return "extract_knowledge"
    raise RuntimeError("agent_node did not produce a validated outcome")

# ─── Graph Construction ───────────────────────────────────────────────────

from tools import ALL_TOOLS
from services.tool_observations import observe_tool_call


def tool_error_message(error: Exception) -> str:
    """Keep a failed tool call paired and visible so the model can finish the turn."""
    if isinstance(error, (MissingUserContextError, UserContextMismatchError, MissingThreadContextError)):
        raise error  # Authentication and thread isolation failures are not recoverable tool data.
    logger.error(
        "Tool execution failed: %s", type(error).__name__,
        exc_info=(type(error), error, error.__traceback__),
    )
    if isinstance(error, ToolInvocationError):
        reason = "工具参数不符合定义，请根据工具参数结构修正"
    elif isinstance(error, PermissionError):
        reason = "访问被拒绝"
    elif isinstance(error, ValueError):
        reason = "输入或存储的数据格式无效"
    elif isinstance(error, (ConnectionError, TimeoutError)):
        reason = "外部服务连接失败或超时"
    else:
        reason = "内部执行错误"
    return (
        f"工具执行失败（{type(error).__name__}：{reason}）。本次调用没有可信结果，"
        "不得宣称任务已成功；可以继续独立步骤，否则向用户说明失败。"
        "不要自动重复可能有副作用的操作。"
    )


tool_node = ToolNode(ALL_TOOLS, handle_tool_errors=tool_error_message, awrap_tool_call=observe_tool_call)

workflow = StateGraph(AgentState)

workflow.add_node("load_context", load_context_node)
workflow.add_node("prepare_context", prepare_context_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("extract_knowledge", extract_knowledge_node)
workflow.add_node("fail_run", fail_run_node)

workflow.set_entry_point("load_context")
workflow.add_edge("load_context", "prepare_context")
workflow.add_edge("prepare_context", "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "extract_knowledge": "extract_knowledge",
        "fail_run": "fail_run",
    }
)

workflow.add_edge("tools", "prepare_context")
workflow.add_edge("extract_knowledge", END)

graph = workflow.compile()

if __name__ == "__main__":
    print("Graph compiled successfully. Use 'langgraph dev' to run.")
