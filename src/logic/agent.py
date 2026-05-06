"""
This is the main entry point for the text-based LangGraph agent.
It defines the workflow graph using custom nodes and edges.
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from logic.schema import AgentState
from logic.nodes import load_context_node, agent_node, summarize_node, extract_knowledge_node

logger = logging.getLogger("chat-agent")

# ─── Graph Logic ──────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["tools", "extract_knowledge"]:
    """Route to tools if the last message has tool calls, otherwise extract_knowledge."""
    messages = state.messages
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "extract_knowledge"

# ─── Graph Construction ───────────────────────────────────────────────────

from tools import ALL_TOOLS

tool_node = ToolNode(ALL_TOOLS)

workflow = StateGraph(AgentState)

workflow.add_node("load_context", load_context_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("extract_knowledge", extract_knowledge_node)
workflow.add_node("summarize", summarize_node)

workflow.set_entry_point("load_context")
workflow.add_edge("load_context", "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "extract_knowledge": "extract_knowledge"
    }
)

workflow.add_edge("tools", "agent")
workflow.add_edge("extract_knowledge", "summarize")
workflow.add_edge("summarize", END)

graph = workflow.compile()

if __name__ == "__main__":
    print("Graph compiled successfully. Use 'langgraph dev' to run.")
