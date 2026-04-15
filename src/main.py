"""
This is the main entry point for the text-based LangGraph agent.
It defines the workflow graph using custom nodes and edges for maximum efficiency
and long-term companionship features.
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from schema import AgentState
from llm import get_model
from nodes import load_context_node, agent_node, summarize_node

# Import tools
from tools.memory import memTools
from tools.reminder import reminder_tools
from tools.call_user import call_tools
from tools.todos import todo_tools

logger = logging.getLogger("chat-agent")

# ─── Graph Logic ──────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["tools", "summarize"]:
    """
    Route to tools if the last message has tool calls, otherwise summarize.
    """
    messages = state.messages
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "summarize"

# ─── Graph Construction ───────────────────────────────────────────────────

# Define the tools available to the agent
tools = [*todo_tools, *memTools, *reminder_tools, *call_tools]
tool_node = ToolNode(tools)

# Bind tools to the model once
# NOTE: We use gpt-cloud as requested.
model = get_model("gpt-cloud").bind_tools(tools)

def call_model(state: AgentState, config):
    """Wrapper for the agent node to use the bound model."""
    # We pass the bound model to the agent_node logic
    # In nodes.py, agent_node currently fetches the model, but we can override it here
    # or improve nodes.py to be more generic.
    return agent_node(state, config)

# Define the workflow
workflow = StateGraph(AgentState)

workflow.add_node("load_context", load_context_node)
workflow.add_node("agent", agent_node) # We'll update nodes.py to handle the bound model or pass it
workflow.add_node("tools", tool_node)
workflow.add_node("summarize", summarize_node)

workflow.set_entry_point("load_context")
workflow.add_edge("load_context", "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "summarize": "summarize"
    }
)

workflow.add_edge("tools", "agent")
workflow.add_edge("summarize", END)

# Compile the graph
# Note: Persistence is handled by the 'langgraph dev' platform.
graph = workflow.compile()

# For local debugging
if __name__ == "__main__":
    print("Graph compiled successfully. Use 'langgraph dev' to run.")
