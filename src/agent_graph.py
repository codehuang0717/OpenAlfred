"""
Hand-crafted LangGraph StateGraph for the voice pipeline.

Bypasses LangGraph Server HTTP layer and CopilotKit middleware.
Designed for minimal latency in the STT → LLM → TTS loop.

Supports dynamic model switching between cloud GPT and local Ollama Gemma4.
注意，目前该Agent已经被废弃了，不要再使用
"""

import logging
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage
from tools.voice_tools import voice_tool_list
from schema import VoiceAgentState
from llm import get_model
from prompts import VOICE_SYSTEM_PROMPT
from config import config

logger = logging.getLogger("voice-agent")

# Default LLM (used for initial tool binding)
default_llm = get_model("gpt-cloud")


# ─── Nodes ────────────────────────────────────────────────────────────────────

async def llm_node(state: VoiceAgentState) -> dict:
    """LLM reasoning node with tool bindings and dynamic model selection."""
    messages = state["messages"]

    # Pick model based on state
    model_selection = state.get("model_selection") or "gpt-cloud"
    llm = get_model(model_selection)
    llm_with_tools = llm.bind_tools(voice_tool_list)

    # Inject system prompt + current time at the head if not already present
    has_system = any(
        isinstance(m, SystemMessage) or (isinstance(m, dict) and m.get("role") == "system")
        for m in messages
    )
    if not has_system:
        from zoneinfo import ZoneInfo
        now_uk = datetime.now(ZoneInfo("Europe/London"))
        current_time = now_uk.strftime("%Y-%m-%d %H:%M:%S")
        weekday = now_uk.strftime("%A")
        system_msg = SystemMessage(
            content=f"{VOICE_SYSTEM_PROMPT}\n\nCurrent UK Local Time: {current_time} ({weekday}). Timezone: Europe/London."
        )
        messages = [system_msg] + list(messages)

    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


def should_continue(state: VoiceAgentState) -> str:
    """Route: if LLM wants to call tools, go to tool_node; otherwise, extract response."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return "voice_reply_node"


# Tool node: uses LangGraph's built-in ToolNode for parallel execution
tool_node = ToolNode(voice_tool_list)


async def voice_reply_node(state: VoiceAgentState) -> dict:
    """Extract the final LLM text as tts_text."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage):
        text = last_message.content or "收到"
        # Strip any accidental [TERMINATE] tags
        clean_text = text.replace("[TERMINATE]", "").strip()
        return {"tts_text": clean_text if clean_text else "收到"}
    return {"tts_text": "收到"}


# ─── Graph Assembly ───────────────────────────────────────────────────────────

def build_voice_graph():
    graph = StateGraph(VoiceAgentState)

    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("voice_reply_node", voice_reply_node)

    graph.set_entry_point("llm_node")

    graph.add_conditional_edges(
        "llm_node",
        should_continue,
        {
            "tool_node": "tool_node",
            "voice_reply_node": "voice_reply_node",
        },
    )

    # After tool execution, loop back to LLM for more reasoning
    graph.add_edge("tool_node", "llm_node")

    # After extracting voice reply, we're done
    graph.add_edge("voice_reply_node", END)

    return graph.compile()


voice_graph = build_voice_graph()
