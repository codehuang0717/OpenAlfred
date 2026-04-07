"""
Hand-crafted LangGraph StateGraph for the voice pipeline.

Bypasses LangGraph Server HTTP layer and CopilotKit middleware.
Designed for minimal latency in the STT → LLM → TTS loop.

Supports dynamic model switching between cloud GPT and local Ollama Gemma4.
"""

import logging
import os
import dotenv
from typing import Annotated, Optional
from datetime import datetime

# Load .env from project root (same as langgraph.json "env": "../../.env")
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
dotenv.load_dotenv(_env_path, override=True)
# Also load local .env if it exists
dotenv.load_dotenv(override=True)

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage
from typing_extensions import TypedDict

from tools.voice_tools import voice_tool_list

logger = logging.getLogger("voice-agent")


# ─── State ────────────────────────────────────────────────────────────────────

class VoiceAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    tts_text: Optional[str]
    model_selection: Optional[str]  # "gpt-cloud" or "gemma-local"


# ─── System Prompt ────────────────────────────────────────────────────────────

VOICE_SYSTEM_PROMPT = """\
你是用户的智能助手 Alfred。你正在通过电话与用户对话。

## 回复规则
- 你的回复将被直接转为语音播放给用户
- 用简短、自然的中文口语回复（50字以内）
- 不要使用 markdown、列表、编号、代码块等任何格式标记
- 像朋友之间说话一样，亲切自然
- 不在回复中主动推荐工具，直接执行用户的请求

## 可用工具

### 任务管理
- voice_get_todos: 获取所有任务列表
- voice_add_todo: 添加新任务

### 提醒功能
- voice_add_reminder: 设置定时提醒
  - 重要原则:
    1. 如果用户提到"叫醒"、"起床"、"早点睡"、"紧急"、"别忘了"等词汇，必须设置 delivery_method="call"
    2. 如果 delivery_method="call"，必须提供一个亲切自然的 call_greeting
    3. 普通碎事用 delivery_method="push"
    4. 电话提醒最好提前一点，具体提前多久由你判断
- voice_list_reminders: 列出所有提醒
- voice_cancel_reminder: 取消提醒

### 记忆功能
- voice_search_memory: 搜索用户的记忆和偏好
- voice_add_memory: 存储用户信息到长期记忆

### 电话功能
- voice_make_outbound_call: 主动拨打电话给用户

## 处理逻辑
1. 始终基于系统消息中的"当前时间"来计算相对时间
2. 如果用户要求设置提醒，优先考虑是否需要电话呼叫
"""


# ─── Model Factory ────────────────────────────────────────────────────────────

# Cache for model instances
_model_cache = {}


def get_voice_model(model_selection: str = "gpt-cloud"):
    """Return the appropriate LLM based on model selection, with caching."""
    if model_selection not in _model_cache:
        if model_selection == "gemma-local":
            try:
                from langchain_ollama import ChatOllama
                _model_cache[model_selection] = ChatOllama(
                    model="gemma4:e2b",
                    base_url="http://localhost:11434",
                )
                logger.info("Using local Ollama gemma4:e2b model for voice")
            except ImportError:
                logger.warning("langchain-ollama not installed, falling back to GPT")
                _model_cache[model_selection] = ChatOpenAI(model="gpt-5-mini")
        else:
            _model_cache[model_selection] = ChatOpenAI(model="gpt-5-mini")
    return _model_cache[model_selection]


# Default LLM (used for initial tool binding)
default_llm = ChatOpenAI(model="gpt-5-mini")


# ─── Nodes ────────────────────────────────────────────────────────────────────

async def llm_node(state: VoiceAgentState) -> dict:
    """LLM reasoning node with tool bindings and dynamic model selection."""
    messages = state["messages"]

    # Pick model based on state
    model_selection = state.get("model_selection") or "gpt-cloud"
    llm = get_voice_model(model_selection)
    llm_with_tools = llm.bind_tools(voice_tool_list)

    # Inject system prompt + current time at the head if not already present
    has_system = any(
        isinstance(m, SystemMessage) or (isinstance(m, dict) and m.get("role") == "system")
        for m in messages
    )
    if not has_system:
        from zoneinfo import ZoneInfo
        current_time = datetime.now(ZoneInfo("Europe/London")).strftime("%Y-%m-%d %H:%M:%S")
        system_msg = SystemMessage(
            content=f"{VOICE_SYSTEM_PROMPT}\n\nCurrent UK Local Time: {current_time}. Timezone: Europe/London."
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
