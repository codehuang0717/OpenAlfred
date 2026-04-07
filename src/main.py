"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.

Supports dynamic model switching between cloud GPT and local Ollama Gemma4.
Uses wrap_model_call middleware to swap the model before each LLM invocation.
"""

import asyncio
import logging
from typing import Any, Callable, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from tools.memTool import memTools
from tools.reminder import reminder_tools
from tools.speak import speak_tools
from tools.call_user import call_tools
from tools.todos import AgentState, todo_tools
from database import init_db

logger = logging.getLogger("chat-agent")


async def init():
    await init_db()


if __name__ == "__main__":
    asyncio.run(init())


# ─── Model Cache ─────────────────────────────────────────────────────────

_model_cache: dict[str, BaseChatModel] = {}


def _get_model(selection: str) -> BaseChatModel:
    """Get or create a cached model instance."""
    if selection not in _model_cache:
        if selection == "gemma-local":
            try:
                from langchain_ollama import ChatOllama
                _model_cache[selection] = ChatOllama(
                    model="gemma4:e2b",
                    base_url="http://localhost:11434",
                )
                logger.info("Initialized Ollama gemma4:e2b model")
            except ImportError:
                logger.warning("langchain-ollama not installed, falling back to GPT")
                _model_cache[selection] = ChatOpenAI(model="gpt-5.4-nano")
        else:
            _model_cache[selection] = ChatOpenAI(model="gpt-5.4-nano")
    return _model_cache[selection]


# ─── Model Switching Middleware ──────────────────────────────────────────

class ModelSwitchMiddleware(AgentMiddleware):
    """Middleware that swaps the LLM model based on model_selection in agent state."""

    name = "model_switch"

    def wrap_model_call(self, request: ModelRequest, handler: Callable) -> Any:
        """Intercept model call and swap the model based on state."""
        model_selection = "gpt-cloud"
        if hasattr(request, "state") and request.state:
            model_selection = request.state.get("model_selection", "gpt-cloud")

        target_model = _get_model(model_selection)

        if target_model is not request.model:
            logger.info(f"[ModelSwitch] Swapping model to: {model_selection}")
            request.model = target_model

        return handler(request)

    async def abefore_agent(self, state, **kwargs):
        """Read model_selection from state before agent runs, and record user chat history."""
        model_selection = state.get("model_selection", "gpt-cloud")
        
        # Save messages to database explicitly for the specified session
        from database import add_chat_message
        session_id = state.get("chat_session_id")
        if session_id:
            messages = state.get("messages", [])
            for m in messages:
                if m.type in ("human", "ai"):
                    content = ""
                    if isinstance(m.content, list):
                        content = " ".join(str(c) for c in m.content)
                    else:
                        content = str(m.content)
                        
                    if content.strip():
                        msg_id = getattr(m, "id", "") or str(hash(content))
                        role = "user" if m.type == "human" else "assistant"
                        await add_chat_message(session_id, role, content, msg_id)
        
        return state

    async def aafter_agent(self, state, runtime, **kwargs):
        """Record chat history after agent completes to capture AI responses immediately."""
        from database import add_chat_message
        session_id = state.get("chat_session_id")
        if session_id:
            messages = state.get("messages", [])
            for m in messages:
                if m.type in ("human", "ai"):
                    content = ""
                    if isinstance(m.content, list):
                        content = " ".join(str(c) for c in m.content)
                    else:
                        content = str(m.content)
                        
                    if content.strip():
                        msg_id = getattr(m, "id", "") or str(hash(content))
                        role = "user" if m.type == "human" else "assistant"
                        await add_chat_message(session_id, role, content, msg_id)
        
        return state

    async def awrap_model_call(self, request: ModelRequest, handler: Callable) -> Any:
        """Intercept model call and swap the model based on state asynchronously."""
        model_selection = "gpt-cloud"
        if hasattr(request, "state") and request.state:
            model_selection = request.state.get("model_selection", "gpt-cloud")

        target_model = _get_model(model_selection)

        if target_model is not request.model:
            logger.info(f"[ModelSwitch] Swapping model to: {model_selection}")
            request.model = target_model

        return await handler(request)


# ─── Agent Creation ──────────────────────────────────────────────────────

agent = create_agent(
    model=ChatOpenAI(model="gpt-5.4-nano"),  # Default (will be swapped by middleware)
    tools=[*todo_tools, *memTools, *reminder_tools, *speak_tools, *call_tools],
    middleware=[ModelSwitchMiddleware()],
    state_schema=AgentState,
    system_prompt="""
你是用户的智能助手，可以帮助用户管理任务、设置提醒等。

## 可用工具

### 1. 任务管理 (Todos)
- **get_todos**: 获取所有任务列表
- **add_todo**: 添加新任务

### 2. 提醒功能 (Reminders)
- **add_reminder**: 设置定时提醒 (重要!)
  - 参数: body, scheduled_at, delivery_method, call_greeting
  - **重要原则**: 
    1. 如果用户提到"叫醒"、"起床"、"早点睡"、"紧急"、"别忘了"等词汇，或者你判定该提醒非常重要，**必须**设置 `delivery_method="call"`。
    2. 如果 `delivery_method="call"`，你**必须**提供一个亲切、自然且符合情境的 `call_greeting`（例如："老大，该起床了，太阳都晒屁股了！"）。
    3. 对于普通碎事，使用默认的 `delivery_method="push"`。
    4.电话提醒最好是提前一点，具体提前多久，时间由你自行判断

## 任务处理逻辑
1. 始终基于系统消息中提供的"当前时间"来计算相对时间（如"10分钟后"）。
2. 如果用户要求设置提醒，优先考虑是否需要电话呼叫。
""",
)

graph = agent
