"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.

Supports dynamic model switching between cloud GPT and local Ollama Gemma4.
Uses wrap_model_call middleware to swap the model before each LLM invocation.
"""

import asyncio
import logging
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from schema import AgentState
from llm import get_model
from prompts import AGENT_SYSTEM_PROMPT
from tools.memory import memTools
from tools.reminder import reminder_tools
from tools.call_user import call_tools
from tools.todos import todo_tools

logger = logging.getLogger("chat-agent")

async def init():
    await init_db()

if __name__ == "__main__":
    asyncio.run(init())

# ─── Model Switching Middleware ──────────────────────────────────────────

class ModelSwitchMiddleware(AgentMiddleware):
    """Middleware that swaps the LLM model based on model_selection in agent state."""

    name = "model_switch"

    async def awrap_model_call(self, request: ModelRequest, handler: Callable) -> Any:
        """Intercept model call and swap the model based on state asynchronously."""
        model_selection = "gpt-cloud"
        if hasattr(request, "state") and request.state:
            model_selection = request.state.get("model_selection", "gpt-cloud")

        target_model = get_model(model_selection)

        if target_model is not request.model:
            logger.info(f"[ModelSwitch] Swapping model to: {model_selection}")
            request.model = target_model

        return await handler(request)

    async def abefore_agent(self, state, **kwargs):
        """Inject dynamic time into system prompt."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from langchain_core.messages import SystemMessage
        
        # Inject Current Time
        now_uk = datetime.now(ZoneInfo("Europe/London"))
        time_str = now_uk.strftime("%Y-%m-%d %H:%M:%S")
        weekday = now_uk.strftime("%A")
        
        time_msg = SystemMessage(
            content=f"Current Time: {time_str} ({weekday}). Timezone: Europe/London."
        )
        
        if isinstance(state, dict):
            messages = state.get("messages", [])
            state["messages"] = [time_msg] + list(messages)
        elif hasattr(state, "messages"):
            state.messages = [time_msg] + list(state.messages)

        return state

    async def aafter_agent(self, state, runtime, **kwargs):
        """No manual persistence needed; handled by LangGraph checkpointer."""
        return state

# ─── Agent Creation ──────────────────────────────────────────────────────

agent = create_agent(
    model=get_model("gpt-cloud"),  # Default (will be swapped by middleware)
    tools=[*todo_tools, *memTools, *reminder_tools, *call_tools],
    middleware=[ModelSwitchMiddleware()],
    state_schema=AgentState,
    system_prompt=AGENT_SYSTEM_PROMPT,
)

graph = agent
