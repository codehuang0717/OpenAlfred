"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.

Supports dynamic model switching between cloud GPT and local Ollama Gemma4.
Uses wrap_model_call middleware to swap the model before each LLM invocation.
Integrates the three-layer context management system.
"""

import asyncio
import logging
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import HumanMessage, SystemMessage
from schema import AgentState
from llm import get_model
from prompts import AGENT_SYSTEM_PROMPT
from context_manager import ContextManager
from tools.memory import memTools
from tools.reminder import reminder_tools
from tools.call_user import call_tools
from tools.todos import todo_tools

logger = logging.getLogger("chat-agent")
ctx_manager = ContextManager()

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
        """Inject dynamic time and context memories into the conversation."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        
        # Inject Current Time
        now_uk = datetime.now(ZoneInfo("Europe/London"))
        time_str = now_uk.strftime("%Y-%m-%d %H:%M:%S")
        weekday = now_uk.strftime("%A")
        
        time_msg = SystemMessage(
            content=f"Current Time: {time_str} ({weekday}). Timezone: Europe/London."
        )
        
        if isinstance(state, dict):
            messages = state.get("messages", [])
            summary = state.get("conversation_summary", "")
            
            # Inject summary context if available
            context_parts = [time_msg]
            if summary:
                context_parts.append(
                    SystemMessage(content=f"[对话历史摘要]\n{summary}")
                )
            
            # Apply sliding window
            recent_messages = messages[-ctx_manager.max_messages:] if len(messages) > ctx_manager.max_messages else messages
            state["messages"] = context_parts + list(recent_messages)
        elif hasattr(state, "messages"):
            messages = list(state.messages)
            summary = getattr(state, "conversation_summary", "")
            
            context_parts = [time_msg]
            if summary:
                context_parts.append(
                    SystemMessage(content=f"[对话历史摘要]\n{summary}")
                )
            
            recent_messages = messages[-ctx_manager.max_messages:] if len(messages) > ctx_manager.max_messages else messages
            state.messages = context_parts + recent_messages

        return state

    async def aafter_agent(self, state, runtime, **kwargs):
        """Post-processing: trigger summarization if conversation is long enough."""
        if isinstance(state, dict):
            messages = state.get("messages", [])
        elif hasattr(state, "messages"):
            messages = list(state.messages)
        else:
            return state

        # Check if summarization is needed
        if ctx_manager.should_summarize(messages):
            try:
                to_summarize, to_keep = ctx_manager.get_messages_to_summarize(messages)
                if to_summarize:
                    summary_prompt = ctx_manager.build_summary_prompt(to_summarize)
                    llm = get_model("gpt-cloud")
                    result = await llm.ainvoke([HumanMessage(content=summary_prompt)])
                    new_summary = result.content.strip()
                    
                    # Merge with existing summary
                    existing = state.get("conversation_summary", "") if isinstance(state, dict) else getattr(state, "conversation_summary", "")
                    if existing:
                        new_summary = f"{existing}\n\n{new_summary}"
                    
                    if isinstance(state, dict):
                        state["conversation_summary"] = new_summary
                        state["messages"] = to_keep
                    elif hasattr(state, "conversation_summary"):
                        state.conversation_summary = new_summary
                        state.messages = to_keep
                        
                    logger.info("[ContextManager] Summarized older messages")
            except Exception as e:
                logger.warning(f"[ContextManager] Summarization failed: {e}")

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
