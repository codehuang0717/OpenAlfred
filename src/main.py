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

async def _async_summarize_task(to_summarize, existing_summary, summarized_count, thread_id):
    """Background task to summarize older messages without blocking the user response."""
    try:
        from llm import get_model
        from langchain_core.messages import HumanMessage
        llm = get_model("gpt-cloud")
        summary_prompt = ctx_manager.build_summary_prompt(to_summarize)
        
        # Use config={"callbacks": []} to ensure the LLM thoughts don't stream to chat UI
        result = await llm.ainvoke([HumanMessage(content=summary_prompt)], config={"callbacks": []})
        new_summary = result.content.strip()
        
        combined_summary = new_summary
        if existing_summary:
            combined_summary = f"{existing_summary}\n\n{new_summary}"
        
        # --- Evaluation Logging ---
        import os
        from datetime import datetime
        eval_msg = f"\n=== Summary Evaluation {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        eval_msg += f"EXISTING SUMMARY:\n{existing_summary if existing_summary else 'None'}\n\n"
        eval_msg += "MESSAGES SUMMARIZED:\n"
        for m in to_summarize:
            eval_msg += f"  [{getattr(m, 'type', 'N/A')}] {getattr(m, 'content', 'N/A')}\n"
        eval_msg += f"\nNEWLY GENERATED SUMMARY:\n{new_summary}\n\n"
        eval_msg += f"COMBINED SUMMARY:\n{combined_summary}\n"
        eval_msg += "="*60 + "\n"
        
        # Terminal output
        print(eval_msg)
        
        # File logging
        log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "summary_eval.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(eval_msg)
        logger.info("[ContextManager] Async summarization completed and saved eval log.")
        # --------------------------
        
        # --- Commit to Side Channel Storage ---
        from database import set_thread_memory
        new_count = summarized_count + len(to_summarize)
        await set_thread_memory(thread_id, combined_summary, new_count)
        logger.info(f"[ContextManager] Successfully saved background summary for thread {thread_id} (count: {new_count})")
            
    except Exception as e:
        logger.warning(f"[ContextManager] Async summarization failed: {e}")

class ModelSwitchMiddleware(AgentMiddleware):
    """Middleware that swaps the LLM model based on model_selection in agent state."""

    name = "model_switch"

    async def awrap_model_call(self, request: ModelRequest, handler: Callable) -> Any:
        """Intercept model call and swap the model based on state asynchronously."""
        model_selection = "gpt-cloud"
        if hasattr(request, "state") and request.state:
            state = getattr(request, "state")
            model_selection = state.get("model_selection", "gpt-cloud") if isinstance(state, dict) else getattr(state, "model_selection", "gpt-cloud")
        
        messages = list(request.messages)
        thread_id = "default_thread"
        if hasattr(request, "runtime") and getattr(request.runtime, "config", None):
            thread_id = request.runtime.config.get("configurable", {}).get("thread_id")
        if (not thread_id or thread_id == "default_thread") and messages and getattr(messages[0], "id", None):
            thread_id = str(messages[0].id)
        if not thread_id: thread_id = "default_thread"
            
        from database import get_thread_memory
        summary, _ = await get_thread_memory(thread_id)

        target_model = get_model(model_selection)
        
        # --- Inject Dynamic Info via system_message ---
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now_uk = datetime.now(ZoneInfo("Europe/London"))
        time_str = now_uk.strftime("%Y-%m-%d %H:%M:%S")
        weekday = now_uk.strftime("%A")

        context = f"[系统信息]\nCurrent Time: {time_str} ({weekday}). Timezone: Europe/London."
        if summary:
            context += f"\n\n[对话历史摘要]\n{summary}"

        old_sys = request.system_message.content if getattr(request, "system_message", None) else ""
        new_sys_content = f"{old_sys}\n\n{context}".strip()
        from langchain_core.messages import SystemMessage
        new_sys = SystemMessage(content=new_sys_content)

        # Truncate request messages to slide window (LLM context limits)
        # We do not delete messages from the actual DB State, so UI retains history.
        messages = list(request.messages)
        trimmed_messages = messages[-ctx_manager.max_messages:] if len(messages) > ctx_manager.max_messages else messages

        # Apply override securely to the ModelRequest
        request = request.override(
            model=target_model,
            system_message=new_sys,
            messages=trimmed_messages
        )
        # --------------------------------------------------------

        if target_model is not request.model:
            logger.info(f"[ModelSwitch] Swapping model to: {model_selection}")

        return await handler(request)

    async def abefore_agent(self, state, **kwargs):
        """Pass through; DB-preserving injection happens via ModelRequest in awrap_model_call."""
        return state

    async def aafter_agent(self, state, runtime, **kwargs):
        """Auto summarize asynchronously if conversation is long."""
        if isinstance(state, dict):
            messages = state.get("messages", [])
        elif hasattr(state, "messages"):
            messages = list(state.messages)
        else:
            return state

        thread_id = "default_thread"
        if hasattr(runtime, "config") and runtime.config:
            thread_id = runtime.config.get("configurable", {}).get("thread_id")
        if (not thread_id or thread_id == "default_thread") and messages and getattr(messages[0], "id", None):
            thread_id = str(messages[0].id)
        if not thread_id: thread_id = "default_thread"

        from database import get_thread_memory
        existing_summary, summarized_count = await get_thread_memory(thread_id)
        
        unsummarized = messages[summarized_count:]

        # Delegate logic safely to ContextManager
        if ctx_manager.should_summarize(unsummarized):
            to_summarize, _ = ctx_manager.get_messages_to_summarize(unsummarized)
            
            if to_summarize:
                import asyncio
                # Fire and forget the summary background task against side channel DB
                asyncio.create_task(
                    _async_summarize_task(to_summarize, existing_summary, summarized_count, thread_id)
                )

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
