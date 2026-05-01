from utils.logger import get_logger
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode

from schema import AgentState
from llm import get_model
from prompts import AGENT_SYSTEM_PROMPT
from database import get_thread_memory, set_thread_memory
from context_manager import ContextManager

logger = get_logger("graph-nodes")
ctx_manager = ContextManager()

async def load_context_node(state: AgentState, config):
    """
    Node to inject dynamic context (time, summary) into the message list.
    Replaces the 'awrap_model_call' logic from middleware.
    """
    # 1. Get current time
    now_uk = datetime.now(ZoneInfo("Europe/London"))
    time_str = now_uk.strftime("%Y-%m-%d %H:%M:%S")
    weekday = now_uk.strftime("%A")
    
    # 2. Get Thread Memory (Summary) from DB if not already in state
    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
    summary = state.conversation_summary
    summarized_count = state.summarized_count
    
    if not summary:
        summary, summarized_count = await get_thread_memory(thread_id)
        
    context = f"[系统信息]\nCurrent Time: {time_str} ({weekday}). Timezone: Europe/London."
    if summary:
        context += f"\n\n[对话历史摘要]\n{summary}"
        
    # To avoid polluting history with multiple system messages, we store the current 
    # instruction in a separate state field and prepend it only during the LLM call.
    return {
        "system_instruction": f"{AGENT_SYSTEM_PROMPT}\n\n{context}",
        "conversation_summary": summary,
        "summarized_count": summarized_count
    }

async def agent_node(state: AgentState, config):
    """
    The main reasoning node.
    Binds all tools by default. Excludes browser tasks for voice calls.
    """
    from tools import ALL_TOOLS
    from tools.browser import web_browser_task
    
    model_selection = state.model_selection or "gpt-cloud"
    
    # ── Tool Selection ──
    # Check if this is a voice call scenario
    metadata = config.get("metadata", {}) if isinstance(config, dict) else getattr(config, "metadata", {})
    is_voice = metadata.get("type") == "call"
    
    if is_voice:
        # Exclude browser task for voice to avoid accidental/unstable triggers
        selected_tools = [t for t in ALL_TOOLS if t.name != "web_browser_task"]
        logger.info(f"[AgentNode] Voice call detected. Binding {len(selected_tools)} tools (excluded browser).")
    else:
        selected_tools = list(ALL_TOOLS)
        logger.info(f"[AgentNode] Text chat detected. Binding all {len(selected_tools)} tools.")
    
    # Bind tools to the model
    llm = get_model(model_selection).bind_tools(selected_tools)
    
    # Construction of the dynamic prompt:
    # 1. Prepend the transient system instruction (with current time/summary)
    # 2. Add the truncated message history
    system_msg = SystemMessage(content=state.system_instruction)
    
    raw_messages = state.messages
    # Robust sliding window: Take the last N messages
    if len(raw_messages) > ctx_manager.max_messages:
        recent_history = raw_messages[-ctx_manager.max_messages:]
    else:
        recent_history = raw_messages

    # Combine: [SystemMessage] + [Recent History]
    prompt_messages = [system_msg] + recent_history
        
    # Run the model
    response = await llm.ainvoke(prompt_messages, config)
    return {"messages": [response]}


async def summarize_node(state: AgentState, config):
    """
    Post-processing node to update the conversation summary in the background.
    Replaces the 'aafter_agent' logic from middleware.
    """
    messages = state.messages
    
    # Voice fast-path: skip summarization for voice call threads (minimal latency)
    if hasattr(config, "metadata") or (isinstance(config, dict) and "metadata" in config):
        metadata = config.get("metadata", {}) if isinstance(config, dict) else getattr(config, "metadata", {})
        if metadata.get("type") == "call":
            return {}

    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
    
    # Logic from context_manager.py
    existing_summary, summarized_count = await get_thread_memory(thread_id)
    unsummarized = messages[summarized_count:]
    
    if ctx_manager.should_summarize(unsummarized):
        to_summarize, _ = ctx_manager.get_messages_to_summarize(unsummarized)
        if to_summarize:
            logger.info(f"Summarizing {len(to_summarize)} messages for thread {thread_id}")
            summary_prompt = ctx_manager.build_summary_prompt(to_summarize)
            llm = get_model("gpt-cloud")
            result = await llm.ainvoke([HumanMessage(content=summary_prompt)], config={"callbacks": []})
            new_summary = result.content.strip()
            
            combined_summary = f"{existing_summary}\n\n{new_summary}" if existing_summary else new_summary
            new_count = summarized_count + len(to_summarize)
            
            # --- Evaluation Logging (Restored from middleware logic) ---
            import os
            eval_msg = f"\n=== Summary Evaluation {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            eval_msg += f"EXISTING SUMMARY:\n{existing_summary if existing_summary else 'None'}\n\n"
            eval_msg += "MESSAGES SUMMARIZED:\n"
            for m in to_summarize:
                eval_msg += f"  [{getattr(m, 'type', 'N/A')}] {getattr(m, 'content', 'N/A')}\n"
            eval_msg += f"\nNEWLY GENERATED SUMMARY:\n{new_summary}\n\n"
            eval_msg += f"COMBINED SUMMARY:\n{combined_summary}\n"
            eval_msg += "="*60 + "\n"
            
            print(eval_msg)
            log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "summary_eval.log")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(eval_msg)
            # ----------------------------------------------------------

            await set_thread_memory(thread_id, combined_summary, new_count)
            # Update state so the next turn has the new summary
            return {"conversation_summary": combined_summary, "summarized_count": new_count}
            
    return {}
