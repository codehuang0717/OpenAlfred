import logging
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

logger = logging.getLogger("graph-nodes")
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
    Uses keyword-based dynamic tool selection to minimize token overhead.
    """
    from tools import TOOL_GROUPS, DEFAULT_TOOLS, ALL_TOOLS
    model_selection = state.model_selection or "gpt-cloud"
    
    # ── Dynamic Tool Selection (Strategy 2A) ──
    # Extract the last user message for keyword matching
    selected_tools = _select_tools_by_intent(state.messages)
    
    # Bind only the selected tools to the model
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


def _select_tools_by_intent(messages: list) -> list:
    """Select relevant tools based on keyword matching against the last user message.
    
    Falls back to DEFAULT_TOOLS (todos + reminders) if no keywords match.
    Always deduplicates to avoid binding the same tool twice.
    """
    from tools import TOOL_GROUPS, DEFAULT_TOOLS
    
    # Find the last user message
    user_msg = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            user_msg = content.lower()
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_msg = msg.get("content", "").lower()
            break
    
    if not user_msg:
        return DEFAULT_TOOLS
    
    selected = []
    matched_groups = []
    
    for group_name, group_config in TOOL_GROUPS.items():
        keywords = group_config["keywords"]
        if any(kw in user_msg for kw in keywords):
            selected.extend(group_config["tools"])
            matched_groups.append(group_name)
    
    # Fallback: no keyword matched → use default core tools
    if not selected:
        selected = list(DEFAULT_TOOLS)
        matched_groups = ["default(todos+reminders)"]
    
    # Deduplicate while preserving order
    seen = set()
    unique_tools = []
    for tool in selected:
        tool_id = id(tool)
        if tool_id not in seen:
            seen.add(tool_id)
            unique_tools.append(tool)
    
    logger.info(f"[ToolRouter] Matched groups: {matched_groups} → {len(unique_tools)} tools bound (vs {len(DEFAULT_TOOLS)} default)")
    return unique_tools

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
