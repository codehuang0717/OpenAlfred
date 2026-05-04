from utils.logger import get_logger
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal
import json
import re
import traceback

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode

from logic.schema import AgentState
from services.llm import get_model
from logic.prompts import AGENT_SYSTEM_PROMPT, KNOWLEDGE_EXTRACTION_PROMPT
from core.database import get_thread_memory, set_thread_memory
from logic.context_manager import ContextManager
from logic.memory_manager import memory_manager
from core.config import config as app_config

logger = get_logger("graph-nodes")
ctx_manager = ContextManager()

def _get_user_id_from_config(config) -> str:
    """Extract user_id from LangGraph config."""
    if isinstance(config, dict):
        conf = config.get("configurable", {})
        auth_user = conf.get("langgraph_auth_user", {})
        if isinstance(auth_user, dict) and "identity" in auth_user:
            return auth_user["identity"]
        metadata = config.get("metadata", {})
        if "owner" in metadata:
            return metadata["owner"]
        # Voice agent passes "owner" in configurable, not "thread_owner"
        if "owner" in conf:
            return conf["owner"]
        if "thread_owner" in conf:
            return conf["thread_owner"]
        if "user_id" in conf:
            return conf["user_id"]
    return "default"


async def load_context_node(state: AgentState, config):
    """
    Node to inject dynamic context (time, summary, L1 memories) into the message list.
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

    # 3. Resolve user_id: always prefer JWT/auth identity over cached state
    resolved = _get_user_id_from_config(config)
    if resolved != "default":
        user_id = resolved
    else:
        # Fallback: use state value (e.g. voice calls where auth may not fire)
        user_id = state.user_id or "default"
    logger.debug(
        f"[load_context] user_id={user_id} "
        f"(resolved={resolved} state_uid={state.user_id})"
    )
    l1_memories = memory_manager.build_injection_text(user_id)

    context = f"[系统信息]\nCurrent Time: {time_str} ({weekday}). Timezone: Europe/London."
    if l1_memories:
        context += f"\n\n{l1_memories}"
    if summary:
        context += f"\n\n[对话历史摘要]\n{summary}"

    return {
        "system_instruction": f"{AGENT_SYSTEM_PROMPT}\n\n{context}",
        "conversation_summary": summary,
        "summarized_count": summarized_count,
        "user_id": user_id,
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
        # Slim tool set for lower latency: exclude browser, outbound call, and UI-oriented email tools
        voice_exclude = {"web_browser_task", "make_outbound_call", "get_recent_emails", "read_email", "get_email_accounts"}
        selected_tools = [t for t in ALL_TOOLS if t.name not in voice_exclude]
        logger.info(f"[AgentNode] Voice call detected. Binding {len(selected_tools)}/{len(ALL_TOOLS)} tools (excluded: {voice_exclude}).")
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
            
            # Log summary evaluation via logger (non-blocking)
            logger.info(
                f"Summary evaluation: {len(to_summarize)} msgs summarized | "
                f"New summary length: {len(new_summary)} chars | "
                f"Combined length: {len(combined_summary)} chars"
            )
            # ----------------------------------------------------------

            await set_thread_memory(thread_id, combined_summary, new_count)
            # Update state so the next turn has the new summary
            return {"conversation_summary": combined_summary, "summarized_count": new_count}

    return {}


async def extract_knowledge_node(state: AgentState, config):
    """
    Post-processing node to extract user facts/preferences from ALL unextracted
    conversation turns and merge into L1 local memory files.

    Key design:
    - Accumulates messages since last extraction (no information is lost)
    - Uses a turn counter to throttle frequency (default: every 3 turns)
    - Feeds all unextracted turns into a single extraction call (batch extraction)
    """
    interval = app_config.EXTRACTION_INTERVAL
    counter = getattr(state, "extraction_counter", 0) + 1

    # Trigger gate: accumulate turns, only extract every N turns
    if counter < interval:
        return {"extraction_counter": counter}

    messages = state.messages
    if len(messages) < 2:
        return {"extraction_counter": 0}

    # Collect all user+assistant pairs since last extraction
    prev_extracted = getattr(state, "extracted_msg_count", 0)
    unextracted = messages[prev_extracted:]

    # Build conversation transcript from unextracted messages
    turns: list[str] = []
    i = 0
    while i < len(unextracted):
        m = unextracted[i]
        if isinstance(m, HumanMessage) and m.content:
            user_text = str(m.content).strip()
            if len(user_text) >= app_config.EXTRACTION_MIN_MSG_LENGTH:
                # Find the next assistant reply (text only, not tool calls)
                assistant_text = ""
                for j in range(i + 1, len(unextracted)):
                    am = unextracted[j]
                    if isinstance(am, AIMessage) and am.content and not getattr(am, "tool_calls", None):
                        assistant_text = str(am.content).strip()
                        break
                if assistant_text:
                    turns.append(f"用户：{user_text}\n助手：{assistant_text}")
            i += 1
        else:
            i += 1

    if not turns:
        return {"extraction_counter": 0, "extracted_msg_count": len(messages)}

    resolved = _get_user_id_from_config(config)
    user_id = resolved if resolved != "default" else (state.user_id or "default")
    conversation_text = "\n---\n".join(turns)

    try:
        llm = get_model("gpt-cloud")
        prompt = KNOWLEDGE_EXTRACTION_PROMPT.format(conversation=conversation_text)
        result = await llm.ainvoke(
            [HumanMessage(content=prompt)],
            config={"callbacks": []}
        )
        raw_content = result.content
        if not raw_content:
            return {"extraction_counter": 0, "extracted_msg_count": len(messages)}
        if not isinstance(raw_content, str):
            raw_content = str(raw_content)
        text = raw_content.strip()

        if "```" in text:
            text = re.sub(r"```\w*", "", text).replace("```", "").strip()

        facts = json.loads(text)
        if not isinstance(facts, list) or len(facts) == 0:
            return {"extraction_counter": 0, "extracted_msg_count": len(messages)}

        valid_facts = []
        for f in facts:
            if isinstance(f, dict) and "fact" in f:
                valid_facts.append({
                    "category": f.get("category", "profile"),
                    "fact": f["fact"],
                    "importance": f.get("importance", "medium"),
                })

        if not valid_facts:
            return {"extraction_counter": 0, "extracted_msg_count": len(messages)}

        added = await memory_manager.extract_and_merge(user_id, valid_facts, llm)
        logger.info(f"Knowledge extraction (counter={counter}, {len(turns)} turns): "
                     f"{len(valid_facts)} facts found, added to {added} for '{user_id}'")

        if app_config.MEM0_ENABLED:
            try:
                from mem0 import MemoryClient
                mem0 = MemoryClient(api_key=app_config.MEM0_API_KEY)
                for f in valid_facts:
                    mem0.add(
                        messages=[{"role": "user", "content": f['fact']}],
                        user_id=user_id
                    )
                logger.debug(f"L2 Mem0 updated with {len(valid_facts)} facts")
            except Exception as e:
                logger.warning(f"L2 Mem0 write failed (non-blocking): {e}")

    except json.JSONDecodeError:
        logger.debug(f"Failed to parse knowledge extraction JSON. Raw: {text[:200]}")
    except Exception as e:
        logger.warning(f"Knowledge extraction failed: {e}\n{traceback.format_exc()}")

    return {"extraction_counter": 0, "extracted_msg_count": len(messages)}
