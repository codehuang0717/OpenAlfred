from utils.logger import get_logger
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from logic.schema import AgentState
from services.llm import get_model, get_bound_model
from logic.prompts import AGENT_SYSTEM_PROMPT, KNOWLEDGE_EXTRACTION_PROMPT
from core.database import get_thread_memory, set_thread_memory
from logic.context_manager import ContextManager
from logic.memory_manager import memory_manager
from core.config import config as app_config
from services.weather import format_weather_prompt_context, get_weather_summary

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
    weather_context = ""
    try:
        weather_context = format_weather_prompt_context(
            await get_weather_summary(user_id=user_id)
        )
    except Exception as e:
        logger.debug("[load_context] weather context skipped: %s", e)

    context = f"[系统信息]\nCurrent Time: {time_str} ({weekday}). Timezone: Europe/London."
    if weather_context:
        context += f"\n\n{weather_context}"
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
    
    # model_selection priority: config.configurable > state > default
    conf = config.get("configurable", {}) if isinstance(config, dict) else {}
    model_selection = conf.get("model_selection") or state.model_selection or "gpt-cloud"

    # ── Error mapping helper ──
    def _map_llm_error(e: Exception) -> str:
        """Map LLM provider exceptions to user-facing Chinese messages."""
        name = type(e).__name__
        msg = str(e)

        # OpenAI context overflow
        if "context_length_exceeded" in msg or "context overflow" in msg.lower():
            return f"上下文过长，超出模型限制。请缩短对话或开启新会话。"

        # Generic context / token limit clues
        if "reduce the length" in msg.lower() or "limit is" in msg.lower():
            return f"上下文超过模型 token 上限，请精简消息或切换支持更长上下文的模型。"

        # Gemini model not found
        if "not found" in msg.lower() and ("model" in msg.lower() or "gemini" in msg.lower()):
            return f"模型不存在：{msg.split(chr(10))[0][:120]}"

        # Auth errors (401/403)
        if "401" in msg or "403" in msg or "unauthorized" in msg.lower() or "permission" in msg.lower():
            return f"API 认证失败，请检查对应模型的 API Key 是否正确配置。"

        # Rate limit
        if "429" in msg or "rate limit" in msg.lower() or "quota" in msg.lower():
            return f"API 调用频率超限，请稍后重试。"

        # Bad request — pass through the API's own message
        if "400" in msg or "bad request" in msg.lower():
            brief = msg.split("\n")[0] if "\n" in msg else msg
            return f"请求参数错误：{brief[:200]}"

        # Fallback: include exception type + first line
        brief = msg.split("\n")[0] if "\n" in msg else msg
        return f"{name}：{brief[:200]}"

    # ── Tool Selection ──
    # Check if this is a voice call scenario
    metadata = config.get("metadata", {}) if isinstance(config, dict) else getattr(config, "metadata", {})
    is_voice = metadata.get("type") == "call"
    
    if is_voice:
        # Slim tool set for lower latency: exclude browser, outbound call, and UI-oriented email tools
        voice_exclude = {"make_outbound_call", "get_recent_emails", "read_email", "get_email_accounts"}
        selected_tools = [t for t in ALL_TOOLS if t.name not in voice_exclude]
        logger.info(f"[AgentNode] Voice call detected. Binding {len(selected_tools)}/{len(ALL_TOOLS)} tools (excluded: {voice_exclude}).")
    else:
        selected_tools = list(ALL_TOOLS)
        logger.info(f"[AgentNode] Text chat detected. Binding all {len(selected_tools)} tools.")
    
    # Bind tools to the model (cached by model + tool set)
    tool_names = frozenset(t.name for t in selected_tools)
    llm = get_bound_model(model_selection, tool_names, ALL_TOOLS)
    
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
    try:
        response = await llm.ainvoke(prompt_messages, config)
    except Exception as e:
        friendly = _map_llm_error(e)
        logger.error(f"[AgentNode] LLM error (model={model_selection}): {friendly}")
        error_msg = AIMessage(content=f"❌ {friendly}")
        return {"messages": [error_msg]}
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
    Every N turns, ask the LLM to review the conversation and update memory files.
    The LLM sees ALL existing memories first, so it won't duplicate.
    """
    interval = app_config.EXTRACTION_INTERVAL
    counter = getattr(state, "extraction_counter", 0) + 1

    if counter < interval:
        logger.debug(
            "[extract_knowledge] SKIP (counter=%d < interval=%d)",
            counter, interval,
        )
        return {"extraction_counter": counter}

    messages = state.messages
    if len(messages) < 2:
        logger.debug("[extract_knowledge] SKIP (messages=%d < 2)", len(messages))
        return {"extraction_counter": 0}

    prev_extracted = getattr(state, "extracted_msg_count", 0)
    unextracted = messages[prev_extracted:]
    if len(unextracted) < 2:
        logger.debug(
            "[extract_knowledge] SKIP (unextracted=%d < 2, prev=%d)",
            len(unextracted), prev_extracted,
        )
        return {"extraction_counter": 0, "extracted_msg_count": len(messages)}

    # Build conversation transcript
    turns: list[str] = []
    i = 0
    while i < len(unextracted):
        m = unextracted[i]
        if isinstance(m, HumanMessage) and m.content:
            user_text = str(m.content).strip()
            if len(user_text) >= app_config.EXTRACTION_MIN_MSG_LENGTH:
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
        logger.debug(
            "[extract_knowledge] SKIP (no turns built | unextracted=%d)",
            len(unextracted),
        )
        return {"extraction_counter": 0, "extracted_msg_count": len(messages)}

    resolved = _get_user_id_from_config(config)
    user_id = resolved if resolved != "default" else (state.user_id or "default")
    conversation_text = "\n---\n".join(turns)

    # Load existing memories so the LLM can avoid duplicates
    existing = memory_manager.load_all_memories(user_id)
    existing_block = f"\n[已有记忆]\n{existing}\n" if existing else ""

    logger.debug(
        "[extract_knowledge] ENTER extraction | user_id=%s | turns=%d | "
        "existing_mem_len=%d | conv_len=%d",
        user_id, len(turns), len(existing), len(conversation_text),
    )

    try:
        from utils.structured_output import structured_invoke, StructuredOutputError
        from logic.schema import KnowledgeExtractionResult

        llm = get_model("gpt-cloud")
        prompt = KNOWLEDGE_EXTRACTION_PROMPT.format(
            existing_memories=existing_block,
            conversation=conversation_text,
        )
        logger.debug(
            "[extract_knowledge] Calling structured_invoke | schema=KnowledgeExtractionResult",
        )
        result = await structured_invoke(
            llm,
            [HumanMessage(content=prompt)],
            KnowledgeExtractionResult,
            max_retries=2,
            config={"callbacks": []},
        )
        logger.debug(
            "[extract_knowledge] structured_invoke returned | facts=%d",
            len(result.facts),
        )

        if not result.facts:
            logger.debug("[extract_knowledge] No facts extracted (empty result)")
            return {"extraction_counter": 0, "extracted_msg_count": len(messages)}

        # Map category to filename, append each fact
        cat_to_file = {
            "profile": "profile.md",
            "preferences": "preferences.md",
            "relationship": "relationship.md",
            "patterns": "learned_patterns.md",
        }
        timestamp = datetime.now().strftime("%Y-%m-%d")
        added = 0
        skipped_dup = 0
        for f in result.facts:
            fname = cat_to_file.get(f.category, "profile.md")
            line = f"- [{timestamp}] {f.fact}"
            if line.strip().lower() in existing.lower():
                skipped_dup += 1
                logger.debug(
                    "[extract_knowledge] SKIP duplicate | cat=%s | fact=%.80s",
                    f.category, f.fact,
                )
                continue
            memory_manager.append_to_memory_file(user_id, fname, line)
            added += 1
            logger.debug(
                "[extract_knowledge] WROTE | cat=%s | file=%s | fact=%.80s",
                f.category, fname, f.fact,
            )

        logger.info(
            "[extract_knowledge] DONE | user_id=%s | found=%d | added=%d | "
            "skipped_dup=%d",
            user_id, len(result.facts), added, skipped_dup,
        )

    except StructuredOutputError as e:
        logger.warning(
            "[extract_knowledge] StructuredOutputError | user_id=%s | err=%s",
            user_id, e,
        )
    except Exception as e:
        logger.warning(
            "[extract_knowledge] Unexpected error | user_id=%s | err=%s | type=%s",
            user_id, e, type(e).__name__,
        )

    return {"extraction_counter": 0, "extracted_msg_count": len(messages)}
