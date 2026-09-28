from utils.logger import get_logger
from datetime import datetime
from zoneinfo import ZoneInfo
import time

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from logic.schema import AgentState
from services.llm import get_model, get_bound_model, output_limit_kwargs
from logic.prompts import AGENT_SYSTEM_PROMPT, KNOWLEDGE_EXTRACTION_PROMPT
from logic.context_manager import ContextManager
from logic.context_metrics import model_usage_metrics
from logic.context_payload import serialize
from logic.memory_manager import memory_manager
from core.config import config as app_config
from services.weather import format_weather_prompt_context, get_weather_summary
from utils.auth_utils import require_thread_id, require_user_id

logger = get_logger("graph-nodes")
ctx_manager = ContextManager()

def _is_voice_channel(config) -> bool:
    """Voice runs set configurable.channel=voice; metadata.type=call is a fallback."""
    if isinstance(config, dict):
        conf = config.get("configurable", {}) or {}
        if conf.get("channel") == "voice":
            return True
        metadata = config.get("metadata", {}) or {}
        return metadata.get("type") == "call"
    metadata = getattr(config, "metadata", {}) or {}
    return metadata.get("type") == "call"


async def load_context_node(state: AgentState, config):
    """
    Node to inject dynamic context (time, summary, L1 memories) into the message list.
    """
    # 1. Get current time
    now_uk = datetime.now(ZoneInfo(app_config.TIMEZONE))
    time_str = now_uk.strftime("%Y-%m-%d %H:%M:%S")
    weekday = now_uk.strftime("%A")

    # Legacy append-only summaries are not injected. The budget planner loads
    # a versioned, owner-scoped rolling summary immediately before each model call.
    require_thread_id(config)

    # 3. Resolve user_id from authenticated request metadata. Fail closed when
    # ownership is missing or conflicting; cached graph state is not authority.
    user_id = require_user_id(config)
    logger.debug(
        f"[load_context] user_id={user_id} state_uid={state.user_id}"
    )

    l1_memories = memory_manager.build_injection_text(user_id)
    weather_context = ""
    try:
        weather_context = format_weather_prompt_context(
            await get_weather_summary(user_id=user_id)
        )
    except Exception as e:
        logger.debug("[load_context] weather context skipped: %s", e)

    context = f"[系统信息]\nCurrent Time: {time_str} ({weekday}). Timezone: {app_config.TIMEZONE}."
    if weather_context:
        context += f"\n\n{weather_context}"
    return {
        "system_instruction": f"{AGENT_SYSTEM_PROMPT}\n\n{l1_memories}" if l1_memories else AGENT_SYSTEM_PROMPT,
        "runtime_context": context,
        "user_id": user_id,
    }

async def agent_node(state: AgentState, config):
    """
    The main reasoning node.
    Binds all tools by default. Excludes browser tasks for voice calls.
    """
    from tools import ALL_TOOLS
    if state.context_error:
        return {"messages": [AIMessage(content=f"上下文准备失败：{state.context_error}。已停止后续主模型调用，请检查配置或拆分输入后重试。") ]}
    
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
    selected_tools = selected_context_tools(config)
    
    # Bind tools to the model (cached by model + tool set)
    tool_names = frozenset(t.name for t in selected_tools)
    llm = get_bound_model(model_selection, tool_names, ALL_TOOLS)
    
    if not state.prepared_messages:
        raise RuntimeError("Budgeted context preparation must run before agent_node")
    prompt_messages = state.prepared_messages
        
    # Run the model
    started = time.monotonic()
    try:
        response = await llm.ainvoke(prompt_messages, config, **output_limit_kwargs(model_selection, ctx_manager.output_reserve))
    except Exception as e:
        friendly = _map_llm_error(e)
        logger.error(f"[AgentNode] LLM error (model={model_selection}): {friendly}")
        error_msg = AIMessage(content=f"❌ {friendly}")
        return {"messages": [error_msg]}
    usage = model_usage_metrics(response, model_selection, round((time.monotonic() - started) * 1000))
    logger.info("context usage: %s", serialize(usage))
    return {"messages": [response], "context_metrics": {**state.context_metrics, "model_usage": usage}}


def selected_context_tools(config) -> list:
    from tools import ALL_TOOLS
    excluded = {"make_outbound_call", "get_recent_emails", "read_email", "get_email_accounts"} if _is_voice_channel(config) else set()
    return [tool for tool in ALL_TOOLS if tool.name not in excluded]


async def prepare_context_node(state: AgentState, config) -> dict:
    """Run before EVERY invocation, including tools loops and voice turns."""
    user_id = require_user_id(config)
    thread_id = require_thread_id(config)
    try:
        prepared = await ctx_manager.prepare(
            state.messages, state.system_instruction, selected_context_tools(config), user_id, thread_id,
            runtime_context=state.runtime_context,
        )
        return {"prepared_messages": prepared.messages, "conversation_summary": prepared.summary,
                "summarized_count": prepared.covered_count, "context_metrics": prepared.metrics,
                "context_error": ""}
    except Exception as exc:
        logger.exception("context preparation failed; no main-model request sent")
        return {"prepared_messages": [], "context_error": f"{type(exc).__name__}: {exc}",
                "context_metrics": {"event": "context.failed", "error_type": type(exc).__name__}}


async def extract_knowledge_node(state: AgentState, config):
    """Conservatively extract evidence-backed user facts, never assistant wording."""
    import json
    from logic.memory_policy import CATEGORY_FILES, user_sources, validate_candidate
    from logic.schema import KnowledgeExtractionResult

    counter = state.extraction_counter + 1
    if counter < app_config.EXTRACTION_INTERVAL:
        return {"extraction_counter": counter}
    sources = user_sources(state.messages, state.extracted_msg_count)
    complete = {"extraction_counter": 0, "extracted_msg_count": len(state.messages)}
    if not sources:
        return complete
    user_id = require_user_id(config)
    try:
        existing = memory_manager.load_all_memories(user_id)
        payload = json.dumps({
            "existing_memories": existing,
            "user_messages": [{"message_index": index, "text": text} for index, text in sources.items()],
        }, ensure_ascii=False)
        model = get_model(app_config.MEMORY_MODEL_SELECTION)
        result = await model.with_structured_output(KnowledgeExtractionResult).ainvoke(
            [SystemMessage(content=KNOWLEDGE_EXTRACTION_PROMPT), HumanMessage(content=payload)],
            config={"callbacks": []},
        )
        if not isinstance(result, KnowledgeExtractionResult):
            raise TypeError("Memory extraction returned an invalid structured result")
        added = rejected = duplicates = 0
        for fact in result.facts:
            try:
                validate_candidate(fact, sources)
                written = memory_manager.append_to_memory_file(
                    user_id, CATEGORY_FILES[fact.category],
                    f"- [{datetime.now().strftime('%Y-%m-%d')}] {fact.fact.strip()}",
                )
                added += int(written)
                duplicates += int(not written)
            except ValueError as exc:
                rejected += 1
                logger.warning("[extract_knowledge] rejected candidate: %s", exc)
        logger.info("[extract_knowledge] added=%d duplicate=%d rejected=%d", added, duplicates, rejected)
    except Exception:
        logger.exception("[extract_knowledge] extraction failed; cursor preserved")
        return {"extraction_counter": 0}
    return complete
