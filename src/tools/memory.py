from langchain.tools import tool, ToolRuntime
from datetime import datetime
import logging
import json
from langchain_core.messages import HumanMessage, SystemMessage
from logic.memory_policy import CATEGORY_DESCRIPTIONS, CATEGORY_FILES, MEMORY_POLICY, user_sources, validate_candidate
from logic.schema import KnowledgeExtractionResult
from core.config import config
from services.llm import get_model
from utils.auth_utils import require_runtime_user_id
from services.tool_observations import observed, failed, fields, action, CATEGORIES

logger = logging.getLogger("memory-tools")


def _get_user_id(runtime: ToolRuntime) -> str:
    """Extract a verified user_id from LangGraph request metadata."""
    return require_runtime_user_id(runtime)


VALID_CATEGORIES = list(CATEGORY_DESCRIPTIONS.keys())


@tool
def get_user_profile(runtime: ToolRuntime) -> str:
    """Read the user's core L1 memory profile.

    Returns only the always-relevant profile and preferences memories. Use
    get_user_memory_category for relationship or behavior pattern memories.
    """
    from logic.memory_manager import memory_manager
    user_id = _get_user_id(runtime)
    memories = memory_manager.load_memories(user_id, ["profile.md", "preferences.md"])
    if not memories:
        return observed("暂无用户画像信息。", "暂无个人资料和偏好记忆", outcome="empty")
    header = "## 用户画像类别说明\n"
    for cat in ("profile", "preferences"):
        desc = CATEGORY_DESCRIPTIONS[cat]
        header += f"- **{cat}**: {desc}\n"
    return observed(
        header + "\n" + memories,
        "已读取个人资料与偏好",
        details=fields(范围="个人资料、个人偏好", 记忆内容=memories),
        actions=[action("settings", "管理记忆", "profile")],
    )


@tool
def get_user_memory_category(category: str, runtime: ToolRuntime) -> str:
    """Read one user memory category on demand.

    Use this when relationship history or behavioral patterns are relevant to
    the user's request. Category must be one of: profile, preferences,
    relationship, patterns.
    """
    from logic.memory_manager import memory_manager

    user_id = _get_user_id(runtime)
    if category not in VALID_CATEGORIES:
        return failed(
            f"无效类别 '{category}'。可选类别: {', '.join(VALID_CATEGORIES)}",
            "记忆类别无效",
        )

    filename = CATEGORY_FILES[category]

    memories = memory_manager.load_memories(user_id, [filename])
    if not memories:
        return observed(
            f"暂无 {category} 类记忆。",
            f"暂无{CATEGORIES[category]}记忆",
            outcome="empty",
        )
    return observed(
        f"## {CATEGORY_DESCRIPTIONS[category]}\n\n{memories}",
        f"已读取{CATEGORIES[category]}",
        details=fields(记忆内容=memories),
        actions=[action("settings", "管理记忆", "profile")],
    )


@tool
async def update_user_memory(category: str, content: str, runtime: ToolRuntime) -> str:
    """Save a durable fact ONLY when the current user explicitly asks to remember it.

    Not for one-off tasks, transient formatting, inferred preferences, screen
    observations, or assistant suggestions. A shared evidence reviewer must approve
    the fact against the latest user message. Existing conflicting facts require
    clarification; this tool does not overwrite or delete memories.

    Args:
        category: One of 'profile', 'preferences', 'relationship', 'patterns'
        content: The new fact or information to append to this category.
    """
    from logic.memory_manager import memory_manager

    user_id = _get_user_id(runtime)

    if category not in VALID_CATEGORIES:
        return failed(
            f"无效类别 '{category}'。可选类别: {', '.join(VALID_CATEGORIES)}",
            "记忆类别无效",
        )

    filename = CATEGORY_FILES[category]

    try:
        state = runtime.state
        messages = state.get("messages", []) if isinstance(state, dict) else state.messages
        sources = user_sources(messages)
        if not sources:
            return observed(
                "未保存：没有可核验的本轮用户原话",
                "未保存记忆：缺少本轮用户证据",
                outcome="blocked",
            )
        latest = max(sources)
        sources = {latest: sources[latest]}
        result = await get_model(config.MEMORY_MODEL_SELECTION).with_structured_output(KnowledgeExtractionResult).ainvoke(
            [SystemMessage(content=MEMORY_POLICY + "\n额外要求：这是显式记忆工具。用户本轮必须明确要求长期记住此事；否则返回空 facts。只审核 proposed_fact，批准时 fact 与 category 必须原样返回，不得改写成其他事实。"),
             HumanMessage(content=json.dumps({
                 "user_messages": [{"message_index": latest, "text": sources[latest]}],
                 "existing_memories": memory_manager.load_all_memories(user_id),
                 "proposed_fact": {"category": category, "fact": content},
             }, ensure_ascii=False))], config={"callbacks": []},
        )
        if not isinstance(result, KnowledgeExtractionResult):
            raise TypeError("Invalid memory review result")
        if len(result.facts) != 1 or result.facts[0].category != category or result.facts[0].fact != content:
            return observed(
                "未保存：不符合长期记忆规则、已有相同信息或需要先确认冲突",
                "未保存：规则审核未批准，需确认重复或冲突",
                outcome="blocked",
                details=fields(候选记忆=content),
            )
        validate_candidate(result.facts[0], sources)
        timestamp = datetime.now().strftime("%Y-%m-%d")
        entry = f"- [{timestamp}] {content}"
        if not memory_manager.append_to_memory_file(user_id, filename, entry):
            return observed(
                "已有相同记忆，未重复添加",
                "已有相同记忆，未重复添加",
                outcome="exists",
                details=fields(候选记忆=content),
            )
        logger.info(f"L1 memory updated: user={user_id}, category={category}")
        return observed(
            f"已更新用户画像 [{category}]: {content}",
            f"已追加{CATEGORIES[category]}记忆",
            details=fields(
                新增记忆=content,
                用户原话=sources[latest],
                写入方式="追加；没有覆盖或删除已有记忆",
            ),
            actions=[action("settings", "管理记忆", "profile")],
        )
    except Exception as e:
        logger.error(f"Failed to update L1 memory: {e}")
        return failed(f"更新失败: {str(e)}", "保存记忆失败，未确认写入成功")


memTools = [get_user_profile, get_user_memory_category, update_user_memory]
