from langchain.tools import tool, ToolRuntime
from datetime import datetime
import logging

logger = logging.getLogger("memory-tools")


def _get_user_id(runtime: ToolRuntime) -> str:
    """Extract user_id from RunnableConfig populated by LangGraph Auth."""
    if hasattr(runtime, "config") and runtime.config:
        conf = runtime.config.get("configurable", {})
        auth_user = conf.get("langgraph_auth_user", {})
        if isinstance(auth_user, dict) and "identity" in auth_user:
            return auth_user["identity"]

        metadata = runtime.config.get("metadata", {})
        if "owner" in metadata:
            return metadata["owner"]

        if "thread_owner" in conf:
            return conf["thread_owner"]
    if hasattr(runtime, "state") and runtime.state:
        if isinstance(runtime.state, dict): return runtime.state.get("user_id", "default")
        return getattr(runtime.state, "user_id", "default")
    return "default"


CATEGORY_DESCRIPTIONS = {
    "profile": "基本信息 — 姓名、身份、重要日期、核心信息",
    "preferences": "偏好 — 喜欢/讨厌的事物、口味、兴趣",
    "relationship": "关系 — 关系状态、与他人的互动历史",
    "patterns": "行为模式 — 习惯、工作方式、作息规律",
}

VALID_CATEGORIES = list(CATEGORY_DESCRIPTIONS.keys())


@tool
def get_user_profile(runtime: ToolRuntime) -> str:
    """Read the user's L1 local memory profile (all categories).
    Returns the full user profile from local .md files."""
    from logic.memory_manager import memory_manager
    user_id = _get_user_id(runtime)
    memories = memory_manager.load_all_memories(user_id)
    if not memories:
        return "暂无用户画像信息。"
    header = "## 用户画像类别说明\n"
    for cat, desc in CATEGORY_DESCRIPTIONS.items():
        header += f"- **{cat}**: {desc}\n"
    return header + "\n" + memories


@tool
def update_user_memory(category: str, content: str, runtime: ToolRuntime) -> str:
    """Update the user's L1 local memory in a specific category.

    Args:
        category: One of 'profile', 'preferences', 'relationship', 'patterns'
        content: The new fact or information to append to this category.
    """
    from logic.memory_manager import memory_manager

    user_id = _get_user_id(runtime)

    if category not in VALID_CATEGORIES:
        return f"无效类别 '{category}'。可选类别: {', '.join(VALID_CATEGORIES)}"

    filename = {
        "profile": "profile.md",
        "preferences": "preferences.md",
        "relationship": "relationship.md",
        "patterns": "learned_patterns.md",
    }[category]

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        entry = f"- [{timestamp}] {content}"
        memory_manager.append_to_memory_file(user_id, filename, entry)
        logger.info(f"L1 memory updated: user={user_id}, category={category}")
        return f"已更新用户画像 [{category}]: {content}"
    except Exception as e:
        logger.error(f"Failed to update L1 memory: {e}")
        return f"更新失败: {str(e)}"


memTools = [get_user_profile, update_user_memory]
