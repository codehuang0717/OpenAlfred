"""Read bounded excerpts from this authenticated thread, never another user's log."""

from langchain.tools import tool, ToolRuntime
import tiktoken
from services.tool_observations import observed, fields
from core.config import config
from logic.context_payload import payload, reference, serialize
from utils.auth_utils import require_runtime_user_id, require_thread_id


@tool
def read_context_excerpt(ref: str, runtime: ToolRuntime, offset: int = 0, limit: int = 1200) -> str:
    """Read an original history/tool-result excerpt by its ctx:index:hash reference.

    Only the current authenticated thread is accessible. Offsets and limits are
    characters. Follow next_offset to page through omitted evidence; a preview
    alone does not establish that an operation succeeded.
    """
    user_id = require_runtime_user_id(runtime)
    require_thread_id(runtime.config)
    state = runtime.state
    state_user = state.get("user_id") if isinstance(state, dict) else state.user_id
    if state_user != user_id:
        raise PermissionError("History owner does not match authenticated user")
    messages = state["messages"] if isinstance(state, dict) else state.messages
    parts = ref.split(":")
    if len(parts) != 3 or parts[0] != "ctx" or not parts[1].isdigit():
        raise ValueError("Invalid context reference")
    index = int(parts[1])
    if index >= len(messages) or reference(index, messages[index]) != ref:
        raise ValueError("Context reference is stale or not from this thread")
    if offset < 0 or not 1 <= limit <= 1200:
        raise ValueError("offset >= 0 and 1 <= limit <= 1200 required")
    text = serialize(payload(messages[index]))
    if offset > len(text):
        raise ValueError("Offset exceeds original content")
    end = min(len(text), offset + limit)
    encoding = tiktoken.get_encoding("cl100k_base")
    while True:
        result = serialize({"ref": ref, "offset": offset, "total_chars": len(text),
                            "excerpt": text[offset:end], "next_offset": end if end < len(text) else None})
        if len(encoding.encode(result, disallowed_special=())) <= config.CONTEXT_TOOL_RESULT_TOKENS - 200:
            return observed(
                result,
                f"已读取会话原文第 {offset + 1}–{end} 字符"
                + ("，还有后续" if end < len(text) else "")
                if end > offset
                else "已到会话原文末尾，没有更多内容",
                outcome="empty"
                if end == offset
                else "truncated"
                if end < len(text)
                else "completed",
                details=fields(
                    原文片段=text[offset:end],
                    原文总字符=len(text),
                    下一页起点=end if end < len(text) else "已到末尾",
                ),
            )
        if end - offset <= 1:
            raise ValueError("Tool-result budget too small for a history excerpt")
        end = offset + (end - offset) // 2


context_tools = [read_context_excerpt]
