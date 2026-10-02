"""Voice-only request to end a SIP call after the spoken reply."""

from langchain.tools import ToolRuntime, tool

from logic.voice_control import END_CALL_APPROVED
from services.tool_observations import observed, fields
from utils.auth_utils import require_runtime_user_id


@tool
async def request_end_call(runtime: ToolRuntime) -> str:
    """Request to end this phone call when the caller clearly says goodbye or asks to hang up.

    Do not use for thanks, pauses, or the end of a topic. After this tool returns,
    say one brief, natural goodbye. The call stays open briefly for a follow-up.
    """
    config = (runtime.config or {}).get("configurable", {})
    if config.get("channel") != "voice" or config.get("call_type") not in {"inbound", "outbound"}:
        raise PermissionError("Only SIP voice calls can request a hangup")
    require_runtime_user_id(runtime)
    return observed(
        END_CALL_APPROVED,
        "已请求结束当前通话，等待实际挂断",
        outcome="queued",
        details=fields(说明="告别播报后会保留短暂追问时间；此时未确认挂断"),
    )


end_call_tools = [request_end_call]
