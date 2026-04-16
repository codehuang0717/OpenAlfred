import os
import time
import json
import uuid
import jwt as pyjwt
import httpx
import logging
from langchain.tools import tool
from langchain.tools import ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

from config import config

logger = logging.getLogger("call-user-tool")

# 最新的云端 Outbound Trunk ID
OUTBOUND_TRUNK_ID = "ST_Bcj2LDXqL4J7"


def generate_sip_token():
    """根据 LiveKit 官方规范生成 SIP 专用 Token"""
    now = int(time.time())
    payload = {
        "iss": config.LIVEKIT_API_KEY,
        "iat": now,
        "nbf": now - 60,  # 提前一分钟防止服务器时间误差
        "exp": now + 3600,
        "sip": {"admin": True, "call": True},
    }
    return pyjwt.encode(payload, config.LIVEKIT_API_SECRET, algorithm="HS256")


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


def _room_name_to_thread_uuid(room_name: str) -> str:
    """Deterministically map a LiveKit room name to a valid UUID for LangGraph threads."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"call:{room_name}"))


async def dial_user(phone_number: str = "100", initial_speech: str = "", user_id: str = "default") -> str:
    """ Core logic to initiate an outbound SIP call. Returns status message. """
    room_name = f"outbound-{user_id}-{int(time.time())}"
    thread_uuid = _room_name_to_thread_uuid(room_name)
    
    thread_metadata = {
        "owner": user_id,
        "type": "call",
        "title": "语音外拨呼叫 (主动监督)",
        "room_name": room_name,
    }
    if initial_speech:
        thread_metadata["initial_speech"] = initial_speech

    # Step 1: Pre-create thread
    try:
        now = int(time.time())
        svc_jwt = pyjwt.encode(
            {"sub": user_id, "username": "supervisor", "service": True,
             "iat": now, "exp": now + 3600},
            config.JWT_SECRET, algorithm=config.JWT_ALGORITHM,
        )
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{config.LANGGRAPH_API_URL}/threads",
                headers={"Authorization": f"Bearer {svc_jwt}", "Content-Type": "application/json"},
                json={"thread_id": thread_uuid, "metadata": thread_metadata},
                timeout=5.0,
            )
    except Exception as e:
        logger.warning(f"Failed to pre-create call thread: {e}")

    # Step 2: Dial
    jwt_token = generate_sip_token()
    sip_headers = {"Authorization": f"Bearer {jwt_token}", "Content-Type": "application/json"}
    
    api_url = config.LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")
    url = f"{api_url.rstrip('/')}/twirp/livekit.SIP/CreateSIPParticipant"

    dial_data = {
        "sipTrunkId": OUTBOUND_TRUNK_ID,
        "sipCallTo": phone_number,
        "roomName": room_name,
        "participantIdentity": "agent_caller",
        "participantName": "Alfred Supervisor",
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=sip_headers, json=dial_data, timeout=10.0)
            if resp.status_code == 200:
                return f"呼叫已发起 ({phone_number})"
            return f"呼叫失败: {resp.text}"
    except Exception as e:
        return f"系统异常: {str(e)}"

@tool
async def make_outbound_call(
    runtime: ToolRuntime, phone_number: str = "100", initial_speech: str = ""
) -> Command:
    """
    主动拨打电话给用户。当你需要紧急提醒用户或者用户要求你打电话时使用。
    """
    user_id = _get_user_id(runtime)
    msg = await dial_user(phone_number, initial_speech, user_id)

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=msg,
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


call_tools = [make_outbound_call]
