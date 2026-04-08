import os
import time
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


@tool
async def make_outbound_call(
    runtime: ToolRuntime, phone_number: str = "100"
) -> Command:
    """
    主动拨打电话给用户。当你需要紧急提醒用户或者用户要求你打电话时使用。

    Args:
        phone_number: 拨打的目标号码，默认为 "100"（分机号）。
    """
    jwt_token = generate_sip_token()

    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }

    # 修正后的 LiveKit SIP 官方 API 路径
    api_url = config.LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")
    if api_url.endswith("/"):
        api_url = api_url[:-1]

    # 官方标准的 Twirp 路径
    url = f"{api_url}/twirp/livekit.SIP/CreateSIPParticipant"

    # 负载必须使用驼峰命名以匹配 Protobuf 定义
    dial_data = {
        "sipTrunkId": OUTBOUND_TRUNK_ID,
        "sipCallTo": phone_number,
        "roomName": f"outbound-{int(time.time())}",
        "participantIdentity": f"agent_caller",
        "participantName": "AI Assistant",
    }

    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"SIP API Request: {url}")
            resp = await client.post(url, headers=headers, json=dial_data, timeout=10.0)

            if resp.status_code == 200:
                logger.info("Outbound call successful")
                msg = f"呼叫已发起，请注意接听分机 {phone_number}。"
            else:
                logger.error(f"LiveKit SIP Error ({resp.status_code}): {resp.text}")
                msg = f"发起呼叫失败 (API返回: {resp.status_code})。"

    except Exception as e:
        logger.error(f"Call Tool API Exception: {e}")
        msg = f"呼叫系统请求异常: {str(e)}"

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
