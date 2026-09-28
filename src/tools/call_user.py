import os
import asyncio
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

from core.config import config
from core.database import get_user_bark_url
from utils.auth_utils import (
    require_config_value,
    require_explicit_user_id,
    require_runtime_user_id,
)
from services.notification import notification_service

logger = logging.getLogger("call-user-tool")

# 最新的云端 Outbound Trunk ID
OUTBOUND_TRUNK_ID = config.LIVEKIT_SIP_TRUNK_ID


def generate_sip_token():
    """根据 LiveKit 官方规范生成 SIP 专用 Token"""
    api_key = require_config_value("LIVEKIT_API_KEY", config.LIVEKIT_API_KEY)
    api_secret = require_config_value("LIVEKIT_API_SECRET", config.LIVEKIT_API_SECRET)
    now = int(time.time())
    payload = {
        "iss": api_key,
        "sub": "openalfred-call-user",
        "iat": now,
        "nbf": now - 60,  # 提前一分钟防止服务器时间误差
        "exp": now + 3600,
        "video": {
            "roomAdmin": True,
            "roomCreate": True,
            "roomList": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
        },
        "sip": {"admin": True, "call": True},
    }
    return pyjwt.encode(payload, api_secret, algorithm="HS256")


def generate_room_admin_token(room_name: str):
    """Generate a room-scoped token for LiveKit RoomService polling."""
    api_key = require_config_value("LIVEKIT_API_KEY", config.LIVEKIT_API_KEY)
    api_secret = require_config_value("LIVEKIT_API_SECRET", config.LIVEKIT_API_SECRET)
    now = int(time.time())
    payload = {
        "iss": api_key,
        "sub": "openalfred-call-monitor",
        "iat": now,
        "nbf": now - 60,
        "exp": now + 3600,
        "video": {
            "room": room_name,
            "roomAdmin": True,
            "roomList": True,
            "canSubscribe": True,
        },
    }
    return pyjwt.encode(payload, api_secret, algorithm="HS256")


async def _get_user_id(runtime: ToolRuntime) -> str:
    """Extract a verified user_id from LangGraph request metadata."""
    return require_runtime_user_id(runtime)


def _room_name_to_thread_uuid(room_name: str) -> str:
    """Deterministically map a LiveKit room name to a valid UUID for LangGraph threads."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"call:{room_name}"))


def _fallback_title(reminder_id: str | None = None, supervisor_id: str | None = None) -> str:
    if reminder_id:
        return "OpenAlfred reminder call missed"
    if supervisor_id:
        return "OpenAlfred supervisor call missed"
    return "OpenAlfred call missed"


async def _send_call_fallback_bark(
    *,
    user_id: str,
    target_number: str,
    initial_speech: str,
    reason: str,
    reminder_id: str | None = None,
    supervisor_id: str | None = None,
) -> bool:
    """Send a Bark fallback when a SIP call cannot be connected."""
    user_id = require_explicit_user_id(user_id)
    bark_url = await get_user_bark_url(user_id)

    body_parts = [f"Voice call to {target_number} was not connected."]
    if initial_speech:
        body_parts.append(f"Message: {initial_speech}")
    body_parts.append(f"Reason: {reason}")

    success = await notification_service.send_bark_notification(
        body="\n".join(body_parts),
        title=_fallback_title(reminder_id, supervisor_id),
        subtitle=f"Target: {target_number}",
        level="timeSensitive",
        sound="birdsong",
        group="OpenAlfred-Calls",
        icon="https://cdn-icons-png.flaticon.com/512/597/597177.png",
        bark_url=bark_url,
    )
    logger.info(
        f"[call-fallback] bark={'success' if success else 'failed'} "
        f"user_id={user_id} target={target_number} reason={reason}"
    )
    return success


def _format_call_failure(reason: str, bark_sent: bool) -> str:
    fallback = "Bark fallback sent" if bark_sent else "Bark fallback failed"
    return f"Call failed: {reason}. {fallback}."


async def _wait_for_outbound_answer(
    client: httpx.AsyncClient,
    api_url: str,
    room_name: str,
    timeout_seconds: float = 20.0,
    interval_seconds: float = 1.0,
) -> tuple[bool, str]:
    """Poll LiveKit until the outbound SIP call is answered or fails.

    For outbound calls, `sip.callStatus=active` can mean the trunk answered.
    The real user-answer signal is `sip.callTag`.
    """
    url = f"{api_url.rstrip('/')}/twirp/livekit.RoomService/ListParticipants"
    headers = {
        "Authorization": f"Bearer {generate_room_admin_token(room_name)}",
        "Content-Type": "application/json",
    }
    deadline = time.monotonic() + timeout_seconds
    last_status = "waiting for answer"

    while time.monotonic() < deadline:
        try:
            resp = await client.post(
                url,
                headers=headers,
                json={"room": room_name},
                timeout=5.0,
            )
            if resp.status_code != 200:
                last_status = f"participant poll API {resp.status_code}: {resp.text}"
                await asyncio.sleep(interval_seconds)
                continue

            data = resp.json()
            participants = data.get("participants", [])
            sip_participants = [
                p for p in participants
                if (p.get("attributes") or {}).get("sip.callID")
                or p.get("identity") == "agent_caller"
                or p.get("identity", "").startswith("sip_")
            ]

            if not sip_participants:
                last_status = "SIP participant not present"
                await asyncio.sleep(interval_seconds)
                continue

            for participant in sip_participants:
                attrs = participant.get("attributes") or {}
                if attrs.get("sip.callTag"):
                    return True, "answered"
                status = attrs.get("sip.callStatus")
                if status:
                    last_status = f"sip.callStatus={status}"
                if status in {"hangup", "failed", "busy"}:
                    return False, last_status
        except Exception as e:
            last_status = f"participant poll error: {e}"

        await asyncio.sleep(interval_seconds)

    return False, f"no answer within {int(timeout_seconds)}s ({last_status})"


async def dial_user(
    *,
    user_id: str,
    phone_number: str = "",
    initial_speech: str = "",
    reminder_id: str = None,
    supervisor_id: str = None,
) -> str:
    """ Core logic to initiate an outbound SIP call. Returns status message. """
    user_id = require_explicit_user_id(user_id)
    trunk_id = require_config_value("LIVEKIT_SIP_TRUNK_ID", OUTBOUND_TRUNK_ID)
    if reminder_id:
        room_name = f"outbound-reminder-{reminder_id}-{user_id}"
    elif supervisor_id:
        room_name = f"outbound-supervisor-{supervisor_id}-{user_id}"
    else:
        room_name = f"outbound-{user_id}-{int(time.time())}"
        
    thread_uuid = _room_name_to_thread_uuid(room_name)
    
    thread_metadata = {
        "owner": user_id,
        "type": "call",
        "title": "语音外拨呼叫 (主动监督)" if not reminder_id else "语音提醒呼叫",
        "room_name": room_name,
        "initial_speech": initial_speech,
    }

    # Step 1: Pre-create thread
    try:
        now = int(time.time())
        jwt_secret = require_config_value("JWT_SECRET", config.JWT_SECRET)
        svc_jwt = pyjwt.encode(
            {"sub": user_id, "username": "supervisor", "service": True,
             "iat": now, "exp": now + 3600},
            jwt_secret, algorithm=config.JWT_ALGORITHM,
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

    # Auto-resolve the user's SIP extension if no explicit number given
    target_number = phone_number or await _resolve_phone_number(user_id)

    # Step 2: Dial
    jwt_token = generate_sip_token()
    sip_headers = {"Authorization": f"Bearer {jwt_token}", "Content-Type": "application/json"}

    api_url = config.LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")
    url = f"{api_url.rstrip('/')}/twirp/livekit.SIP/CreateSIPParticipant"

    dial_data = {
        "sip_trunk_id": trunk_id,
        "sipCallTo": target_number,
        "roomName": room_name,
        "participantIdentity": "agent_caller",
        "participantName": "Alfred Supervisor",
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=sip_headers, json=dial_data, timeout=10.0)
            if resp.status_code == 200:
                answered, answer_reason = await _wait_for_outbound_answer(
                    client=client,
                    api_url=api_url,
                    room_name=room_name,
                )
                if answered:
                    return f"Call answered ({target_number})"
                bark_sent = await _send_call_fallback_bark(
                    user_id=user_id,
                    target_number=target_number,
                    initial_speech=initial_speech,
                    reason=answer_reason,
                    reminder_id=reminder_id,
                    supervisor_id=supervisor_id,
                )
                return _format_call_failure(answer_reason, bark_sent)
            reason = f"LiveKit SIP API {resp.status_code}: {resp.text}"
            bark_sent = await _send_call_fallback_bark(
                user_id=user_id,
                target_number=target_number,
                initial_speech=initial_speech,
                reason=reason,
                reminder_id=reminder_id,
                supervisor_id=supervisor_id,
            )
            return _format_call_failure(reason, bark_sent)
    except Exception as e:
        reason = str(e)
        bark_sent = await _send_call_fallback_bark(
            user_id=user_id,
            target_number=target_number,
            initial_speech=initial_speech,
            reason=reason,
            reminder_id=reminder_id,
            supervisor_id=supervisor_id,
        )
        return _format_call_failure(reason, bark_sent)

async def _resolve_phone_number(user_id: str) -> str:
    """Resolve a dialable number for the given user.
    Raises when the user has no SIP extension; never dials a global target.
    """
    user_id = require_explicit_user_id(user_id)
    from core.database import get_user_by_id
    user = await get_user_by_id(user_id)
    if not user:
        raise ValueError(f"User not found: {user_id}")
    extension = str(user.get("sip_extension") or "").strip()
    if not extension:
        raise ValueError(f"User {user_id} has no SIP extension configured")
    logger.info(f"[_resolve_phone_number] user_id={user_id} -> sip_extension={extension}")
    return extension


@tool
async def make_outbound_call(
    runtime: ToolRuntime, phone_number: str = "", initial_speech: str = ""
) -> Command:
    """Make an outbound phone call to the user for urgent reminders or when requested.

    Args:
        phone_number: Target phone/extension. Leave empty to auto-resolve from user profile.
        initial_speech: What to say when the user picks up.
    """
    user_id = await _get_user_id(runtime)
    # Auto-resolve the user's SIP extension if no explicit number given
    target = phone_number or await _resolve_phone_number(user_id)
    logger.info(
        f"[make_outbound_call] user_id={user_id} phone={target} "
        f"speech={initial_speech[:50]}"
    )
    msg = await dial_user(user_id=user_id, phone_number=target, initial_speech=initial_speech)

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
