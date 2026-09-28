import uuid
import time
import httpx
import json
from core.config import config
from utils.logger import get_logger
from utils.latency import latency_tracker
from utils.auth_utils import mint_service_jwt
from logic.prompts import (
    CALL_INBOUND_PROMPT,
    CALL_OUTBOUND_PROMPT,
    build_outbound_motivation_prompt,
)

logger = get_logger("livekit-agent-client")

# Session-level metadata cache to pass info from entrypoint to call_agent
# In a real decoupled system, this might be passed as arguments or stored in Redis
session_metadata_cache = {}

async def _ensure_thread(session_id: str, user_id: str, title: str, call_type: str = "inbound") -> dict:
    """Create or get a LangGraph thread for this voice session."""
    thread_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"call:{session_id}"))
    result = {"thread_uuid": thread_uuid, "initial_speech": ""}
    
    token = mint_service_jwt(user_id)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient() as client:
            # Try to get the thread first
            resp = await client.get(
                f"{config.LANGGRAPH_API_URL}/threads/{thread_uuid}",
                headers=headers,
                timeout=5.0,
            )
            if resp.status_code == 200:
                thread_data = resp.json()
                metadata = thread_data.get("metadata", {})
                result["initial_speech"] = metadata.get("initial_speech", "")
                logger.info(f"Found existing call thread: {thread_uuid} (initial_speech={bool(result['initial_speech'])})")
                return result
            
            # Create the thread
            resp = await client.post(
                f"{config.LANGGRAPH_API_URL}/threads",
                headers=headers,
                json={
                    "thread_id": thread_uuid,
                    "metadata": {
                        "owner": user_id,
                        "type": "call",
                        "call_type": call_type,
                        "title": title,
                        "room_name": session_id,
                    },
                },
                timeout=5.0,
            )
            if resp.status_code in (200, 201):
                logger.info(f"Created LG thread for call: {thread_uuid} (room: {session_id})")
            else:
                logger.warning(f"Thread creation returned {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"Failed to ensure thread {thread_uuid}: {e}")
    return result

async def call_agent(session_id: str, text: str, user_id: str, model_selection: str = None):
    """Send a message to the unified Agent via LangGraph Server HTTP API."""
    latency_tracker.start("llm_total")
    
    session_data = session_metadata_cache.get(session_id, {})
    unique_session_id = session_data.get("unique_session_id", session_id)
    thread_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"call:{unique_session_id}"))
    
    call_type = session_data.get("call_type", "inbound")
    initial_speech = session_data.get("initial_speech", "")
    is_fresh = session_data.get("is_fresh", True)
    
    input_messages = []
    
    if is_fresh:
        if call_type == "outbound" and initial_speech:
            context = build_outbound_motivation_prompt(initial_speech)
            input_messages.append({"role": "system", "content": context})
            input_messages.append({"role": "assistant", "content": initial_speech})
        elif call_type == "outbound":
            input_messages.append({"role": "system", "content": CALL_OUTBOUND_PROMPT})
        else:
            input_messages.append({"role": "system", "content": CALL_INBOUND_PROMPT})
        
        session_data["is_fresh"] = False
    
    input_messages.append({"role": "user", "content": text})

    try:
        from core.database import get_setting
        global_model_selection = await get_setting("model_selection", "gpt-cloud")
    except Exception:
        global_model_selection = "gpt-cloud"

    model = model_selection or global_model_selection
    token = mint_service_jwt(user_id)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        latency_tracker.start("llm_graph_invoke")
        final_text = ""
        message_yielded = False
        llm_timed = False
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{config.LANGGRAPH_API_URL}/threads/{thread_uuid}/runs/stream",
                headers=headers,
                json={
                    "assistant_id": "agent",
                    "input": {
                        "messages": input_messages,
                        "model_selection": model,
                        "user_id": user_id,
                    },
                    "stream_mode": ["updates"],
                    "config": {
                        "configurable": {
                            "thread_id": thread_uuid,
                            "user_id": user_id,
                            "owner": user_id,
                            "channel": "voice",
                        },
                    },
                    "metadata": {
                        "owner": user_id,
                        "type": "call",
                        "call_type": call_type,
                        "room_name": session_id,
                        "initial_speech": initial_speech,
                    },
                },
                timeout=60.0,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"LG Server error ({response.status_code}): {error_text[:200]}")
                    yield "message", "抱歉，我暂时无法处理你的请求。"
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            for node_name, state_update in _iter_update_nodes(data):
                                if node_name != "agent" or not isinstance(state_update, dict):
                                    continue
                                messages = state_update.get("messages") or []
                                if not messages:
                                    continue
                                last_msg = messages[-1]
                                if last_msg.get("type") != "ai":
                                    continue

                                tool_calls = last_msg.get("tool_calls") or []
                                if tool_calls:
                                    for tc in tool_calls:
                                        name = tc.get("name") if isinstance(tc, dict) else None
                                        if name:
                                            yield "tool_call", name

                                content = last_msg.get("content", "")
                                if isinstance(content, list):
                                    content = "".join(
                                        b.get("text", "") if isinstance(b, dict) else str(b)
                                        for b in content
                                    )
                                content = (content or "").strip()
                                if content and not tool_calls and not message_yielded:
                                    final_text = content
                                    message_yielded = True
                                    if not llm_timed:
                                        latency_tracker.end("llm_graph_invoke")
                                        latency_tracker.end("llm_total")
                                        llm_timed = True
                                    logger.info(
                                        "[TIMING][Agent] EARLY_MESSAGE | chars=%d",
                                        len(final_text),
                                    )
                                    yield "message", final_text
                        except json.JSONDecodeError:
                            continue

        if not llm_timed:
            latency_tracker.end("llm_graph_invoke")
            latency_tracker.end("llm_total")

        if not message_yielded:
            yield "message", final_text or "收到"
            
    except Exception as e:
        logger.error(f"Voice Agent Error: {e}", exc_info=True)
        yield "message", "抱歉，我暂时无法处理你的请求。"


def _iter_update_nodes(data):
    """Yield (node_name, state_update) from a LangGraph updates SSE payload."""
    if not isinstance(data, dict):
        return
    payload = data.get("data") if data.get("event") == "updates" or data.get("type") == "updates" else data
    if not isinstance(payload, dict):
        return
    for node_name, state_update in payload.items():
        if node_name in {"event", "type", "data"}:
            continue
        yield node_name, state_update
