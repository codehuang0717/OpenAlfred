# Runtime Monkey Patch to fix opentelemetry compatibility with livekit-agents
import opentelemetry.sdk._logs as sdk_logs
if not hasattr(sdk_logs, "LogData"):
    class LogData:
        def __init__(self, log_record, instrumentation_scope):
            self.log_record = log_record
            self.instrumentation_scope = instrumentation_scope
    sdk_logs.LogData = LogData

import asyncio
import json
import random
import glob
import logging
import io
import wave
import httpx
import collections
import os
import uuid
import time
import numpy as np
from datetime import datetime
from config import config

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, vad, stt, tts, llm
from livekit.plugins import silero
from livekit import rtc
from tts import get_tts_stream
from database import (
    AUDIO_CACHE_DIR,
    init_db,
)

from utils.logger import setup_logging, get_logger

# Initialize unified logging
setup_logging(log_file="livekit.log")
logger = get_logger("livekit-worker")

# Silence some noisy third-party loggers already handled by setup_logging, 
# but adding a few more specific to livekit.
logging.getLogger("aiosqlite").setLevel(logging.WARNING)

logger.info("Pre-loading Silero VAD model...")
GLOBAL_VAD = silero.VAD.load(min_silence_duration=1.0)
logger.info("Silero VAD model loaded successfully.")

# --- Custom Plugin Wrappers for VoiceAssistant ---

class SenseVoiceSTT(stt.STT):
    def __init__(self):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False))

    async def _transcribe(self, buffer: rtc.AudioFrame, language: str = None) -> stt.SpeechEvent:
        # Re-use existing transcription logic
        # For non-streaming STT, we buffer and send to HTTP
        audio_data = b"".join([f.data for f in buffer])
        text = await transcribe_audio(audio_data, buffer[0].sample_rate, buffer[0].num_channels)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=language or "zh")]
        )

class OpenAlfredTTS(tts.TTS):
    def __init__(self):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=True))

    def synthesize(self, text: str) -> tts.ChunkedStream:
        return OpenAlfredTTSStream(text)

class OpenAlfredTTSStream(tts.ChunkedStream):
    def __init__(self, text: str):
        super().__init__()
        self.text = text

    async def _run(self):
        async for audio_chunk in get_tts_stream(self.text, target_sample_rate=24000):
            # Convert raw bytes to AudioFrame
            # VoiceAssistant expects AudioFrames
            self._event_ch.send_nowait(tts.SynthesizedAudio(
                frame=rtc.AudioFrame(
                    data=audio_chunk,
                    sample_rate=24000,
                    num_channels=1,
                    samples_per_channel=len(audio_chunk) // 2
                )
            ))
        self._event_ch.close()

# --- LangGraph LLM Wrapper ---

class LangGraphLLM(llm.LLM):
    def __init__(self, user_id: str):
        super().__init__()
        self.user_id = user_id

    def chat(self, chat_ctx: llm.ChatContext, fnet: llm.FunctionNetwork = None) -> llm.ChatStream:
        return LangGraphChatStream(chat_ctx, self.user_id)

class LangGraphChatStream(llm.ChatStream):
    def __init__(self, chat_ctx: llm.ChatContext, user_id: str):
        super().__init__()
        self.chat_ctx = chat_ctx
        self.user_id = user_id

    async def _run(self):
        # Extract the last user message
        last_msg = self.chat_ctx.messages[-1]
        text = last_msg.content
        session_id = "livekit-session" # Simplified for now
        
        # Use existing call_agent logic but adapt to ChatStream events
        async for event_type, payload in call_agent(session_id, text, self.user_id):
            if event_type == "message":
                self._event_ch.send_nowait(llm.ChatChunk(
                    choices=[llm.Choice(delta=llm.ChoiceDelta(content=payload, role="assistant"))]
                ))
            elif event_type == "tool_call":
                # VoiceAssistant doesn't handle our custom transition audio directly via tool_calls here,
                # but we can trigger it or let the assistant handle standard tool calls.
                logger.info(f"LangGraph requested tool: {payload}")
                
        self._event_ch.close()

# --- End of LangGraph LLM ---

db_initialized = False
# Session-level metadata cache to pass info from entrypoint to call_agent
session_metadata_cache = {}



def _mint_service_jwt(user_id: str) -> str:
    """Mint a short-lived JWT for the voice worker to call LangGraph Server.
    
    Uses the same JWT_SECRET as the main auth system so the LG auth handler
    can validate it. The 'sub' claim is the actual user_id so thread
    ownership is correctly attributed.
    """
    import jwt as pyjwt
    now = int(time.time())
    payload = {
        "sub": user_id,
        "username": "voice-worker",
        "service": True,
        "iat": now,
        "exp": now + 3600,
    }
    return pyjwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


class LatencyTracker:
    def __init__(self):
        self.timings = {}

    def start(self, name):
        self.timings[name] = {"start": time.perf_counter()}

    def end(self, name):
        if name in self.timings:
            self.timings[name]["end"] = time.perf_counter()
            self.timings[name]["duration"] = (
                self.timings[name]["end"] - self.timings[name]["start"]
            )

    def get(self, name):
        return self.timings.get(name, {}).get("duration", 0)

    def log_summary(self):
        logger.info("=== Latency Summary ===")
        for name, data in self.timings.items():
            logger.info(f"  {name}: {data.get('duration', 0) * 1000:.0f}ms")
        logger.info("=======================")

    def reset(self):
        self.timings = {}


latency_tracker = LatencyTracker()


async def _ensure_thread(session_id: str, user_id: str, title: str, call_type: str = "inbound") -> dict:
    """Create or get a LangGraph thread for this voice session.

    
    Returns a dict with:
      - thread_uuid: the UUID used as LangGraph thread_id
      - initial_speech: pre-generated greeting from thread metadata (if any)
    """
    thread_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"call:{session_id}"))
    result = {"thread_uuid": thread_uuid, "initial_speech": ""}
    
    token = _mint_service_jwt(user_id)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient() as client:
            # Try to get the thread first — may have been pre-created by call_user.py
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
            
            # Create the thread (for inbound calls or if pre-creation failed)
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


async def call_agent(session_id: str, text: str, user_id: str, model_selection: str = None) -> str:
    """Send a message to the unified Agent via LangGraph Server HTTP API."""
    latency_tracker.start("llm_total")
    
    # Retrieve session-specific metadata (initial_speech, etc.)
    session_data = session_metadata_cache.get(session_id, {})
    unique_session_id = session_data.get("unique_session_id", session_id)
    
    # Map session to UUID
    thread_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"call:{unique_session_id}"))
    
    call_type = session_data.get("call_type", "inbound")

    initial_speech = session_data.get("initial_speech", "")
    is_fresh = session_data.get("is_fresh", True)
    
    # Construct input messages
    input_messages = []
    
    # Atomic Push: Inject context and greeting history IF it's the first turn
    if is_fresh:
        if call_type == "outbound" and initial_speech:
            context = (
                f"[系统指示] 你主动拨打了此电话。拨号动机: \"{initial_speech}\"。"
                "请基于此动机与用户对话。使用简洁、自然的口语回复，严禁使用Markdown或-或空格等特殊符号，对于日期时间等信息，请使用中文口语方式表达，比如14:27转换成下午两点二十七分。"
            )
            input_messages.append({"role": "system", "content": context})
            input_messages.append({"role": "assistant", "content": initial_speech})
        elif call_type == "outbound":
            input_messages.append({"role": "system", "content": "[系统指示] 你主动呼叫了用户。请以友好的方式开始对话。对于日期时间等信息，请使用中文口语方式表达，比如14:27转换成下午两点二十七分，严禁使用Markdown或-或空格等特殊符号。"})
        else:
            input_messages.append({"role": "system", "content": "[系统指示] 用户呼入了你的热线。请以友好的方式接待。对于日期时间等信息，请使用中文口语方式表达，比如14:27转换成下午两点二十七分，严禁使用Markdown或-或空格等特殊符号。"})
        
        # Mark as no longer fresh
        session_data["is_fresh"] = False
    
    # Add current user speech
    input_messages.append({"role": "user", "content": text})

    try:
        from database import get_setting
        global_model_selection = await get_setting("model_selection", "gpt-cloud")
    except Exception:
        global_model_selection = "gpt-cloud"


        
    model = model_selection or global_model_selection

    token = _mint_service_jwt(user_id)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        latency_tracker.start("llm_graph_invoke")
        
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
                    },
                    "stream_mode": ["updates"],
                    "config": {
                        "configurable": {
                            "thread_id": thread_uuid,
                            "user_id": user_id,
                            "owner": user_id
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

                final_text = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            # logger.info(f"[call_agent Debug] {line[6:]}")
                            data = json.loads(line[6:])
                            if isinstance(data, dict):
                                for node_name, state_update in data.items():
                                    if node_name == "agent" and "messages" in state_update:
                                        # Check the newly added messages
                                        messages = state_update["messages"]
                                        if messages:
                                            last_msg = messages[-1]
                                            if last_msg.get("type") == "ai":
                                                tool_calls = last_msg.get("tool_calls", [])
                                                if tool_calls:
                                                    for tc in tool_calls:
                                                        name = tc.get("name")
                                                        if name:
                                                            yield "tool_call", name
                                                
                                                # Check if there is text content
                                                content = last_msg.get("content", "")
                                                if isinstance(content, list):
                                                    content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
                                                if content and content.strip():
                                                    final_text = content.strip()
                        except json.JSONDecodeError:
                            continue

        latency_tracker.end("llm_graph_invoke")
        latency_tracker.end("llm_total")
        
        if final_text:
            yield "message", final_text
        else:
            yield "message", "收到"
            
    except Exception as e:
        logger.error(f"Voice Agent Error: {e}", exc_info=True)
        yield "message", "抱歉，我暂时无法处理你的请求。"


async def transcribe_audio(audio_data: bytes, sample_rate: int, channels: int) -> str:
    latency_tracker.start("stt_total")
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    wav_bytes = wav_io.getvalue()
    try:
        latency_tracker.start("stt_http_request")
        async with httpx.AsyncClient() as client:
            files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
            resp = await client.post(
                "http://127.0.0.1:8000/extract_text", files=files, timeout=15.0
            )
            resp.raise_for_status()
            latency_tracker.end("stt_http_request")
            result = resp.json()
            latency_tracker.end("stt_total")
            return result.get("results", "")
    except Exception as e:
        logger.error(f"SenseVoice Error: {e}")
        return ""


# Removed old text_to_speech function to use streaming src/tts.py


        await lkapi.aclose()
        await asyncio.sleep(1.0)
    except Exception as e:
        logger.error(f"Definitive hangup logic failed: {e}")

    logger.info("Disconnecting agent from room and closing job...")
    await room.disconnect()



async def entrypoint(ctx: JobContext):
    global db_initialized
    if not db_initialized:
        await init_db()
        db_initialized = True

    # Elegant Fix: Use SUBSCRIBE_NONE to take control of the subscription lifecycle.
    # This avoids the race condition where the SDK tries to subscribe before it has
    # fully indexed the participant (KeyError: 'sip_100').
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)

    room = ctx.room
    should_exit = asyncio.Event()

    call_type = "outbound" if room.name.startswith("outbound-") else "inbound"
    
    # For inbound calls, we MUST Use a unique session ID to avoid grouping separate calls into one thread
    # room.name for inbound is often just 'inbound'. room.sid or ctx.job.id is unique.
    unique_session_id = room.name
    if call_type == "inbound":
        unique_session_id = f"{room.name}-{ctx.job.id}"
    
    user_id = "default"


    if call_type == "outbound":
        # Format 1: outbound-{user_id}-{timestamp}
        # Format 2: outbound-reminder-{reminder_id}-{user_id}
        # Format 3: outbound-supervisor-{supervisor_id}-{user_id}
        if room.name.startswith("outbound-reminder-"):
            # room.name looks like outbound-reminder-UUID-USERID
            # "outbound-reminder-" is 18 chars. UUID is 36 chars.
            # Then we have a '-' which is 1 char. The rest is user_id.
            user_id = room.name[18 + 36 + 1:]
            if not user_id:
                user_id = "default"
        elif room.name.startswith("outbound-supervisor-"):
            # outbound-supervisor-sup_1234567890-userid
            # parts[0]=outbound, parts[1]=supervisor, parts[2]=sup_id, parts[3:]=user_id
            parts = room.name.split("-")
            if len(parts) >= 4:
                user_id = "-".join(parts[3:])
            else:
                user_id = "default"
        else:
            parts = room.name.split("-")
            if len(parts) >= 3:
                # Format 1: Reconstruct the user_id (which might have hyphens) from the middle parts
                user_id = "-".join(parts[1:-1])
    else:
        # TODO: 暂时先绑定到FlyingPig账号的名下，下一轮修改为动态映射
        try:
            from database import get_user_by_username
            fp_user = await get_user_by_username("FlyingPig")
            if fp_user:
                user_id = fp_user["id"]
        except Exception as e:
            logger.error(f"Failed to find inbound user FlyingPig: {e}")

    # ── CRITICAL: Register event handlers IMMEDIATELY ──
    answered_event = asyncio.Event() 
    greeting_played = False
    greeting_lock = asyncio.Lock()

    async def trigger_greeting():
        nonlocal greeting_played
        async with greeting_lock:
            if greeting_played:
                return
            greeting_played = True
            logger.info(f"Triggering greeting for {call_type} call (session: {unique_session_id})")
            if call_type == "outbound":

                await asyncio.sleep(0.5)
            await play_greeting(room, initial_speech, should_exit, user_id, call_type)

    # ── Create / retrieve LangGraph thread and read initial_speech ──
    call_title = "语音外拨呼叫" if call_type == "outbound" else "语音呼入接待"
    thread_info = await _ensure_thread(unique_session_id, user_id, call_title, call_type)
    thread_uuid = thread_info["thread_uuid"]

    initial_speech = thread_info.get("initial_speech", "")

    # Cache metadata for call_agent to use in Run Metadata
    session_metadata_cache[room.name] = {
        "call_type": call_type,
        "initial_speech": initial_speech,
        "is_fresh": True,
        "unique_session_id": unique_session_id,
    }



    async def subscribe_to_audio(p: rtc.RemoteParticipant):
        """Manually subscribe to audio tracks for a participant."""
        logger.info(f"Scanning tracks for participant {p.identity}...")
        for pub in p.track_publications.values():
            if pub.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"Manually subscribing to audio track {pub.sid} from {p.identity}")
                pub.set_subscribed(True)

    @room.on("participant_connected")
    def on_participant_connected(p: rtc.RemoteParticipant):
        logger.info(f"Participant connected: {p.identity}. Triggering manual subscription.")
        asyncio.create_task(subscribe_to_audio(p))

    @room.on("participant_attributes_changed")

    def on_attributes_changed(changed_attributes: dict, participant: rtc.RemoteParticipant):
        if "sip.callStatus" in changed_attributes:
            new_status = participant.attributes.get("sip.callStatus")
            logger.info(f"SIP Participant {participant.identity} status: {new_status}")
            if new_status == "active":
                logger.info(f"SIP Call confirmed ACTIVE for {participant.identity}")
                answered_event.set()
                asyncio.create_task(trigger_greeting())

    @room.on("participant_disconnected")
    def on_participant_disconnected(p):
        if not room.remote_participants:
            logger.info("All remote participants disconnected. Exiting job...")
            should_exit.set()

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if isinstance(track, rtc.AudioTrack):
            logger.info(f"Subscribed to audio track {track.sid} from {participant.identity}")
            asyncio.create_task(process_audio_safe(track, room, should_exit, user_id))
            # Some SIP providers don't update callStatus but start audio.
            # We'll use track subscription as a fallback, but only if it's the right identity
            # or if we are desperate. For now, let's trust callStatus FIRST.
            if call_type == "inbound":
                answered_event.set()
                asyncio.create_task(trigger_greeting())

    # ── Scan for already existing participants/tracks ──
    # Now that handlers are registered, we manually trigger subscription for anyone
    # who managed to join between room.connect and this point.
    for participant in room.remote_participants.values():
        asyncio.create_task(subscribe_to_audio(participant))
        
        status = participant.attributes.get("sip.callStatus")
        if status == "active":
            logger.info(f"Found already active SIP participant: {participant.identity}")
            answered_event.set()
            asyncio.create_task(trigger_greeting())


    logger.info(f"Call session ready: {unique_session_id} ({call_type})")

    from livekit.agents.voice_assistant import VoiceAssistant

    # Initialize our high-level Assistant
    assistant = VoiceAssistant(
        vad=GLOBAL_VAD,
        stt=SenseVoiceSTT(),
        llm=LangGraphLLM(user_id=user_id),
        tts=OpenAlfredTTS(),
        chat_ctx=llm.ChatContext().append(role="system", text="你是一个友好的语音助手 Alfred。使用简洁自然的口语回复。"),
    )

    # Start the assistant
    assistant.start(ctx.room)

    if call_type == "outbound":
        logger.info("Outbound call: Waiting for user to answer (physical pick up)...")
        try:
            await asyncio.wait_for(answered_event.wait(), timeout=30)
            logger.info("User picked up! Starting communication...")
            # Greet the user
            if initial_speech:
                await assistant.say(initial_speech, allow_interruptions=True)
        except asyncio.TimeoutError:
            logger.warning("No answer detected within 30s. Terminating.")
            should_exit.set()
            return
    else:
        # Inbound greeting
        if initial_speech:
            await assistant.say(initial_speech, allow_interruptions=True)
        else:
            await assistant.say("你好！我是 Alfred，请问有什么我可以帮你的？", allow_interruptions=True)

    try:
        await should_exit.wait()
    finally:
        # Clear session cache on exit
        session_metadata_cache.pop(room.name, None)

    logger.info("Termination signal received. Forcing SIP hangup...")

    logger.info(f"Call session ended: {room.name}")

    try:
        from livekit import api

        lkapi = api.LiveKitAPI()

        logger.info(
            f"Attempting to remove SIP participants or delete room to force hangup: {room.name}"
        )

        found_sip = False
        for p in room.remote_participants.values():
            if p.identity.startswith("sip_"):
                logger.info(f"Removing SIP participant: {p.identity}")
                await lkapi.room.remove_participant(
                    api.RoomParticipantIdentity(room=room.name, identity=p.identity)
                )
                found_sip = True

        if not found_sip:
            logger.info("No specific SIP identity found, deleting entire room...")
            await lkapi.room.delete_room(api.DeleteRoomRequest(room=room.name))
        else:
            await asyncio.sleep(0.5)
            await lkapi.room.delete_room(api.DeleteRoomRequest(room=room.name))

        await lkapi.aclose()
        await asyncio.sleep(1.0)
    except Exception as e:
        logger.error(f"Definitive hangup logic failed: {e}")

    logger.info("Disconnecting agent from room and closing job...")
    await room.disconnect()


    logger.info(f"Call session ended: {room.name}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
