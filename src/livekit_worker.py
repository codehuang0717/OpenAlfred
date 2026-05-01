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

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, vad
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


async def play_tts(room: rtc.Room, text: str, should_exit_event: asyncio.Event, interrupt_event: asyncio.Event):
    latency_tracker.start("tts_generate")
    source = rtc.AudioSource(24000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("tts", source)
    publication = await room.local_participant.publish_track(track)
    try:
        latency_tracker.start("tts_first_chunk")
        first_chunk_found = False
        
        jitter_buffer_bytes = b""
        jitter_threshold_bytes = int(24000 * (config.TTS_JITTER_BUFFER_MS / 1000) * 2) 
        playback_started = False
        total_samples_pushed = 0
        start_time = 0

        async for audio_chunk in get_tts_stream(text, target_sample_rate=24000):
            if interrupt_event.is_set():
                logger.info("[VoiceInterrupt] TTS audio playback aborted due to interrupt during stream.")
                break

            if not first_chunk_found:
                latency_tracker.end("tts_first_chunk")
                latency_tracker.end("tts_generate")
                first_chunk_found = True
            
            jitter_buffer_bytes += audio_chunk
            
            if not playback_started and len(jitter_buffer_bytes) >= jitter_threshold_bytes:
                latency_tracker.start("tts_playback")
                latency_tracker.start("tts_audio_stream")
                playback_started = True
                start_time = time.time()
            
            if playback_started:
                audio_np = np.frombuffer(jitter_buffer_bytes, dtype=np.int16)
                jitter_buffer_bytes = b"" 
                
                chunk_size = 480 
                for i in range(0, len(audio_np), chunk_size):
                    if interrupt_event.is_set():
                        break
                    chunk = audio_np[i : i + chunk_size]
                    if len(chunk) > 0:
                        frame = rtc.AudioFrame(data=chunk.tobytes(), sample_rate=24000, num_channels=1, samples_per_channel=len(chunk))
                        await source.capture_frame(frame)
                        total_samples_pushed += len(chunk)
                await asyncio.sleep(0.01)

        # Final cleanup for remaining small chunks
        if jitter_buffer_bytes and not interrupt_event.is_set():
            if not playback_started:
                playback_started = True
                start_time = time.time()
            audio_np = np.frombuffer(jitter_buffer_bytes, dtype=np.int16)
            for i in range(0, len(audio_np), 480):
                if interrupt_event.is_set():
                    break
                chunk = audio_np[i : i + 480]
                frame = rtc.AudioFrame(data=chunk.tobytes(), sample_rate=24000, num_channels=1, samples_per_channel=len(chunk))
                await source.capture_frame(frame)
                total_samples_pushed += len(chunk)

        latency_tracker.end("tts_audio_stream")
        
        # Wait for audio to drain
        if total_samples_pushed > 0 and not interrupt_event.is_set():
            total_duration = total_samples_pushed / 24000
            elapsed = time.time() - start_time
            remaining = total_duration - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining + 0.3) 
        
        latency_tracker.end("tts_playback")
    except asyncio.CancelledError:
        logger.info("[VoiceInterrupt] TTS audio task cancelled gracefully.")
    finally:
        await room.local_participant.unpublish_track(publication.sid)
        if "[TERMINATE]" in text:
            logger.info("Termination signal detected in Agent response. Hanging up...")
            should_exit_event.set()


async def play_greeting(room: rtc.Room, initial_speech: str = "", should_exit_event: asyncio.Event = None, user_id: str = "default", call_type: str = "inbound"):
    """Play the greeting. Context injection is now handled dynamically by the Graph/Middleware."""
    if initial_speech:
        logger.info(f"Playing initial speech: {initial_speech}")
        dummy_interrupt = asyncio.Event()
        await play_tts(room, initial_speech, should_exit_event, dummy_interrupt)
        return


    # Updated greeting path to assets/
    wav_path = str(config.ASSETS_DIR / "greeting.wav")
    if "outbound-reminder-" in room.name:
        start_idx = room.name.find("outbound-reminder-") + len("outbound-reminder-")
        reminder_id = room.name[start_idx : start_idx + 36]
        specific_wav = os.path.join(AUDIO_CACHE_DIR, f"reminder_{reminder_id}.wav")
        if os.path.exists(specific_wav):
            wav_path = specific_wav
            logger.info(f"Using specific reminder audio: {wav_path}")

    if not os.path.exists(wav_path):
        return

    source = rtc.AudioSource(24000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("greeting", source)
    publication = await room.local_participant.publish_track(track)

    try:
        with wave.open(wav_path, "rb") as wf:
            framerate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())

        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        if framerate != 24000:
            audio_np = np.interp(
                np.linspace(
                    0, len(audio_np) - 1, int(len(audio_np) * (24000 / framerate))
                ),
                np.arange(len(audio_np)),
                audio_np,
            ).astype(np.int16)

        silence = np.zeros(24000 + 12000, dtype=np.int16)
        chunk_size = 480
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i : i + chunk_size]
            if len(chunk) > 0:
                frame = rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=24000,
                    num_channels=1,
                    samples_per_channel=len(chunk),
                )
                await source.capture_frame(frame)

        for i in range(0, len(audio_np), chunk_size):
            chunk = audio_np[i : i + chunk_size]
            if len(chunk) > 0:
                frame = rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=24000,
                    num_channels=1,
                    samples_per_channel=len(chunk),
                )
                await source.capture_frame(frame)
        await asyncio.sleep(len(audio_np) / 24000 + 0.5)
    except Exception as e:
        logger.error(f"Error in play_greeting: {e}")
    finally:
        await room.local_participant.unpublish_track(publication.sid)



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
        # Format: outbound-{user_id}-{timestamp}
        parts = room.name.split("-")
        if len(parts) >= 3:
            # Reconstruct the UUID (which has hyphens)
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

    if call_type == "outbound":
        logger.info("Outbound call: Waiting for user to answer (physical pick up)...")
        try:
            # Wait for SIP status or track (with preference for status)
            await asyncio.wait_for(answered_event.wait(), timeout=30)
            logger.info("User picked up! Starting communication...")
        except asyncio.TimeoutError:
            logger.warning("No answer detected within 30s. Terminating.")
            should_exit.set()
            return
    else:
        # Inbound can trigger greeting immediately
        asyncio.create_task(trigger_greeting())

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


async def play_transition_audio(room: rtc.Room, interrupt_event: asyncio.Event, tool_name: str = None):
    """Play a transition audio file based on the tool called, or a generic fallback."""
    transition_dir = config.ASSETS_DIR / "transitions"
    if not transition_dir.exists():
        logger.info("[Transition] Transition directory missing, skipping.")
        return
        
    wav_path = None
    if tool_name:
        specific_path = transition_dir / f"{tool_name}.wav"
        if specific_path.exists():
            wav_path = specific_path
            
    if not wav_path:
        generic_files = ["checking.wav", "hmm.wav", "thinking.wav", "wait.wav", "working.wav"]
        available_generics = [f for f in generic_files if (transition_dir / f).exists()]
        if available_generics:
            wav_path = transition_dir / random.choice(available_generics)
        else:
            logger.info("[Transition] No valid generic transition audio files found, skipping.")
            return
            
    logger.info(f"[Transition] Selected transition audio: {os.path.basename(wav_path)}")
    
    source = rtc.AudioSource(24000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("transition", source)
    publication = await room.local_participant.publish_track(track)
    
    try:
        with wave.open(str(wav_path), "rb") as wf:
            framerate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
            
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        if framerate != 24000:
            audio_np = np.interp(
                np.linspace(0, len(audio_np) - 1, int(len(audio_np) * (24000 / framerate))),
                np.arange(len(audio_np)),
                audio_np,
            ).astype(np.int16)

        chunk_size = 480
        for i in range(0, len(audio_np), chunk_size):
            if interrupt_event.is_set():
                logger.info("[VoiceInterrupt] Transition audio playback aborted due to interrupt.")
                break
                
            chunk = audio_np[i : i + chunk_size]
            if len(chunk) > 0:
                frame = rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=24000,
                    num_channels=1,
                    samples_per_channel=len(chunk),
                )
                await source.capture_frame(frame)
            await asyncio.sleep(0.01)
            
        if not interrupt_event.is_set():
            await asyncio.sleep(len(audio_np) / 24000 + 0.1)
            
    except asyncio.CancelledError:
        logger.info("[VoiceInterrupt] Transition audio task cancelled.")
    except Exception as e:
        logger.error(f"[Transition] Error playing transition audio: {e}")
    finally:
        await room.local_participant.unpublish_track(publication.sid)


async def process_audio_safe(
    track: rtc.AudioTrack, room: rtc.Room, should_exit: asyncio.Event, user_id: str
):
    audio_stream = rtc.AudioStream(track)
    vad_stream = GLOBAL_VAD.stream()
    all_frames = []
    pre_buffer = collections.deque(maxlen=300)
    is_speaking = False
    session_id = room.name

    interrupt_event = asyncio.Event()
    current_tts_task = None
    current_transition_task = None

    async def vad_logic():
        nonlocal is_speaking, all_frames, current_tts_task, current_transition_task
        async for event in vad_stream:
            if event.type == vad.VADEventType.START_OF_SPEECH:
                latency_tracker.start("vad_silence")
                is_speaking = True
                
                logger.info("[VoiceInterrupt] Detected START_OF_SPEECH. Interrupting AI...")
                interrupt_event.set()
                if current_tts_task and not current_tts_task.done():
                    current_tts_task.cancel()
                if current_transition_task and not current_transition_task.done():
                    current_transition_task.cancel()
                    
                all_frames = list(pre_buffer)
                pre_buffer.clear()
            elif event.type == vad.VADEventType.END_OF_SPEECH:
                latency_tracker.end("vad_silence")
                latency_tracker.start("vad_speech")
                latency_tracker.start("end_to_end")
                is_speaking = False
                interrupt_event.clear()
                if all_frames:
                    audio_data = b"".join([f.data for f in all_frames])
                    latency_tracker.end("vad_speech")
                    text = await transcribe_audio(audio_data, 48000, 1)
                    if text:
                        logger.info(f"========> [User Said]: {text}")
                        
                        # Option A: No immediate random transition audio. Wait for tool call.
                        
                        latency_tracker.start("agent_response")
                        
                        final_resp_text = ""
                        async for event_type, payload in call_agent(session_id, text, user_id):
                            if interrupt_event.is_set():
                                break
                                
                            if event_type == "tool_call":
                                logger.info(f"[Agent Tool Call]: {payload}")
                                if current_transition_task and not current_transition_task.done():
                                    current_transition_task.cancel()
                                current_transition_task = asyncio.create_task(
                                    play_transition_audio(room, interrupt_event, tool_name=payload)
                                )
                            elif event_type == "message":
                                final_resp_text = payload
                                
                        latency_tracker.end("agent_response")
                        latency_tracker.end("end_to_end")
                        logger.info(f"========> [Agent Response]: {final_resp_text}")
                        
                        if not interrupt_event.is_set() and final_resp_text:
                            # User requested: transition words should NOT be interrupted by the final TTS
                            if current_transition_task and not current_transition_task.done():
                                logger.info("Waiting for transition audio to finish before playing TTS...")
                                await current_transition_task
                                
                            if not interrupt_event.is_set():
                                current_tts_task = asyncio.create_task(
                                    play_tts(room, final_resp_text, should_exit, interrupt_event)
                                )

                        logger.info("========> [Latency Summary] =========")
                        logger.info(
                            f"  VAD沉默检测: {latency_tracker.get('vad_silence') * 1000:.0f}ms"
                        )
                        logger.info(
                            f"  VAD语音检测: {latency_tracker.get('vad_speech') * 1000:.0f}ms"
                        )
                        logger.info(
                            f"  STT语音识别: {latency_tracker.get('stt_total') * 1000:.0f}ms"
                        )
                        logger.info(
                            f"    - HTTP请求: {latency_tracker.get('stt_http_request') * 1000:.0f}ms"
                        )
                        logger.info(
                            f"  LLM总延迟: {latency_tracker.get('llm_total') * 1000:.0f}ms"
                        )
                        logger.info(
                            f"    - 加载历史: {latency_tracker.get('llm_load_history') * 1000:.0f}ms"
                        )
                        logger.info(
                            f"    - 图推理: {latency_tracker.get('llm_graph_invoke') * 1000:.0f}ms"
                        )
                        logger.info(
                            f"  TTS首包延迟: {latency_tracker.get('tts_first_chunk') * 1000:.0f}ms"
                        )
                        logger.info(
                            f"  TTS生成(全): {latency_tracker.get('tts_generate') * 1000:.0f}ms"
                        )
                        logger.info(
                            f"  TTS播放: {latency_tracker.get('tts_playback') * 1000:.0f}ms"
                        )
                        logger.info(
                            f"    - 音频流: {latency_tracker.get('tts_audio_stream') * 1000:.0f}ms"
                        )
                        total = latency_tracker.get("end_to_end")
                        logger.info(f"  端到端延迟: {total * 1000:.0f}ms")
                        logger.info("=======================================")
                        latency_tracker.reset()
                all_frames = []

    vad_task = asyncio.create_task(vad_logic())
    try:
        async for frame_event in audio_stream:
            vad_stream.push_frame(frame_event.frame)
            if is_speaking:
                all_frames.append(frame_event.frame)
            else:
                pre_buffer.append(frame_event.frame)
    finally:
        vad_task.cancel()
        await vad_stream.aclose()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
