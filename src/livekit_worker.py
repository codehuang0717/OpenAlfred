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
import logging
import io
import wave
import httpx
import collections
import os
import uuid
import time
from datetime import datetime
from config import config

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, vad
from livekit.plugins import silero
from livekit import rtc
from agent_graph import voice_graph
from tts import get_tts_stream
from database import (
    AUDIO_CACHE_DIR,
    init_db,
    create_call_session,
    end_call_session,
    add_call_message,
    get_call_session_history,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("livekit-worker")
logging.getLogger("aiosqlite").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

db_initialized = False


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


async def call_agent(session_id: str, text: str, model_selection: str = None) -> str:
    latency_tracker.start("llm_total")
    try:
        from database import get_setting
        global_model_selection = await get_setting("model_selection", "gpt-cloud")
    except Exception:
        global_model_selection = "gpt-cloud"
        
    model = model_selection or global_model_selection

    try:
        # 1. Persist user message
        await add_call_message(session_id, "user", text)

        # 2. Load conversation history (last 10 turns)
        latency_tracker.start("llm_load_history")
        history = await get_call_session_history(session_id, max_turns=10)
        latency_tracker.end("llm_load_history")

        # 3. Build messages for the graph (history already has role/content dicts)
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        # Add current user message if not already in history
        if not messages or messages[-1].get("content") != text:
            messages.append({"role": "user", "content": text})

        # 4. Invoke voice graph directly (no HTTP!)
        latency_tracker.start("llm_graph_invoke")
        result = await voice_graph.ainvoke({
            "messages": messages,
            "session_id": session_id,
            "model_selection": model,
        })
        latency_tracker.end("llm_graph_invoke")

        latency_tracker.end("llm_total")

        tts_text = result.get("tts_text") or "收到"

        # 5. Persist assistant response
        await add_call_message(session_id, "assistant", tts_text)

        return tts_text
    except Exception as e:
        logger.error(f"Voice Agent Error: {e}", exc_info=True)
        return "抱歉，我暂时无法处理你的请求。"


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


async def play_tts(room: rtc.Room, text: str, should_exit_event: asyncio.Event):
    latency_tracker.start("tts_generate")
    source = rtc.AudioSource(48000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("tts", source)
    publication = await room.local_participant.publish_track(track)
    try:
        latency_tracker.start("tts_first_chunk")
        first_chunk_found = False
        
        jitter_buffer_bytes = b""
        # Using 250ms as the stable default for RTX 2060
        jitter_threshold_bytes = int(48000 * (config.TTS_JITTER_BUFFER_MS / 1000) * 2) 
        playback_started = False
        total_samples_pushed = 0
        start_time = 0

        async for audio_chunk in get_tts_stream(text, target_sample_rate=48000):
            if not first_chunk_found:
                latency_tracker.end("tts_first_chunk")
                latency_tracker.end("tts_generate")
                first_chunk_found = True
            
            jitter_buffer_bytes += audio_chunk
            
            # Start playing once jitter buffer threshold is reached
            if not playback_started and len(jitter_buffer_bytes) >= jitter_threshold_bytes:
                latency_tracker.start("tts_playback")
                latency_tracker.start("tts_audio_stream")
                playback_started = True
                start_time = time.time()
            
            if playback_started:
                import numpy as np
                audio_np = np.frombuffer(jitter_buffer_bytes, dtype=np.int16)
                jitter_buffer_bytes = b"" 
                
                chunk_size = 1920 
                for i in range(0, len(audio_np), chunk_size):
                    chunk = audio_np[i : i + chunk_size]
                    if len(chunk) > 0:
                        frame = rtc.AudioFrame(data=chunk.tobytes(), sample_rate=48000, num_channels=1, samples_per_channel=len(chunk))
                        await source.capture_frame(frame)
                        total_samples_pushed += len(chunk)
                await asyncio.sleep(0.01)

        # Final cleanup for remaining small chunks
        if jitter_buffer_bytes:
            if not playback_started:
                playback_started = True
                start_time = time.time()
            import numpy as np
            audio_np = np.frombuffer(jitter_buffer_bytes, dtype=np.int16)
            for i in range(0, len(audio_np), 1920):
                chunk = audio_np[i : i + 1920]
                frame = rtc.AudioFrame(data=chunk.tobytes(), sample_rate=48000, num_channels=1, samples_per_channel=len(chunk))
                await source.capture_frame(frame)
                total_samples_pushed += len(chunk)

        latency_tracker.end("tts_audio_stream")
        
        # Wait for audio to drain
        if total_samples_pushed > 0:
            total_duration = total_samples_pushed / 48000
            elapsed = time.time() - start_time
            remaining = total_duration - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining + 0.3) 
        
        latency_tracker.end("tts_playback")
    finally:
        await room.local_participant.unpublish_track(publication.sid)
        if "[TERMINATE]" in text:
            logger.info("Termination signal detected in Agent response. Hanging up...")
            should_exit_event.set()


async def play_greeting(room: rtc.Room):
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

    source = rtc.AudioSource(48000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("greeting", source)
    publication = await room.local_participant.publish_track(track)

    try:
        with wave.open(wav_path, "rb") as wf:
            framerate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
        import numpy as np

        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        if framerate != 48000:
            audio_np = np.interp(
                np.linspace(
                    0, len(audio_np) - 1, int(len(audio_np) * (48000 / framerate))
                ),
                np.arange(len(audio_np)),
                audio_np,
            ).astype(np.int16)

        silence = np.zeros(48000 + 24000, dtype=np.int16)
        chunk_size = 1920
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i : i + chunk_size]
            if len(chunk) > 0:
                frame = rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=48000,
                    num_channels=1,
                    samples_per_channel=len(chunk),
                )
                await source.capture_frame(frame)

        for i in range(0, len(audio_np), chunk_size):
            chunk = audio_np[i : i + chunk_size]
            if len(chunk) > 0:
                frame = rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=48000,
                    num_channels=1,
                    samples_per_channel=len(chunk),
                )
                await source.capture_frame(frame)
        await asyncio.sleep(len(audio_np) / 48000 + 0.5)
    except Exception as e:
        logger.error(f"Error in play_greeting: {e}")
    finally:
        await room.local_participant.unpublish_track(publication.sid)



async def entrypoint(ctx: JobContext):
    global db_initialized
    if not db_initialized:
        await init_db()
        db_initialized = True

    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    room = ctx.room
    should_exit = asyncio.Event()

    # CRITICAL: Register event handlers IMMEDIATELY after connect,
    # BEFORE any await calls, to avoid missing track_subscribed events.
    @room.on("participant_disconnected")
    def on_participant_disconnected(p):
        remote_sip = [
            p
            for p in room.remote_participants.values()
            if p.identity.startswith("sip_")
        ]
        if not remote_sip:
            logger.info("All SIP participants disconnected. Exiting job...")
            should_exit.set()

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if isinstance(track, rtc.AudioTrack):
            logger.info(
                f"Subscribed to audio track {track.sid} from {participant.identity}"
            )
            asyncio.create_task(process_audio_safe(track, room, should_exit))

    # Now safe to do async I/O — event handlers are already registered
    call_type = "inbound"
    if room.name.startswith("outbound-reminder-"):
        call_type = "outbound-reminder"
    elif room.name.startswith("outbound-"):
        call_type = "outbound"
    await create_call_session(room.name, call_type)
    logger.info(f"Call session created: {room.name} ({call_type})")

    if room.name.startswith("outbound-"):
        logger.info("Outbound call: Waiting for Answer Status...")
        answered = False
        while not answered:
            for p in room.remote_participants.values():
                if p.identity.startswith("sip_"):
                    status = p.attributes.get("sip.callStatus")
                    sip_meta = None
                    try:
                        if p.metadata:
                            sip_meta = (
                                json.loads(p.metadata).get("sip", {}).get("call_status")
                            )
                    except:
                        pass
                    if status == "active" or sip_meta == "active":
                        logger.info(f"User {p.identity} answered!")
                        answered = True
                        break
            if not answered:
                await asyncio.sleep(0.2)
        await asyncio.sleep(0.5)
        await play_greeting(room)
    else:
        await play_greeting(room)

    await should_exit.wait()
    logger.info("Termination signal received. Forcing SIP hangup...")

    # End call session in database
    await end_call_session(room.name)
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


async def process_audio_safe(
    track: rtc.AudioTrack, room: rtc.Room, should_exit: asyncio.Event
):
    audio_stream = rtc.AudioStream(track)
    silero_vad = silero.VAD.load()
    vad_stream = silero_vad.stream()
    all_frames = []
    pre_buffer = collections.deque(maxlen=300)
    is_speaking = False
    session_id = room.name

    async def vad_logic():
        nonlocal is_speaking, all_frames
        async for event in vad_stream:
            if event.type == vad.VADEventType.START_OF_SPEECH:
                latency_tracker.start("vad_silence")
                is_speaking = True
                all_frames = list(pre_buffer)
                pre_buffer.clear()
            elif event.type == vad.VADEventType.END_OF_SPEECH:
                latency_tracker.end("vad_silence")
                latency_tracker.start("vad_speech")
                latency_tracker.start("end_to_end")
                is_speaking = False
                if all_frames:
                    audio_data = b"".join([f.data for f in all_frames])
                    latency_tracker.end("vad_speech")
                    text = await transcribe_audio(audio_data, 48000, 1)
                    if text:
                        logger.info(f"========> [User Said]: {text}")
                        latency_tracker.start("agent_response")
                        resp_text = await call_agent(session_id, text)
                        latency_tracker.end("agent_response")
                        latency_tracker.end("end_to_end")
                        logger.info(f"========> [Agent Response]: {resp_text}")
                        asyncio.create_task(play_tts(room, resp_text, should_exit))

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
