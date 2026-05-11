import asyncio
import time
import os
import wave
import random
import numpy as np
from livekit import rtc
from core.config import config
from utils.logger import get_logger
from utils.latency import latency_tracker
from services.tts import get_tts_stream
from core.database import AUDIO_CACHE_DIR

logger = get_logger("livekit-audio")

async def play_tts(room: rtc.Room, text: str, should_exit_event: asyncio.Event, interrupt_event: asyncio.Event, start_event: asyncio.Event = None):
    """Generate and play TTS audio in the LiveKit room.

    If start_event is provided and not yet set, audio is prefetched in the
    background and playback is deferred until start_event is set — eliminating
    the TTS cold-start gap after a transition audio finishes.
    """
    t0 = time.time()
    defer_playback = start_event is not None and not start_event.is_set()
    logger.info(f"[TIMING][TTS] START | text_len={len(text)} | deferred={defer_playback} | t={t0:.3f}")

    latency_tracker.start("tts_generate")
    prefetch_buffer = bytearray() if defer_playback else None

    source = None
    track = None
    publication = None

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
                t_first = time.time()
                logger.info(f"[TIMING][TTS] FIRST_CHUNK | dt={t_first - t0:.3f}s")
                first_chunk_found = True

            if defer_playback and not start_event.is_set():
                prefetch_buffer.extend(audio_chunk)
                continue

            # start_event just fired: publish track and merge prefetched audio
            if defer_playback and len(prefetch_buffer) > 0:
                t_pub = time.time()
                logger.info(f"[TIMING][TTS] START_EVENT_FIRED | prefetch_bytes={len(prefetch_buffer)} | dt={t_pub - t0:.3f}s")
                if source is None:
                    source = rtc.AudioSource(24000, 1)
                    track = rtc.LocalAudioTrack.create_audio_track("tts", source)
                    publication = await room.local_participant.publish_track(track)
                    logger.info(f"[TIMING][TTS] TRACK_PUBLISHED | dt={time.time() - t0:.3f}s")

                jitter_buffer_bytes = bytes(prefetch_buffer) + audio_chunk
                prefetch_buffer = bytearray()
                defer_playback = False
                continue

            # Lazy publish: defer track creation until ready to play
            if source is None:
                source = rtc.AudioSource(24000, 1)
                track = rtc.LocalAudioTrack.create_audio_track("tts", source)
                publication = await room.local_participant.publish_track(track)
                logger.info(f"[TIMING][TTS] TRACK_PUBLISHED | dt={time.time() - t0:.3f}s")

            jitter_buffer_bytes += audio_chunk

            if not playback_started and len(jitter_buffer_bytes) >= jitter_threshold_bytes:
                t_play = time.time()
                logger.info(f"[TIMING][TTS] PLAYBACK_START | dt={t_play - t0:.3f}s")
                latency_tracker.start("tts_playback")
                latency_tracker.start("tts_audio_stream")
                playback_started = True
                start_time = t_play

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

        # Flush prefetched audio if we deferred
        if defer_playback and len(prefetch_buffer) > 0 and not interrupt_event.is_set():
            if source is None:
                source = rtc.AudioSource(24000, 1)
                track = rtc.LocalAudioTrack.create_audio_track("tts", source)
                publication = await room.local_participant.publish_track(track)
                logger.info(f"DEBUG-PUBLISHING: Published NEW TTS track (deferred): {publication.sid}")

            if not playback_started:
                latency_tracker.start("tts_playback")
                latency_tracker.start("tts_audio_stream")
                playback_started = True
                start_time = time.time()

            prefetch_np = np.frombuffer(prefetch_buffer, dtype=np.int16)
            for i in range(0, len(prefetch_np), 480):
                if interrupt_event.is_set():
                    break
                chunk = prefetch_np[i : i + 480]
                if len(chunk) > 0:
                    frame = rtc.AudioFrame(data=chunk.tobytes(), sample_rate=24000, num_channels=1, samples_per_channel=len(chunk))
                    await source.capture_frame(frame)
                    total_samples_pushed += len(chunk)

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

        if playback_started:
            latency_tracker.end("tts_audio_stream")
            logger.info(f"[TIMING][TTS] STREAM_DONE | total_samples={total_samples_pushed} | dt={time.time() - t0:.3f}s")

        # Wait for audio to drain
        if total_samples_pushed > 0 and not interrupt_event.is_set():
            total_duration = total_samples_pushed / 24000
            elapsed = time.time() - start_time
            remaining = total_duration - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining + 0.3)

        if playback_started:
            latency_tracker.end("tts_playback")
    except asyncio.CancelledError:
        logger.info(f"[TIMING][TTS] CANCELLED | dt={time.time() - t0:.3f}s")
    finally:
        if publication:
            logger.info(f"[TIMING][TTS] DONE | total_dt={time.time() - t0:.3f}s")
            await room.local_participant.unpublish_track(publication.sid)
        if "[TERMINATE]" in text:
            logger.info("Termination signal detected in Agent response. Hanging up...")
            should_exit_event.set()

async def play_greeting(room: rtc.Room, initial_speech: str = "", should_exit_event: asyncio.Event = None, user_id: str = "default", call_type: str = "inbound"):
    """Play the greeting. Prioritize pre-rendered wav if available, otherwise TTS."""
    
    # 1. Check for specific pre-rendered wav file first
    wav_path = None
    if "outbound-reminder-" in room.name:
        start_idx = room.name.find("outbound-reminder-") + len("outbound-reminder-")
        reminder_id = room.name[start_idx : start_idx + 36]
        specific_wav = os.path.join(AUDIO_CACHE_DIR, f"reminder_{reminder_id}.wav")
        if os.path.exists(specific_wav):
            wav_path = specific_wav
            logger.info(f"Prioritizing specific reminder audio: {wav_path}")
    elif "outbound-supervisor-" in room.name:
        start_idx = room.name.find("outbound-supervisor-") + len("outbound-supervisor-")
        end_idx = room.name.find("-", start_idx)
        if end_idx != -1:
            sup_id = room.name[start_idx : end_idx]
            specific_wav = os.path.join(AUDIO_CACHE_DIR, f"supervisor_{sup_id}.wav")
            if os.path.exists(specific_wav):
                wav_path = specific_wav
                logger.info(f"Prioritizing pre-generated supervisor audio: {wav_path}")
    
    # 2. If wav exists, play it and return
    if wav_path:
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
                    np.linspace(0, len(audio_np) - 1, int(len(audio_np) * (24000 / framerate))),
                    np.arange(len(audio_np)),
                    audio_np,
                ).astype(np.int16)

            silence = np.zeros(24000 + 12000, dtype=np.int16)
            chunk_size = 480
            for i in range(0, len(silence), chunk_size):
                chunk = silence[i : i + chunk_size]
                if len(chunk) > 0:
                    frame = rtc.AudioFrame(data=chunk.tobytes(), sample_rate=24000, num_channels=1, samples_per_channel=len(chunk))
                    await source.capture_frame(frame)

            for i in range(0, len(audio_np), chunk_size):
                chunk = audio_np[i : i + chunk_size]
                if len(chunk) > 0:
                    frame = rtc.AudioFrame(data=chunk.tobytes(), sample_rate=24000, num_channels=1, samples_per_channel=len(chunk))
                    await source.capture_frame(frame)
            await asyncio.sleep(len(audio_np) / 24000 + 0.5)
        except Exception as e:
            logger.error(f"Error in play_greeting (file): {e}")
        finally:
            await room.local_participant.unpublish_track(publication.sid)
        return

    # 3. Fallback: play_tts
    if initial_speech:
        logger.info(f"Playing initial speech via TTS: {initial_speech}")
        dummy_interrupt = asyncio.Event()
        await play_tts(room, initial_speech, should_exit_event, dummy_interrupt)
        return

    # 4. Final fallback: default greeting.wav
    wav_path = str(config.ASSETS_DIR / "greeting.wav")
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
                np.linspace(0, len(audio_np) - 1, int(len(audio_np) * (24000 / framerate))),
                np.arange(len(audio_np)),
                audio_np,
            ).astype(np.int16)

        silence = np.zeros(24000 + 12000, dtype=np.int16)
        chunk_size = 480
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i : i + chunk_size]
            if len(chunk) > 0:
                frame = rtc.AudioFrame(data=chunk.tobytes(), sample_rate=24000, num_channels=1, samples_per_channel=len(chunk))
                await source.capture_frame(frame)

        for i in range(0, len(audio_np), chunk_size):
            chunk = audio_np[i : i + chunk_size]
            if len(chunk) > 0:
                frame = rtc.AudioFrame(data=chunk.tobytes(), sample_rate=24000, num_channels=1, samples_per_channel=len(chunk))
                await source.capture_frame(frame)
        await asyncio.sleep(len(audio_np) / 24000)
    except Exception as e:
        logger.error(f"Error in play_greeting: {e}")
    finally:
        await room.local_participant.unpublish_track(publication.sid)

async def play_transition_audio(room: rtc.Room, interrupt_event: asyncio.Event, tool_name: str = None):
    """Play a single transition audio file based on the tool called, or a generic fallback."""
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
            return

    t0 = time.time()
    logger.info(f"[TIMING][Transition] START | tool={tool_name} | file={os.path.basename(wav_path)} | t={t0:.3f}")

    source = rtc.AudioSource(24000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("transition", source)
    publication = await room.local_participant.publish_track(track)
    logger.info(f"[TIMING][Transition] TRACK_PUBLISHED | dt={time.time() - t0:.3f}s")

    try:
        with wave.open(str(wav_path), "rb") as wf:
            framerate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())

        audio_duration = len(audio_data) / (framerate * wf.getsampwidth() * wf.getnchannels())
        logger.info(f"[TIMING][Transition] AUDIO_LOADED | duration={audio_duration:.2f}s | dt={time.time() - t0:.3f}s")

        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        if framerate != 24000:
            audio_np = np.interp(
                np.linspace(0, len(audio_np) - 1, int(len(audio_np) * (24000 / framerate))),
                np.arange(len(audio_np)),
                audio_np,
            ).astype(np.int16)

        t_push_start = time.time()
        chunk_size = 480
        frames_pushed = 0
        for i in range(0, len(audio_np), chunk_size):
            if interrupt_event.is_set():
                break
            chunk = audio_np[i : i + chunk_size]
            if len(chunk) > 0:
                frame = rtc.AudioFrame(data=chunk.tobytes(), sample_rate=24000, num_channels=1, samples_per_channel=len(chunk))
                await source.capture_frame(frame)
                frames_pushed += 1

        t_push_end = time.time()
        push_elapsed = t_push_end - t_push_start
        audio_dur = len(audio_np) / 24000
        # logger.info(f"[TIMING][Transition] FRAMES_PUSHED | frames={frames_pushed} | push_time={push_elapsed:.3f}s | audio_dur={audio_dur:.3f}s | dt={t_push_end - t0:.3f}s")

        if not interrupt_event.is_set():
            remaining = audio_dur - push_elapsed
            if remaining > 0:
                logger.info(f"[TIMING][Transition] DRAIN_WAIT | remaining={remaining:.3f}s")
                await asyncio.sleep(remaining + 0.1)

        logger.info(f"[TIMING][Transition] DONE | total_dt={time.time() - t0:.3f}s")

    except asyncio.CancelledError:
        logger.info(f"[TIMING][Transition] CANCELLED | dt={time.time() - t0:.3f}s")
        pass
    except Exception as e:
        logger.error(f"[Transition] Error playing transition audio: {e}")
    finally:
        await room.local_participant.unpublish_track(publication.sid)


async def play_transition_audio_loop(room: rtc.Room, interrupt_event: asyncio.Event, stop_event: asyncio.Event, tool_name: str = None):
    """Play transition audio in a loop until interrupted or stopped.

    Keeps playing random transition audio files back-to-back to fill the
    silence during tool execution + LLM response. Stops when either
    interrupt_event (user speaks) or stop_event (message arrived) is set.
    """
    transition_dir = config.ASSETS_DIR / "transitions"
    if not transition_dir.exists():
        logger.info("[Transition] Transition directory missing, skipping.")
        return

    generic_files = ["checking.wav", "hmm.wav", "thinking.wav", "wait.wav", "working.wav"]
    available_generics = [f for f in generic_files if (transition_dir / f).exists()]
    if not available_generics:
        return

    logger.info("[Transition] Starting transition audio loop")
    iteration = 0

    while not interrupt_event.is_set() and not stop_event.is_set():
        iteration += 1
        wav_path = transition_dir / random.choice(available_generics)
        logger.info(f"[Transition] Loop iteration {iteration}: {os.path.basename(wav_path)}")

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
                if interrupt_event.is_set() or stop_event.is_set():
                    break
                chunk = audio_np[i : i + chunk_size]
                if len(chunk) > 0:
                    frame = rtc.AudioFrame(data=chunk.tobytes(), sample_rate=24000, num_channels=1, samples_per_channel=len(chunk))
                    await source.capture_frame(frame)

            if not interrupt_event.is_set() and not stop_event.is_set():
                await asyncio.sleep(len(audio_np) / 24000 + 0.1)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[Transition] Error in loop: {e}")
        finally:
            await room.local_participant.unpublish_track(publication.sid)

    logger.info(f"[Transition] Transition audio loop stopped after {iteration} iterations")
