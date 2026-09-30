import asyncio
import time

import numpy as np
from livekit import rtc
from livekit.agents import vad
from livekit.plugins import silero
from utils.logger import get_logger
from utils.latency import latency_tracker
from .stt import transcribe_audio
from .agent_client import call_agent
from .audio_playback import play_tts, play_transition_audio, play_transition_audio_loop
from .end_call import EndCallCountdown

logger = get_logger("livekit-session")

# Ignore very brief VAD spikes while keeping barge-in responsive.
VAD_PREFIX_SECONDS = 0.3
VAD_MODEL = silero.VAD.load(
    min_speech_duration=0.2,
    min_silence_duration=0.5,
    prefix_padding_duration=VAD_PREFIX_SECONDS,
    activation_threshold=0.6,
)
MIN_SPEECH_SECONDS = 0.25
MIN_AUDIO_RMS = 80.0


def _prepare_utterance(event: vad.VADEvent) -> tuple[bytes, int] | None:
    """Use the VAD's own bounded speech segment, not a shared rolling buffer."""
    frames = event.frames
    if not frames:
        logger.info("[voice-input] discarded segment without audio frames")
        return None
    sample_rate = frames[0].sample_rate
    if sample_rate <= 0 or any(
        frame.sample_rate != sample_rate or frame.num_channels != 1 for frame in frames
    ):
        logger.warning("[voice-input] discarded inconsistent audio frames")
        return None
    audio_data = b"".join(bytes(frame.data) for frame in frames)
    if not audio_data or len(audio_data) % 2:
        logger.info("[voice-input] discarded empty or malformed audio")
        return None
    samples = np.frombuffer(audio_data, dtype=np.int16)
    audio_seconds = len(samples) / sample_rate
    # Silero resets speech_duration to zero in END_OF_SPEECH events. Its frame
    # contains prefix padding and trailing silence; subtract those instead.
    speech_seconds = max(0.0, audio_seconds - VAD_PREFIX_SECONDS - event.silence_duration)
    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
    logger.info(
        "[voice-input] segment speech=%.2fs audio=%.2fs rms=%.0f",
        speech_seconds, audio_seconds, rms,
    )
    if speech_seconds < MIN_SPEECH_SECONDS:
        logger.info("[voice-input] discarded short segment")
        return None
    if rms < MIN_AUDIO_RMS:
        logger.info("[voice-input] discarded low-energy segment")
        return None
    return audio_data, sample_rate


def _has_words(text: str) -> bool:
    return any(character.isalnum() for character in text)

class VoiceSession:
    def __init__(self, room: rtc.Room, should_exit: asyncio.Event, user_id: str,
                 is_sip: bool = False, answered_event: asyncio.Event = None,
                 interrupt_event: asyncio.Event = None):
        self.room = room
        self.should_exit = should_exit
        self.user_id = user_id
        self.is_sip = is_sip
        self.answered_event = answered_event or asyncio.Event()
        if not is_sip: # Non-SIP (Web) is always "answered"
            self.answered_event.set()
        self.session_id = room.name
        
        self.interrupt_event = interrupt_event if interrupt_event is not None else asyncio.Event()
        self.transition_stop_event = asyncio.Event()
        self.current_tts_task = None
        self.current_transition_task = None
        self.current_response_task = None
        self._response_tasks: set[asyncio.Task] = set()
        self.last_activity_time = time.time()
        self.SILENCE_TIMEOUT = 25.0 # Standard conversation timeout
        self.is_greeting_playing = False # New flag to track initial greeting
        self.is_speaking = False
        self.is_agent_processing = False
        self.vad_stream = VAD_MODEL.stream()
        self.end_call_countdown = EndCallCountdown(
            should_exit, lambda: self.is_speaking,
        )

    async def run(self, track: rtc.AudioTrack):
        """Main loop for processing audio from a track."""
        audio_stream = rtc.AudioStream(track)
        vad_task = asyncio.create_task(self._vad_logic_loop())
        
        try:
            async for frame_event in audio_stream:
                # Elegant state-aware timeout check
                if self.answered_event.is_set():
                    # Only check for silence if we're not busy and the call is active
                    is_busy = (self.current_tts_task and not self.current_tts_task.done()) or \
                              (self.current_transition_task and not self.current_transition_task.done()) or \
                              (self.current_response_task and not self.current_response_task.done()) or \
                              self.is_agent_processing or self.is_speaking or self.is_greeting_playing
                    
                    if is_busy:
                        self.last_activity_time = time.time()
                    
                    if time.time() - self.last_activity_time > self.SILENCE_TIMEOUT:
                        logger.info(f"Session Timeout: {self.SILENCE_TIMEOUT}s of total silence in active call. Hanging up {self.session_id}...")
                        self.should_exit.set()
                        break
                else:
                    # While waiting for answer, we just keep the activity time fresh
                    self.last_activity_time = time.time()

                self.vad_stream.push_frame(frame_event.frame)
        finally:
            self.end_call_countdown.cancel()
            vad_task.cancel()
            pending = {vad_task, *self._response_tasks}
            pending.update(task for task in (self.current_tts_task,
                                             self.current_transition_task) if task is not None)
            for task in pending:
                if task is not None and not task.done() and not task.cancelling():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await self.vad_stream.aclose()
            await audio_stream.aclose()

    async def _vad_logic_loop(self):
        """Processes VAD events and triggers agent responses."""
        async for event in self.vad_stream:
            if event.type == vad.VADEventType.START_OF_SPEECH:
                self._handle_start_of_speech()
            elif event.type == vad.VADEventType.END_OF_SPEECH:
                self.is_speaking = False
                task = asyncio.create_task(self._handle_end_of_speech(event))
                self.current_response_task = task
                self._response_tasks.add(task)
                task.add_done_callback(self._on_response_done)

    def _on_response_done(self, task: asyncio.Task) -> None:
        self._response_tasks.discard(task)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error("[voice-input] response task failed", exc_info=(
                    type(error), error, error.__traceback__,
                ))

    def _handle_start_of_speech(self):
        self.end_call_countdown.cancel()
        self.last_activity_time = time.time()
        self.user_has_spoken = True
        latency_tracker.start("vad_silence")
        self.is_speaking = True
        
        logger.info("[VoiceInterrupt] Detected START_OF_SPEECH. Interrupting AI...")
        self.interrupt_event.set()
        if (self.current_response_task and not self.current_response_task.done()
                and not self.current_response_task.cancelling()):
            self.current_response_task.cancel()
        if self.current_tts_task and not self.current_tts_task.done():
            self.current_tts_task.cancel()
        if self.current_transition_task and not self.current_transition_task.done():
            self.current_transition_task.cancel()
            
    async def _handle_end_of_speech(self, event: vad.VADEvent):
        latency_tracker.end("vad_silence")
        latency_tracker.start("vad_speech")
        latency_tracker.start("end_to_end")
        prepared = _prepare_utterance(event)
        if prepared is None:
            latency_tracker.reset()
            return
        audio_data, sample_rate = prepared
        interrupt_event = self.interrupt_event = asyncio.Event()
        latency_tracker.end("vad_speech")

        # 1. Transcribe
        text = (await transcribe_audio(audio_data, sample_rate, 1)).strip()
        if not _has_words(text):
            logger.info("[voice-input] discarded non-lexical transcript: %r", text)
            latency_tracker.reset()
            return

        logger.info(f"========> [User Said]: {text}")
        latency_tracker.start("agent_response")

        # 2. Call Agent
        final_resp_text = ""
        end_call_requested = False
        speech_task = None
        tts_start_event = asyncio.Event()
        playback_finished = False
        self.is_agent_processing = True
        agent_stream = call_agent(self.session_id, text, self.user_id)
        try:
            async for event_type, payload in agent_stream:
                if interrupt_event.is_set():
                    break

                if event_type == "tool_call":
                    logger.info(f"[Agent Tool Call]: {payload}")
                    if self.current_transition_task and not self.current_transition_task.done():
                        self.current_transition_task.cancel()
                    self.current_transition_task = asyncio.create_task(
                        play_transition_audio(self.room, interrupt_event, tool_name=payload)
                    )
                elif event_type == "end_call_requested":
                    end_call_requested = bool(payload)
                    if end_call_requested:
                        await self._finish_playback(
                            speech_task, tts_start_event, interrupt_event,
                            final_resp_text, end_call_requested=True,
                        )
                        playback_finished = True
                elif event_type == "message":
                    final_resp_text = payload
                    if interrupt_event.is_set() or not final_resp_text:
                        continue
                    if self.current_tts_task and not self.current_tts_task.done():
                        continue
                    t_msg = time.time()
                    logger.info(f"========> [Agent Response]: {final_resp_text}")
                    logger.info(f"[TIMING][Session] MESSAGE_ARRIVED | t={t_msg:.3f}")
                    latency_tracker.end("agent_response")
                    latency_tracker.end("end_to_end")
                    transition_busy = (
                        self.current_transition_task is not None
                        and not self.current_transition_task.done()
                    )
                    if transition_busy:
                        speech_task = self.current_tts_task = asyncio.create_task(
                            play_tts(
                                self.room,
                                final_resp_text,
                                interrupt_event,
                                start_event=tts_start_event,
                            )
                        )
                    else:
                        speech_task = self.current_tts_task = asyncio.create_task(
                            play_tts(
                                self.room,
                                final_resp_text,
                                interrupt_event,
                            )
                        )
                    logger.info(f"[TIMING][Session] TTS_TASK_CREATED | dt={time.time() - t_msg:.3f}s")
        finally:
            try:
                # Also close the run if interruption occurs in the loop body,
                # while the async generator is suspended at a yielded event.
                await agent_stream.aclose()
            finally:
                self.is_agent_processing = False

        if "agent_response" in latency_tracker.timings and "end" not in latency_tracker.timings.get("agent_response", {}):
            latency_tracker.end("agent_response")
        if "end_to_end" in latency_tracker.timings and "end" not in latency_tracker.timings.get("end_to_end", {}):
            latency_tracker.end("end_to_end")

        if not playback_finished:
            await self._finish_playback(
                speech_task, tts_start_event, interrupt_event,
                final_resp_text, end_call_requested,
            )
        self._log_latency_summary()
        latency_tracker.reset()

    async def _finish_playback(
        self, speech_task: asyncio.Task | None, tts_start_event: asyncio.Event,
        interrupt_event: asyncio.Event, final_resp_text: str,
        end_call_requested: bool,
    ) -> None:
        """Finish speech and start the grace period independently of graph tail work."""
        # Wait for transition to finish, then release deferred TTS.
        t_wait_start = time.time()
        if self.current_transition_task and not self.current_transition_task.done():
            logger.info(f"[TIMING][Session] WAIT_TRANSITION_START | t={t_wait_start:.3f}")
            await self.current_transition_task
            logger.info(f"[TIMING][Session] WAIT_TRANSITION_DONE | dt={time.time() - t_wait_start:.3f}s")
        else:
            logger.info(
                f"[TIMING][Session] NO_TRANSITION_TO_WAIT | "
                f"transition_active={self.current_transition_task is not None} | dt=0s"
            )

        if not interrupt_event.is_set() and final_resp_text:
            tts_start_event.set()
            logger.info(f"[TIMING][Session] START_EVENT_SET | t={time.time():.3f}")

        played_to_end = False
        if speech_task is not None:
            try:
                played_to_end = await speech_task
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
                logger.info("[TIMING][Session] TTS_TASK_CANCELLED")
        # A TTS task can catch CancelledError while returning its playback result.
        # Do not let that swallow cancellation of the parent response task.
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError

        if end_call_requested:
            if self.is_sip and played_to_end and not interrupt_event.is_set():
                logger.info("[hangup] goodbye played; waiting two seconds for follow-up")
                self.end_call_countdown.arm()
            else:
                logger.warning("[hangup] request discarded because goodbye did not finish")

    def _log_latency_summary(self):
        logger.info("========> [Latency Summary] =========")
        logger.info(f"  VAD沉默检测: {latency_tracker.get('vad_silence') * 1000:.0f}ms")
        logger.info(f"  VAD语音检测: {latency_tracker.get('vad_speech') * 1000:.0f}ms")
        logger.info(f"  STT语音识别: {latency_tracker.get('stt_total') * 1000:.0f}ms")
        logger.info(f"    - HTTP请求: {latency_tracker.get('stt_http_request') * 1000:.0f}ms")
        logger.info(f"  LLM总延迟: {latency_tracker.get('llm_total') * 1000:.0f}ms")
        logger.info(f"    - 图推理: {latency_tracker.get('llm_graph_invoke') * 1000:.0f}ms")
        logger.info(f"  TTS首包延迟: {latency_tracker.get('tts_first_chunk') * 1000:.0f}ms")
        logger.info(f"  TTS生成(全): {latency_tracker.get('tts_generate') * 1000:.0f}ms")
        logger.info(f"  TTS播放: {latency_tracker.get('tts_playback') * 1000:.0f}ms")
        logger.info(f"    - 音频流: {latency_tracker.get('tts_audio_stream') * 1000:.0f}ms")
        total = latency_tracker.get("end_to_end")
        logger.info(f"  端到端延迟: {total * 1000:.0f}ms")
        logger.info("=======================================")
