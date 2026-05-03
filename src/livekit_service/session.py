import asyncio
import collections
import time
from livekit import rtc
from livekit.agents import vad
from livekit.plugins import silero
from utils.logger import get_logger
from utils.latency import latency_tracker
from .stt import transcribe_audio
from .agent_client import call_agent
from .audio_playback import play_tts, play_transition_audio

logger = get_logger("livekit-session")

# Pre-load VAD model as a class or module level variable
VAD_MODEL = silero.VAD.load(min_silence_duration=1.0)

class VoiceSession:
    def __init__(self, room: rtc.Room, should_exit: asyncio.Event, user_id: str, is_sip: bool = False):
        self.room = room
        self.should_exit = should_exit
        self.user_id = user_id
        self.is_sip = is_sip
        self.session_id = room.name
        
        self.interrupt_event = asyncio.Event()
        self.current_tts_task = None
        self.current_transition_task = None
        self.last_activity_time = time.time()
        self.SILENCE_TIMEOUT = 10.0
        
        self.is_speaking = False
        self.is_agent_processing = False
        self.all_frames = []
        self.pre_buffer = collections.deque(maxlen=300)
        
        self.vad_stream = VAD_MODEL.stream()

    async def run(self, track: rtc.AudioTrack):
        """Main loop for processing audio from a track."""
        audio_stream = rtc.AudioStream(track)
        vad_task = asyncio.create_task(self._vad_logic_loop())
        
        try:
            async for frame_event in audio_stream:
                # Update activity if agent is busy
                if (self.current_tts_task and not self.current_tts_task.done()) or \
                   (self.current_transition_task and not self.current_transition_task.done()) or \
                   self.is_agent_processing or self.is_speaking:
                    self.last_activity_time = time.time()

                # Check for silence timeout
                if time.time() - self.last_activity_time > self.SILENCE_TIMEOUT:
                    logger.info(f"DEBUG-TIMEOUT: 10s silence detected. Hanging up session {self.session_id}...")
                    self.should_exit.set()
                    break

                self.vad_stream.push_frame(frame_event.frame)
                if self.is_speaking:
                    self.all_frames.append(frame_event.frame)
                else:
                    self.pre_buffer.append(frame_event.frame)
        finally:
            vad_task.cancel()
            await self.vad_stream.aclose()

    async def _vad_logic_loop(self):
        """Processes VAD events and triggers agent responses."""
        async for event in self.vad_stream:
            if event.type == vad.VADEventType.START_OF_SPEECH:
                self._handle_start_of_speech()
            elif event.type == vad.VADEventType.END_OF_SPEECH:
                await self._handle_end_of_speech()

    def _handle_start_of_speech(self):
        self.last_activity_time = time.time()
        latency_tracker.start("vad_silence")
        self.is_speaking = True
        
        logger.info("[VoiceInterrupt] Detected START_OF_SPEECH. Interrupting AI...")
        self.interrupt_event.set()
        if self.current_tts_task and not self.current_tts_task.done():
            self.current_tts_task.cancel()
        if self.current_transition_task and not self.current_transition_task.done():
            self.current_transition_task.cancel()
            
        self.all_frames = list(self.pre_buffer)
        self.pre_buffer.clear()

    async def _handle_end_of_speech(self):
        latency_tracker.end("vad_silence")
        latency_tracker.start("vad_speech")
        latency_tracker.start("end_to_end")
        self.is_speaking = False
        self.interrupt_event.clear()
        
        if not self.all_frames:
            return
            
        audio_data = b"".join([f.data for f in self.all_frames])
        self.all_frames = []
        latency_tracker.end("vad_speech")
        
        # 1. Transcribe
        text = await transcribe_audio(audio_data, 48000, 1)
        if not text:
            return
            
        logger.info(f"========> [User Said]: {text}")
        latency_tracker.start("agent_response")
        
        # 2. Call Agent
        final_resp_text = ""
        self.is_agent_processing = True
        try:
            async for event_type, payload in call_agent(self.session_id, text, self.user_id):
                if self.interrupt_event.is_set():
                    break
                    
                if event_type == "tool_call":
                    logger.info(f"[Agent Tool Call]: {payload}")
                    if self.current_transition_task and not self.current_transition_task.done():
                        self.current_transition_task.cancel()
                    self.current_transition_task = asyncio.create_task(
                        play_transition_audio(self.room, self.interrupt_event, tool_name=payload)
                    )
                elif event_type == "message":
                    final_resp_text = payload
        finally:
            self.is_agent_processing = False
        
        latency_tracker.end("agent_response")
        latency_tracker.end("end_to_end")
        
        # 3. Play TTS
        if not self.interrupt_event.is_set() and final_resp_text:
            logger.info(f"========> [Agent Response]: {final_resp_text}")
            # Wait for transition audio to finish before playing TTS
            if self.current_transition_task and not self.current_transition_task.done():
                logger.info("Waiting for transition audio to finish before playing TTS...")
                await self.current_transition_task
                
            if not self.interrupt_event.is_set():
                self.current_tts_task = asyncio.create_task(
                    play_tts(self.room, final_resp_text, self.should_exit, self.interrupt_event)
                )

        self._log_latency_summary()
        latency_tracker.reset()

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
