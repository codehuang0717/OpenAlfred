"""Speech events must interrupt playback without waiting for the old turn."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from livekit.agents import vad

from livekit_service.session import VoiceSession, _has_words, _prepare_utterance


def speech_event(text_amplitude: int = 10000, duration: float = 0.5):
    audio_samples = int((0.3 + duration + 0.5) * 48000)
    return SimpleNamespace(
        type=vad.VADEventType.END_OF_SPEECH,
        speech_duration=0.0,  # Silero resets this on END_OF_SPEECH.
        silence_duration=0.5,
        frames=[SimpleNamespace(
            data=text_amplitude.to_bytes(2, "little", signed=True) * audio_samples,
            sample_rate=48000, num_channels=1,
        )],
    )


class TestVoiceInput(unittest.IsolatedAsyncioTestCase):
    def test_initial_greeting_shares_interrupt_signal_with_session(self):
        shared_interrupt = asyncio.Event()
        with patch("livekit_service.session.VAD_MODEL.stream", return_value=object()):
            session = VoiceSession(
                SimpleNamespace(name="call-room"), asyncio.Event(), "user-1",
                interrupt_event=shared_interrupt,
            )
        self.assertIs(session.interrupt_event, shared_interrupt)
        session._handle_start_of_speech()
        self.assertTrue(shared_interrupt.is_set())

    def test_uses_vad_event_audio_and_rejects_short_or_quiet_segments(self):
        event = speech_event()
        self.assertEqual(_prepare_utterance(event), (bytes(event.frames[0].data), 48000))
        self.assertIsNone(_prepare_utterance(speech_event(duration=0.1)))
        self.assertIsNone(_prepare_utterance(speech_event(text_amplitude=0)))

    def test_punctuation_only_is_not_a_user_turn(self):
        self.assertFalse(_has_words("。 .!?"))
        self.assertTrue(_has_words("停。"))
        self.assertTrue(_has_words("The."))

    async def test_invalid_transcript_never_calls_agent(self):
        session = VoiceSession.__new__(VoiceSession)
        session.user_id = "fixture"
        session.interrupt_event = asyncio.Event()
        with patch("livekit_service.session.transcribe_audio", new_callable=AsyncMock,
                   return_value=".") as stt, patch("livekit_service.session.call_agent") as agent:
            await session._handle_end_of_speech(speech_event())
        stt.assert_awaited_once()
        agent.assert_not_called()

    async def test_vad_start_is_handled_while_response_is_pending(self):
        class EventStream:
            def __init__(self):
                self.queue = asyncio.Queue()

            async def __aiter__(self):
                while True:
                    yield await self.queue.get()

        session = VoiceSession.__new__(VoiceSession)
        session.vad_stream = EventStream()
        session.user_id = "fixture"
        session.end_call_countdown = SimpleNamespace(cancel=Mock())
        session.interrupt_event = asyncio.Event()
        session.current_response_task = None
        session.current_tts_task = None
        session.current_transition_task = None
        session._response_tasks = set()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def pending_response(_event):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        end_event = speech_event()
        start_event = SimpleNamespace(type=vad.VADEventType.START_OF_SPEECH)
        with patch.object(session, "_handle_end_of_speech", side_effect=pending_response), patch("livekit_service.session.get_voice_settings", AsyncMock(return_value=SimpleNamespace(stt_enabled=True))) as settings:
            loop_task = asyncio.create_task(session._vad_logic_loop())
            try:
                await session.vad_stream.queue.put(SimpleNamespace(type=vad.VADEventType.INFERENCE_DONE))
                await session.vad_stream.queue.put(end_event)
                await asyncio.wait_for(started.wait(), timeout=0.2)
                settings.assert_awaited_once_with("fixture")
                await session.vad_stream.queue.put(start_event)
                await asyncio.wait_for(cancelled.wait(), timeout=0.2)
                self.assertTrue(session.interrupt_event.is_set())
                self.assertTrue(session.is_speaking)
            finally:
                loop_task.cancel()
                await asyncio.gather(loop_task, return_exceptions=True)


class TestPlaybackInterrupt(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_discards_queued_tts_audio(self):
        from livekit_service.audio_playback import play_tts

        waiting = asyncio.Event()

        async def blocked_playout():
            waiting.set()
            await asyncio.Event().wait()

        source = SimpleNamespace(
            capture_frame=AsyncMock(), wait_for_playout=AsyncMock(side_effect=blocked_playout),
            clear_queue=Mock(),
        )
        participant = SimpleNamespace(
            publish_track=AsyncMock(return_value=SimpleNamespace(sid="track-1")),
            unpublish_track=AsyncMock(),
        )
        room = SimpleNamespace(local_participant=participant)

        async def pcm_chunks(*_args, **_kwargs):
            yield b"\x00\x01" * 240

        with patch("livekit_service.audio_playback.get_tts_stream", new=pcm_chunks), \
                patch("livekit_service.audio_playback._make_tts_source", return_value=source), \
                patch("livekit_service.audio_playback.rtc.LocalAudioTrack.create_audio_track",
                      return_value=object()):
            task = asyncio.create_task(play_tts(room, "正在说话", asyncio.Event(), user_id="fixture"))
            await asyncio.wait_for(waiting.wait(), timeout=0.2)
            task.cancel()
            self.assertFalse(await task)
        source.clear_queue.assert_called_once()
        participant.unpublish_track.assert_awaited_once_with("track-1")
