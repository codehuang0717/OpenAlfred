"""Voice lifecycle, tenant isolation and STT privacy regressions."""

import asyncio
import io
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from db import connection
from schemas.voice import VoiceSettings
from services import voice_store, voice_runtime


def recording(seconds: float = 2, silent: bool = False) -> bytes:
    output = io.BytesIO()
    samples = np.zeros(int(seconds * 16000)) if silent else np.sin(np.arange(int(seconds * 16000)) * 0.05) * 0.1
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
        audio.writeframes((samples * 32767).astype("<i2").tobytes())
    return output.getvalue()


class TestVoiceStorage(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [patch.object(connection, "DATABASE_PATH", str(self.root / "db.sqlite")), patch.object(voice_store, "VOICE_DATA", self.root / "voice")]
        for item in self.patches: item.start()
        await connection.init_db()

    async def asyncTearDown(self):
        for item in reversed(self.patches): item.stop()
        self.temp.cleanup()

    async def test_reference_and_settings_are_owned_by_one_account(self):
        profile = await voice_store.create_profile("alice", "我的声音", "测试参考文本", recording())
        saved, path = await voice_store.get_profile("alice", profile.id)
        self.assertTrue(path.is_file())
        self.assertEqual(saved.transcript, "测试参考文本")
        with self.assertRaises(ValueError):
            await voice_store.get_profile("bob", profile.id)
        settings = VoiceSettings(profile_id=profile.id, stt_enabled=False)
        await voice_store.save_voice_settings("alice", settings)
        self.assertFalse((await voice_store.get_voice_settings("alice")).stt_enabled)
        self.assertTrue((await voice_store.get_voice_settings("bob")).stt_enabled)
        with self.assertRaises(ValueError):
            await voice_store.save_voice_settings("bob", settings)
        await voice_store.delete_profile("alice", profile.id)
        self.assertIsNone((await voice_store.get_voice_settings("alice")).profile_id)
        self.assertFalse(path.exists())

    async def test_invalid_audio_is_rejected_before_persistence(self):
        for content in [b"not audio", recording(0.1), recording(silent=True)]:
            with self.assertRaises(ValueError):
                await voice_store.create_profile("alice", "voice", "text", content)
        self.assertFalse(voice_store.user_directory("alice").exists())

    async def test_path_traversal_cannot_read_another_file(self):
        with self.assertRaises(ValueError):
            await voice_store.get_profile("alice", "../../db.sqlite")

    async def test_disabled_stt_never_sends_recording_to_the_service(self):
        from livekit_service.stt import transcribe_audio
        from routers.multimodal import transcribe_audio_api
        from fastapi import HTTPException
        await voice_store.save_voice_settings("alice", VoiceSettings(profile_id=None, stt_enabled=False))
        with patch("livekit_service.stt.httpx.AsyncClient") as client:
            self.assertEqual(await transcribe_audio(b"recording", 16000, 1, "alice"), "")
            client.assert_not_called()
        file = MagicMock()
        file.read = AsyncMock()
        with self.assertRaises(HTTPException) as caught:
            await transcribe_audio_api(file, {"id": "alice"})
        self.assertEqual(caught.exception.status_code, 409)
        file.read.assert_not_awaited()

    async def test_missing_tts_account_never_calls_another_engine(self):
        from services.tts import get_tts_stream
        from utils.auth_utils import MissingUserContextError
        with patch("services.tts.httpx.AsyncClient") as client:
            with self.assertRaises(MissingUserContextError):
                await anext(get_tts_stream("hello", user_id=""))
            client.assert_not_called()

    async def test_call_greeting_uses_the_account_voice(self):
        from livekit_service.audio_playback import play_greeting
        await voice_store.save_voice_settings("alice", VoiceSettings(profile_id=None))
        with patch("livekit_service.audio_playback.os.path.exists", return_value=False), patch("livekit_service.audio_playback.play_tts", AsyncMock()) as play:
            room = SimpleNamespace(name="inbound-fixture")
            interrupt = asyncio.Event()
            await play_greeting(room, "你好", interrupt, user_id="alice")
            play.assert_awaited_once_with(room, "你好", interrupt, user_id="alice")

    async def test_greeting_cache_is_versioned_and_preserves_custom_words(self):
        from services.voice_cache import cached_voice, render_cached_voice, saved_voice
        from livekit_service.audio_playback import play_greeting
        settings = VoiceSettings(engine="voxcpm", profile_id=None)
        await voice_store.save_voice_settings("alice", settings)

        async def pcm(*_args, **_kwargs):
            yield b"\x00\x01" * 24000

        alias = self.root / "reminder.wav"
        with patch("services.tts.get_tts_stream", side_effect=pcm) as generate:
            original = await render_cached_voice("alice", "具体提醒话术", alias)
            await render_cached_voice("alice", "具体提醒话术")
            self.assertEqual(generate.call_count, 1)
        self.assertEqual(await cached_voice("alice", "具体提醒话术"), original)
        self.assertIsNone(await cached_voice("bob", "具体提醒话术"))
        self.assertIsNone(await saved_voice("bob", alias))
        settings.stt_enabled = False
        await voice_store.save_voice_settings("alice", settings)
        self.assertEqual(await cached_voice("alice", "具体提醒话术"), original)
        settings.vox.cfg_value = 3
        await voice_store.save_voice_settings("alice", settings)
        self.assertIsNone(await cached_voice("alice", "具体提醒话术"))
        self.assertEqual(await saved_voice("alice", alias), ("具体提醒话术", None))
        with patch("services.voice_cache.saved_voice", AsyncMock(return_value=("具体提醒话术", None))), patch("livekit_service.audio_playback.play_tts", AsyncMock()) as play:
            room = SimpleNamespace(name="outbound-reminder-" + "a" * 36)
            await play_greeting(room, "其他文本", user_id="alice")
            self.assertEqual(play.call_args.args[1], "具体提醒话术")

    async def test_changed_voice_during_generation_never_publishes_old_cache(self):
        from services.voice_cache import descriptor, render_cached_voice
        settings = VoiceSettings(engine="voxcpm", profile_id=None)
        await voice_store.save_voice_settings("alice", settings)
        _key, expected = await descriptor("alice", "你好")

        async def pcm(*_args, **_kwargs):
            settings.vox.cfg_value = 3
            await voice_store.save_voice_settings("alice", settings)
            yield b"\x00\x01" * 24000

        with patch("services.tts.get_tts_stream", side_effect=pcm):
            with self.assertRaisesRegex(RuntimeError, "已改变"):
                await render_cached_voice("alice", "你好")
        self.assertFalse(expected.exists())

    async def test_cached_greeting_plays_without_inference_and_obeys_tts_switch(self):
        from services.voice_cache import render_cached_voice
        from livekit_service.audio_playback import play_greeting
        settings = VoiceSettings(engine="voxcpm", profile_id=None)
        await voice_store.save_voice_settings("alice", settings)

        async def pcm(*_args, **_kwargs):
            yield b"\x00\x01" * 480

        with patch("services.tts.get_tts_stream", side_effect=pcm):
            await render_cached_voice("alice", settings.greeting_text)
        source = SimpleNamespace(capture_frame=AsyncMock(), wait_for_playout=AsyncMock(), clear_queue=MagicMock())
        participant = SimpleNamespace(publish_track=AsyncMock(return_value=SimpleNamespace(sid="greeting")), unpublish_track=AsyncMock())
        room = SimpleNamespace(name="inbound-test", local_participant=participant)
        with patch("livekit_service.audio_playback._make_tts_source", return_value=source), patch("livekit_service.audio_playback.rtc.LocalAudioTrack.create_audio_track", return_value=object()), patch("livekit_service.audio_playback.play_tts", AsyncMock()) as inference:
            await play_greeting(room, user_id="alice")
            inference.assert_not_awaited()
            source.wait_for_playout.assert_awaited_once()
            self.assertEqual(source.capture_frame.await_count, 1)
            settings.tts_enabled = False
            await voice_store.save_voice_settings("alice", settings)
            await play_greeting(room, user_id="alice")
            self.assertEqual(participant.publish_track.await_count, 1)

    def test_unsupported_or_unbounded_settings_are_rejected(self):
        for values in [{"engine": "shell"}, {"qwen_model": "../../file"}, {"vox": {"inference_timesteps": 100}}, {"qwen": {"temperature": -1}}, {"speed": 2}]:
            with self.assertRaises(ValidationError):
                VoiceSettings.model_validate(values)

    async def test_public_api_upload_select_preview_and_cross_account_access(self):
        import httpx
        from fastapi import FastAPI
        from routers import voice
        from routers.auth import get_current_user

        app = FastAPI()
        app.include_router(voice.router)
        owner = {"id": "alice"}
        app.dependency_overrides[get_current_user] = lambda: owner

        async def audio(user_id, text):
            self.assertEqual(user_id, "alice")
            self.assertEqual(text, "试听文本")
            yield b"\x00\x01" * 24000

        with patch.object(voice_store, "_builtin", return_value=None), patch.object(voice.event_bus, "publish", AsyncMock()), patch.object(voice, "engine_stream", side_effect=audio):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                result = await client.post("/api/voice/profiles", data={"name": "clone", "transcript": "准确参考文本"}, files={"file": ("reference.wav", recording(), "audio/wav")})
                self.assertEqual(result.status_code, 201)
                profile_id = result.json()["id"]
                settings = VoiceSettings(profile_id=profile_id)
                result = await client.put("/api/voice/config", json=settings.model_dump())
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()["settings"]["profile_id"], profile_id)
                result = await client.post("/api/voice/preview", json={"text": "试听文本"})
                self.assertEqual(result.status_code, 200)
                with wave.open(io.BytesIO(result.content)) as output:
                    self.assertEqual(output.getframerate(), 24000)
                    self.assertEqual(output.getnframes(), 24000)
                owner["id"] = "bob"
                result = await client.get(f"/api/voice/profiles/{profile_id}/audio")
                self.assertEqual(result.status_code, 404)
                result = await client.put("/api/voice/config", json=settings.model_dump())
                self.assertEqual(result.status_code, 422)

    async def test_inference_worker_requires_a_per_launch_token(self):
        import httpx
        from services.voice_engine import create_app
        app = create_app({"token": "fixture-token", "engine": "voxcpm", "model": "VoxCPM2"})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            self.assertEqual((await client.get("/health")).status_code, 401)
            self.assertEqual((await client.post("/speech", json={"text": "hello"})).status_code, 401)
            self.assertEqual((await client.get("/health", headers={"X-Engine-Token": "fixture-token"})).status_code, 200)

    async def test_cancelled_response_does_not_overlap_gpu_inference(self):
        from services.voice_engine import EngineSpeech, create_app

        finish = threading.Event()
        started = threading.Event()
        calls = []

        def generate(**_kwargs):
            calls.append(True)
            yield np.ones(240, dtype=np.float32) * 0.1
            started.set()
            finish.wait(5)

        model = SimpleNamespace(generate_streaming=generate, tts_model=SimpleNamespace(sample_rate=24000))
        vendor = SimpleNamespace(VoxCPM=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: model))
        app = create_app({"token": "test", "engine": "voxcpm", "model": "VoxCPM2", "model_path": "fixture"})
        speech = next(route.endpoint for route in app.routes if route.path == "/speech")
        with patch.dict(sys.modules, {"voxcpm": vendor}):
            async with app.router.lifespan_context(app):
                try:
                    response = await speech(EngineSpeech(text="first"))
                    await anext(response.body_iterator)
                    self.assertTrue(await asyncio.to_thread(started.wait, 1))
                    await response.body_iterator.aclose()
                    waiting = asyncio.create_task(speech(EngineSpeech(text="second")))
                    await asyncio.sleep(0.05)
                    self.assertFalse(waiting.done())
                    self.assertEqual(len(calls), 1)
                    finish.set()
                    second = await asyncio.wait_for(waiting, 2)
                    await anext(second.body_iterator)
                    await second.body_iterator.aclose()
                finally:
                    finish.set()


class TestVoiceLifecycle(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patch = patch.object(voice_runtime, "VOICE_DATA", self.root)
        self.patch.start()
        executable = self.root / ".venv/Scripts/python.exe"
        executable.parent.mkdir(parents=True)
        executable.touch()
        self.project_patch = patch.object(voice_runtime.config, "VOXCPM_PROJECT", self.root)
        self.project_patch.start()
        self.manager = voice_runtime.VoiceManager()

    async def asyncTearDown(self):
        await self.manager.close()
        self.patch.stop()
        self.project_patch.stop()
        self.temp.cleanup()

    async def test_cancelling_a_loading_model_cleans_up_and_allows_restart(self):
        settings = VoiceSettings(engine="voxcpm")
        pending = asyncio.Event()

        async def startup(_settings, _path):
            try:
                await pending.wait()
            except asyncio.CancelledError:
                self.manager.state = "stopped"
                raise

        with patch.object(voice_runtime, "model_path", return_value=self.root), patch.object(self.manager, "_start", side_effect=startup):
            result = await self.manager.control("start", settings)
            self.assertEqual(result.state, "starting")
            await asyncio.sleep(0)
            result = await self.manager.control("stop", settings)
            self.assertEqual(result.state, "stopped")
            self.assertTrue(self.manager.transition.cancelled())
            result = await self.manager.control("start", settings)
            self.assertEqual(result.state, "starting")

    async def test_invalid_replacement_does_not_stop_a_running_model(self):
        settings = VoiceSettings(engine="voxcpm")
        self.manager.state = "running"
        self.manager.signature = voice_runtime.launch_signature(settings)
        with patch.object(voice_runtime, "model_path", side_effect=ValueError("Missing model")), patch.object(self.manager, "_stop", AsyncMock()) as stop:
            with self.assertRaises(ValueError):
                await self.manager.control("start", VoiceSettings(engine="fasterqwentts"))
            stop.assert_not_awaited()
        self.assertEqual(self.manager.state, "running")

    def test_changed_context_length_requires_model_reload(self):
        settings = VoiceSettings()
        self.manager.signature = voice_runtime.launch_signature(settings)
        self.manager.state = "running"
        settings.qwen.max_seq_len = 4096
        self.assertTrue(self.manager.status(settings).restart_required)

    def test_pid_reuse_never_terminates_an_unrelated_process(self):
        process = MagicMock()
        process.create_time.return_value = 99
        with patch.object(voice_runtime.psutil, "Process", return_value=process):
            voice_runtime._terminate(123, 88)
        process.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
