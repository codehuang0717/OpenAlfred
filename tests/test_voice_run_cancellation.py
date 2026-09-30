"""Voice interruptions must stop the exact server run and release its stream."""

import asyncio
import json
import unittest
import uuid
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from livekit_service.agent_client import call_agent, session_metadata_cache
from livekit_service.end_call import EndCallCountdown
from livekit_service.session import VoiceSession
from logic.voice_control import END_CALL_APPROVED


RUN_ID = "6778263b-3ce2-4bc2-8b5d-6d08a5ca05d7"
THREAD_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "call:cancel-room"))


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


class RunStream(httpx.AsyncByteStream):
    def __init__(self, chunks, failure=None, block=False):
        self.chunks = chunks
        self.failure = failure
        self.block = block
        self.waiting = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.failure:
            raise self.failure
        if self.block:
            self.waiting.set()
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


class TestVoiceRunCancellation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        session_metadata_cache["cancel-room"] = {
            "unique_session_id": "cancel-room", "is_fresh": False,
            "call_type": "inbound",
        }
        self.cancel_requests = []
        self.cancel_status = 200
        self.run_status = "running"
        self.header_run = True
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.addCleanup(session_metadata_cache.pop, "cancel-room", None)
        self.patches.enter_context(patch(
            "livekit_service.agent_client.mint_service_jwt", return_value="user-token",
        ))
        self.patches.enter_context(patch(
            "core.database.get_setting", new_callable=AsyncMock, return_value="gpt-cloud",
        ))
        self.client_type = httpx.AsyncClient
        self.patches.enter_context(patch(
            "livekit_service.agent_client.httpx.AsyncClient", side_effect=self._client,
        ))

    def _client(self):
        return self.client_type(transport=httpx.MockTransport(self._respond))

    async def _respond(self, request):
        if request.method == "GET":
            self.assertEqual(request.url.path, f"/threads/{THREAD_ID}/runs/{RUN_ID}")
            self.assertEqual(request.headers["Authorization"], "Bearer user-token")
            return httpx.Response(200, json={"status": self.run_status})
        if request.url.path.endswith("/cancel"):
            self.cancel_requests.append(request)
            return httpx.Response(self.cancel_status)
        payload = json.loads(request.content)
        self.assertEqual(payload["on_disconnect"], "cancel")
        self.assertEqual(payload["multitask_strategy"], "interrupt")
        self.assertEqual(payload["input"]["user_id"], "user-1")
        headers = {"content-type": "text/event-stream"}
        if self.header_run:
            headers["Content-Location"] = f"/threads/{THREAD_ID}/runs/{RUN_ID}"
        return httpx.Response(200, stream=self.stream, headers=headers)

    def assert_cancelled(self):
        self.assertEqual(len(self.cancel_requests), 1)
        request = self.cancel_requests[0]
        self.assertEqual(request.url.path, f"/threads/{THREAD_ID}/runs/{RUN_ID}/cancel")
        self.assertEqual(dict(request.url.params), {"action": "interrupt", "wait": "true"})
        self.assertEqual(request.headers["Authorization"], "Bearer user-token")
        self.assertTrue(self.stream.closed)

    async def test_task_interruption_cancels_run_before_first_sse_event(self):
        self.stream = RunStream([], block=True)
        task = asyncio.create_task(self._collect())
        await asyncio.wait_for(self.stream.waiting.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_cancelled()

    async def test_metadata_identifies_run_without_location_header(self):
        self.header_run = False
        self.stream = RunStream([sse("metadata", {"run_id": RUN_ID})], block=True)
        task = asyncio.create_task(self._collect())
        await asyncio.wait_for(self.stream.waiting.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_cancelled()

    async def test_closing_generator_at_tool_event_cancels_run(self):
        self.stream = RunStream([sse("updates", {"agent": {"messages": [{
            "type": "ai", "tool_calls": [{"id": "tool-1", "name": "create_task"}],
        }]}})], block=True)
        stream = call_agent("cancel-room", "安排任务", "user-1")
        self.assertEqual(await anext(stream), ("tool_call", "create_task"))
        await stream.aclose()
        self.assert_cancelled()

    async def test_network_failure_cancels_run(self):
        self.stream = RunStream([], failure=httpx.ReadError("connection dropped"))
        events = await self._collect()
        self.assertEqual(events, [("message", "抱歉，我暂时无法处理你的请求。")])
        self.assert_cancelled()

    async def test_server_error_event_is_not_treated_as_success(self):
        self.stream = RunStream([sse("error", {"error": "RuntimeError"})])
        events = await self._collect()
        self.assertEqual(events, [("message", "抱歉，我暂时无法处理你的请求。")])
        self.assert_cancelled()

    async def test_cancellation_failure_is_logged_and_does_not_swallow_interrupt(self):
        self.cancel_status = 503
        self.stream = RunStream([], block=True)
        task = asyncio.create_task(self._collect())
        await asyncio.wait_for(self.stream.waiting.wait(), timeout=1)
        with self.assertLogs("livekit-agent-client", level="ERROR") as logs:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertIn("could not confirm cancellation", logs.output[0])
        self.assert_cancelled()

    async def test_completed_run_is_not_cancelled(self):
        self.stream = RunStream([sse("updates", {"agent": {"messages": [{
            "type": "ai", "content": "安排好了", "tool_calls": [],
        }]}})])
        self.assertEqual(await self._collect(), [("message", "安排好了")])
        self.assertFalse(self.cancel_requests)
        self.assertTrue(self.stream.closed)

    async def test_disconnect_cancellation_race_confirms_terminal_run(self):
        self.cancel_status = 404
        self.run_status = "interrupted"
        self.stream = RunStream([], block=True)
        task = asyncio.create_task(self._collect())
        await asyncio.wait_for(self.stream.waiting.wait(), timeout=1)
        with self.assertLogs("livekit-agent-client", level="INFO") as logs:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertIn("server run already ended", logs.output[0])
        self.assert_cancelled()

    async def test_cancellation_404_is_not_hidden_if_run_is_still_running(self):
        self.cancel_status = 404
        self.stream = RunStream([], block=True)
        task = asyncio.create_task(self._collect())
        await asyncio.wait_for(self.stream.waiting.wait(), timeout=1)
        with self.assertLogs("livekit-agent-client", level="ERROR") as logs:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertIn("could not confirm cancellation", logs.output[0])
        self.assert_cancelled()

    async def test_goodbye_signal_does_not_detach_from_graph_tail(self):
        self.stream = RunStream([
            sse("updates", {"agent": {"messages": [{"type": "ai", "tool_calls": [
                {"name": "request_end_call", "id": "end-1"},
            ]}]}}),
            sse("updates", {"tools": {"messages": [{
                "name": "request_end_call", "tool_call_id": "end-1",
                "content": END_CALL_APPROVED,
            }]}}),
            sse("updates", {"agent": {"messages": [{
                "type": "ai", "content": "再见", "tool_calls": [],
            }]}}),
        ], block=True)
        stream = call_agent("cancel-room", "拜拜", "user-1")
        self.assertEqual(await anext(stream), ("message", "再见"))
        self.assertEqual(await anext(stream), ("end_call_requested", True))
        pending = asyncio.create_task(anext(stream))
        await asyncio.wait_for(self.stream.waiting.wait(), timeout=1)
        self.assertFalse(pending.done())
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assert_cancelled()

    async def _collect(self):
        return [event async for event in call_agent("cancel-room", "你好", "user-1")]


class TestVoiceSessionRunOwnership(unittest.IsolatedAsyncioTestCase):
    def make_session(self):
        with patch("livekit_service.session.VAD_MODEL.stream", return_value=SimpleNamespace(
            aclose=AsyncMock(), push_frame=Mock(),
        )):
            session = VoiceSession(SimpleNamespace(name="room"), asyncio.Event(),
                                   "user-1", is_sip=True)
        session.end_call_countdown.grace_seconds = 0.01
        return session

    async def test_session_shutdown_waits_for_already_cancelled_response_cleanup(self):
        session = self.make_session()
        audio_waiting = asyncio.Event()
        response_waiting = asyncio.Event()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        class AudioStream:
            aclose = AsyncMock()

            async def __aiter__(self):
                audio_waiting.set()
                await asyncio.Event().wait()
                yield None

        async def response():
            try:
                response_waiting.set()
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await allow_cleanup.wait()

        async def vad_loop():
            await asyncio.Event().wait()

        audio_stream = AudioStream()
        session.current_response_task = asyncio.create_task(response())
        session._response_tasks.add(session.current_response_task)
        with patch("livekit_service.session.rtc.AudioStream", return_value=audio_stream), \
                patch.object(session, "_vad_logic_loop", side_effect=vad_loop):
            task = asyncio.create_task(session.run(object()))
            try:
                await asyncio.wait_for(audio_waiting.wait(), timeout=1)
                await asyncio.wait_for(response_waiting.wait(), timeout=1)
                session._handle_start_of_speech()
                await asyncio.wait_for(cleanup_started.wait(), timeout=1)
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertFalse(session.current_response_task.done())
            finally:
                allow_cleanup.set()
                await asyncio.gather(task, session.current_response_task, return_exceptions=True)
        session.vad_stream.aclose.assert_awaited_once()
        audio_stream.aclose.assert_awaited_once()

    async def test_goodbye_countdown_does_not_wait_for_memory_extraction(self):
        session = self.make_session()
        tail_entered = asyncio.Event()
        closed = asyncio.Event()

        async def events(*args):
            try:
                yield "message", "再见"
                yield "end_call_requested", True
                tail_entered.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

        with patch("livekit_service.session._prepare_utterance", return_value=(b"audio", 48000)), \
                patch("livekit_service.session.transcribe_audio", new_callable=AsyncMock, return_value="拜拜"), \
                patch("livekit_service.session.call_agent", side_effect=events), \
                patch("livekit_service.session.play_tts", new_callable=AsyncMock, return_value=True):
            task = asyncio.create_task(session._handle_end_of_speech(object()))
            try:
                await asyncio.wait_for(tail_entered.wait(), timeout=1)
                await asyncio.wait_for(session.should_exit.wait(), timeout=1)
                self.assertFalse(task.done())
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(closed.is_set())

    async def test_interruption_during_goodbye_playback_closes_suspended_generator(self):
        session = self.make_session()
        playback_started = asyncio.Event()
        closed = asyncio.Event()

        async def events(*args):
            try:
                yield "message", "再见"
                yield "end_call_requested", True
                await asyncio.Event().wait()
            finally:
                closed.set()

        async def playback(*args, **kwargs):
            playback_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return False  # Match real play_tts cancellation behavior.

        with patch("livekit_service.session._prepare_utterance", return_value=(b"audio", 48000)), \
                patch("livekit_service.session.transcribe_audio", new_callable=AsyncMock, return_value="拜拜"), \
                patch("livekit_service.session.call_agent", side_effect=events), \
                patch("livekit_service.session.play_tts", side_effect=playback):
            session.current_response_task = asyncio.create_task(
                session._handle_end_of_speech(object()),
            )
            await asyncio.wait_for(playback_started.wait(), timeout=1)
            session._handle_start_of_speech()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(session.current_response_task, timeout=1)
        self.assertTrue(closed.is_set())
        self.assertIsNone(session.end_call_countdown.task)
