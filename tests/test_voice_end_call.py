"""A requested hangup must wait for speech and remain cancellable."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from typing import Annotated, TypedDict
from unittest.mock import AsyncMock, patch

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from livekit_service.agent_client import call_agent, session_metadata_cache
from livekit_service.end_call import EndCallCountdown
from logic.nodes import selected_context_tools
from logic.prompts import CALL_ENDING_INSTRUCTION
from tools.end_call import END_CALL_APPROVED, request_end_call
from utils.auth_utils import MissingUserContextError


class TestEndCallCountdown(unittest.IsolatedAsyncioTestCase):
    async def test_exits_after_grace_period(self):
        should_exit = asyncio.Event()
        countdown = EndCallCountdown(should_exit, lambda: False, grace_seconds=0.01)
        countdown.arm()
        await asyncio.wait_for(should_exit.wait(), timeout=0.2)
        self.assertIsNone(countdown.task)

    async def test_follow_up_cancels_hangup(self):
        should_exit = asyncio.Event()
        countdown = EndCallCountdown(should_exit, lambda: False, grace_seconds=0.02)
        countdown.arm()
        countdown.cancel()
        await asyncio.sleep(0.04)
        self.assertFalse(should_exit.is_set())

    async def test_active_speech_prevents_hangup(self):
        should_exit = asyncio.Event()
        speaking = True
        countdown = EndCallCountdown(should_exit, lambda: speaking, grace_seconds=0.01)
        countdown.arm()
        self.assertIsNone(countdown.task)
        speaking = False
        countdown.arm()
        speaking = True
        await asyncio.sleep(0.03)
        self.assertFalse(should_exit.is_set())


class TestEndCallTool(unittest.IsolatedAsyncioTestCase):
    def test_prompt_uses_tool_without_legacy_control_marker(self):
        self.assertIn("request_end_call", CALL_ENDING_INSTRUCTION)
        self.assertNotIn("[TERMINATE]", CALL_ENDING_INSTRUCTION)
        self.assertNotIn("控制标记", CALL_ENDING_INSTRUCTION)

    async def test_only_sip_voice_calls_can_request_end(self):
        voice = SimpleNamespace(config={"configurable": {
            "channel": "voice", "call_type": "inbound", "owner": "user-1",
        }})
        self.assertEqual(await request_end_call.coroutine(voice), END_CALL_APPROVED)
        for channel, call_type in (("chat", "inbound"), ("voice", "local")):
            with self.subTest(channel=channel, call_type=call_type):
                other = SimpleNamespace(config={"configurable": {
                    "channel": channel, "call_type": call_type, "owner": "user-1",
                }})
                with self.assertRaises(PermissionError):
                    await request_end_call.coroutine(other)

    async def test_missing_user_is_rejected(self):
        voice = SimpleNamespace(config={"configurable": {
            "channel": "voice", "call_type": "outbound",
        }})
        with self.assertRaises(MissingUserContextError):
            await request_end_call.coroutine(voice)

    async def test_tool_is_only_offered_for_phone_calls(self):
        for channel, call_type, expected in (
            ("voice", "inbound", True),
            ("voice", "outbound", True),
            ("voice", "local", False),
            ("chat", "inbound", False),
        ):
            with self.subTest(channel=channel, call_type=call_type):
                names = {tool.name for tool in selected_context_tools({
                    "configurable": {"channel": channel, "call_type": call_type}
                })}
                self.assertEqual("request_end_call" in names, expected)

    async def test_tool_node_emits_named_approval_result(self):
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        graph = StateGraph(State)
        graph.add_node("tools", ToolNode([request_end_call]))
        graph.set_entry_point("tools")
        graph.add_edge("tools", END)
        result = await graph.compile().ainvoke({"messages": [AIMessage(
            content="", tool_calls=[{"name": "request_end_call", "args": {}, "id": "call-1"}],
        )]}, config={"configurable": {"channel": "voice", "call_type": "inbound", "owner": "user-1"}})
        observation = next(message for message in result["messages"] if isinstance(message, ToolMessage))
        self.assertEqual((observation.name, observation.content, observation.tool_call_id),
                         ("request_end_call", END_CALL_APPROVED, "call-1"))


class TestVoiceAgentEndSignal(unittest.IsolatedAsyncioTestCase):
    async def _events(self, tool_result: str, include_goodbye: bool = True):
        events = [
            {"agent": {"messages": [{"type": "ai", "content": "", "tool_calls": [
                {"id": "call-1", "name": "request_end_call", "args": {}}
            ]}]}},
            {"tools": {"messages": [{"type": "tool", "name": "request_end_call",
                "tool_call_id": "call-1", "content": tool_result}]}},
        ]
        if include_goodbye:
            events.append({"agent": {"messages": [{"type": "ai", "content": "好，回头见。",
                "tool_calls": []}]}})
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

        async def respond(_request):
            self.assertEqual(json.loads(_request.content)["on_disconnect"], "cancel")
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        session_metadata_cache["test-room"] = {
            "call_type": "inbound", "is_fresh": False, "unique_session_id": "test-room",
        }
        try:
            with patch("livekit_service.agent_client.httpx.AsyncClient", return_value=client), \
                    patch("livekit_service.agent_client.mint_service_jwt", return_value="test-token"), \
                    patch("core.database.get_setting", new_callable=AsyncMock, return_value="gpt-cloud"):
                return [event async for event in call_agent("test-room", "拜拜", "user-1")]
        finally:
            session_metadata_cache.pop("test-room", None)

    async def test_approved_tool_and_spoken_goodbye_request_hangup(self):
        self.assertEqual(await self._events(END_CALL_APPROVED), [
            ("message", "好，回头见。"), ("end_call_requested", True),
        ])

    async def test_failed_tool_does_not_request_hangup(self):
        self.assertEqual(await self._events("工具执行失败"), [
            ("message", "好，回头见。"),
        ])

    async def test_missing_goodbye_does_not_request_hangup(self):
        self.assertEqual(await self._events(END_CALL_APPROVED, include_goodbye=False), [
            ("message", "抱歉，我暂时无法结束通话。"),
        ])


class TestVoiceSessionHangup(unittest.IsolatedAsyncioTestCase):
    async def _respond(self, playback_succeeded: bool):
        from livekit_service.session import VoiceSession

        session = VoiceSession.__new__(VoiceSession)
        session.room = object()
        session.session_id = "test-room"
        session.user_id = "user-1"
        session.is_sip = True
        session.should_exit = asyncio.Event()
        session.interrupt_event = asyncio.Event()
        session.current_tts_task = None
        session.current_transition_task = None
        session.current_response_task = None
        session.is_speaking = False
        session.is_agent_processing = False
        session.end_call_countdown = EndCallCountdown(
            session.should_exit, lambda: session.is_speaking, grace_seconds=0.02,
        )

        async def agent_events(*_args):
            yield "message", "好，回头见。"
            yield "end_call_requested", True

        with patch("livekit_service.session.transcribe_audio", new_callable=AsyncMock, return_value="拜拜"), \
                patch("livekit_service.session.call_agent", side_effect=agent_events), \
                patch("livekit_service.session.play_tts", new_callable=AsyncMock, return_value=playback_succeeded), \
                patch.object(session, "_log_latency_summary"):
            await session._handle_end_of_speech(SimpleNamespace(
                speech_duration=0.0, silence_duration=0.5,
                frames=[SimpleNamespace(data=b"\x10\x27" * 62400,
                                        sample_rate=48000, num_channels=1)],
            ))
        return session

    async def test_goodbye_played_then_hangs_up(self):
        session = await self._respond(True)
        self.assertFalse(session.should_exit.is_set())
        await asyncio.wait_for(session.should_exit.wait(), timeout=0.2)

    async def test_follow_up_cancels_pending_hangup(self):
        session = await self._respond(True)
        session._handle_start_of_speech()
        await asyncio.sleep(0.04)
        self.assertFalse(session.should_exit.is_set())

    async def test_failed_playout_does_not_hang_up(self):
        session = await self._respond(False)
        await asyncio.sleep(0.04)
        self.assertFalse(session.should_exit.is_set())


class TestTtsCompletion(unittest.IsolatedAsyncioTestCase):
    async def test_empty_success_response_does_not_emit_fake_speech(self):
        from services.tts import get_tts_stream

        client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=b""),
        ))
        with patch("services.tts.httpx.AsyncClient", return_value=client):
            chunks = [chunk async for chunk in get_tts_stream("再见")]
        self.assertEqual(chunks, [])

    async def test_reports_success_only_after_source_queue_drains(self):
        from livekit_service.audio_playback import play_tts

        source = SimpleNamespace(capture_frame=AsyncMock(), wait_for_playout=AsyncMock())
        participant = SimpleNamespace(
            publish_track=AsyncMock(return_value=SimpleNamespace(sid="track-1")),
            unpublish_track=AsyncMock(),
        )
        room = SimpleNamespace(local_participant=participant)

        async def pcm_chunks(*_args, **_kwargs):
            yield b"\x00\x01" * 240

        with patch("livekit_service.audio_playback.get_tts_stream", new=pcm_chunks), \
                patch("livekit_service.audio_playback._make_tts_source", return_value=source), \
                patch("livekit_service.audio_playback.rtc.LocalAudioTrack.create_audio_track", return_value=object()):
            completed = await play_tts(room, "再见", asyncio.Event())
        self.assertTrue(completed)
        source.wait_for_playout.assert_awaited_once()
        participant.unpublish_track.assert_awaited_once_with("track-1")
