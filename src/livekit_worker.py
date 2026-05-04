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
import os
import sys
from core.config import config

# Force local mode if --local flag is present
if "--local" in sys.argv:
    os.environ["LIVEKIT_URL"] = "ws://localhost:7880"
    if "--url" not in sys.argv:
        sys.argv.extend(["--url", "ws://localhost:7880"])
    sys.argv.remove("--local")

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit import rtc
from core.database import init_db
from utils.logger import setup_logging, get_logger

# Import decoupled components
from livekit_service.agent_client import _ensure_thread, session_metadata_cache
from livekit_service.audio_playback import play_greeting
from livekit_service.session import VoiceSession

# Initialize unified logging
setup_logging(log_file="livekit.log")
logger = get_logger("livekit-worker")

# Silence noisy third-party loggers
logging.getLogger("aiosqlite").setLevel(logging.WARNING)

db_initialized = False


def _extract_caller_extension(attrs: dict, identity: str, room_name: str) -> str | None:
    """Extract the caller's extension number. Tries multiple sources."""

    # 1. Room name: LiveKit SIP creates rooms like _101_xxx from dispatch rule
    import re
    m = re.match(r"^_(\d+)_", room_name)
    if m:
        ext = m.group(1)
        logger.info(f"[sip-extract] ext={ext} from room_name={room_name!r}")
        return ext

    # 2. sip.callerId (works when Asterisk does NOT spoof CALLERID)
    raw = attrs.get("sip.callerId", "")
    if raw:
        ext = raw.replace("sip:", "").split("@")[0].strip()
        if ext and ext != "+1234567890":  # skip spoofed caller ID
            logger.info(f"[sip-extract] ext={ext} from sip.callerId={raw!r}")
            return ext

    # 3. sip.from header
    raw = attrs.get("sip.from", "")
    if raw:
        m = re.search(r"sip:(\d+)@", raw)
        if m:
            ext = m.group(1)
            logger.info(f"[sip-extract] ext={ext} from sip.from={raw!r}")
            return ext

    # 4. participant identity (e.g., sip_101)
    if identity.startswith("sip_"):
        num = identity[4:]
        if num.isdigit():
            logger.info(f"[sip-extract] ext={num} from identity={identity!r}")
            return num

    logger.warning(f"[sip-extract] FAILED room={room_name!r} attrs={attrs} identity={identity}")
    return None


async def entrypoint(ctx: JobContext):
    global db_initialized
    if not db_initialized:
        await init_db()
        db_initialized = True

    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)

    room = ctx.room
    should_exit = asyncio.Event()

    # ── Call type ──────────────────────────────────────────────────
    if room.name.startswith("outbound-"):
        call_type = "outbound"
    elif room.name.startswith("local-"):
        call_type = "local"
    else:
        call_type = "inbound"

    logger.info(f"[entrypoint] call_type={call_type} room={room.name}")

    unique_session_id = room.name
    if call_type in ("inbound", "local"):
        unique_session_id = f"{room.name}-{ctx.job.id}"

    # ── User resolution ────────────────────────────────────────────
    user_id: str | None = None
    thread_info: dict = {}
    initial_speech = ""

    if call_type == "outbound":
        # Outbound: user_id embedded in room name by dial_user()
        if room.name.startswith("outbound-reminder-"):
            user_id = room.name[18 + 36 + 1:] or "default"
        elif room.name.startswith("outbound-supervisor-"):
            parts = room.name.split("-")
            user_id = "-".join(parts[3:]) if len(parts) >= 4 else "default"
        else:
            parts = room.name.split("-")
            user_id = "-".join(parts[1:-1]) if len(parts) >= 3 else "default"
        logger.info(f"[outbound] user_id={user_id}")

    elif call_type == "local":
        # Local: use active user (ear service wakes local machine)
        try:
            from core.database import get_active_user
            active = await get_active_user()
            if active:
                user_id = active["id"]
                logger.info(f"[local] user_id={user_id} username={active.get('username')}")
        except Exception as e:
            logger.error(f"[local] resolve failed: {e}")

    else:
        # Inbound: DO NOT pre-assign — resolve from SIP participant
        user_id = None
        logger.info("[inbound] waiting for SIP participant to resolve user...")

    # ── Thread creation (deferred for inbound) ─────────────────────
    call_title = {
        "outbound": "语音外拨呼叫",
        "local": "本地语音唤醒",
        "inbound": "语音呼入接待",
    }[call_type]

    if call_type != "inbound":
        thread_info = await _ensure_thread(unique_session_id, user_id or "default", call_title, call_type)
        initial_speech = thread_info.get("initial_speech", "")

    session_metadata_cache[room.name] = {
        "call_type": call_type,
        "initial_speech": initial_speech,
        "is_fresh": True,
        "unique_session_id": unique_session_id,
    }

    # ── Inbound user resolver (called when SIP participant connects) ──
    inbound_resolved = asyncio.Event()

    async def resolve_inbound_user(p: rtc.RemoteParticipant):
        """Awaitable: extract caller extension, look up user, set user_id."""
        nonlocal user_id, thread_info, initial_speech

        if call_type != "inbound":
            return

        attrs = dict(p.attributes)
        logger.info(
            f"[inbound-resolve] identity={p.identity} "
            f"sip.from={attrs.get('sip.from', 'N/A')} "
            f"sip.callerId={attrs.get('sip.callerId', 'N/A')}"
        )

        caller_ext = _extract_caller_extension(attrs, p.identity, room.name)

        if not caller_ext:
            logger.error(
                f"[inbound-resolve] FAILED to extract extension — "
                f"cannot identify caller. attrs={attrs}"
            )
            inbound_resolved.set()
            return

        # Strict lookup: NO fallback to active_user
        try:
            from core.database import get_user_by_sip_extension
            sip_user = await get_user_by_sip_extension(caller_ext)
            if sip_user:
                user_id = sip_user["id"]
                logger.info(
                    f"[inbound-resolve] ext={caller_ext} → "
                    f"username={sip_user['username']} id={user_id}"
                )

                # Now create the thread with correct owner
                thread_info = await _ensure_thread(
                    unique_session_id, user_id, call_title, call_type
                )
                initial_speech = thread_info.get("initial_speech", "")

                # Update session metadata with correct user
                session_metadata_cache[room.name] = {
                    "call_type": call_type,
                    "initial_speech": initial_speech,
                    "is_fresh": True,
                    "unique_session_id": unique_session_id,
                }

                logger.info(f"[inbound-resolve] thread created for user={user_id}")
            else:
                logger.error(
                    f"[inbound-resolve] extension={caller_ext} NOT FOUND in DB — "
                    f"caller cannot be identified. No fallback."
                )
        except Exception as e:
            logger.error(f"[inbound-resolve] lookup error: {e}")
        finally:
            inbound_resolved.set()

    # ── State ──────────────────────────────────────────────────────
    answered_event = asyncio.Event()
    greeting_event = asyncio.Event()  # outbound: only set on callTag (real answer)
    greeting_played = False
    greeting_lock = asyncio.Lock()
    active_sessions: dict = {}

    async def trigger_greeting():
        nonlocal greeting_played
        async with greeting_lock:
            if greeting_played:
                return
            greeting_played = True
            logger.info(f"[greeting] call_type={call_type} session={unique_session_id}")

            try:
                if call_type == "local":
                    return

                current_speech = initial_speech
                session_obj = next(iter(active_sessions.values()), None) if active_sessions else None
                if session_obj and hasattr(session_obj, 'is_greeting_playing'):
                    session_obj.is_greeting_playing = True

                try:
                    await play_greeting(room, current_speech, should_exit,
                                       user_id or "default", call_type)
                finally:
                    if session_obj and hasattr(session_obj, 'is_greeting_playing'):
                        session_obj.is_greeting_playing = False
            except Exception as e:
                logger.error(f"[greeting] error: {e}")

    def make_voice_session(participant):
        """Create a VoiceSession with the CURRENT user_id."""
        is_sip = participant.identity.startswith("sip_")
        return VoiceSession(room, should_exit, user_id or "default",
                           is_sip=is_sip, answered_event=answered_event)

    # ── Room event handlers ────────────────────────────────────────

    async def subscribe_to_audio(p: rtc.RemoteParticipant):
        for pub in p.track_publications.values():
            if pub.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"[audio] subscribing to track {pub.sid} from {p.identity}")
                pub.set_subscribed(True)

    @room.on("participant_connected")
    def on_participant_connected(p: rtc.RemoteParticipant):
        attrs = dict(p.attributes)
        logger.info(
            f"[participant] connected identity={p.identity} "
            f"attrs={json.dumps(attrs, ensure_ascii=False, default=str)}"
        )

        if call_type == "inbound":
            # AWAIT resolution — must finish before audio/track handling
            asyncio.create_task(_on_inbound_participant(p))
        else:
            asyncio.create_task(subscribe_to_audio(p))

    async def _on_inbound_participant(p: rtc.RemoteParticipant):
        """Inbound participant lifecycle: resolve user → subscribe audio."""
        await resolve_inbound_user(p)
        await subscribe_to_audio(p)

    @room.on("participant_attributes_changed")
    def on_attributes_changed(changed_attributes: dict, participant: rtc.RemoteParticipant):
        logger.info(
            f"[sip-attrs] participant={participant.identity} "
            f"changed={json.dumps(changed_attributes, ensure_ascii=False, default=str)}"
        )
        # For outbound calls: sip.callStatus may become "active" due to
        # trunk/Asterisk early-answer BEFORE the human picks up. The true
        # user-answer signal is sip.callTag appearing — this is the far-end
        # UAS To-tag generated in the final 200 OK when the callee answers.
        if call_type == "outbound" and "sip.callTag" in changed_attributes:
            logger.info(f"[sip] USER ANSWERED (callTag): {participant.attributes.get('sip.callTag')}")
            greeting_event.set()
        elif "sip.callStatus" in changed_attributes:
            new_status = participant.attributes.get("sip.callStatus")
            logger.info(f"[sip] callStatus={new_status} from {participant.identity}")
            if new_status == "active":
                answered_event.set()
                if call_type in ("inbound", "local"):
                    asyncio.create_task(trigger_greeting())

    @room.on("participant_disconnected")
    def on_participant_disconnected(p):
        if not room.remote_participants:
            should_exit.set()

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if isinstance(track, rtc.AudioTrack):
            if participant.identity in active_sessions:
                return

            session = make_voice_session(participant)
            task = asyncio.create_task(session.run(track))
            active_sessions[participant.identity] = session

            def on_task_done(t):
                active_sessions.pop(participant.identity, None)
            task.add_done_callback(on_task_done)

            if call_type in ("inbound", "local"):
                answered_event.set()
                asyncio.create_task(trigger_greeting())
            # outbound: do NOT set answered_event here.
            # We must wait for sip.callStatus to become "active"
            # via on_attributes_changed. Otherwise greeting plays
            # while the phone is still ringing.

    # ── Initial scan (for participants already in the room) ────────
    for participant in room.remote_participants.values():
        if call_type == "inbound":
            asyncio.create_task(_on_inbound_participant(participant))
        else:
            asyncio.create_task(subscribe_to_audio(participant))

        # Log full participant state on initial scan
        logger.info(
            f"[sip-scan] participant={participant.identity} "
            f"attrs={json.dumps(dict(participant.attributes), ensure_ascii=False, default=str)}"
        )

        if participant.attributes.get("sip.callStatus") == "active":
            answered_event.set()
            if call_type == "local":
                asyncio.create_task(trigger_greeting())

    if call_type == "outbound":
        # For outbound, ONLY sip.callTag (far-end UAS answer) triggers
        # greeting. sip.callStatus="active" is the trunk answer and is
        # deliberately ignored for outbound.
        try:
            await asyncio.wait_for(greeting_event.wait(), timeout=30)
        except asyncio.TimeoutError:
            logger.warning("[outbound] greeting_event timed out after 30s")
            should_exit.set()
            return
        logger.info("[outbound] greeting_event set, playing greeting")
        await asyncio.sleep(0.3)
        await trigger_greeting()
    else:
        asyncio.create_task(trigger_greeting())

    # ── Wait for call end ──────────────────────────────────────────
    try:
        await should_exit.wait()
    finally:
        session_metadata_cache.pop(room.name, None)

    # ── Hangup ─────────────────────────────────────────────────────
    try:
        from livekit import api
        lkapi = api.LiveKitAPI()
        for p in room.remote_participants.values():
            if p.identity.startswith("sip_"):
                await lkapi.room.remove_participant(
                    api.RoomParticipantIdentity(room=room.name, identity=p.identity))
        await lkapi.room.delete_room(api.DeleteRoomRequest(room=room.name))
        await lkapi.aclose()
    except Exception as e:
        logger.error(f"[hangup] error: {e}")

    await room.disconnect()


if __name__ == "__main__":
    logger.info(f"Starting LiveKit worker... URL: {os.environ.get('LIVEKIT_URL')}")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
