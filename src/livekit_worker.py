# Runtime Monkey Patch to fix opentelemetry compatibility with livekit-agents
import opentelemetry.sdk._logs as sdk_logs
if not hasattr(sdk_logs, "LogData"):
    class LogData:
        def __init__(self, log_record, instrumentation_scope):
            self.log_record = log_record
            self.instrumentation_scope = instrumentation_scope
    sdk_logs.LogData = LogData

import asyncio
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

async def entrypoint(ctx: JobContext):
    global db_initialized
    if not db_initialized:
        await init_db()
        db_initialized = True

    # Use SUBSCRIBE_NONE to take control of the subscription lifecycle
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)

    room = ctx.room
    should_exit = asyncio.Event()

    # Determine call type based on room name prefix
    if room.name.startswith("outbound-"):
        call_type = "outbound"
    elif room.name.startswith("local-"):
        call_type = "local"
    else:
        call_type = "inbound"
    
    logger.info(f"Detected call type: {call_type} for room: {room.name}")

    # Unique session ID for LangGraph threads
    unique_session_id = room.name
    if call_type in ("inbound", "local"):
        unique_session_id = f"{room.name}-{ctx.job.id}"
    
    user_id = "default"

    # Identify user based on call type and room name
    if call_type == "outbound":
        if room.name.startswith("outbound-reminder-"):
            user_id = room.name[18 + 36 + 1:] or "default"
        elif room.name.startswith("outbound-supervisor-"):
            parts = room.name.split("-")
            user_id = "-".join(parts[3:]) if len(parts) >= 4 else "default"
        else:
            parts = room.name.split("-")
            user_id = "-".join(parts[1:-1]) if len(parts) >= 3 else "default"
    elif call_type == "local":
        parts = room.name.split("-")
        user_id = parts[1] if len(parts) >= 2 else "default"
    else:
        # Inbound
        try:
            from core.database import get_user_by_username
            fp_user = await get_user_by_username("FlyingPig")
            if fp_user:
                user_id = fp_user["id"]
        except Exception as e:
            logger.error(f"Failed to find inbound user FlyingPig: {e}")

    answered_event = asyncio.Event() 
    greeting_played = False
    greeting_lock = asyncio.Lock()
    active_sessions = {} # identity -> Task

    async def trigger_greeting():
        nonlocal greeting_played
        async with greeting_lock:
            if greeting_played:
                return
            greeting_played = True
            logger.info(f"Triggering greeting for {call_type} call (session: {unique_session_id})")
            
            try:
                if call_type == "local":
                    logger.info("Skipping cloud greeting for local call.")
                    return
                
                current_initial_speech = initial_speech
                if call_type == "local" and not current_initial_speech:
                    current_initial_speech = "我在，请讲。"
                
                # Link with session to prevent timeout
                session_obj = next(iter(active_sessions.values()), None) if active_sessions else None
                if session_obj and hasattr(session_obj, 'is_greeting_playing'):
                    session_obj.is_greeting_playing = True
                
                try:
                    await play_greeting(room, current_initial_speech, should_exit, user_id, call_type)
                finally:
                    if session_obj and hasattr(session_obj, 'is_greeting_playing'):
                        session_obj.is_greeting_playing = False
            except Exception as e:
                logger.error(f"Error in trigger_greeting: {e}")

    # Setup LangGraph thread
    call_title = "本地语音唤醒" if call_type == "local" else ("语音外拨呼叫" if call_type == "outbound" else "语音呼入接待")
    thread_info = await _ensure_thread(unique_session_id, user_id, call_title, call_type)
    initial_speech = thread_info.get("initial_speech", "")

    # Cache metadata for call_agent to use
    session_metadata_cache[room.name] = {
        "call_type": call_type,
        "initial_speech": initial_speech,
        "is_fresh": True,
        "unique_session_id": unique_session_id,
    }

    async def subscribe_to_audio(p: rtc.RemoteParticipant):
        for pub in p.track_publications.values():
            if pub.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"Manually subscribing to audio track {pub.sid} from {p.identity}")
                pub.set_subscribed(True)

    @room.on("participant_connected")
    def on_participant_connected(p: rtc.RemoteParticipant):
        asyncio.create_task(subscribe_to_audio(p))

    @room.on("participant_attributes_changed")
    def on_attributes_changed(changed_attributes: dict, participant: rtc.RemoteParticipant):
        if "sip.callStatus" in changed_attributes:
            if participant.attributes.get("sip.callStatus") == "active":
                answered_event.set()
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

            is_sip = participant.identity.startswith("sip_")
            session = VoiceSession(room, should_exit, user_id, is_sip=is_sip, answered_event=answered_event)
            task = asyncio.create_task(session.run(track))
            active_sessions[participant.identity] = session # Store the session object
            
            def on_task_done(t):
                active_sessions.pop(participant.identity, None)
            task.add_done_callback(on_task_done)
            
            if call_type in ("inbound", "local") or not participant.identity.startswith("sip_"):
                answered_event.set()
                asyncio.create_task(trigger_greeting())

    # Scan existing participants
    for participant in room.remote_participants.values():
        asyncio.create_task(subscribe_to_audio(participant))
        if participant.attributes.get("sip.callStatus") == "active":
            answered_event.set()
            asyncio.create_task(trigger_greeting())

    if call_type == "outbound":
        try:
            await asyncio.wait_for(answered_event.wait(), timeout=30)
        except asyncio.TimeoutError:
            should_exit.set()
            return
    else:
        asyncio.create_task(trigger_greeting())

    try:
        await should_exit.wait()
    finally:
        session_metadata_cache.pop(room.name, None)

    # Force hangup logic
    try:
        from livekit import api
        lkapi = api.LiveKitAPI()
        for p in room.remote_participants.values():
            if p.identity.startswith("sip_"):
                await lkapi.room.remove_participant(api.RoomParticipantIdentity(room=room.name, identity=p.identity))
        await lkapi.room.delete_room(api.DeleteRoomRequest(room=room.name))
        await lkapi.aclose()
    except Exception as e:
        logger.error(f"Hangup logic failed: {e}")

    await room.disconnect()

if __name__ == "__main__":
    logger.info(f"Starting LiveKit worker... URL: {os.environ.get('LIVEKIT_URL')}")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
