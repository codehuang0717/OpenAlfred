import asyncio
import logging
import time
from livekit import rtc
import pyaudio
import numpy as np
from core.config import config

logger = logging.getLogger("ear-livekit-client")

class LocalLiveKitClient:
    """
    Client that connects to a local LiveKit server after wake-word detection.
    It publishes the microphone stream and plays back the agent's response.
    """
    def __init__(self, url=None, api_key=None, api_secret=None):
        self.url = url or config.LIVEKIT_URL or "ws://localhost:7880"
        self.api_key = api_key or config.LIVEKIT_API_KEY or "devkey"
        self.api_secret = api_secret or config.LIVEKIT_API_SECRET or "secret"
        
        self.room = None
        self.audio_source = rtc.AudioSource(16000, 1) # openWakeWord uses 16kHz
        self.is_connected = False
        self.p = pyaudio.PyAudio()
        self.output_stream = None
        self.active_playback_tasks = {} # track_sid -> Task
        self.on_disconnect = None # Callback when room disconnects

    async def connect(self, room_name="local-user-session", identity="local-ear"):
        if self.is_connected:
            return
        
        # Create a fresh room for each connection to avoid duplicate event listeners
        self.room = rtc.Room()
        
        logger.info(f"Connecting to local LiveKit at {self.url}...")
        try:
            # Generate a token for local dev
            token = self._generate_token(room_name, identity)
            
            await self.room.connect(self.url, token)
            self.is_connected = True
            logger.info(f"Connected to room: {room_name}")

            # Publish the mic track
            track = rtc.LocalAudioTrack.create_audio_track("mic", self.audio_source)
            await self.room.local_participant.publish_track(track)
            logger.info("Published local audio track.")

            @self.room.on("disconnected")
            def on_room_disconnected(reason):
                logger.info(f"DEBUG-CLIENT: Room disconnected. Reason: {reason}")
                self.is_connected = False
                if self.on_disconnect:
                    asyncio.create_task(self.on_disconnect())

            @self.room.on("track_subscribed")
            def on_track_subscribed(track, publication, participant):
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"DEBUG-CLIENT-SUBSCRIPTION: Subscribed to track {track.sid} from {participant.identity}")
                    
                    # CRITICAL FIX: Cancel ALL existing playback tasks to prevent overlapping
                    for sid, task in list(self.active_playback_tasks.items()):
                        if sid != track.sid:
                            logger.info(f"DEBUG-CLIENT-SUBSCRIPTION: Cancelling OLD playback task for {sid}")
                            task.cancel()
                            self.active_playback_tasks.pop(sid, None)

                    if track.sid in self.active_playback_tasks:
                        logger.warning(f"DEBUG-CLIENT-SUBSCRIPTION: Playback ALREADY active for {track.sid}. Ignoring.")
                        return

                    logger.info(f"DEBUG-CLIENT-SUBSCRIPTION: Starting NEW playback task for {track.sid}")
                    task = asyncio.create_task(self._play_audio_stream(track))
                    self.active_playback_tasks[track.sid] = task
                    
                    def on_task_done(t):
                        logger.info(f"DEBUG-CLIENT-SUBSCRIPTION: Playback task for {track.sid} FINISHED.")
                        if self.active_playback_tasks.get(track.sid) == t:
                            self.active_playback_tasks.pop(track.sid, None)
                    task.add_done_callback(on_task_done)

        except Exception as e:
            logger.error(f"Failed to connect to LiveKit: {e}")
            self.is_connected = False

    def _generate_token(self, room_name, identity):
        """Generate a token for the SDK to use."""
        from livekit import api
        token = api.AccessToken(self.api_key, self.api_secret) \
            .with_identity(identity) \
            .with_name("Local Ear Service") \
            .with_grants(api.VideoGrants(room_join=True, room=room_name))
        return token.to_jwt()

    def push_audio(self, audio_data: np.ndarray):
        """Push audio chunks from EarService to LiveKit."""
        if not self.is_connected:
            return
            
        frame = rtc.AudioFrame(
            data=audio_data.tobytes(),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=len(audio_data)
        )
        # Note: capture_frame is a coroutine but usually returns quickly
        asyncio.create_task(self.audio_source.capture_frame(frame))

    async def _play_audio_stream(self, track: rtc.AudioTrack):
        """Subscribes to agent audio and plays it through local speakers."""
        audio_stream = rtc.AudioStream(track)
        track_sid = track.sid
        
        logger.info(f"DEBUG-AUDIT: Starting playback for Track {track_sid}")
        frame_count = 0
        try:
            async for frame_event in audio_stream:
                if not self.is_connected:
                    break
                
                frame = frame_event.frame
                frame_count += 1
                
                # Log a heartbeat every 1 second of audio (approx 50 frames)
                if frame_count % 50 == 1:
                    logger.info(f"DEBUG-AUDIT: Track {track_sid} | Frame #{frame_count} | SysTime: {time.time()}")

                # Initialize stream based on the first frame's properties
                if not self.output_stream:
                    logger.info(f"Initializing output stream: {frame.sample_rate}Hz, {frame.num_channels} channels")
                    self.output_stream = self.p.open(
                        format=pyaudio.paInt16,
                        channels=frame.num_channels,
                        rate=frame.sample_rate,
                        output=True
                    )
                
                # Convert memoryview to bytes for PyAudio
                self.output_stream.write(bytes(frame.data))
        except Exception as e:
            logger.error(f"Error during audio playback for {track_sid}: {e}")
        finally:
            logger.info(f"DEBUG-AUDIT: Playback for Track {track_sid} FINISHED. Total frames: {frame_count}")

    async def disconnect(self):
        if self.is_connected:
            self.is_connected = False
            await self.room.disconnect()
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.output_stream = None
            logger.info("Disconnected from local session.")

    def __del__(self):
        if self.p:
            self.p.terminate()
