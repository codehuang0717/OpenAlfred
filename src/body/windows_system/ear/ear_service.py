import asyncio
import logging
import pyaudio
import numpy as np
import time
import os
import wave
from .wakeword import WakeWordService
from .local_livekit_client import LocalLiveKitClient

logger = logging.getLogger("ear-service")

class EarService:
    """
    High-level service that manages the microphone and wake word detection.
    This is the 'ear' of the system.
    """
    def __init__(self, wakeword_models=None):
        self.wakeword_service = WakeWordService(wakeword_models=wakeword_models)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        
        # Audio configuration (openWakeWord requirements)
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1280 # 80ms chunks

        # LiveKit Integration
        self.lk_client = LocalLiveKitClient(url="ws://localhost:7880")
        self.lk_client.on_disconnect = self.exit_conversation_mode
        self.in_conversation = False
        self.is_playing_sfx = False # Flag to avoid pushing mic data during local playback
        self.last_activity_time = 0
        self.last_exit_time = 0.0 # Cooldown to avoid accidental re-wake
        self.conversation_timeout = 10.0 # Auto-hangup after 10s of silence

    async def start(self, on_detect_callback=None):
        """Start listening for wake words."""
        if self.is_running:
            return
        
        # Set internal callback that bridges to LiveKit
        async def internal_callback(name, score):
            # Check for cooldown
            if time.time() - self.last_exit_time < 1.5:
                logger.info(f"Ignoring wake word '{name}' during cooldown period.")
                return

            logger.info(f"*** Wake Word '{name}' triggered integration flow ***")
            if on_detect_callback:
                on_detect_callback(name, score)
            await self.enter_conversation_mode()

        self.wakeword_service.set_callback(lambda n, s: asyncio.create_task(internal_callback(n, s)))
        self.wakeword_service.initialize()
        
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        self.is_running = True
        logger.info("Ear service started, listening for wake words...")
        
        asyncio.create_task(self._listen_loop())

    async def enter_conversation_mode(self):
        """Switch from wake-word detection to active LiveKit conversation."""
        if self.in_conversation:
            return
            
        logger.info("*** WAKE WORD DETECTED: Entering conversation mode ***")
        
        self.is_playing_sfx = True
        # Play pre-recorded "I'm here" voice locally for ultra-low latency
        try:
            base_dir = r"E:\PythonProjects\OpenAlfred\agent"
            wake_wav = os.path.join(base_dir, "assets", "sounds", "wake_up.wav")
            
            if os.path.exists(wake_wav):
                with wave.open(wake_wav, "rb") as wf:
                    p = pyaudio.PyAudio()
                    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                    channels=wf.getnchannels(),
                                    rate=wf.getframerate(),
                                    output=True)
                    data = wf.readframes(1024)
                    while data:
                        stream.write(data)
                        data = wf.readframes(1024)
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
        except Exception as e:
            logger.error(f"Error playing wake voice: {e}")
        finally:
            self.is_playing_sfx = False

        self.in_conversation = True
        self.last_activity_time = time.time()
        
        # Room name: local-{timestamp}
        room_name = f"local-user-{int(time.time())}"
        await self.lk_client.connect(room_name=room_name)

    async def exit_conversation_mode(self):
        """Switch back to wake-word detection."""
        if not self.in_conversation:
            return
            
        logger.info("Exiting conversation mode, returning to wake-word detection.")
        await self.lk_client.disconnect()
        self.in_conversation = False
        self.last_exit_time = time.time() # Start cooldown
        if self.wakeword_service.model:
            self.wakeword_service.model.reset()

    async def _listen_loop(self):
        try:
            while self.is_running:
                # Read from mic
                data = await asyncio.get_event_loop().run_in_executor(
                    None, self.stream.read, self.CHUNK, False
                )
                audio_np = np.frombuffer(data, dtype=np.int16)

                if self.in_conversation:
                    # Route to LiveKit Agent (unless we are playing a local SFX)
                    if not self.is_playing_sfx:
                        self.lk_client.push_audio(audio_np)
                    
                    # Basic silence timeout logic (optional, Agent usually hangs up)
                    # if time.time() - self.last_activity_time > self.conversation_timeout:
                    #    await self.exit_conversation_mode()
                else:
                    # Route to WakeWord detection
                    self.wakeword_service.process_audio(audio_np)
                
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in ear listen loop: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the ear service."""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Cleanup LiveKit
        if self.lk_client.is_connected:
            asyncio.create_task(self.lk_client.disconnect())
            
        logger.info("Ear service stopped.")

async def main():
    logging.basicConfig(level=logging.INFO)
    
    def on_wake(name, score):
        print(f"\a*** WAKE WORD DETECTED: {name} ({score:.2f}) ***")

    ear = EarService()
    try:
        await ear.start(on_wake)
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        ear.stop()

if __name__ == "__main__":
    asyncio.run(main())
