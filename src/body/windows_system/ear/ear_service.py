import asyncio
import logging
import pyaudio
import numpy as np
from .wakeword import WakeWordService

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

    def start(self, on_detect_callback):
        """Start listening for wake words."""
        if self.is_running:
            return
        
        self.wakeword_service.set_callback(on_detect_callback)
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
        
        # Start the processing loop in a thread or task
        # For simplicity in this step, we'll use a blocking loop that can be run in a task
        asyncio.create_task(self._listen_loop())

    async def _listen_loop(self):
        try:
            while self.is_running:
                # Use run_in_executor to avoid blocking the event loop with PyAudio read
                data = await asyncio.get_event_loop().run_in_executor(
                    None, self.stream.read, self.CHUNK, False
                )
                audio_np = np.frombuffer(data, dtype=np.int16)
                self.wakeword_service.process_audio(audio_np)
                # Small sleep to yield
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
        logger.info("Ear service stopped.")

if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    
    def on_wake(name, score):
        print(f"\a*** WAKE WORD DETECTED: {name} ({score:.2f}) ***")

    ear = EarService()
    loop = asyncio.get_event_loop()
    try:
        ear.start(on_wake)
        loop.run_forever()
    except KeyboardInterrupt:
        ear.stop()
