import os
import logging
import numpy as np
import asyncio
from typing import Optional, Callable
from openwakeword.model import Model

logger = logging.getLogger("ear-wakeword")

class WakeWordService:
    """
    Service for detecting wake words in an audio stream.
    Uses openWakeWord for efficient, local detection.
    """
    def __init__(self, wakeword_models: Optional[list] = None, inference_framework: str = "onnx"):
        # Default to 'hey_jarvis' if no models provided
        if not wakeword_models:
            wakeword_models = ["hey_jarvis"]
        
        self.models = wakeword_models
        self.inference_framework = inference_framework
        self.model = None
        self._is_initialized = False
        self._on_detect_callback: Optional[Callable[[str, float], None]] = None
        self._last_detection_time = 0
        self._cooldown_seconds = 1.5  # 1.5 seconds cooldown

    def initialize(self):
        """Initialize the openWakeWord model."""
        if self._is_initialized:
            return
        
        logger.info(f"Initializing openWakeWord model for: {self.models}")
        try:
            from openwakeword import get_pretrained_model_paths
            # Get all paths and filter for the ones we want
            all_paths = get_pretrained_model_paths()
            model_paths = [p for p in all_paths if any(m in os.path.basename(p) for m in self.models)]
            
            if not model_paths:
                logger.warning(f"No pre-trained models found matching {self.models}. Using all defaults.")
                model_paths = []

            self.model = Model(
                wakeword_model_paths=model_paths
            )
            self._is_initialized = True
            logger.info("WakeWord model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize WakeWord model: {e}")
            raise

    def set_callback(self, callback: Callable[[str, float], None]):
        """Set a callback function to be called when a wake word is detected."""
        self._on_detect_callback = callback

    def process_audio(self, audio_data: np.ndarray):
        """
        Process a chunk of audio data.
        audio_data should be a numpy array of int16 samples.
        """
        if not self._is_initialized:
            self.initialize()

        import time
        current_time = time.time()
        
        # openWakeWord expects 1280 samples at 16kHz (80ms)
        predictions = self.model.predict(audio_data)

        # Check if we are still in cooldown
        if current_time - self._last_detection_time < self._cooldown_seconds:
            return

        for mdl_name, score in predictions.items():
            if score > 0.5: # Confidence threshold
                self._last_detection_time = current_time # Start cooldown
                logger.info(f"Wake word detected: {mdl_name} (score: {score:.2f})")
                if self._on_detect_callback:
                    self._on_detect_callback(mdl_name, score)
                # Reset model internals
                self.model.reset()

    async def run_detection(self, audio_source_iterator):
        """
        Run detection on an async iterator that yields audio chunks.
        """
        self.initialize()
        logger.info("Starting wake word detection loop...")
        async for chunk in audio_source_iterator:
            # Assume chunk is already correctly formatted or converted
            self.process_audio(chunk)

if __name__ == "__main__":
    # Quick test logic
    logging.basicConfig(level=logging.INFO)
    service = WakeWordService()
    service.initialize()
    print("WakeWord service ready for testing.")
