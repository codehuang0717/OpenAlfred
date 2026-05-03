import os
import logging
import numpy as np
import asyncio
from typing import Optional, Callable
from openwakeword.model import Model
from core.config import config

logger = logging.getLogger("ear-wakeword")

class WakeWordService:
    """
    Service for detecting wake words in an audio stream.
    Uses openWakeWord for efficient, local detection.
    """
    def __init__(self, wakeword_models: Optional[list] = None, inference_framework: str = "onnx"):
        # Default to 'hey_jarvis' if no models provided
        if not wakeword_models:
            wakeword_models = ["alfred_fast"]
        
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
            
            # 1. Get library pre-trained paths
            all_paths = get_pretrained_model_paths()
            model_paths = [p for p in all_paths if any(m in os.path.basename(p) for m in self.models)]

            # 2. Check local assets directory (via config)
            local_models_dir = os.path.join(str(config.PROJECT_ROOT), "assets", "models")
            
            logger.info(f"Checking local models directory: {local_models_dir}")
            if os.path.exists(local_models_dir):
                for f in os.listdir(local_models_dir):
                    if f.endswith((".onnx", ".tflite")):
                        # Normalize names for comparison (remove underscores and convert to lowercase)
                        name_no_ext = os.path.splitext(f)[0].lower().replace("_", "")
                        for m in self.models:
                            target_name = m.lower().replace("_", "")
                            if target_name in name_no_ext:
                                full_path = os.path.join(local_models_dir, f)
                                if full_path not in model_paths:
                                    logger.info(f"Found custom model: {full_path}")
                                    model_paths.append(full_path)
            else:
                logger.warning(f"Local models directory not found: {local_models_dir}")

            if not model_paths:
                logger.warning(f"No models found matching {self.models} in library or {local_models_dir}")
                model_paths = []

            # Initialize Model. Note: Some versions of openwakeword don't support inference_framework argument
            # or hardcode ONNX. We'll pass model_paths and hope for the best.
            self.model = Model(
                wakeword_model_paths=model_paths
            )
            self._is_initialized = True
            logger.info(f"WakeWord model initialized successfully with {len(model_paths)} models.")
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

        # Debug: Print max score periodically or if it's significant
        max_score = 0
        if predictions:
            max_score = max(predictions.values())
            if max_score > 0.1: # Only print if there's some activity
                 logger.debug(f"Current max wake-word score: {max_score:.3f}")

        for mdl_name, score in predictions.items():
            if score > 0.32: # Lowered threshold for personalized model
                self._last_detection_time = current_time # Start cooldown
                logger.info(f"*** WAKE WORD DETECTED: {mdl_name} (score: {score:.3f}) ***")
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
