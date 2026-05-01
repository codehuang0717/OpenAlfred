import httpx
import json
import asyncio
import numpy as np
import logging
import wave
import io
import os
from typing import AsyncGenerator
from config import config

logger = logging.getLogger("tts-client")

async def get_tts_stream(text: str, target_sample_rate: int = 24000) -> AsyncGenerator[bytes, None]:
    """
    Generate audio stream from local TTS service and resample to target_sample_rate.
    Yields raw PCM chunks (int16).
    """
    url = config.TTS_URL
    payload = {
        "model": config.TTS_MODEL,
        "input": text,
        "voice": config.TTS_VOICE,
        "response_format": "pcm",
        "stream": True
    }

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=payload, timeout=30.0) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"TTS Request failed: {response.status_code}, {error_text}")
                    return

                # PCM data from service is 16-bit LE, Mono
                # Buffer for incomplete frames if needed, but PCM usually datang in chunks of bytes
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    if not chunk:
                        continue
                    
                    # Convert bytes to int16 numpy array
                    audio_np = np.frombuffer(chunk, dtype=np.int16)
                    yield audio_np.tobytes()

                # Final padding: 250ms of silence to ensure the last word isn't cut off by audio pipelines
                silence_padding = np.zeros(int(target_sample_rate * 0.25), dtype=np.int16)
                yield silence_padding.tobytes()

    except Exception as e:
        logger.error(f"Error in TTS streaming: {e}")

async def save_tts_to_file(text: str, output_path: str):
    """
    Generate full audio and save it as a WAV file.
    Always saves at 24000Hz Mono for consistency with current system.
    """
    audio_chunks = []
    # Add simple generator to verify data
    async for chunk in get_tts_stream(text, target_sample_rate=24000):
        # Debug: Check chunk size
        # logger.info(f"DEBUG: TTS chunk size: {len(chunk)}")
        audio_chunks.append(chunk)
    
    if not audio_chunks:
        logger.error(f"Failed to generate audio for file: '{text[:20]}'")
        return

    # Debug: Print the path being used for saving
    logger.info(f"DEBUG: Saving TTS audio to: {output_path}")

    full_audio = b"".join(audio_chunks)
    
    def save_blocking():
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(full_audio)
            
    await asyncio.to_thread(save_blocking)
    logger.info(f"TTS saved to {output_path}")
