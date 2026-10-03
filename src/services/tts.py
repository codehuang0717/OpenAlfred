import httpx
import time
import numpy as np
from typing import AsyncGenerator
from core.config import config
from utils.auth_utils import mint_service_jwt, require_explicit_user_id
from utils.logger import get_logger

logger = get_logger("tts-client")

async def get_tts_stream(text: str, target_sample_rate: int = 24000, *, user_id: str) -> AsyncGenerator[bytes, None]:
    """
    Generate audio from the account's managed local TTS engine.
    Yields raw PCM chunks (int16).
    """
    if target_sample_rate != 24000:
        raise ValueError("TTS 输出固定为 24000Hz，不能直接重标采样率")
    user_id = require_explicit_user_id(user_id)
    url = f"{config.VOICE_API_URL}/api/voice/speech"
    headers = {"Authorization": f"Bearer {mint_service_jwt(user_id)}"}
    payload = {"text": text}

    try:
        t_prev = None
        received_audio = False
        timeout = httpx.Timeout(120.0, connect=10.0)
        async with httpx.AsyncClient(trust_env=False) as client:
            async with client.stream("POST", url, json=payload, headers=headers, timeout=timeout) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"TTS Request failed: {response.status_code}, {error_text}")
                    response.raise_for_status()

                # PCM data from service is 16-bit LE, Mono
                # Buffer for incomplete frames if needed, but PCM usually datang in chunks of bytes
                async for chunk in response.aiter_bytes(chunk_size=19200):
                    if not chunk:
                        continue
                    now = time.perf_counter()
                    if t_prev is not None:
                        gap_ms = (now - t_prev) * 1000
                        audio_ms = len(chunk) / 2 / target_sample_rate * 1000
                        if gap_ms > audio_ms + 50:
                            logger.warning(
                                "TTS recv gap=%.0fms audio=%.0fms (producer slower than playback)",
                                gap_ms,
                                audio_ms,
                            )
                    else:
                        logger.info("TTS first network bytes=%d", len(chunk))
                    t_prev = now
                    received_audio = True
                    yield chunk

                if received_audio:
                    # Padding only makes sense after real speech; an empty stream
                    # must not count as a successfully played goodbye.
                    silence_padding = np.zeros(int(target_sample_rate * 0.25), dtype=np.int16)
                    yield silence_padding.tobytes()

    except Exception as e:
        logger.error(f"Error in TTS streaming: {e}")
        raise

async def save_tts_to_file(text: str, output_path: str, *, user_id: str) -> None:
    """Atomically save audio and its account/voice fingerprint metadata."""
    from pathlib import Path
    from services.voice_cache import render_cached_voice
    await render_cached_voice(user_id, text, Path(output_path))
    logger.info("TTS saved to %s", output_path)
