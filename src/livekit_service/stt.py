import io
import wave
import httpx
from utils.logger import get_logger
from utils.latency import latency_tracker

logger = get_logger("livekit-stt")

async def transcribe_audio(audio_data: bytes, sample_rate: int, channels: int) -> str:
    """Send audio data to the SenseVoice service for transcription."""
    latency_tracker.start("stt_total")
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    wav_bytes = wav_io.getvalue()
    
    try:
        latency_tracker.start("stt_http_request")
        async with httpx.AsyncClient() as client:
            files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
            resp = await client.post(
                "http://127.0.0.1:8000/extract_text", files=files, timeout=15.0
            )
            resp.raise_for_status()
            latency_tracker.end("stt_http_request")
            result = resp.json()
            latency_tracker.end("stt_total")
            return result.get("results", "")
    except Exception as e:
        logger.error(f"SenseVoice Error: {e}")
        return ""
