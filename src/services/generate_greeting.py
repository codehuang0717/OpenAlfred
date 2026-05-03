import httpx
import wave
import io
import asyncio
import numpy as np
from core.config import config


async def generate():
    print("Requesting TTS for greeting...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                config.GREETING_TTS_URL,
                data={"text": "Hello. 你好啊! 老大", "voice": "default"},
            )
            resp.raise_for_status()
            audio_data = resp.content

            # The API returns raw PCM 16kHz 16-bit mono.
            # We will save it directly as a proper WAV file so `play_greeting` can read it easily.

            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            out_path = "greeting.wav"
            with wave.open(out_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)
                wf.writeframes(audio_data)

            print(f"Successfully generated {out_path} with {len(audio_np)} samples.")
    except Exception as e:
        print(f"Failed to generate greeting: {e}")


if __name__ == "__main__":
    asyncio.run(generate())
