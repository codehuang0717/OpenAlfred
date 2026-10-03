"""Manual GPU integration probe; only the managed worker is stopped.

Run with the API stopped: uv run python tests/probe_voice_lifecycle.py
"""

import asyncio
import sys
import wave
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core.config import config
from schemas.voice import VoiceSettings
from services.voice_runtime import VoiceManager


async def main() -> None:
    manager = VoiceManager()
    await manager.initialize()
    try:
        for engine, size in [("voxcpm", "1.7B"), ("fasterqwentts", "0.6B"), ("fasterqwentts", "1.7B")]:
            settings = VoiceSettings(engine=engine, qwen_model=size)
            print("START", engine, size, flush=True)
            await manager.control("start", settings)
            await manager.transition
            status = manager.status(settings)
            print("READY", status.model_dump(), flush=True)
            if status.state != "running":
                raise RuntimeError(status.error)
            payload = {
                "text": "你好，我是阿尔弗雷德。语音模型切换测试成功。",
                "reference_path": str(config.VOXCPM_PROJECT / "voices/yingxue.wav"),
                "reference_text": "希望你以后能够做的比我还好哟",
                "vox": settings.vox.model_dump(), "qwen": settings.qwen.model_dump(),
            }
            async with httpx.AsyncClient(trust_env=False, timeout=180) as client:
                response = await client.post(f"http://127.0.0.1:{config.VOICE_ENGINE_PORT}/speech", json=payload, headers={"X-Engine-Token": manager.token})
                response.raise_for_status()
            if len(response.content) < 48000 or len(response.content) % 2:
                raise RuntimeError("Invalid or empty PCM output")
            path = config.ASSETS_DIR / "audio_cache" / f"probe-{engine}-{size}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1); output.setsampwidth(2); output.setframerate(24000)
                output.writeframes(response.content)
            print("AUDIO", round(len(response.content) / 48000, 2), "seconds", path, flush=True)
        await manager.control("stop", settings)
        print("STOPPED", manager.status(settings).state, flush=True)
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
