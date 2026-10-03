"""Real GPU + local API greeting-cache probe, isolated from personal settings.

Run with the business API stopped: uv run python tests/probe_voice_greeting.py
"""

import asyncio
import sys
import tempfile
import time
import wave
from pathlib import Path
from unittest.mock import patch

import httpx
import uvicorn
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core.config import config
from db import connection
from routers.auth import get_current_user
from routers.voice import router
from schemas.voice import VoiceSettings
from services import voice_runtime, voice_store
from services.voice_cache import cached_voice, greeting_cache


async def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        with patch.object(connection, "DATABASE_PATH", str(root / "db.sqlite")), patch.object(voice_store, "VOICE_DATA", root / "voice"), patch.object(voice_runtime, "VOICE_DATA", root / "voice"), patch.object(config, "VOICE_API_URL", "http://127.0.0.1:7799"):
            await connection.init_db()
            app = FastAPI()
            app.include_router(router)
            app.dependency_overrides[get_current_user] = lambda: {"id": "voice-probe"}
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=7799, log_level="warning"))
            serving = asyncio.create_task(server.serve())
            settings = VoiceSettings(engine="fasterqwentts", qwen_model="0.6B", greeting_text="你好，我是阿尔弗雷德。这是缓存的问候语。")
            try:
                while not server.started:
                    if serving.done():
                        await serving
                        raise RuntimeError("Probe API could not start")
                    await asyncio.sleep(0.1)
                async with httpx.AsyncClient(base_url=config.VOICE_API_URL, trust_env=False, timeout=20) as client:
                    response = await client.put("/api/voice/config", json=settings.model_dump())
                    response.raise_for_status()
                    response = await client.post("/api/voice/runtime", json={"action": "start"})
                    response.raise_for_status()

                    async def ready() -> Path:
                        async with asyncio.timeout(300):
                            while True:
                                response = await client.get("/api/voice/config")
                                response.raise_for_status()
                                state = response.json()
                                if state["runtime"]["state"] == "failed" or state["greeting"]["state"] == "failed":
                                    raise RuntimeError(str(state))
                                if state["greeting"]["state"] == "ready":
                                    path = await cached_voice("voice-probe", settings.greeting_text)
                                    with wave.open(str(path)) as audio:
                                        assert audio.getframerate() == 24000 and audio.getnframes() > 24000
                                    return path
                                await asyncio.sleep(0.5)

                    first = await ready()
                    started = time.perf_counter()
                    assert await cached_voice("voice-probe", settings.greeting_text) == first
                    print("CACHE READY", round((time.perf_counter() - started) * 1000, 1), "ms lookup", flush=True)
                    settings.qwen.temperature = 0.8
                    response = await client.put("/api/voice/config", json=settings.model_dump())
                    response.raise_for_status()
                    assert response.json()["settings"]["qwen"]["temperature"] == 0.8
                    # Background generation may finish before PUT returns;
                    # verify the cache version instead of assuming its timing.
                    second = await ready()
                    assert second != first
                    assert await cached_voice("another-account", settings.greeting_text) is None
                    print("PARAMETER CHANGE REGENERATED CACHE; ACCOUNT ISOLATION PASSED", flush=True)
                    response = await client.post("/api/voice/runtime", json={"action": "stop"})
                    response.raise_for_status()
                    assert response.json()["state"] == "stopped"
            finally:
                await greeting_cache.close()
                await voice_runtime.voice_manager.close()
                server.should_exit = True
                await serving


if __name__ == "__main__":
    asyncio.run(main())
