"""Private, versioned speech cache and cancellable greeting prewarming."""

import asyncio
import hashlib
import json
import os
import uuid
import wave
from pathlib import Path

from schemas.voice import VoiceGreetingStatus
from services import voice_store
from utils.auth_utils import require_explicit_user_id
from utils.logger import get_logger

logger = get_logger("voice-cache")


def owner_key(user_id: str) -> str:
    return hashlib.sha256(require_explicit_user_id(user_id).encode()).hexdigest()


async def descriptor(user_id: str, text: str) -> tuple[str, Path]:
    settings = await voice_store.get_voice_settings(user_id)
    reference = None
    if settings.profile_id:
        profile, path = await voice_store.get_profile(user_id, settings.profile_id)
        info = await asyncio.to_thread(path.stat)
        reference = [profile.id, profile.transcript, info.st_size, info.st_mtime_ns]
    payload = {"version": 1, "owner": owner_key(user_id), "text": text.strip(),
               "engine": settings.engine, "reference": reference,
               "model": settings.qwen_model if settings.engine == "fasterqwentts" else "VoxCPM2",
               "options": (settings.qwen if settings.engine == "fasterqwentts" else settings.vox).model_dump()}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return key, voice_store.VOICE_DATA / "greetings" / owner_key(user_id) / f"{key}.wav"


async def cached_voice(user_id: str, text: str) -> Path | None:
    _key, path = await descriptor(user_id, text)
    return path if await asyncio.to_thread(path.is_file) else None


async def saved_voice(user_id: str, path: Path) -> tuple[str, Path | None] | None:
    """Old reminder aliases are usable only with matching owner and voice metadata."""
    metadata = path.with_suffix(".voice.json")
    if not await asyncio.to_thread(metadata.is_file):
        return None
    value = json.loads(await asyncio.to_thread(metadata.read_text, encoding="utf-8"))
    if value.get("owner") != owner_key(user_id):
        return None
    text = value["text"]
    key, _cache = await descriptor(user_id, text)
    valid = key == value.get("key") and await asyncio.to_thread(path.is_file)
    return text, path if valid else None


async def render_cached_voice(user_id: str, text: str, destination: Path | None = None) -> Path:
    from services.tts import get_tts_stream

    if not (await voice_store.get_voice_settings(user_id)).tts_enabled:
        raise ValueError("语音合成已关闭")
    key, path = await descriptor(user_id, text)
    if not await asyncio.to_thread(path.is_file):
        audio = bytearray()
        async for chunk in get_tts_stream(text, user_id=user_id):
            audio.extend(chunk)
            if len(audio) > 24000 * 2 * 120:
                raise ValueError("问候音频超过两分钟，请缩短文本")
        if not audio or len(audio) % 2:
            raise RuntimeError("语音引擎没有返回有效音频")
        current, _ = await descriptor(user_id, text)
        if current != key:
            raise RuntimeError("音色设置已改变，本次缓存已丢弃")

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                with wave.open(str(temporary), "wb") as output:
                    output.setnchannels(1); output.setsampwidth(2); output.setframerate(24000)
                    output.writeframes(audio)
                os.replace(temporary, path)
                # Bound duplicated cached greetings; reminder aliases retain
                # their own audio and verified metadata outside this directory.
                existing = []
                for item in path.parent.glob("*.wav"):
                    try:
                        existing.append((item.stat().st_mtime_ns, item))
                    except FileNotFoundError:
                        continue
                for _modified, obsolete in sorted(existing, reverse=True)[32:]:
                    if obsolete != path:
                        obsolete.unlink(missing_ok=True)
            finally:
                temporary.unlink(missing_ok=True)

        await asyncio.to_thread(write)
    if destination is not None:
        def alias() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
            metadata = destination.with_suffix(".voice.json")
            temporary_meta = metadata.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_bytes(path.read_bytes())
                temporary_meta.write_text(json.dumps({"owner": owner_key(user_id), "key": key, "text": text.strip()}, ensure_ascii=False), encoding="utf-8")
                metadata.unlink(missing_ok=True)
                os.replace(temporary, destination)
                os.replace(temporary_meta, metadata)
            finally:
                temporary.unlink(missing_ok=True)
                temporary_meta.unlink(missing_ok=True)
        await asyncio.to_thread(alias)
    return path


class GreetingCache:
    def __init__(self) -> None:
        self.tasks: dict[str, asyncio.Task] = {}
        self.states: dict[str, tuple[str, VoiceGreetingStatus]] = {}

    async def status(self, user_id: str) -> VoiceGreetingStatus:
        settings = await voice_store.get_voice_settings(user_id)
        try:
            key, path = await descriptor(user_id, settings.greeting_text)
        except ValueError:
            return VoiceGreetingStatus(state="failed", error="所选音色不存在，请重新选择")
        if await asyncio.to_thread(path.is_file):
            return VoiceGreetingStatus(state="ready")
        previous = self.states.get(user_id)
        return previous[1] if previous and previous[0] == key else VoiceGreetingStatus()

    def schedule(self, user_id: str, transition: asyncio.Task | None = None) -> None:
        require_explicit_user_id(user_id)
        previous = self.tasks.get(user_id)
        if previous and not previous.done():
            previous.cancel()
        self.tasks[user_id] = asyncio.create_task(self._warm(user_id, transition), name="greeting-cache")

    async def _warm(self, user_id: str, transition: asyncio.Task | None) -> None:
        from services.voice_runtime import voice_manager
        key = None
        try:
            if transition is not None:
                await asyncio.shield(transition)
            settings = await voice_store.get_voice_settings(user_id)
            runtime = voice_manager.status(settings)
            if not settings.tts_enabled or runtime.state != "running" or runtime.restart_required:
                return
            key, _path = await descriptor(user_id, settings.greeting_text)
            self.states[user_id] = (key, VoiceGreetingStatus(state="generating"))
            await render_cached_voice(user_id, settings.greeting_text)
            self.states[user_id] = (key, VoiceGreetingStatus(state="ready"))
        except asyncio.CancelledError:
            if key and self.states.get(user_id, (None,))[0] == key:
                self.states.pop(user_id, None)
            raise
        except Exception:
            logger.exception("Greeting cache generation failed for account %s", user_id)
            if key:
                self.states[user_id] = (key, VoiceGreetingStatus(state="failed", error="问候语缓存生成失败，请检查语音日志后重新加载模型"))

    async def close(self) -> None:
        pending = list(self.tasks.values())
        self.tasks.clear()
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self.states.clear()


greeting_cache = GreetingCache()
