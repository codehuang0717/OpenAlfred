"""Private voice settings and normalized reference recordings, scoped by user."""

import asyncio
import hashlib
import io
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from core.config import config
from db.settings import get_setting, set_setting
from schemas.voice import VoiceProfile, VoiceSettings
from utils.auth_utils import require_explicit_user_id

VOICE_DATA = config.PROJECT_ROOT / "data" / "voice"
MAX_REFERENCE_BYTES = 20 * 1024 * 1024


def user_directory(user_id: str) -> Path:
    user_id = require_explicit_user_id(user_id)
    return VOICE_DATA / "profiles" / hashlib.sha256(user_id.encode()).hexdigest()


async def get_voice_settings(user_id: str) -> VoiceSettings:
    user_id = require_explicit_user_id(user_id)
    raw = await get_setting(f"voice:{user_id}")
    return VoiceSettings.model_validate_json(raw) if raw else VoiceSettings()


async def save_voice_settings(user_id: str, settings: VoiceSettings) -> None:
    user_id = require_explicit_user_id(user_id)
    if not settings.greeting_text.strip():
        raise ValueError("问候语不能为空")
    if settings.profile_id is not None:
        await get_profile(user_id, settings.profile_id)
    if settings.qwen.max_new_tokens > settings.qwen.max_seq_len - 256:
        raise ValueError("最大生成长度需为上下文留出至少 256 个 token")
    await set_setting(f"voice:{user_id}", settings.model_dump_json())


def _builtin() -> tuple[VoiceProfile, Path] | None:
    path = config.VOXCPM_PROJECT / "voices" / "yingxue.wav"
    if not path.is_file():
        path = config.QWEN_TTS_PROJECT / "yingxue.wav"
    if not path.is_file():
        return None
    info = sf.info(path)
    return VoiceProfile(id="builtin", name="映雪（预置）", transcript="希望你以后能够做的比我还好哟", duration=info.duration, builtin=True), path


async def list_profiles(user_id: str) -> list[VoiceProfile]:
    directory = user_directory(user_id)

    def read() -> list[VoiceProfile]:
        builtin = _builtin()
        profiles = [builtin[0]] if builtin else []
        for path in sorted(directory.glob("*.json")):
            profiles.append(VoiceProfile.model_validate_json(path.read_text(encoding="utf-8")))
        return profiles

    return await asyncio.to_thread(read)


async def get_profile(user_id: str, profile_id: str) -> tuple[VoiceProfile, Path]:
    directory = user_directory(user_id)

    def read() -> tuple[VoiceProfile, Path]:
        if profile_id == "builtin":
            builtin = _builtin()
            if builtin:
                return builtin
        elif len(profile_id) == 32 and all(c in "0123456789abcdef" for c in profile_id):
            metadata, audio = directory / f"{profile_id}.json", directory / f"{profile_id}.wav"
            if metadata.is_file() and audio.is_file():
                return VoiceProfile.model_validate_json(metadata.read_text(encoding="utf-8")), audio
        raise ValueError("音色不存在，请重新选择或上传参考音频")

    return await asyncio.to_thread(read)


async def create_profile(user_id: str, name: str, transcript: str, content: bytes) -> VoiceProfile:
    directory = user_directory(user_id)
    name, transcript = name.strip(), transcript.strip()
    if not name or len(name) > 80 or not transcript or len(transcript) > 2000:
        raise ValueError("请填写音色名称（最多 80 字）和参考音频的逐字文本（最多 2000 字）")
    if not content or len(content) > MAX_REFERENCE_BYTES:
        raise ValueError("参考音频不能为空或超过 20MB")

    def write() -> VoiceProfile:
        try:
            with sf.SoundFile(io.BytesIO(content)) as source:
                duration = len(source) / source.samplerate
                if not 1 <= duration <= 60 or not 8000 <= source.samplerate <= 192000 or source.channels > 2:
                    raise ValueError("参考音频需为 1–60 秒、单声道或双声道，采样率 8–192kHz")
                samples = source.read(dtype="float32", always_2d=True).mean(axis=1)
                sample_rate = source.samplerate
        except sf.LibsndfileError as error:
            raise ValueError("无法解码参考音频，请上传有效的 WAV、FLAC、MP3 或 OGG 文件") from error
        if not np.isfinite(samples).all() or float(np.max(np.abs(samples))) < 0.001:
            raise ValueError("参考音频无有效人声，请检查文件")
        profile = VoiceProfile(id=uuid.uuid4().hex, name=name, transcript=transcript, duration=duration)
        directory.mkdir(parents=True, exist_ok=True)
        audio = directory / f"{profile.id}.wav"
        metadata = directory / f"{profile.id}.json"
        try:
            sf.write(audio, samples, sample_rate, subtype="PCM_16")
            temporary = metadata.with_suffix(".tmp")
            temporary.write_text(profile.model_dump_json(), encoding="utf-8")
            temporary.replace(metadata)
        except BaseException:
            audio.unlink(missing_ok=True)
            metadata.unlink(missing_ok=True)
            metadata.with_suffix(".tmp").unlink(missing_ok=True)
            raise
        return profile

    return await asyncio.to_thread(write)


async def delete_profile(user_id: str, profile_id: str) -> None:
    profile, audio = await get_profile(user_id, profile_id)
    if profile.builtin:
        raise ValueError("预置音色不能删除")
    settings = await get_voice_settings(user_id)
    if settings.profile_id == profile_id:
        settings.profile_id = None
        await save_voice_settings(user_id, settings)
    await asyncio.to_thread(audio.with_suffix(".json").unlink, missing_ok=True)
    await asyncio.to_thread(audio.unlink, missing_ok=True)
