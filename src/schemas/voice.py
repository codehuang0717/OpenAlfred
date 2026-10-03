"""Validated public voice settings and lifecycle contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VoiceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VoxOptions(VoiceModel):
    inference_timesteps: int = Field(9, ge=1, le=30)
    cfg_value: float = Field(2.0, ge=0.5, le=5.0)
    normalize: bool = False
    seed: int | None = Field(None, ge=0, le=2147483647)


class QwenOptions(VoiceModel):
    language: Literal["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Italian", "Portuguese", "Russian", "Spanish"] = "Auto"
    chunk_size: int = Field(10, ge=1, le=48)
    temperature: float = Field(0.9, ge=0.1, le=2)
    top_k: int = Field(50, ge=1, le=200)
    top_p: float = Field(1.0, gt=0, le=1)
    repetition_penalty: float = Field(1.05, ge=1, le=2)
    max_new_tokens: int = Field(1024, ge=128, le=2048)
    max_seq_len: Literal[2048, 4096] = 2048
    xvec_only: bool = False


class VoiceSettings(VoiceModel):
    engine: Literal["voxcpm", "fasterqwentts"] = "fasterqwentts"
    qwen_model: Literal["0.6B", "1.7B"] = "1.7B"
    tts_enabled: bool = True
    stt_enabled: bool = True
    profile_id: str | None = Field("builtin", max_length=80, pattern=r"^(builtin|[0-9a-f]{32})$")
    greeting_text: str = Field("Hello. 你好啊！老大。", min_length=1, max_length=200)
    vox: VoxOptions = Field(default_factory=VoxOptions)
    qwen: QwenOptions = Field(default_factory=QwenOptions)


class VoiceProfile(VoiceModel):
    id: str
    name: str
    transcript: str
    duration: float
    builtin: bool = False


class VoiceRuntime(VoiceModel):
    state: Literal["stopped", "starting", "running", "stopping", "failed"]
    engine: Literal["voxcpm", "fasterqwentts"] | None = None
    model: str | None = None
    error: str | None = None
    restart_required: bool = False


class VoiceEngineInfo(VoiceModel):
    id: Literal["voxcpm", "fasterqwentts"]
    name: str
    available: bool
    reason: str | None = None


class VoiceConfigResponse(VoiceModel):
    settings: VoiceSettings
    runtime: VoiceRuntime
    engines: list[VoiceEngineInfo]
    profiles: list[VoiceProfile]
    greeting: "VoiceGreetingStatus"


class VoiceGreetingStatus(VoiceModel):
    state: Literal["missing", "generating", "ready", "failed"] = "missing"
    error: str | None = None


class VoiceControl(VoiceModel):
    action: Literal["start", "stop", "restart"]


class VoicePreview(VoiceModel):
    text: str = Field(min_length=1, max_length=500)


class VoiceSpeech(VoiceModel):
    text: str = Field(min_length=1, max_length=4000)
