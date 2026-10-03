"""Account-scoped voice controls; inference stays in a separate GPU process."""

import io
import wave
from typing import AsyncIterator

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from core.config import config
from core.event_bus import EventType, event_bus
from routers.auth import get_current_user
from schemas.voice import (
    VoiceConfigResponse, VoiceControl, VoicePreview, VoiceProfile,
    VoiceRuntime, VoiceSettings, VoiceSpeech,
)
from services.voice_runtime import engine_info, voice_manager
from services.voice_cache import greeting_cache
from services.voice_store import (
    MAX_REFERENCE_BYTES, create_profile, delete_profile, get_profile,
    get_voice_settings, list_profiles, save_voice_settings,
)

router = APIRouter(prefix="/api/voice", tags=["voice"])


@router.get("/config", response_model=VoiceConfigResponse)
async def get_voice_config(user: dict = Depends(get_current_user)) -> VoiceConfigResponse:
    settings = await get_voice_settings(user["id"])
    return VoiceConfigResponse(settings=settings, runtime=voice_manager.status(settings), engines=engine_info(), profiles=await list_profiles(user["id"]), greeting=await greeting_cache.status(user["id"]))


@router.put("/config", response_model=VoiceConfigResponse)
async def update_voice_config(settings: VoiceSettings, user: dict = Depends(get_current_user)) -> VoiceConfigResponse:
    try:
        await save_voice_settings(user["id"], settings)
        if not settings.tts_enabled:
            await greeting_cache.close()
            await voice_manager.control("stop", settings)
        else:
            greeting_cache.schedule(user["id"], voice_manager.transition)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    await event_bus.publish(EventType.VOICE_UPDATED, {"user_id": user["id"]})
    return await get_voice_config(user)


@router.post("/runtime", response_model=VoiceRuntime)
async def control_voice_runtime(request: VoiceControl, user: dict = Depends(get_current_user)) -> VoiceRuntime:
    settings = await get_voice_settings(user["id"])
    try:
        result = await voice_manager.control(request.action, settings)
        if request.action == "stop" or result.state == "starting":
            await greeting_cache.close()
        if request.action != "stop":
            greeting_cache.schedule(user["id"], voice_manager.transition)
        return result
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error


@router.post("/profiles", response_model=VoiceProfile, status_code=201)
async def upload_voice_profile(
    name: str = Form(..., max_length=80), transcript: str = Form(..., max_length=2000),
    file: UploadFile = File(...), user: dict = Depends(get_current_user),
) -> VoiceProfile:
    try:
        content = await file.read(MAX_REFERENCE_BYTES + 1)
        return await create_profile(user["id"], name, transcript, content)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    finally:
        await file.close()


@router.get("/profiles/{profile_id}/audio", response_class=FileResponse, responses={200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}}})
async def voice_reference_audio(profile_id: str, user: dict = Depends(get_current_user)) -> FileResponse:
    try:
        _profile, path = await get_profile(user["id"], profile_id)
    except ValueError as error:
        raise HTTPException(404, "音色不存在") from error
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "private, no-store"})


@router.delete("/profiles/{profile_id}", status_code=204)
async def remove_voice_profile(profile_id: str, user: dict = Depends(get_current_user)) -> Response:
    try:
        await delete_profile(user["id"], profile_id)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return Response(status_code=204)


async def engine_stream(user_id: str, text: str) -> AsyncIterator[bytes]:
    settings = await get_voice_settings(user_id)
    if not settings.tts_enabled:
        raise HTTPException(409, "语音合成已关闭")
    runtime = voice_manager.status(settings)
    if runtime.state != "running" or runtime.restart_required:
        raise HTTPException(409, "请启动所选语音模型，切换模型后需重新加载")
    reference_path, reference_text = None, ""
    if settings.profile_id is not None:
        try:
            profile, path = await get_profile(user_id, settings.profile_id)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        reference_path, reference_text = str(path), profile.transcript
    if settings.engine == "fasterqwentts" and reference_path is None:
        raise HTTPException(422, "Qwen Base 需要参考音色，请上传并选择参考音频")
    payload = {"text": text, "reference_path": reference_path, "reference_text": reference_text, "vox": settings.vox.model_dump(), "qwen": settings.qwen.model_dump()}
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(180, connect=5)) as client:
            async with client.stream("POST", f"http://127.0.0.1:{config.VOICE_ENGINE_PORT}/speech", json=payload, headers={"X-Engine-Token": voice_manager.token}) as response:
                if not response.is_success:
                    raise HTTPException(502, "语音生成失败，请检查语音运行日志或稍后重试")
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
    except httpx.HTTPError as error:
        raise HTTPException(502, "语音引擎连接中断，请重新启动模型后重试") from error


@router.post("/speech", response_class=StreamingResponse, responses={200: {"content": {"audio/pcm": {"schema": {"type": "string", "format": "binary"}}}}})
async def voice_speech(request: VoiceSpeech, user: dict = Depends(get_current_user)) -> StreamingResponse:
    stream = engine_stream(user["id"], request.text)
    try:
        first = await anext(stream)
    except StopAsyncIteration as error:
        await stream.aclose()
        raise HTTPException(502, "语音引擎未生成音频") from error

    async def response_stream():
        try:
            yield first
            async for chunk in stream:
                yield chunk
        finally:
            await stream.aclose()

    return StreamingResponse(response_stream(), media_type="audio/pcm", headers={"X-Audio-Sample-Rate": "24000", "Cache-Control": "no-store"})


@router.post("/preview", response_class=Response, responses={200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}}})
async def voice_preview(request: VoicePreview, user: dict = Depends(get_current_user)) -> Response:
    chunks = bytearray()
    async for chunk in engine_stream(user["id"], request.text):
        chunks.extend(chunk)
        if len(chunks) > 24000 * 2 * 120:
            raise HTTPException(422, "试听音频超过两分钟，请缩短文本")
    if not chunks:
        raise HTTPException(502, "语音引擎未生成音频")
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(chunks)
    return Response(output.getvalue(), media_type="audio/wav", headers={"Cache-Control": "private, no-store"})
