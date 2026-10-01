import os
import asyncio
import logging
import httpx
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from PIL import Image

from core.database import get_setting, set_setting, get_user_bark_url, set_user_bark_url, get_onboarding_seen, set_onboarding_seen
from routers.auth import get_current_user
from services.screen_monitor import monitor_status, write_preferences
from services.screen_binding import configured_owner, require_local_binding_request, bind_on_desktop
from services.weather import (
    clear_weather_location,
    get_saved_weather_location,
    get_weather_summary,
    save_weather_location,
)

from schemas.responses import (
    AgentAvatarResponse,
    AgentConfigResponse,
    ModelResponse,
    ModelSelectionResponse,
    ModelSelectionUpdateResponse,
    NotifyConfigResponse,
    NotifyTestResponse,
    NotifyUpdateResponse,
    OnboardingResponse,
    OnboardingUpdateResponse,
    OnlineResponse,
    SavedWeatherLocationResponse,
    StatusResponse,
    SupervisorResponse,
    TimezoneResponse,
    WeatherLocationUpdateResponse,
    WeatherResponse,
    json_response,
)

router = APIRouter(prefix="/api", tags=["settings"])


class UserTimezoneRequest(BaseModel):
    timezone: str


@router.put("/settings/timezone", responses=json_response(TimezoneResponse, 200))
async def update_user_timezone(req: UserTimezoneRequest, user: dict = Depends(get_current_user)):
    from services.user_time import save_user_timezone
    try:
        return {"timezone": await save_user_timezone(user["id"], req.timezone)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

AGENT_AVATAR_DIR = Path(__file__).parent.parent / "uploads" / "agents"


@router.get("/agent/config", responses=json_response(AgentConfigResponse, 200))
async def get_agent_config(user: dict = Depends(get_current_user)):
    avatar_path = AGENT_AVATAR_DIR / f"{user['id']}.jpg"
    return {
        "agent_avatar_url": f"/static/agents/{user['id']}.jpg" if avatar_path.exists() else "",
    }


@router.post("/agent/avatar", responses=json_response(AgentAvatarResponse, 200))
async def upload_agent_avatar(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="图片大小不能超过 5MB")

    AGENT_AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = AGENT_AVATAR_DIR / f"tmp_{user['id']}"
    tmp_path.write_bytes(content)

    try:
        img = Image.open(tmp_path).convert("RGB")
        size = min(img.size)
        left = (img.size[0] - size) // 2
        top = (img.size[1] - size) // 2
        img = img.crop((left, top, left + size, top + size))
        img = img.resize((256, 256), Image.LANCZOS)
        img.save(AGENT_AVATAR_DIR / f"{user['id']}.jpg", "JPEG", quality=85)
    except Exception:
        raise HTTPException(status_code=400, detail="无法处理的图片格式")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {"status": "uploaded", "agent_avatar_url": f"/static/agents/{user['id']}.jpg"}
logger = logging.getLogger("settings-router")

class ModelSelectionRequest(BaseModel):
    model_selection: str = "gpt-cloud"

class SupervisorConfigRequest(BaseModel):
    recording_enabled: bool
    smart_supervision_enabled: bool

@router.get("/models", responses=json_response(list[ModelResponse], 200))
async def get_models():
    """Get list of available LLM models across all providers."""
    from core.config import config
    models = []

    # OpenAI GPT
    if config.OPENAI_API_KEY:
        models.append({
            "id": "gpt-cloud",
            "name": config.CLOUD_CHAT_MODEL,
            "provider": "OpenAI",
            "icon": "zap",
            "description": f"GPT model ({config.CLOUD_CHAT_MODEL})"
        })

    # Cerebras (OpenAI-compatible via Cerebras API)
    if config.CEREBRAS_API_KEY:
        models.append({
            "id": "cerebras",
            "name": config.CEREBRAS_CHAT_MODEL,
            "provider": "Cerebras",
            "icon": "cpu",
            "description": f"Cerebras Llama ({config.CEREBRAS_CHAT_MODEL})"
        })

    # Google Gemini
    if config.GOOGLE_API_KEY:
        models.append({
            "id": "gemini",
            "name": config.GEMINI_CHAT_MODEL,
            "provider": "Google",
            "icon": "sparkles",
            "description": f"Gemini model ({config.GEMINI_CHAT_MODEL})"
        })

    # DeepSeek (OpenAI-compatible)
    if config.DEEPSEEK_API_KEY:
        models.append({
            "id": "deepseek",
            "name": config.DEEPSEEK_FLASH_MODEL,
            "provider": "DeepSeek",
            "icon": "zap",
            "description": f"DeepSeek Flash ({config.DEEPSEEK_FLASH_MODEL})"
        })
        models.append({
            "id": "deepseek-pro",
            "name": config.DEEPSEEK_PRO_MODEL,
            "provider": "DeepSeek",
            "icon": "star",
            "description": f"DeepSeek Pro ({config.DEEPSEEK_PRO_MODEL})"
        })

    # Xiaomi MiMo (OpenAI-compatible)
    if config.MIMO_API_KEY:
        models.append({
            "id": "mimo",
            "name": config.MIMO_CHAT_MODEL,
            "provider": "Xiaomi",
            "icon": "sparkles",
            "description": f"Xiaomi MiMo ({config.MIMO_CHAT_MODEL})"
        })

    # Local Ollama
    models.append({
        "id": "gemma-local",
        "name": config.LOCAL_MODEL_NAME,
        "provider": "Ollama",
        "icon": "hard-drive",
        "description": "Local model for privacy and offline use."
    })

    return models

@router.get("/model/selection", responses=json_response(ModelSelectionResponse, 200))
async def get_model_selection_api(user: dict = Depends(get_current_user)):
    """Get the current globally selected LLM model."""
    selection = await get_setting("model_selection", "gpt-cloud")
    return {"model_selection": selection}

@router.post("/model/selection", responses=json_response(ModelSelectionUpdateResponse, 200))
async def set_model_selection_api(data: ModelSelectionRequest, user: dict = Depends(get_current_user)):
    """Update the globally selected LLM model."""
    await set_setting("model_selection", data.model_selection)
    return {"status": "updated", "model_selection": data.model_selection}

@router.get("/ollama/status", responses=json_response(OnlineResponse, 200))
async def check_ollama_status():
    """Check if local Ollama server is reachable."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags", timeout=2.0)
            return {"online": resp.status_code == 200}
    except Exception:
        return {"online": False}

@router.get("/supervisor/config", responses=json_response(SupervisorResponse, 200))
async def get_supervisor_config_api(user: dict = Depends(get_current_user)):
    """Get the current supervisor enabled status."""
    try:
        if configured_owner() is None:
            return {"binding_required": True, "recording_enabled": False,
                    "smart_supervision_enabled": False, "supervisor_running": False,
                    "screenpipe_running": False, "analysis_running": False, "error": None}
        return {"binding_required": False, **await monitor_status(user["id"])}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

@router.post("/supervisor/config", responses=json_response(SupervisorResponse, 200))
async def set_supervisor_config_api(data: SupervisorConfigRequest, user: dict = Depends(get_current_user)):
    """Set the supervisor enabled status."""
    try:
        await write_preferences(user["id"], data.recording_enabled, data.smart_supervision_enabled)
        return {"binding_required": False, **await monitor_status(user["id"])}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/supervisor/bind", responses=json_response(SupervisorResponse, 200))
async def bind_supervisor_account(request: Request, user: dict = Depends(get_current_user)):
    """Bind only the authenticated identity, with explicit physical desktop consent."""
    try:
        require_local_binding_request(request.client.host if request.client else "", request.headers)
        await asyncio.to_thread(bind_on_desktop, user["id"], user["username"])
        return {"binding_required": False, **await monitor_status(user["id"])}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


class NotifyConfigRequest(BaseModel):
    bark_url: str = ""


class WeatherLocationRequest(BaseModel):
    latitude: float
    longitude: float
    accuracy: Optional[float] = None
    label: str = "当前位置"
    source: str = "browser"


@router.get("/notify/config", responses=json_response(NotifyConfigResponse, 200))
async def get_notify_config(user: dict = Depends(get_current_user)):
    """Get the current user's push notification configuration."""
    bark_url = await get_user_bark_url(user["id"])
    return {"bark_url": bark_url}


@router.post("/notify/config", responses=json_response(NotifyUpdateResponse, 200))
async def set_notify_config(data: NotifyConfigRequest, user: dict = Depends(get_current_user)):
    """Update the current user's push notification configuration."""
    await set_user_bark_url(user["id"], data.bark_url)
    return {"status": "updated", "bark_url": data.bark_url}


@router.delete("/notify/config", responses=json_response(StatusResponse, 200))
async def unbind_notify_config(user: dict = Depends(get_current_user)):
    """Unbind (clear) the current user's Bark URL."""
    await set_user_bark_url(user["id"], "")
    return {"status": "unbound"}


@router.post("/notify/test", responses=json_response(NotifyTestResponse, 200))
async def test_notify(user: dict = Depends(get_current_user)):
    """Send a test notification to the current user's Bark device."""
    from services.notification import notification_service
    bark_url = await get_user_bark_url(user["id"])
    if not bark_url:
        raise HTTPException(status_code=400, detail="未配置 Bark URL，请先填入设备地址")

    success = await notification_service.send_bark_notification(
        body="如果你收到这条消息，说明 Bark 推送配置成功！",
        title="✅ OpenAlfred 连接测试",
        level="active",
        sound="birdsong",
        group="OpenAlfred-Test",
        bark_url=bark_url,
    )
    if success:
        return {"status": "ok", "message": "测试通知已发送"}
    else:
        raise HTTPException(status_code=502, detail="Bark 服务不可达，请检查 URL 是否正确")


@router.get("/weather/location", responses=json_response(SavedWeatherLocationResponse, 200))
async def get_weather_location(user: dict = Depends(get_current_user)):
    """Get the current user's default weather location."""
    return {"location": await get_saved_weather_location(user["id"])}


@router.post("/weather/location", responses=json_response(WeatherLocationUpdateResponse, 200))
async def set_weather_location(data: WeatherLocationRequest, user: dict = Depends(get_current_user)):
    """Save the current user's default weather location."""
    if not -90 <= data.latitude <= 90:
        raise HTTPException(status_code=400, detail="纬度必须在 -90 到 90 之间")
    if not -180 <= data.longitude <= 180:
        raise HTTPException(status_code=400, detail="经度必须在 -180 到 180 之间")

    payload = {
        "latitude": data.latitude,
        "longitude": data.longitude,
        "accuracy": data.accuracy,
        "label": data.label.strip() or "当前位置",
        "source": data.source.strip() or "browser",
    }
    saved = await save_weather_location(user["id"], payload)
    return {"status": "updated", "location": saved}


@router.delete("/weather/location", responses=json_response(StatusResponse, 200))
async def delete_weather_location(user: dict = Depends(get_current_user)):
    """Clear the current user's default weather location."""
    await clear_weather_location(user["id"])
    return {"status": "deleted"}


@router.get("/weather/current", responses=json_response(WeatherResponse, 200))
async def get_current_weather(user: dict = Depends(get_current_user)):
    """Get the current user's saved-location weather summary."""
    try:
        summary = await get_weather_summary(user_id=user["id"])
    except Exception as e:
        logger.warning("weather/current failed: %s", e)
        raise HTTPException(status_code=502, detail="天气服务暂时不可用")
    if not summary:
        return {"weather": None, "needs_location": True}
    return {"weather": summary, "needs_location": False}


class OnboardingRequest(BaseModel):
    seen: bool = True


@router.get("/onboarding", responses=json_response(OnboardingResponse, 200))
async def get_onboarding_status(user: dict = Depends(get_current_user)):
    """Check if the current user has seen the onboarding tutorial prompt."""
    seen = await get_onboarding_seen(user["id"])
    return {"seen": seen}


@router.post("/onboarding", responses=json_response(OnboardingUpdateResponse, 200))
async def set_onboarding_status(data: OnboardingRequest, user: dict = Depends(get_current_user)):
    """Mark the current user as having seen/dismissed the onboarding tutorial prompt."""
    await set_onboarding_seen(user["id"], data.seen)
    return {"status": "updated", "seen": data.seen}
