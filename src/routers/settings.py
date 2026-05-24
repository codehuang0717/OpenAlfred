import os
import logging
import httpx
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image

from core.database import get_setting, set_setting, get_user_bark_url, set_user_bark_url
from routers.auth import get_current_user

router = APIRouter(prefix="/api", tags=["settings"])

AGENT_AVATAR_DIR = Path(__file__).parent.parent / "uploads" / "agents"


@router.get("/agent/config")
async def get_agent_config(user: dict = Depends(get_current_user)):
    avatar_path = AGENT_AVATAR_DIR / f"{user['id']}.jpg"
    return {
        "agent_avatar_url": f"/static/agents/{user['id']}.jpg" if avatar_path.exists() else "",
    }


@router.post("/agent/avatar")
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

@router.get("/models")
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

    # Local Ollama
    models.append({
        "id": "gemma-local",
        "name": config.LOCAL_MODEL_NAME,
        "provider": "Ollama",
        "icon": "hard-drive",
        "description": "Local model for privacy and offline use."
    })

    return models

@router.get("/model/selection")
async def get_model_selection_api(user: dict = Depends(get_current_user)):
    """Get the current globally selected LLM model."""
    selection = await get_setting("model_selection", "gpt-cloud")
    return {"model_selection": selection}

@router.post("/model/selection")
async def set_model_selection_api(data: ModelSelectionRequest, user: dict = Depends(get_current_user)):
    """Update the globally selected LLM model."""
    await set_setting("model_selection", data.model_selection)
    return {"status": "updated", "model_selection": data.model_selection}

@router.get("/ollama/status")
async def check_ollama_status():
    """Check if local Ollama server is reachable."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags", timeout=2.0)
            return {"online": resp.status_code == 200}
    except Exception:
        return {"online": False}

@router.get("/supervisor/config")
async def get_supervisor_config_api(user: dict = Depends(get_current_user)):
    """Get the current supervisor enabled status."""
    recording_enabled_str = await get_setting("recording_enabled", "true")
    smart_supervision_enabled_str = await get_setting("smart_supervision_enabled", "true")
    
    # Fallback for backward compatibility
    if not recording_enabled_str and not smart_supervision_enabled_str:
        old_enabled_str = await get_setting("supervisor_enabled", "true")
        recording_enabled_str = old_enabled_str
        smart_supervision_enabled_str = old_enabled_str

    return {
        "recording_enabled": recording_enabled_str.lower() == "true",
        "smart_supervision_enabled": smart_supervision_enabled_str.lower() == "true"
    }

@router.post("/supervisor/config")
async def set_supervisor_config_api(data: SupervisorConfigRequest, user: dict = Depends(get_current_user)):
    """Set the supervisor enabled status."""
    await set_setting("recording_enabled", str(data.recording_enabled).lower())
    await set_setting("smart_supervision_enabled", str(data.smart_supervision_enabled).lower())
    
    # Update legacy setting for compatibility
    await set_setting("supervisor_enabled", str(data.smart_supervision_enabled).lower())

    from core.event_bus import event_bus, EventType
    await event_bus.publish(EventType.SUPERVISOR_WAKEUP)

    return {
        "status": "updated",
        "recording_enabled": data.recording_enabled,
        "smart_supervision_enabled": data.smart_supervision_enabled
    }


class NotifyConfigRequest(BaseModel):
    bark_url: str = ""


@router.get("/notify/config")
async def get_notify_config(user: dict = Depends(get_current_user)):
    """Get the current user's push notification configuration."""
    bark_url = await get_user_bark_url(user["id"])
    return {"bark_url": bark_url}


@router.post("/notify/config")
async def set_notify_config(data: NotifyConfigRequest, user: dict = Depends(get_current_user)):
    """Update the current user's push notification configuration."""
    await set_user_bark_url(user["id"], data.bark_url)
    return {"status": "updated", "bark_url": data.bark_url}


@router.delete("/notify/config")
async def unbind_notify_config(user: dict = Depends(get_current_user)):
    """Unbind (clear) the current user's Bark URL."""
    await set_user_bark_url(user["id"], "")
    return {"status": "unbound"}


@router.post("/notify/test")
async def test_notify(user: dict = Depends(get_current_user)):
    """Send a test notification to the current user's Bark device."""
    from services.notification import notification_service
    from core.config import config as app_config

    bark_url = await get_user_bark_url(user["id"]) or app_config.BARK_URL
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
