import logging
import httpx
from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from database import get_setting, set_setting
from routers.auth import get_current_user

router = APIRouter(prefix="/api", tags=["settings"])
logger = logging.getLogger("settings-router")

class ModelSelectionRequest(BaseModel):
    model_selection: str = "gpt-cloud"

class SupervisorConfigRequest(BaseModel):
    recording_enabled: bool
    smart_supervision_enabled: bool

@router.get("/models")
async def get_models():
    """Get list of available LLM models."""
    from config import config
    return [
        {
            "id": "gpt-cloud",
            "name": config.CLOUD_CHAT_MODEL,
            "provider": "OpenAI",
            "icon": "zap",
            "description": "Cloud model for complex tasks."
        },
        {
            "id": "gemma-local",
            "name": config.LOCAL_MODEL_NAME,
            "provider": "Ollama",
            "icon": "cpu",
            "description": "Local model for privacy and offline use."
        }
    ]

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

    from event_bus import event_bus, EventType
    await event_bus.publish(EventType.SUPERVISOR_WAKEUP)

    return {
        "status": "updated", 
        "recording_enabled": data.recording_enabled,
        "smart_supervision_enabled": data.smart_supervision_enabled
    }
