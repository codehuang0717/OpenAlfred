from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import httpx
from database import (
    get_all_todos,
    init_db,
    get_all_reminders,
    get_setting,
    set_setting,
)
from tools.reminder import check_and_send_pending_reminders


async def run_scheduler():
    """Background task to check and send pending reminders."""
    print("Scheduler started!")
    while True:
        try:
            print("Checking pending reminders...")
            await check_and_send_pending_reminders()
        except Exception as e:
            print(f"Scheduler error: {e}")
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    asyncio.create_task(run_scheduler())
    print("Scheduler task created")

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/todos")
async def get_todos():
    """Get all todos from database."""
    todos = await get_all_todos()
    return todos


@app.get("/api/reminders")
async def get_reminders():
    """Get all reminders from database."""
    reminders = await get_all_reminders()
    return reminders


@app.post("/api/reminders/check")
async def check_reminders():
    """Manually trigger checking pending reminders."""
    await check_and_send_pending_reminders()
    return {"status": "checked"}



@app.get("/api/models")
async def get_available_models():
    """Return available model options."""
    return [
        {
            "id": "gpt-cloud",
            "name": "GPT-5.4 Nano",
            "provider": "openai",
            "icon": "cloud",
            "description": "OpenAI 云端模型，响应快速稳定",
        },
        {
            "id": "gemma-local",
            "name": "Gemma4 E2B",
            "provider": "ollama",
            "icon": "computer",
            "description": "本地 Ollama 部署，隐私安全，无网络延迟",
        },
    ]


@app.post("/api/model/check-ollama")
async def check_ollama_status():
    """Check if local Ollama is running and responsive."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                has_gemma = any("gemma4" in m for m in models)
                return {
                    "online": True,
                    "models": models,
                    "has_gemma4": has_gemma,
                }
    except Exception:
        pass
    return {"online": False, "models": [], "has_gemma4": False}


@app.get("/api/model/selection")
async def get_model_selection_api():
    """Get the globally selected model type."""
    selection = await get_setting("model_selection", "gpt-cloud")
    return {"model_selection": selection}


@app.post("/api/model/selection")
async def set_model_selection_api(data: dict):
    """Set the globally selected model type."""
    selection = data.get("model_selection", "gpt-cloud")
    await set_setting("model_selection", selection)
    
    # Notifying main.py is no longer needed; ModelSwitchMiddleware reads from CopilotKit state directly.
        
    return {"status": "updated", "model_selection": selection}
