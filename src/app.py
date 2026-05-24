"""
FastAPI application — Business API layer for OpenAlfred.
Now refactored to use modular routers.
"""

import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.database import init_db
from core.event_bus import event_bus
from routers import auth, todos, reminders, threads, calls, email, settings, multimodal, events, rag, memory

from utils.logger import setup_logging, get_logger

# Initialize unified logging
setup_logging(log_file="api.log")
logger = get_logger("api")

# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await event_bus.connect()

    logger.info("Database initialized. EventBus connected.")

    yield

    await event_bus.close()


# --- App Instance ---

app = FastAPI(
    title="OpenAlfred API",
    description="Modular business API layer for the OpenAlfred AI agent ecosystem.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Middleware ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Ensure uploads dir exists at import time ---

UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
(UPLOADS_DIR / "avatars").mkdir(parents=True, exist_ok=True)

# --- Register Routers ---

app.include_router(auth.router)
app.include_router(todos.router)
app.include_router(reminders.router)
app.include_router(threads.router)
app.include_router(calls.router)
app.include_router(email.router)
app.include_router(settings.router)
app.include_router(multimodal.router)
app.include_router(events.router)
app.include_router(rag.router)
app.include_router(rag.images_router)
app.include_router(memory.router)

# --- Static Files ---

app.mount("/static", StaticFiles(directory=str(UPLOADS_DIR)), name="static")


@app.get("/")
async def root():
    return {"status": "online", "message": "OpenAlfred API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7788, reload=True)
