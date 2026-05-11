"""
FastAPI application — Business API layer for OpenAlfred.
Now refactored to use modular routers.
"""

import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.database import init_db
from core.event_bus import event_bus
from routers import auth, todos, reminders, threads, calls, email, settings, multimodal, events

from utils.logger import setup_logging, get_logger

# Initialize unified logging
setup_logging(log_file="api.log")
logger = get_logger("api")

# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB
    await init_db()

    # Initialize EventBus (Redis)
    await event_bus.connect()

    # Load MCP tools (deferred from import time when event loop was running)
    from tools import ensure_tools_loaded
    await ensure_tools_loaded()

    logger.info("Database initialized. EventBus connected. MCP tools loaded. Reminder scheduling is handled by worker.py via Redis events.")

    yield

    # Cleanup
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


@app.get("/")
async def root():
    return {"status": "online", "message": "OpenAlfred API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7788, reload=True)
