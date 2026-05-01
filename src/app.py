"""
FastAPI application — Business API layer for OpenAlfred.
Now refactored to use modular routers.
"""

import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from scheduler import check_and_send_pending_reminders
from routers import auth, todos, reminders, threads, calls, email, settings, multimodal

from utils.logger import setup_logging, get_logger

# Initialize unified logging
setup_logging(log_file="api.log")
logger = get_logger("api")

# --- Lifespan ---

async def run_scheduler():
    """Background task to check and send pending reminders."""
    logger.info("Internal scheduler started!")
    while True:
        try:
            await check_and_send_pending_reminders()
        except Exception as e:
            logger.error(f"Internal scheduler error: {e}")
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB
    await init_db()
    
    # Start internal scheduler for reminders (as a fallback or for simple standalone runs)
    asyncio.create_task(run_scheduler())
    logger.info("Internal scheduler task created")

    yield

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

@app.get("/")
async def root():
    return {"status": "online", "message": "OpenAlfred API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7788, reload=True)
