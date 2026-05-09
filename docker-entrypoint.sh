#!/usr/bin/env bash
# OpenAlfred Agent - Docker Entrypoint
# Starts all Python backend services inside the container.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[entrypoint]${NC} $1"; }
warn() { echo -e "${YELLOW}[entrypoint]${NC} $1"; }
die()  { echo -e "${RED}[entrypoint]${NC} $1"; exit 1; }

PIDS=()

cleanup() {
    log "Shutting down all services..."
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    log "All services stopped."
    exit 0
}
trap cleanup SIGTERM SIGINT SIGQUIT

# Ensure .env exists
if [ ! -f /app/.env ]; then
    warn ".env not found at /app/.env — copying from .env.example"
    cp /app/.env.example /app/.env
fi

cd /app

log "============================================"
log "  OpenAlfred Agent - Starting Services      "
log "============================================"

# 1. LangGraph API Server (port 2024)
log "Starting LangGraph API on :2024..."
langgraph dev --allow-blocking --host 0.0.0.0 --port 2024 &
PIDS+=($!)
sleep 3

# 2. FastAPI Business API (port 7788)
log "Starting FastAPI on :7788..."
cd /app/src
python -m uvicorn app:app --host 0.0.0.0 --port 7788 &
PIDS+=($!)
cd /app

# 3. Background Worker
log "Starting Background Worker..."
python /app/src/worker.py &
PIDS+=($!)

# 4. LiveKit Cloud Worker (port 5883)
log "Starting LiveKit Cloud Worker on :5883..."
LIVEKIT_HTTP_SERVER_PORT=5883 python /app/src/livekit_worker.py dev &
PIDS+=($!)

# 5. LiveKit Local Worker (port 5884)
log "Starting LiveKit Local Worker on :5884..."
LIVEKIT_HTTP_SERVER_PORT=5884 python /app/src/livekit_worker.py dev --local &
PIDS+=($!)

# 6. Supervisor
log "Starting Proactive Supervisor..."
python /app/src/supervisor.py &
PIDS+=($!)

log "============================================"
log "  All services started!                     "
log "  LangGraph:  http://localhost:2024          "
log "  FastAPI:    http://localhost:7788          "
log "  LK Cloud:   http://localhost:5883          "
log "  LK Local:   http://localhost:5884          "
log "============================================"

# Wait for all background processes
wait
