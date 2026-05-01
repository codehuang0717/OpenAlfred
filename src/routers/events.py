import asyncio
import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from routers.auth import get_current_user
from event_bus import event_bus
from utils.logger import get_logger

logger = get_logger("router.events")

router = APIRouter(prefix="/api/events", tags=["events"])

@router.get("/stream")
async def event_stream(user: dict = Depends(get_current_user)):
    """SSE endpoint — subscribes to Redis and streams events to the frontend."""
    
    async def generate():
        logger.info(f"User {user['id']} connected to event stream")
        
        # Initial keep-alive or connection confirmation
        yield f"data: {json.dumps({'type': 'connected', 'user_id': user['id']})}\n\n"
        
        try:
            # Subscribe to all relevant patterns
            # Note: EventBus.subscribe handles the CHANNEL_PREFIX
            async for event in event_bus.subscribe("todo.*", "reminder.*", "supervisor.*"):
                # Filter by user_id if present in event data (optional but good for multi-user)
                event_user_id = event.get("data", {}).get("user_id")
                if event_user_id and event_user_id != user["id"]:
                    continue
                
                logger.debug(f"Streaming event to {user['id']}: {event['type']}")
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            logger.info(f"Event stream for user {user['id']} disconnected")
            raise
        except Exception as e:
            logger.error(f"Error in event stream for user {user['id']}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no", # For Nginx compatibility
        }
    )
