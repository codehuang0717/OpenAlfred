import httpx
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials

from config import config
from routers.auth import get_current_user, security
from routers.threads import _lg_headers

router = APIRouter(prefix="/api/calls", tags=["calls"])

@router.get("/threads")
async def list_calls(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user: dict = Depends(get_current_user),
):
    """List all voice call threads owned by the current user."""
    headers = _lg_headers(credentials.credentials)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{config.LANGGRAPH_API_URL}/threads/search",
            headers=headers,
            json={
                "metadata": {"owner": user["id"], "type": "call"},
                "limit": 100,
            },
            timeout=10.0,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Failed to fetch calls")
        threads = resp.json()

    result = []
    for t in threads:
        metadata = t.get("metadata", {})
        room_name = metadata.get("room_name", "")
        title = metadata.get("title", "")
        is_outbound = room_name.startswith("outbound-") or "外拨" in title
        direction = "outbound" if is_outbound else "inbound"
        result.append({
            "thread_id": t["thread_id"],
            "title": metadata.get("title", "语音通话记录"),
            "updated_at": t.get("updated_at", t.get("created_at", "")),
            "created_at": t.get("created_at", ""),
            "direction": direction,
            "room_name": room_name,
        })

    result.sort(key=lambda x: x["updated_at"], reverse=True)
    return result
