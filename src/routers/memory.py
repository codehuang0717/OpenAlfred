"""
Memory router — REST API for reading/writing L1 user memory files.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from routers.auth import get_current_user
from logic.memory_manager import memory_manager, ALL_L1_FILES

logger = logging.getLogger("memory-router")

router = APIRouter(prefix="/api", tags=["memory"])

CATEGORY_MAP = {
    "profile": "profile.md",
    "preferences": "preferences.md",
    "learned_patterns": "learned_patterns.md",
    "relationship": "relationship.md",
}

CATEGORY_TITLES = {
    "profile": "用户基本信息",
    "preferences": "用户偏好",
    "learned_patterns": "学习到的行为模式",
    "relationship": "关系状态",
}


class UpdateMemoryRequest(BaseModel):
    content: str


@router.get("/memory")
async def get_memories(user: dict = Depends(get_current_user)):
    """Return all L1 memory files for the authenticated user."""
    user_id = user["id"]
    memory_manager._ensure_user_dir(user_id)
    user_dir = memory_manager._user_dir(user_id)

    result = {}
    for key, filename in CATEGORY_MAP.items():
        raw = memory_manager._read_file(user_dir / filename)
        content = memory_manager._strip_comments(raw).strip()
        result[key] = {
            "filename": filename,
            "title": CATEGORY_TITLES.get(key, filename),
            "content": content,
        }

    return result


@router.put("/memory/{category}")
async def update_memory(category: str, req: UpdateMemoryRequest, user: dict = Depends(get_current_user)):
    """Overwrite a specific L1 memory file for the authenticated user."""
    if category not in CATEGORY_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid memory category: {category}")

    user_id = user["id"]
    memory_manager._ensure_user_dir(user_id)
    user_dir = memory_manager._user_dir(user_id)
    filepath = user_dir / CATEGORY_MAP[category]

    filepath.write_text(req.content, encoding="utf-8")
    logger.info(f"[update_memory] user={user_id} file={CATEGORY_MAP[category]}")

    return {"status": "updated", "filename": CATEGORY_MAP[category]}
