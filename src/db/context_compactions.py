"""Versioned, owner-scoped rolling summaries; legacy free-text summaries stay untouched."""

from datetime import datetime, timezone
from db.connection import get_db
from utils.auth_utils import require_explicit_user_id


async def load_compaction(user_id: str, thread_id: str) -> dict | None:
    user_id = require_explicit_user_id(user_id)
    if not thread_id or thread_id == "default_thread":
        raise ValueError("Explicit thread_id required")
    async with get_db() as db:
        async with db.execute("SELECT * FROM context_compactions WHERE user_id=? AND thread_id=?", (user_id, thread_id)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def save_compaction(user_id: str, thread_id: str, summary: str, count: int, digest: str, revision: int) -> None:
    user_id = require_explicit_user_id(user_id)
    if not thread_id or thread_id == "default_thread" or count < 0:
        raise ValueError("Invalid compaction scope or cursor")
    async with get_db() as db:
        now = datetime.now(timezone.utc).isoformat()
        if revision == 0:
            cursor = await db.execute(
                "INSERT OR IGNORE INTO context_compactions VALUES (?, ?, ?, ?, ?, 1, ?)",
                (user_id, thread_id, summary, count, digest, now),
            )
        else:
            cursor = await db.execute(
                "UPDATE context_compactions SET summary=?, covered_count=?, covered_hash=?, revision=revision+1, updated_at=? WHERE user_id=? AND thread_id=? AND revision=?",
                (summary, count, digest, now, user_id, thread_id, revision),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("上下文摘要发生并发更新，请重试本轮；未覆盖其他请求")
        await db.commit()
