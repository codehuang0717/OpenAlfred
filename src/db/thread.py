"""
Thread memory repository — conversation summary persistence.
"""

import aiosqlite
from datetime import datetime, timezone
from db.connection import DATABASE_PATH


async def get_thread_memory(thread_id: str) -> tuple[str, int]:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        async with db.execute(
            "SELECT conversation_summary, summarized_count FROM thread_memories WHERE thread_id = ?",
            (thread_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return row[0], row[1]
            return "", 0

async def set_thread_memory(thread_id: str, summary: str, count: int):
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            INSERT INTO thread_memories (thread_id, conversation_summary, summarized_count, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                conversation_summary = excluded.conversation_summary,
                summarized_count = excluded.summarized_count,
                updated_at = excluded.updated_at
        """, (thread_id, summary, count, now))
        await db.commit()
