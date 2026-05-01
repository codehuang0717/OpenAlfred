"""
User repository — CRUD operations for the users table.
"""

import aiosqlite
from typing import Optional
from datetime import datetime, timezone
from db.connection import DATABASE_PATH


async def create_user(id: str, username: str, display_name: str, password_hash: str) -> dict:
    created_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            """INSERT INTO users (id, username, display_name, password_hash, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (id, username, display_name, password_hash, created_at),
        )
        await db.commit()
    return {
        "id": id,
        "username": username,
        "display_name": display_name,
        "created_at": created_at,
    }


async def get_user_by_username(username: str) -> Optional[dict]:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def get_user_by_id(user_id: str) -> Optional[dict]:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, username, display_name, created_at, last_login_at FROM users WHERE id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def update_user_last_login(user_id: str):
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?", (now, user_id)
        )
        await db.commit()

async def get_active_user() -> Optional[dict]:
    """Retrieve the user who logged in most recently."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, username, display_name FROM users ORDER BY last_login_at DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None
