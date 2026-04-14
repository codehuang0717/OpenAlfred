import aiosqlite
import os
from typing import Optional, Literal
from datetime import datetime, timezone
from schema import TodoDict
from config import config

DATABASE_PATH = str(config.DB_PATH)
AUDIO_CACHE_DIR = str(config.ASSETS_DIR / "audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)


async def init_db():
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS todos (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                emoji TEXT DEFAULT '🎯',
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL,
                completed_at TEXT,
                deleted INTEGER DEFAULT 0,
                notes TEXT DEFAULT '',
                expected_completion_at TEXT
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY,
                title TEXT,
                subtitle TEXT,
                body TEXT NOT NULL,
                scheduled_at TEXT NOT NULL,
                sent INTEGER DEFAULT 0,
                level TEXT DEFAULT 'active',
                sound TEXT,
                created_at TEXT NOT NULL,
                delivery_method TEXT DEFAULT 'push',
                audio_path TEXT DEFAULT ''
            )
        """)

        await db.commit()

        try:
            await db.execute(
                "ALTER TABLE reminders ADD COLUMN delivery_method TEXT DEFAULT 'push'"
            )
        except Exception:
            pass

        try:
            await db.execute(
                "ALTER TABLE reminders ADD COLUMN audio_path TEXT DEFAULT ''"
            )
        except Exception:
            pass

        # Bark fields migrations
        for col in ["title", "subtitle", "sound"]:
            try:
                await db.execute(f"ALTER TABLE reminders ADD COLUMN {col} TEXT")
            except Exception:
                pass
        try:
            await db.execute("ALTER TABLE reminders ADD COLUMN level TEXT DEFAULT 'active'")
        except Exception:
            pass


        await db.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                display_name TEXT DEFAULT '',
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login_at TEXT
            )
        """)

        await db.commit()


# ─── User Functions ────────────────────────────────────────────────────────

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


# ─── Settings Functions ───────────────────────────────────────────────────

async def set_setting(key: str, value: str):
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value)
        )
        await db.commit()

async def get_setting(key: str, default: str = None) -> str:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        async with db.execute("SELECT value FROM settings WHERE key = ?", (key,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return row[0]
            return default



async def get_all_todos() -> list[TodoDict]:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM todos WHERE deleted = 0 ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def add_todo(
    id: str,
    title: str,
    description: str = "",
    emoji: str = "🎯",
    status: str = "pending",
    notes: str = "",
    expected_completion_at: Optional[str] = None,
):
    created_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            """
            INSERT INTO todos (id, title, description, emoji, status, created_at, completed_at, deleted, notes, expected_completion_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                id,
                title,
                description,
                emoji,
                status,
                created_at,
                None,
                notes,
                expected_completion_at,
            ),
        )
        await db.commit()


async def update_todo(
    id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    emoji: Optional[str] = None,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    expected_completion_at: Optional[str] = None,
):
    updates = []
    params = []

    if title is not None:
        updates.append("title = ?")
        params.append(title)
    if description is not None:
        updates.append("description = ?")
        params.append(description)
    if emoji is not None:
        updates.append("emoji = ?")
        params.append(emoji)
    if status is not None:
        updates.append("status = ?")
        params.append(status)
        if status == "completed":
            updates.append("completed_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())
        elif status == "pending":
            updates.append("completed_at = NULL")
    if notes is not None:
        updates.append("notes = ?")
        params.append(notes)
    if expected_completion_at is not None:
        updates.append("expected_completion_at = ?")
        params.append(expected_completion_at)

    if not updates:
        return

    params.append(id)
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            f"UPDATE todos SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        await db.commit()


async def delete_todo(id: str):
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "UPDATE todos SET deleted = 1 WHERE id = ?",
            (id,),
        )
        await db.commit()


async def get_todo_by_id(id: str):
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM todos WHERE id = ? AND deleted = 0",
            (id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def add_reminder(
    id: str,
    body: str,
    scheduled_at: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    level: str = "active",
    sound: Optional[str] = None,
    delivery_method: str = "push",
    audio_path: str = "",
):
    created_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            """
            INSERT INTO reminders (id, title, subtitle, body, scheduled_at, sent, level, sound, created_at, delivery_method, audio_path)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
            """,
            (
                id,
                title,
                subtitle,
                body,
                scheduled_at,
                level,
                sound,
                created_at,
                delivery_method,
                audio_path,
            ),
        )
        await db.commit()


async def get_pending_reminders():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    now_str = now.isoformat()

    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM reminders WHERE sent = 0 ORDER BY scheduled_at ASC"
        ) as cursor:
            rows = await cursor.fetchall()
            reminders = [dict(row) for row in rows]

    filtered = []
    for r in reminders:
        try:
            scheduled = datetime.fromisoformat(r["scheduled_at"].replace("Z", "+00:00"))
            if scheduled <= now:
                filtered.append(r)
        except:
            if r["scheduled_at"] <= now_str:
                filtered.append(r)

    return filtered


async def mark_reminder_sent(id: str):
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "UPDATE reminders SET sent = 1 WHERE id = ?",
            (id,),
        )
        await db.commit()


async def get_all_reminders():
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM reminders ORDER BY scheduled_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def delete_reminder(id: str):
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("DELETE FROM reminders WHERE id = ?", (id,))
        await db.commit()


