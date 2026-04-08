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

        await db.execute("""
            CREATE TABLE IF NOT EXISTS call_sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT DEFAULT 'active',
                call_type TEXT DEFAULT 'inbound'
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS call_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES call_sessions(id)
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT DEFAULT 'active',
                model_used TEXT DEFAULT 'gpt-cloud'
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
            )
        """)

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

        try:
            await db.execute(
                "ALTER TABLE chat_messages ADD COLUMN msg_id TEXT DEFAULT ''"
            )
        except Exception:
            pass

        try:
            await db.execute(
                "ALTER TABLE chat_sessions ADD COLUMN title TEXT DEFAULT '新对话'"
            )
        except Exception:
            pass

        await db.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

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


# ─── Call Session Functions ───────────────────────────────────────────────

async def create_call_session(id: str, call_type: str = "inbound"):
    started_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            """
            INSERT OR IGNORE INTO call_sessions (id, started_at, status, call_type)
            VALUES (?, ?, 'active', ?)
            """,
            (id, started_at, call_type),
        )
        await db.commit()


async def end_call_session(id: str):
    ended_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "UPDATE call_sessions SET status = 'ended', ended_at = ? WHERE id = ?",
            (ended_at, id),
        )
        await db.commit()


async def add_call_message(session_id: str, role: str, content: str):
    created_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            """
            INSERT INTO call_messages (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content, created_at),
        )
        await db.commit()


async def get_call_sessions() -> list[dict]:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM call_sessions ORDER BY started_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def get_call_messages(session_id: str) -> list[dict]:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM call_messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def get_call_session_history(session_id: str, max_turns: int = 10) -> list[dict]:
    """Get recent conversation history for LLM context (latest max_turns exchanges)."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        # Get last N*2 messages (each turn = user + assistant)
        async with db.execute(
            """
            SELECT role, content FROM call_messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, max_turns * 2),
        ) as cursor:
            rows = await cursor.fetchall()
            # Reverse to chronological order
            messages = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
            return messages


# ─── Chat Session Functions ───────────────────────────────────────────────

async def create_chat_session(id: str, model_used: str = "gpt-cloud"):
    started_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            """
            INSERT OR IGNORE INTO chat_sessions (id, started_at, status, model_used)
            VALUES (?, ?, 'active', ?)
            """,
            (id, started_at, model_used),
        )
        await db.commit()


async def end_chat_session(id: str):
    ended_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "UPDATE chat_sessions SET status = 'ended', ended_at = ? WHERE id = ?",
            (ended_at, id),
        )
        await db.commit()

async def get_latest_active_chat_session() -> str:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        async with db.execute(
            "SELECT id FROM chat_sessions WHERE status = 'active' ORDER BY started_at DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return row[0]
            return None

async def add_chat_message(session_id: str, role: str, content: str, msg_id: str = ""):
    created_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        if msg_id:
            async with db.execute("SELECT id FROM chat_messages WHERE msg_id = ?", (msg_id,)) as cursor:
                if await cursor.fetchone():
                    return
        await db.execute(
            """
            INSERT INTO chat_messages (session_id, role, content, created_at, msg_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, created_at, msg_id),
        )
        
        # Auto-update title on first user message
        if role == "user" and content and content.strip():
            async with db.execute("SELECT title FROM chat_sessions WHERE id = ?", (session_id,)) as cursor:
                row = await cursor.fetchone()
                if row and (not row[0] or row[0] == "新对话"):
                    # Use first 15 chars as title
                    new_title = content.strip()[:15]
                    if len(content.strip()) > 15:
                        new_title += "..."
                    await db.execute("UPDATE chat_sessions SET title = ? WHERE id = ?", (new_title, session_id))
                    
        await db.commit()


async def get_chat_sessions() -> list[dict]:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM chat_sessions ORDER BY started_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def get_chat_messages(session_id: str) -> list[dict]:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
