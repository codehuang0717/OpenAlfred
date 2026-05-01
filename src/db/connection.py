"""
Database connection management and schema initialization.
"""

import aiosqlite
import os
from config import config

DATABASE_PATH = str(config.DB_PATH)
AUDIO_CACHE_DIR = str(config.ASSETS_DIR / "audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)


async def init_db():
    """Create all tables and run migrations."""
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
                expected_completion_at TEXT,
                scheduled_start_at TEXT,
                notification_sent INTEGER DEFAULT 0,
                user_id TEXT DEFAULT 'default'
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
            
        await db.execute("""
            CREATE TABLE IF NOT EXISTS thread_memories (
                thread_id TEXT PRIMARY KEY,
                conversation_summary TEXT DEFAULT '',
                summarized_count INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS supervisor_sessions (
                user_id TEXT PRIMARY KEY,
                is_distracted INTEGER DEFAULT 0,
                distraction_start_time TEXT,
                last_alert_time TEXT,
                consecutive_distractions INTEGER DEFAULT 0,
                last_decision TEXT,
                updated_at TEXT NOT NULL
            )
        """)
        await db.commit()

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
                "ALTER TABLE todos ADD COLUMN scheduled_start_at TEXT"
            )
        except Exception:
            pass
        # ── user_id migration (multi-user isolation) ──
        try:
            await db.execute(
                "ALTER TABLE todos ADD COLUMN user_id TEXT DEFAULT 'default'"
            )
        except Exception:
            pass
        try:
            await db.execute(
                "ALTER TABLE todos ADD COLUMN notification_sent INTEGER DEFAULT 0"
            )
        except Exception:
            pass

        try:
            await db.execute(
                "ALTER TABLE reminders ADD COLUMN user_id TEXT DEFAULT 'default'"
            )
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

        await db.execute("""
            CREATE TABLE IF NOT EXISTS email_credentials (
                account_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                email_address TEXT NOT NULL,
                provider TEXT NOT NULL,
                imap_server TEXT,
                imap_port INTEGER,
                smtp_server TEXT,
                smtp_port INTEGER,
                encrypted_password TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        await db.commit()
