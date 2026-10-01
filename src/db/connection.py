"""
Database connection management and schema initialization.
"""

import aiosqlite
import os
from core.config import config

DATABASE_PATH = str(config.DB_PATH)
AUDIO_CACHE_DIR = str(config.ASSETS_DIR / "audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)


async def _table_columns(db, table: str) -> set[str]:
    async with db.execute(f"PRAGMA table_info({table})") as cursor:
        return {row[1] for row in await cursor.fetchall()}


async def _ensure_column(
    db,
    table: str,
    column: str,
    definition: str,
    *,
    require_empty_table: bool = False,
) -> bool:
    """Add a known schema column only when absent; propagate all real DB errors."""
    if column in await _table_columns(db, table):
        return False
    if require_empty_table:
        async with db.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
            count = (await cursor.fetchone())[0]
        if count:
            raise RuntimeError(
                f"Cannot add required {table}.{column}: {count} existing rows need explicit ownership"
            )
    await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    return True


async def _reject_invalid_user_ids(db, table: str) -> None:
    for operation in ("INSERT", "UPDATE"):
        trigger = f"reject_{table}_invalid_user_{operation.lower()}"
        await db.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS {trigger}
            BEFORE {operation} ON {table}
            WHEN NEW.user_id IS NULL
              OR trim(NEW.user_id) = ''
              OR NEW.user_id = 'default'
            BEGIN
                SELECT RAISE(ABORT, '{table}.user_id must be a concrete user');
            END
            """
        )


async def init_db():
    """Create all tables, run migrations, and enable WAL mode."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Enable WAL mode for concurrent read/write and set busy timeout
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("PRAGMA foreign_keys=ON")
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
                user_id TEXT NOT NULL
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
                audio_path TEXT DEFAULT '',
                user_id TEXT NOT NULL
            )
        """)

        await db.commit()

        await _ensure_column(db, "reminders", "delivery_method", "TEXT DEFAULT 'push'")
        await _ensure_column(db, "reminders", "audio_path", "TEXT DEFAULT ''")
            
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
            await _ensure_column(db, "reminders", col, "TEXT")
        await _ensure_column(db, "reminders", "level", "TEXT DEFAULT 'active'")

        await _ensure_column(db, "todos", "scheduled_start_at", "TEXT")
        # ── user_id migration (multi-user isolation) ──
        await _ensure_column(
            db, "todos", "user_id", "TEXT NOT NULL", require_empty_table=True
        )
        await _ensure_column(
            db, "todos", "notification_sent", "INTEGER DEFAULT 0"
        )
        await _ensure_column(
            db, "reminders", "user_id", "TEXT NOT NULL", require_empty_table=True
        )

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
                last_login_at TEXT,
                sip_extension TEXT DEFAULT '',
                sip_password TEXT DEFAULT ''
            )
        """)

        # Migration: add sip columns to existing users table
        for col in ["sip_extension", "sip_password"]:
            await _ensure_column(db, "users", col, "TEXT DEFAULT ''")

        # Migration: add bark_url for per-user push notifications
        await _ensure_column(db, "users", "bark_url", "TEXT DEFAULT ''")

        # Migration: add onboarding_seen for new-user tutorial prompt
        onboarding_added = await _ensure_column(
            db, "users", "onboarding_seen", "INTEGER DEFAULT 0"
        )
        if onboarding_added:
            # Existing users don't need the onboarding popup
            await db.execute("UPDATE users SET onboarding_seen = 1")

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

        await db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                title TEXT DEFAULT '',
                file_type TEXT DEFAULT 'txt',
                chunk_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS image_lookup (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                url TEXT NOT NULL DEFAULT '',
                alt TEXT DEFAULT '',
                filename TEXT DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS context_compactions (
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                covered_count INTEGER NOT NULL,
                covered_hash TEXT NOT NULL,
                revision INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, thread_id)
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_apps (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, kind)
            )
        """)
        await _ensure_column(db, "user_apps", "published_revision_id", "TEXT")
        await _ensure_column(
            db, "user_apps", "status", "TEXT NOT NULL DEFAULT 'published'"
        )
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_app_revisions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                app_id TEXT NOT NULL,
                revision_number INTEGER NOT NULL,
                renderer TEXT NOT NULL CHECK(renderer IN ('catalog', 'html')),
                source_json TEXT NOT NULL,
                validation_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL CHECK(status IN ('ready', 'published')),
                created_at TEXT NOT NULL,
                UNIQUE(app_id, revision_number),
                FOREIGN KEY(app_id) REFERENCES user_apps(id) ON DELETE CASCADE
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_app_jobs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                app_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued', 'generating', 'ready', 'failed', 'cancelled', 'interrupted')),
                error TEXT,
                revision_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(app_id) REFERENCES user_apps(id) ON DELETE CASCADE
            )
        """)
        await _ensure_column(db, "user_app_jobs", "stage", "TEXT NOT NULL DEFAULT 'model'")
        await _ensure_column(db, "user_app_jobs", "origin", "TEXT NOT NULL DEFAULT 'tool'")
        # SQLite cannot ALTER a CHECK constraint. Preserve legacy job records
        # while extending the lifecycle; this migration is transactional.
        async with db.execute("SELECT sql FROM sqlite_master WHERE name = 'user_app_jobs'") as cursor:
            jobs_sql = (await cursor.fetchone())[0]
        if "'cancelled'" not in jobs_sql:
            await db.execute("ALTER TABLE user_app_jobs RENAME TO user_app_jobs_legacy")
            await db.execute("""
                CREATE TABLE user_app_jobs (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, app_id TEXT NOT NULL,
                    prompt TEXT NOT NULL, model TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                        ('queued', 'generating', 'ready', 'failed', 'cancelled', 'interrupted')),
                    error TEXT, revision_id TEXT, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, stage TEXT NOT NULL DEFAULT 'model',
                    origin TEXT NOT NULL DEFAULT 'tool',
                    FOREIGN KEY(app_id) REFERENCES user_apps(id) ON DELETE CASCADE
                )
            """)
            await db.execute("""
                INSERT INTO user_app_jobs SELECT id, user_id, app_id, prompt, model,
                    status, error, revision_id, created_at, updated_at, stage, origin
                FROM user_app_jobs_legacy
            """)
            await db.execute("DROP TABLE user_app_jobs_legacy")
        for column, definition in {
            "context_json": "TEXT NOT NULL DEFAULT '{}'",
            "report": "TEXT",
            "metrics_json": "TEXT NOT NULL DEFAULT '{}'",
            "epoch": "INTEGER NOT NULL DEFAULT 0",
            "runner_id": "TEXT",
            "lease_until": "REAL",
        }.items():
            await _ensure_column(db, "user_app_jobs", column, definition)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_coding_jobs_queue ON user_app_jobs(status, created_at)")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_app_job_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                user_id TEXT NOT NULL, stage TEXT NOT NULL, message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES user_app_jobs(id) ON DELETE CASCADE
            )
        """)
        # Existing trusted timeline panels become revision 1 without changing
        # their IDs or published behavior.
        await db.execute("""
            INSERT OR IGNORE INTO user_app_revisions
                (id, user_id, app_id, revision_number, renderer, source_json,
                 validation_json, status, created_at)
            SELECT id || ':v1', user_id, id, 1, 'catalog', spec_json,
                   '{"catalog":true}', 'published', created_at
            FROM user_apps
            WHERE kind = 'todo_timeline' AND published_revision_id IS NULL
        """)
        await db.execute("""
            UPDATE user_apps SET published_revision_id = id || ':v1'
            WHERE kind = 'todo_timeline' AND published_revision_id IS NULL
        """)

        image_user_added = await _ensure_column(
            db, "image_lookup", "user_id", "TEXT"
        )
        if image_user_added:
            await db.execute(
                """
                UPDATE image_lookup
                SET user_id = (
                    SELECT documents.user_id
                    FROM documents
                    WHERE documents.id = image_lookup.document_id
                )
                """
            )

        await _reject_invalid_user_ids(db, "todos")
        await _reject_invalid_user_ids(db, "reminders")
        await _reject_invalid_user_ids(db, "documents")
        await _reject_invalid_user_ids(db, "image_lookup")
        await _reject_invalid_user_ids(db, "user_apps")
        await _reject_invalid_user_ids(db, "user_app_revisions")
        await _reject_invalid_user_ids(db, "user_app_jobs")

        from db.email_drafts import init_schema as init_mail_schema
        await init_mail_schema(db)
        await _reject_invalid_user_ids(db, "email_drafts")
        await _reject_invalid_user_ids(db, "email_send_jobs")

        await db.commit()


from contextlib import asynccontextmanager


@asynccontextmanager
async def get_db() -> aiosqlite.Connection:
    """Async context manager: yields a DB connection with recommended PRAGMAs."""
    db = await aiosqlite.connect(DATABASE_PATH)
    try:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute("PRAGMA synchronous=NORMAL")
        yield db
    finally:
        await db.close()
