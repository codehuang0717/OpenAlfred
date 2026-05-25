"""
User repository — CRUD operations for the users table.
"""

import secrets
import logging
from typing import Optional
from datetime import datetime, timezone
from db.connection import get_db

logger = logging.getLogger("db-user")

SIP_EXTENSION_START = 101  # reserve 100 for supervisor


async def _get_next_extension() -> str:
    """Auto-assign the next available SIP extension number."""
    async with get_db() as db:
        async with db.execute(
            "SELECT MAX(CAST(sip_extension AS INTEGER)) FROM users WHERE sip_extension != ''"
        ) as cursor:
            row = await cursor.fetchone()
            max_ext = row[0] if row and row[0] else SIP_EXTENSION_START - 1
            return str(max_ext + 1)


def _generate_sip_password(length: int = 12) -> str:
    """Generate a random SIP password."""
    return secrets.token_hex(length // 2)[:length]


async def create_user(
    id: str, username: str, display_name: str, password_hash: str,
    sip_extension: Optional[str] = None, sip_password: Optional[str] = None,
) -> dict:
    created_at = datetime.now(timezone.utc).isoformat()
    ext = sip_extension or await _get_next_extension()
    pwd = sip_password or _generate_sip_password()
    async with get_db() as db:
        await db.execute(
            """INSERT INTO users (id, username, display_name, password_hash,
                                  created_at, sip_extension, sip_password)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (id, username, display_name, password_hash, created_at, ext, pwd),
        )
        await db.commit()
    logger.info(
        f"[create_user] username={username} id={id} "
        f"sip_extension={ext} sip_password={pwd[:3]}***"
    )
    return {
        "id": id,
        "username": username,
        "display_name": display_name,
        "created_at": created_at,
        "sip_extension": ext,
        "sip_password": pwd,
    }


async def get_user_by_username(username: str) -> Optional[dict]:
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def get_user_by_sip_extension(extension: str) -> Optional[dict]:
    """Lookup user by their SIP extension number."""
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM users WHERE sip_extension = ?", (extension,)
        ) as cursor:
            row = await cursor.fetchone()
            result = dict(row) if row else None
            logger.debug(
                f"[get_user_by_sip_extension] extension={extension} "
                f"-> {'found: ' + result['username'] if result else 'NOT FOUND'}"
            )
            return result


async def get_user_by_id(user_id: str) -> Optional[dict]:
    async with get_db() as db:
        async with db.execute(
            "SELECT id, username, display_name, sip_extension, sip_password, "
            "bark_url, created_at, last_login_at FROM users WHERE id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def update_user(user_id: str, display_name: str = None):
    """Update user profile fields. Only updates provided (non-None) fields."""
    updates = []
    params = []
    if display_name is not None:
        updates.append("display_name = ?")
        params.append(display_name)
    if not updates:
        return
    params.append(user_id)
    async with get_db() as db:
        await db.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )
        await db.commit()
    logger.info(f"[update_user] id={user_id} fields={list(dict(zip(updates, params)).keys())}")


async def update_user_last_login(user_id: str):
    now = datetime.now(timezone.utc).isoformat()
    async with get_db() as db:
        await db.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?", (now, user_id)
        )
        await db.commit()

async def get_user_password_hash(user_id: str) -> Optional[str]:
    async with get_db() as db:
        async with db.execute(
            "SELECT password_hash FROM users WHERE id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def update_user_password(user_id: str, password_hash: str):
    async with get_db() as db:
        await db.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id)
        )
        await db.commit()
    logger.info(f"[update_user_password] id={user_id}")

async def get_user_bark_url(user_id: str) -> str:
    """Get the Bark push notification URL for a specific user. Returns empty string if not set."""
    async with get_db() as db:
        async with db.execute(
            "SELECT bark_url FROM users WHERE id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row and row[0] else ""


async def set_user_bark_url(user_id: str, bark_url: str):
    """Set the Bark push notification URL for a specific user."""
    async with get_db() as db:
        await db.execute(
            "UPDATE users SET bark_url = ? WHERE id = ?", (bark_url, user_id)
        )
        await db.commit()
    logger.info(f"[set_user_bark_url] user_id={user_id}")


async def get_onboarding_seen(user_id: str) -> bool:
    """Check if the user has seen/dismissed the onboarding tutorial prompt."""
    async with get_db() as db:
        async with db.execute(
            "SELECT onboarding_seen FROM users WHERE id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return bool(row and row[0])


async def set_onboarding_seen(user_id: str, seen: bool = True):
    """Mark the user as having seen/dismissed the onboarding tutorial prompt."""
    async with get_db() as db:
        await db.execute(
            "UPDATE users SET onboarding_seen = ? WHERE id = ?",
            (1 if seen else 0, user_id),
        )
        await db.commit()


async def get_active_user() -> Optional[dict]:
    """Retrieve the user who logged in most recently."""
    async with get_db() as db:
        async with db.execute(
            "SELECT id, username, display_name, sip_extension "
            "FROM users ORDER BY last_login_at DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None
