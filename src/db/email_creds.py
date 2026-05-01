"""
Email credentials repository — encrypted credential storage.
"""

import aiosqlite
from datetime import datetime, timezone
from db.connection import DATABASE_PATH


async def set_email_credentials(
    account_id: str,
    user_id: str,
    email_address: str,
    provider: str,
    imap_server: str,
    imap_port: int,
    smtp_server: str,
    smtp_port: int,
    encrypted_password: str
):
    created_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO email_credentials 
            (account_id, user_id, email_address, provider, imap_server, imap_port, smtp_server, smtp_port, encrypted_password, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM email_credentials WHERE account_id = ?), ?))
            """,
            (account_id, user_id, email_address, provider, imap_server, imap_port, smtp_server, smtp_port, encrypted_password, account_id, created_at)
        )
        await db.commit()

async def get_email_credentials(user_id: str) -> list[dict]:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM email_credentials WHERE user_id = ?",
            (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def delete_email_credentials(account_id: str, user_id: str):
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "DELETE FROM email_credentials WHERE account_id = ? AND user_id = ?",
            (account_id, user_id)
        )
        await db.commit()
