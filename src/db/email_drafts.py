"""Owner-scoped mail drafts, immutable receipts and a durable SMTP queue."""
import time
import uuid
from datetime import datetime, timezone
from email.headerregistry import Address
from email.errors import HeaderParseError
from email.utils import make_msgid

from db.connection import get_db
from utils.auth_utils import require_explicit_user_id

LEASE_SECONDS = 45
EDITABLE = {"draft", "failed", "cancelled"}


class MailConflict(ValueError):
    pass


class MailNotFound(ValueError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def init_schema(db) -> None:
    await db.executescript("""
        CREATE TABLE IF NOT EXISTS email_drafts (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, client_key TEXT NOT NULL,
            account_id TEXT NOT NULL, to_address TEXT NOT NULL, subject TEXT NOT NULL,
            body TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'draft', proposal_for TEXT, base_revision INTEGER,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(user_id, client_key)
        );
        CREATE INDEX IF NOT EXISTS idx_mail_drafts_owner ON email_drafts(user_id, updated_at);
        CREATE TABLE IF NOT EXISTS email_send_jobs (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, draft_id TEXT NOT NULL,
            revision INTEGER NOT NULL, idempotency_key TEXT NOT NULL,
            account_id TEXT NOT NULL, from_address TEXT NOT NULL, to_address TEXT NOT NULL,
            subject TEXT NOT NULL, body TEXT NOT NULL, message_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued', phase TEXT NOT NULL DEFAULT 'queued',
            runner_id TEXT, lease_until REAL, error TEXT, accepted_at TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(user_id, idempotency_key), FOREIGN KEY(draft_id) REFERENCES email_drafts(id)
        );
        CREATE INDEX IF NOT EXISTS idx_mail_jobs_queue ON email_send_jobs(status, created_at);
        CREATE INDEX IF NOT EXISTS idx_mail_jobs_draft ON email_send_jobs(user_id, draft_id, created_at);
    """)


async def _draft(db, owner: str, identity: str) -> dict:
    async with db.execute("SELECT * FROM email_drafts WHERE id = ? AND user_id = ?", (identity, owner)) as cursor:
        row = await cursor.fetchone()
    if row is None:
        raise MailNotFound("草稿不存在")
    return dict(row)


async def _receipt(db, draft: dict) -> dict:
    async with db.execute("SELECT * FROM email_send_jobs WHERE draft_id = ? AND user_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1", (draft["id"], draft["user_id"])) as cursor:
        row = await cursor.fetchone()
    return {**draft, "last_job": dict(row) if row else None}


async def _account(db, owner: str, identity: str, *, required: bool = False) -> str:
    if not identity and not required:
        return ""
    async with db.execute("SELECT email_address FROM email_credentials WHERE account_id = ? AND user_id = ?", (identity, owner)) as cursor:
        row = await cursor.fetchone()
    if not row:
        raise MailNotFound("发件邮箱不存在，请重新选择")
    return row[0]


async def get_draft(user_id: str, draft_id: str) -> dict:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        return await _receipt(db, await _draft(db, user_id, draft_id))


async def list_drafts(user_id: str) -> list[dict]:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        async with db.execute("SELECT * FROM email_drafts WHERE user_id = ? ORDER BY updated_at DESC LIMIT 100", (user_id,)) as cursor:
            rows = await cursor.fetchall()
        return [await _receipt(db, dict(row)) for row in rows]


async def create_draft(user_id: str, fields: dict, client_key: str, *, proposal_for: str | None = None, base_revision: int | None = None) -> dict:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        async with db.execute("SELECT * FROM email_drafts WHERE user_id = ? AND client_key = ?", (user_id, client_key)) as cursor:
            existing = await cursor.fetchone()
        if existing:
            return await _receipt(db, dict(existing))
        await _account(db, user_id, fields["account_id"])
        if proposal_for:
            original = await _draft(db, user_id, proposal_for)
            if original["revision"] != base_revision or original["status"] not in EDITABLE:
                raise MailConflict("草稿已修改或已提交发送，请重新获取后再提出建议")
        await db.execute("""INSERT OR IGNORE INTO email_drafts
            (id, user_id, client_key, account_id, to_address, subject, body, status, proposal_for, base_revision, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), user_id, client_key, fields["account_id"], fields["to_address"], fields["subject"], fields["body"],
             "proposal" if proposal_for else "draft", proposal_for, base_revision, now(), now()))
        async with db.execute("SELECT id FROM email_drafts WHERE user_id = ? AND client_key = ?", (user_id, client_key)) as cursor:
            draft_id = (await cursor.fetchone())[0]
        await db.commit()
    return await get_draft(user_id, draft_id)


async def update_draft(user_id: str, draft_id: str, revision: int, fields: dict) -> dict:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        draft = await _draft(db, user_id, draft_id)
        if draft["revision"] != revision or draft["status"] not in EDITABLE:
            raise MailConflict("草稿已在其他位置修改或已提交发送；请保留输入并加载最新版本")
        await _account(db, user_id, fields["account_id"])
        await db.execute("""UPDATE email_drafts SET account_id = ?, to_address = ?, subject = ?, body = ?,
            revision = revision + 1, status = 'draft', updated_at = ? WHERE id = ? AND user_id = ?""",
            (fields["account_id"], fields["to_address"], fields["subject"], fields["body"], now(), draft_id, user_id))
        await db.commit()
    return await get_draft(user_id, draft_id)


async def adopt_proposal(user_id: str, proposal_id: str, revision: int) -> dict:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        proposal = await _draft(db, user_id, proposal_id)
        if proposal["status"] != "proposal":
            raise MailConflict("此建议已经处理")
        original = await _draft(db, user_id, proposal["proposal_for"])
        if original["revision"] != revision or proposal["base_revision"] != revision or original["status"] not in EDITABLE:
            raise MailConflict("原草稿已修改；请重新提出建议，避免覆盖你的编辑")
        await db.execute("""UPDATE email_drafts SET account_id = ?, to_address = ?, subject = ?, body = ?,
            revision = revision + 1, status = 'draft', updated_at = ? WHERE id = ? AND user_id = ?""",
            (proposal["account_id"], proposal["to_address"], proposal["subject"], proposal["body"], now(), original["id"], user_id))
        await db.execute("UPDATE email_drafts SET status = 'adopted', updated_at = ? WHERE id = ? AND user_id = ?", (now(), proposal_id, user_id))
        await db.commit()
    return await get_draft(user_id, original["id"])


async def enqueue_send(user_id: str, draft_id: str, revision: int, key: str) -> dict:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        async with db.execute("SELECT * FROM email_send_jobs WHERE user_id = ? AND idempotency_key = ?", (user_id, key)) as cursor:
            previous = await cursor.fetchone()
        if previous:
            if previous["draft_id"] != draft_id or previous["revision"] != revision:
                raise MailConflict("发送请求标识已被其他版本使用")
            return dict(previous)
        draft = await _draft(db, user_id, draft_id)
        if draft["revision"] != revision:
            raise MailConflict("草稿版本已经更新；请查看最新内容后再发送")
        if draft["status"] in {"queued", "sending", "accepted", "partial", "unknown"}:
            return (await _receipt(db, draft))["last_job"]
        if draft["status"] not in EDITABLE:
            raise MailConflict("请先采用此建议，再确认发送")
        try:
            address = Address(addr_spec=draft["to_address"])
            if not address.username or not address.domain or "." not in address.domain:
                raise ValueError()
        except (ValueError, IndexError, HeaderParseError):
            raise ValueError("请输入一个有效的收件邮箱地址") from None
        if not draft["subject"] or not draft["body"].strip():
            raise ValueError("主题和正文不能为空")
        sender = await _account(db, user_id, draft["account_id"], required=True)
        job_id = str(uuid.uuid4())
        await db.execute("""INSERT INTO email_send_jobs (id, user_id, draft_id, revision, idempotency_key, account_id,
            from_address, to_address, subject, body, message_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, user_id, draft_id, revision, key, draft["account_id"], sender, draft["to_address"], draft["subject"], draft["body"], make_msgid(domain="openalfred.local"), now(), now()))
        await db.execute("UPDATE email_drafts SET status = 'queued', updated_at = ? WHERE id = ? AND user_id = ?", (now(), draft_id, user_id))
        await db.commit()
    return await get_job(user_id, job_id)


async def get_job(user_id: str, job_id: str) -> dict:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        async with db.execute("SELECT * FROM email_send_jobs WHERE id = ? AND user_id = ?", (job_id, user_id)) as cursor:
            row = await cursor.fetchone()
    if row is None:
        raise MailNotFound("发送记录不存在")
    return dict(row)


async def list_jobs(user_id: str) -> list[dict]:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        async with db.execute("SELECT * FROM email_send_jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT 100", (user_id,)) as cursor:
            return [dict(row) for row in await cursor.fetchall()]


async def cancel_job(user_id: str, job_id: str) -> dict:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        async with db.execute("SELECT * FROM email_send_jobs WHERE id = ? AND user_id = ?", (job_id, user_id)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise MailNotFound("发送记录不存在")
        if row["status"] != "queued":
            raise MailConflict("任务已经开始提交，不能撤回；请查看发送结果")
        await db.execute("UPDATE email_send_jobs SET status = 'cancelled', updated_at = ? WHERE id = ? AND user_id = ?", (now(), job_id, user_id))
        await db.execute("UPDATE email_drafts SET status = 'cancelled', updated_at = ? WHERE id = ? AND user_id = ?", (now(), row["draft_id"], user_id))
        await db.commit()
    return await get_job(user_id, job_id)


async def claim_job(runner_id: str) -> dict | None:
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        async with db.execute("SELECT * FROM email_send_jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1") as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        job = dict(row)
        await db.execute("UPDATE email_send_jobs SET status = 'sending', phase = 'connecting', runner_id = ?, lease_until = ?, updated_at = ? WHERE id = ?", (runner_id, time.time() + LEASE_SECONDS, now(), job["id"]))
        await db.execute("UPDATE email_drafts SET status = 'sending', updated_at = ? WHERE id = ? AND user_id = ?", (now(), job["draft_id"], job["user_id"]))
        await db.commit()
    return {**job, "status": "sending", "phase": "connecting", "runner_id": runner_id}


async def heartbeat(job: dict) -> bool:
    async with get_db() as db:
        cursor = await db.execute("UPDATE email_send_jobs SET lease_until = ? WHERE id = ? AND runner_id = ? AND status = 'sending'", (time.time() + LEASE_SECONDS, job["id"], job["runner_id"]))
        await db.commit()
        return cursor.rowcount == 1


async def submitting(job: dict) -> None:
    async with get_db() as db:
        cursor = await db.execute("UPDATE email_send_jobs SET phase = 'submitting', updated_at = ? WHERE id = ? AND runner_id = ? AND status = 'sending' AND lease_until > ?", (now(), job["id"], job["runner_id"], time.time()))
        if cursor.rowcount != 1:
            raise MailConflict("发送任务已失去执行权，未提交邮件")
        await db.commit()
    job["phase"] = "submitting"


async def finish_job(job: dict, status: str, error: str | None = None) -> None:
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        cursor = await db.execute("UPDATE email_send_jobs SET status = ?, error = ?, accepted_at = ?, updated_at = ?, lease_until = NULL WHERE id = ? AND runner_id = ? AND status = 'sending'", (status, error, now() if status in {"accepted", "partial"} else None, now(), job["id"], job["runner_id"]))
        if cursor.rowcount == 1:
            await db.execute("UPDATE email_drafts SET status = ?, updated_at = ? WHERE id = ? AND user_id = ?", (status, now(), job["draft_id"], job["user_id"]))
        await db.commit()


async def expire_jobs() -> None:
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        async with db.execute("SELECT * FROM email_send_jobs WHERE status = 'sending' AND lease_until <= ?", (time.time(),)) as cursor:
            rows = await cursor.fetchall()
        for row in rows:
            status = "unknown" if row["phase"] == "submitting" else "failed"
            error = "服务中断，邮件可能已提交；请核对邮箱，未自动重发" if status == "unknown" else "服务中断，邮件尚未提交；可以明确重试"
            await db.execute("UPDATE email_send_jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?", (status, error, now(), row["id"]))
            await db.execute("UPDATE email_drafts SET status = ?, updated_at = ? WHERE id = ? AND user_id = ?", (status, now(), row["draft_id"], row["user_id"]))
        await db.commit()
