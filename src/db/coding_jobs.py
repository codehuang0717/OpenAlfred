"""Durable coding queue, owner checks and fencing for late worker results."""

import json
import time
from datetime import datetime, timezone

from db.connection import get_db
from utils.auth_utils import require_explicit_user_id

LEASE_SECONDS = 45
ACTIVE = {"queued", "generating"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_job(row) -> dict:
    job = dict(row)
    job["context"] = json.loads(job.pop("context_json"))
    job["metrics"] = json.loads(job.pop("metrics_json"))
    return job


async def _event(db, job: dict, stage: str, message: str) -> None:
    await db.execute(
        "INSERT INTO user_app_job_events(job_id, user_id, stage, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (job["id"], job["user_id"], stage, message[:600], now()),
    )
    await db.execute(
        "UPDATE user_apps SET updated_at = ? WHERE id = ? AND user_id = ?",
        (now(), job["app_id"], job["user_id"]),
    )


async def get_job(user_id: str, job_id: str) -> dict | None:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        async with db.execute(
            "SELECT j.*, a.title FROM user_app_jobs j JOIN user_apps a ON a.id = j.app_id AND a.user_id = j.user_id WHERE j.id = ? AND j.user_id = ?",
            (job_id, user_id),
        ) as cursor:
            row = await cursor.fetchone()
    return decode_job(row) if row else None


async def app_jobs(user_id: str, app_id: str) -> list[dict]:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        async with db.execute(
            "SELECT j.*, a.title FROM user_app_jobs j JOIN user_apps a ON a.id = j.app_id AND a.user_id = j.user_id WHERE j.app_id = ? AND j.user_id = ?",
            (app_id, user_id),
        ) as cursor:
            return [decode_job(row) for row in await cursor.fetchall()]


async def claim_job(runner_id: str, concurrency: int) -> dict | None:
    """Claim across API processes, with global and per-user concurrency limits."""
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        async with db.execute(
            "SELECT COUNT(*) FROM user_app_jobs WHERE status = 'generating' AND lease_until > ?",
            (time.time(),),
        ) as cursor:
            if (await cursor.fetchone())[0] >= concurrency:
                await db.rollback()
                return None
        async with db.execute("""
            SELECT j.*, a.title FROM user_app_jobs j
            JOIN user_apps a ON a.id = j.app_id AND a.user_id = j.user_id
            WHERE j.status = 'queued' AND NOT EXISTS (
                SELECT 1 FROM user_app_jobs r WHERE r.user_id = j.user_id
                AND r.status = 'generating' AND r.lease_until > ?
            ) ORDER BY j.created_at, j.id LIMIT 1
        """, (time.time(),)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            await db.rollback()
            return None
        job = decode_job(row)
        require_explicit_user_id(job["user_id"])
        await db.execute(
            "UPDATE user_app_jobs SET status = 'generating', runner_id = ?, lease_until = ?, updated_at = ? WHERE id = ? AND status = 'queued'",
            (runner_id, time.time() + LEASE_SECONDS, now(), job["id"]),
        )
        await _event(db, job, "planning", "编码任务已开始")
        await db.commit()
    job.update(status="generating", runner_id=runner_id)
    return job


async def keep_lease(job: dict) -> bool:
    async with get_db() as db:
        cursor = await db.execute(
            "UPDATE user_app_jobs SET lease_until = ? WHERE id = ? AND user_id = ? AND epoch = ? AND runner_id = ? AND status = 'generating'",
            (time.time() + LEASE_SECONDS, job["id"], job["user_id"], job["epoch"], job["runner_id"]),
        )
        await db.commit()
        return cursor.rowcount == 1


async def progress(job: dict, stage: str, message: str, *, metrics: dict | None = None) -> None:
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        cursor = await db.execute(
            "UPDATE user_app_jobs SET stage = ?, metrics_json = ?, updated_at = ? WHERE id = ? AND user_id = ? AND epoch = ? AND status = 'generating' AND runner_id = ?",
            (stage, json.dumps(metrics or {}), now(), job["id"], job["user_id"], job["epoch"], job["runner_id"]),
        )
        if cursor.rowcount != 1:
            raise ValueError("编码任务已取消、删除或被新的执行取代")
        await _event(db, job, stage, message)
        await db.commit()


async def stop_job(user_id: str, job_id: str, status: str = "cancelled", error: str | None = None,
                   *, expected_epoch: int | None = None) -> bool:
    user_id = require_explicit_user_id(user_id)
    if status not in {"cancelled", "interrupted"}:
        raise ValueError("Invalid stop status")
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        async with db.execute("SELECT * FROM user_app_jobs WHERE id = ? AND user_id = ?", (job_id, user_id)) as cursor:
            row = await cursor.fetchone()
        if row is None or row["status"] not in ACTIVE or (expected_epoch is not None and row["epoch"] != expected_epoch):
            await db.rollback()
            return False
        job = dict(row)
        await db.execute(
            "UPDATE user_app_jobs SET status = ?, error = ?, epoch = epoch + 1, runner_id = NULL, lease_until = NULL, updated_at = ? WHERE id = ? AND user_id = ?",
            (status, error, now(), job_id, user_id),
        )
        await _event(db, job, status, error or "任务已取消；没有发布代码")
        await db.commit()
    return True


async def expire_jobs() -> int:
    """Expired work becomes visible interrupted work, never an implicit retry."""
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        async with db.execute(
            "SELECT * FROM user_app_jobs WHERE status = 'generating' AND (lease_until IS NULL OR lease_until <= ?)",
            (time.time(),),
        ) as cursor:
            rows = await cursor.fetchall()
        for row in rows:
            job = dict(row)
            error = "服务中断了编码任务；可恢复已有检查点，或明确重试。没有发布代码。"
            await db.execute(
                "UPDATE user_app_jobs SET status = 'interrupted', error = ?, epoch = epoch + 1, runner_id = NULL, lease_until = NULL, updated_at = ? WHERE id = ?",
                (error, now(), job["id"]),
            )
            await _event(db, job, "interrupted", error)
        await db.commit()
    return len(rows)


async def requeue_job(user_id: str, job_id: str, *, resume: bool) -> dict | None:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        async with db.execute("SELECT * FROM user_app_jobs WHERE id = ? AND user_id = ?", (job_id, user_id)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            await db.rollback()
            return None
        job = decode_job(row)
        if job["status"] not in {"failed", "cancelled", "interrupted"}:
            raise ValueError("只有已停止的任务可以恢复或重试")
        async with db.execute(
            "SELECT 1 FROM user_app_jobs WHERE app_id = ? AND id != ? AND status IN ('queued', 'generating')",
            (job["app_id"], job_id),
        ) as cursor:
            if await cursor.fetchone():
                raise ValueError("这个小程序已有正在进行的生成任务")
        context = job["context"]
        # Resumption keeps the checkpoint identity; retry gets a fresh identity.
        if not resume:
            context["checkpoint_generation"] = context.get("checkpoint_generation", 0) + 1
        context["resume"] = resume
        await db.execute(
            "UPDATE user_app_jobs SET status = 'queued', stage = 'planning', error = NULL, report = NULL, epoch = epoch + 1, context_json = ?, metrics_json = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (json.dumps(context, ensure_ascii=False), json.dumps(job["metrics"] if resume else {}), now(), job_id, user_id),
        )
        await _event(db, job, "queued", "已请求恢复编码" if resume else "已请求重新编码")
        await db.commit()
    return {"job_id": job_id, "app_id": job["app_id"], "status": "queued"}


async def job_events(user_id: str, job_id: str, after: int = 0) -> list[dict] | None:
    if await get_job(user_id, job_id) is None:
        return None
    async with get_db() as db:
        async with db.execute(
            "SELECT seq, stage, message, created_at FROM user_app_job_events WHERE job_id = ? AND user_id = ? AND seq > ? ORDER BY seq LIMIT 100",
            (job_id, user_id, after),
        ) as cursor:
            return [dict(row) for row in await cursor.fetchall()]
