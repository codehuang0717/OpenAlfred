"""Tenant-scoped storage for trusted panels and generated app revisions."""

import json
import uuid
from datetime import datetime, timezone

from db.connection import get_db
from utils.auth_utils import require_explicit_user_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _app(row) -> dict:
    result = dict(row)
    result.pop("spec_json")
    published_source = result.pop("published_source_json", None)
    if result["kind"] == "todo_timeline":
        if published_source is None:
            raise RuntimeError("Published timeline revision is missing")
        result["spec"] = json.loads(published_source)
    else:
        result["spec"] = None
    return result


async def list_user_apps(user_id: str) -> list[dict]:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        async with db.execute(
            """
            SELECT a.*,
                (SELECT source_json FROM user_app_revisions r
                 WHERE r.id = a.published_revision_id AND r.user_id = a.user_id
                 AND r.renderer = 'catalog') AS published_source_json,
                (SELECT status FROM user_app_jobs j WHERE j.app_id = a.id
                 ORDER BY j.created_at DESC LIMIT 1) AS job_status,
                (SELECT stage FROM user_app_jobs j WHERE j.app_id = a.id
                 ORDER BY j.created_at DESC LIMIT 1) AS job_stage,
                (SELECT error FROM user_app_jobs j WHERE j.app_id = a.id
                 ORDER BY j.created_at DESC LIMIT 1) AS job_error,
                (SELECT COUNT(*) FROM user_app_revisions r
                 WHERE r.app_id = a.id AND r.user_id = a.user_id
                   AND r.status = 'ready'
                   AND (a.published_revision_id IS NULL OR r.id != a.published_revision_id)
                ) AS has_unpublished_revision
            FROM user_apps a WHERE a.user_id = ? ORDER BY a.updated_at DESC
            """,
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_app(row) for row in rows]


async def get_user_app(user_id: str, app_id: str) -> dict | None:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM user_apps WHERE id = ? AND user_id = ?", (app_id, user_id)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        async with db.execute(
            """
            SELECT * FROM user_app_revisions WHERE app_id = ? AND user_id = ?
            ORDER BY revision_number DESC
            """,
            (app_id, user_id),
        ) as cursor:
            revisions = await cursor.fetchall()
        async with db.execute(
            """
            SELECT id, status, stage, model, error, revision_id, created_at, updated_at
            FROM user_app_jobs WHERE app_id = ? AND user_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (app_id, user_id),
        ) as cursor:
            job = await cursor.fetchone()
    published_source = next(
        (item["source_json"] for item in revisions
         if item["id"] == row["published_revision_id"] and item["renderer"] == "catalog"),
        None,
    )
    result = _app({**dict(row), "published_source_json": published_source})
    result["revisions"] = []
    for item in revisions:
        revision = dict(item)
        revision["source"] = json.loads(revision.pop("source_json"))
        revision["validation"] = json.loads(revision.pop("validation_json"))
        result["revisions"].append(revision)
    result["has_unpublished_revision"] = sum(
        1 for item in result["revisions"]
        if item["status"] == "ready" and item["id"] != result["published_revision_id"]
    )
    result["latest_job"] = dict(job) if job else None
    return result


async def save_user_app(user_id: str, kind: str, title: str, spec: dict) -> dict:
    """Publish a trusted catalog revision, retaining a stable panel ID."""
    user_id = require_explicit_user_id(user_id)
    if kind != "todo_timeline":
        raise ValueError("Only the trusted todo timeline may use catalog publishing")
    now = _now()
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        await db.execute(
            """
            INSERT INTO user_apps
                (id, user_id, kind, title, spec_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'published', ?, ?)
            ON CONFLICT(user_id, kind) DO UPDATE SET
                title = excluded.title,
                spec_json = excluded.spec_json,
                updated_at = excluded.updated_at
            """,
            (str(uuid.uuid4()), user_id, kind, title, json.dumps(spec, ensure_ascii=False), now, now),
        )
        async with db.execute(
            "SELECT id FROM user_apps WHERE user_id = ? AND kind = ?", (user_id, kind)
        ) as cursor:
            app_id = (await cursor.fetchone())["id"]
        async with db.execute(
            "SELECT COALESCE(MAX(revision_number), 0) + 1 FROM user_app_revisions WHERE app_id = ?",
            (app_id,),
        ) as cursor:
            revision_number = (await cursor.fetchone())[0]
        revision_id = str(uuid.uuid4())
        await db.execute(
            """
            INSERT INTO user_app_revisions
                (id, user_id, app_id, revision_number, renderer, source_json,
                 validation_json, status, created_at)
            VALUES (?, ?, ?, ?, 'catalog', ?, '{"catalog":true}', 'published', ?)
            """,
            (revision_id, user_id, app_id, revision_number,
             json.dumps(spec, ensure_ascii=False), now),
        )
        await db.execute(
            "UPDATE user_apps SET published_revision_id = ?, status = 'published' WHERE id = ? AND user_id = ?",
            (revision_id, app_id, user_id),
        )
        await db.commit()
    result = await get_user_app(user_id, app_id)
    assert result is not None
    return result


async def create_code_app_job(user_id: str, title: str, prompt: str, model: str) -> dict:
    user_id = require_explicit_user_id(user_id)
    now = _now()
    app_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        await db.execute(
            """
            INSERT INTO user_apps
                (id, user_id, kind, title, spec_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, '{}', 'draft', ?, ?)
            """,
            (app_id, user_id, f"code_app:{app_id}", title, now, now),
        )
        await db.execute(
            """
            INSERT INTO user_app_jobs
                (id, user_id, app_id, prompt, model, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'generating', ?, ?)
            """,
            (job_id, user_id, app_id, prompt, model, now, now),
        )
        await db.commit()
    return {"app_id": app_id, "job_id": job_id}


async def create_code_app_revision_job(user_id: str, app_id: str) -> dict | None:
    """Start a new draft for an owned code app, preserving its published revision."""
    user_id = require_explicit_user_id(user_id)
    now = _now()
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        async with db.execute(
            "SELECT title, kind FROM user_apps WHERE id = ? AND user_id = ?",
            (app_id, user_id),
        ) as cursor:
            app = await cursor.fetchone()
        if app is None or not app["kind"].startswith("code_app:"):
            await db.rollback()
            return None
        async with db.execute(
            "SELECT 1 FROM user_app_jobs WHERE app_id = ? AND user_id = ? AND status IN ('queued', 'generating') LIMIT 1",
            (app_id, user_id),
        ) as cursor:
            active = await cursor.fetchone()
        if active:
            await db.rollback()
            raise ValueError("这个小程序已有正在进行的生成任务")
        async with db.execute(
            "SELECT prompt, model FROM user_app_jobs WHERE app_id = ? AND user_id = ? AND status = 'ready' ORDER BY created_at DESC LIMIT 1",
            (app_id, user_id),
        ) as cursor:
            previous_job = await cursor.fetchone()
        async with db.execute(
            "SELECT source_json FROM user_app_revisions WHERE app_id = ? AND user_id = ? AND renderer = 'html' ORDER BY revision_number DESC LIMIT 1",
            (app_id, user_id),
        ) as cursor:
            revision = await cursor.fetchone()
        if previous_job is None or revision is None:
            await db.rollback()
            raise ValueError("没有可修订的已验证代码版本")
        job_id = str(uuid.uuid4())
        await db.execute(
            """
            INSERT INTO user_app_jobs
                (id, user_id, app_id, prompt, model, status, stage, origin, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'generating', 'model', 'web_revision', ?, ?)
            """,
            (job_id, user_id, app_id, previous_job["prompt"], previous_job["model"], now, now),
        )
        await db.commit()
    return {
        "app_id": app_id, "job_id": job_id, "title": app["title"],
        "prompt": previous_job["prompt"], "model": previous_job["model"],
        "previous_source": json.loads(revision["source_json"]),
    }


async def fail_interrupted_web_revisions() -> int:
    """In-process revision tasks cannot survive an API process restart."""
    now = _now()
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        cursor = await db.execute(
            """
            UPDATE user_app_jobs
            SET status = 'failed', error = '服务重启中断了生成任务，请重新生成响应式草稿', updated_at = ?
            WHERE origin = 'web_revision' AND status IN ('queued', 'generating')
            """,
            (now,),
        )
        await db.execute(
            """
            UPDATE user_apps SET updated_at = ?
            WHERE id IN (
                SELECT app_id FROM user_app_jobs
                WHERE origin = 'web_revision' AND status = 'failed' AND updated_at = ?
            )
            """,
            (now, now),
        )
        await db.commit()
    return cursor.rowcount


async def set_code_app_job_stage(user_id: str, job_id: str, stage: str) -> None:
    user_id = require_explicit_user_id(user_id)
    if stage not in {"model", "validation"}:
        raise ValueError("Invalid code generation stage")
    async with get_db() as db:
        cursor = await db.execute(
            "UPDATE user_app_jobs SET stage = ?, updated_at = ? WHERE id = ? AND user_id = ? AND status = 'generating'",
            (stage, _now(), job_id, user_id),
        )
        if cursor.rowcount != 1:
            await db.rollback()
            raise ValueError("Generation job is missing or not active")
        await db.commit()


async def finish_code_app_job(
    user_id: str, job_id: str, *, source: dict | None = None,
    validation: dict | None = None, error: str | None = None,
) -> str | None:
    """Store a validated candidate or an explicit failure; never auto-publish."""
    user_id = require_explicit_user_id(user_id)
    if (source is None) == (error is None):
        raise ValueError("Provide exactly one of source or error")
    if source is not None and (
        not validation
        or validation.get("javascript_syntax") != "passed"
        or validation.get("data_access") != "none"
    ):
        raise ValueError("A code revision requires passing validation")
    now = _now()
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        async with db.execute(
            "SELECT app_id, status FROM user_app_jobs WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        ) as cursor:
            job = await cursor.fetchone()
        if job is None or job["status"] != "generating":
            raise ValueError("Generation job is missing or not active")
        app_id = job["app_id"]
        revision_id = None
        if source is not None:
            async with db.execute(
                "SELECT COALESCE(MAX(revision_number), 0) + 1 FROM user_app_revisions WHERE app_id = ?",
                (app_id,),
            ) as cursor:
                revision_number = (await cursor.fetchone())[0]
            revision_id = str(uuid.uuid4())
            await db.execute(
                """
                INSERT INTO user_app_revisions
                    (id, user_id, app_id, revision_number, renderer, source_json,
                     validation_json, status, created_at)
                VALUES (?, ?, ?, ?, 'html', ?, ?, 'ready', ?)
                """,
                (revision_id, user_id, app_id, revision_number,
                 json.dumps(source, ensure_ascii=False), json.dumps(validation or {}), now),
            )
        await db.execute(
            """
            UPDATE user_app_jobs SET status = ?, error = ?, revision_id = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            ("ready" if source is not None else "failed", error, revision_id, now, job_id, user_id),
        )
        await db.execute(
            """
            UPDATE user_apps
            SET status = CASE WHEN published_revision_id IS NOT NULL THEN 'published' ELSE ? END,
                updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            ("ready" if source is not None else "failed", now, app_id, user_id),
        )
        await db.commit()
    return revision_id


async def publish_user_app_revision(user_id: str, app_id: str, revision_id: str) -> dict | None:
    user_id = require_explicit_user_id(user_id)
    now = _now()
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        async with db.execute(
            """
            SELECT r.status FROM user_app_revisions r
            JOIN user_apps a ON a.id = r.app_id AND a.user_id = r.user_id
            WHERE r.id = ? AND r.app_id = ? AND r.user_id = ? AND r.renderer = 'html'
              AND json_extract(r.validation_json, '$.javascript_syntax') = 'passed'
              AND json_extract(r.validation_json, '$.data_access') = 'none'
            """,
            (revision_id, app_id, user_id),
        ) as cursor:
            revision = await cursor.fetchone()
        if revision is None or revision["status"] not in ("ready", "published"):
            await db.rollback()
            return None
        await db.execute(
            "UPDATE user_app_revisions SET status = 'published' WHERE id = ? AND user_id = ?",
            (revision_id, user_id),
        )
        await db.execute(
            """
            UPDATE user_apps SET published_revision_id = ?, status = 'published', updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (revision_id, now, app_id, user_id),
        )
        await db.commit()
    return await get_user_app(user_id, app_id)


async def delete_user_app(user_id: str, app_id: str) -> bool:
    user_id = require_explicit_user_id(user_id)
    async with get_db() as db:
        cursor = await db.execute(
            "DELETE FROM user_apps WHERE id = ? AND user_id = ?", (app_id, user_id)
        )
        await db.commit()
    return cursor.rowcount > 0
