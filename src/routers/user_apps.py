"""Authenticated APIs for a user's generated toolbox apps."""

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.event_bus import EventType, event_bus
from db.user_apps import (
    create_code_app_revision_job, delete_user_app, get_user_app, list_user_apps,
    publish_user_app_revision,
)
from routers.auth import get_current_user
from db.coding_jobs import app_jobs, get_job, job_events, requeue_job, stop_job
from services.code_apps import CodeAppSource, coding_context, validate_code_source

from schemas.responses import (
    CodingJobEventsResponse,
    CodingJobResponse,
    QueuedRevisionResponse,
    StatusResponse,
    UserAppDetailsResponse,
    UserAppResponse,
    json_response,
)

router = APIRouter(prefix="/api/user-apps", tags=["user-apps"])


@router.get("", responses=json_response(list[UserAppResponse], 200))
async def get_user_apps(user: dict = Depends(get_current_user)):
    return await list_user_apps(user["id"])


@router.get("/{app_id}", responses=json_response(UserAppDetailsResponse, 200))
async def get_user_app_details(app_id: str, user: dict = Depends(get_current_user)):
    app = await get_user_app(user["id"], app_id)
    if app is None:
        raise HTTPException(status_code=404, detail="App not found")
    return app


class PublishRequest(BaseModel):
    revision_id: str


@router.post("/{app_id}/publish", responses=json_response(UserAppDetailsResponse, 200))
async def publish_user_app(
    app_id: str, payload: PublishRequest, user: dict = Depends(get_current_user),
):
    details = await get_user_app(user["id"], app_id)
    revision = next((item for item in details["revisions"] if item["id"] == payload.revision_id), None) if details else None
    if revision is None or revision["renderer"] != "html":
        raise HTTPException(status_code=404, detail="Ready revision not found")
    try:
        source = CodeAppSource.model_validate(revision["source"])
        await asyncio.to_thread(validate_code_source, source)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"此版本不符合当前源码安全规则：{exc}；请生成兼容草稿") from exc
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail=f"源码验证服务不可用：{exc}") from exc
    app = await publish_user_app_revision(user["id"], app_id, payload.revision_id)
    if app is None:
        raise HTTPException(status_code=404, detail="Ready revision not found")
    await event_bus.publish(EventType.USER_APP_UPDATED, {"id": app_id, "user_id": user["id"]})
    return app


@router.post("/{app_id}/responsive-revision", status_code=202, responses=json_response(QueuedRevisionResponse, 202))
async def create_responsive_revision(
    app_id: str,
    user: dict = Depends(get_current_user),
):
    try:
        app = await get_user_app(user["id"], app_id)
        if app is None or not app["latest_job"]:
            raise HTTPException(status_code=404, detail="App not found")
        context = await coding_context(user["id"], app["latest_job"]["model"])
        job = await create_code_app_revision_job(user["id"], app_id, context=context, queued=True)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="App not found")
    await event_bus.publish(EventType.USER_APP_UPDATED, {"id": app_id, "user_id": user["id"]})
    return {"app_id": app_id, "job_id": job["job_id"], "status": "queued"}


def public_job(job: dict) -> dict:
    return {key: job[key] for key in (
        "id", "app_id", "title", "status", "stage", "model", "error", "report",
        "metrics", "revision_id", "created_at", "updated_at",
    )}


@router.get("/jobs/{job_id}", responses=json_response(CodingJobResponse, 200))
async def get_coding_job(job_id: str, user: dict = Depends(get_current_user)):
    job = await get_job(user["id"], job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Coding job not found")
    return public_job(job)


@router.get("/jobs/{job_id}/events", responses=json_response(CodingJobEventsResponse, 200))
async def coding_job_events(
    job_id: str, after: int = Query(0, ge=0), user: dict = Depends(get_current_user),
):
    job = await get_job(user["id"], job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Coding job not found")
    records = await job_events(user["id"], job_id, after)
    return {"events": records, "job": public_job(job)}


@router.get("/jobs/{job_id}/stream")
async def coding_job_stream(
    job_id: str, after: int = Query(0, ge=0), user: dict = Depends(get_current_user),
):
    if await get_job(user["id"], job_id) is None:
        raise HTTPException(status_code=404, detail="Coding job not found")

    async def generate():
        cursor = after
        while True:
            records = await job_events(user["id"], job_id, cursor)
            if records is None:
                yield "event: deleted\ndata: {}\n\n"
                return
            for record in records:
                cursor = record["seq"]
                yield f"id: {cursor}\nevent: progress\ndata: {json.dumps(record, ensure_ascii=False)}\n\n"
            job = await get_job(user["id"], job_id)
            if job is None:
                return
            yield f"event: snapshot\ndata: {json.dumps(public_job(job), ensure_ascii=False)}\n\n"
            if job["status"] not in {"queued", "generating"}:
                return
            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })


@router.post("/jobs/{job_id}/cancel", responses=json_response(StatusResponse, 200))
async def cancel_coding_job(job_id: str, user: dict = Depends(get_current_user)):
    if await get_job(user["id"], job_id) is None:
        raise HTTPException(status_code=404, detail="Coding job not found")
    if not await stop_job(user["id"], job_id):
        raise HTTPException(status_code=409, detail="Task is no longer running")
    return {"status": "cancelled"}


class RestartRequest(BaseModel):
    resume: bool = True


@router.post("/jobs/{job_id}/restart", status_code=202, responses=json_response(QueuedRevisionResponse, 202))
async def restart_coding_job(job_id: str, payload: RestartRequest, user: dict = Depends(get_current_user)):
    try:
        job = await requeue_job(user["id"], job_id, resume=payload.resume)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Coding job not found")
    return job


@router.delete("/{app_id}", responses=json_response(StatusResponse, 200))
async def remove_user_app(app_id: str, request: Request, user: dict = Depends(get_current_user)):
    jobs = await app_jobs(user["id"], app_id)
    if not await delete_user_app(user["id"], app_id):
        raise HTTPException(status_code=404, detail="App not found")
    worker = getattr(request.app.state, "coding_worker", None)
    if worker is not None:
        await worker.purge_jobs(jobs)
    await event_bus.publish(EventType.USER_APP_DELETED, {"id": app_id, "user_id": user["id"]})
    return {"status": "deleted"}
