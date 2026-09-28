"""Authenticated APIs for a user's generated toolbox apps."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from core.event_bus import EventType, event_bus
from db.user_apps import (
    create_code_app_revision_job, delete_user_app, get_user_app, list_user_apps,
    publish_user_app_revision,
)
from routers.auth import get_current_user
from services.code_apps import complete_code_app_revision

router = APIRouter(prefix="/api/user-apps", tags=["user-apps"])


@router.get("")
async def get_user_apps(user: dict = Depends(get_current_user)):
    return await list_user_apps(user["id"])


@router.get("/{app_id}")
async def get_user_app_details(app_id: str, user: dict = Depends(get_current_user)):
    app = await get_user_app(user["id"], app_id)
    if app is None:
        raise HTTPException(status_code=404, detail="App not found")
    return app


class PublishRequest(BaseModel):
    revision_id: str


@router.post("/{app_id}/publish")
async def publish_user_app(
    app_id: str, payload: PublishRequest, user: dict = Depends(get_current_user),
):
    app = await publish_user_app_revision(user["id"], app_id, payload.revision_id)
    if app is None:
        raise HTTPException(status_code=404, detail="Ready revision not found")
    await event_bus.publish(EventType.USER_APP_UPDATED, {"id": app_id, "user_id": user["id"]})
    return app


@router.post("/{app_id}/responsive-revision", status_code=202)
async def create_responsive_revision(
    app_id: str, background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    try:
        job = await create_code_app_revision_job(user["id"], app_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="App not found")
    background_tasks.add_task(complete_code_app_revision, user["id"], job)
    await event_bus.publish(EventType.USER_APP_UPDATED, {"id": app_id, "user_id": user["id"]})
    return {"app_id": app_id, "job_id": job["job_id"], "status": "generating"}


@router.delete("/{app_id}")
async def remove_user_app(app_id: str, user: dict = Depends(get_current_user)):
    if not await delete_user_app(user["id"], app_id):
        raise HTTPException(status_code=404, detail="App not found")
    await event_bus.publish(EventType.USER_APP_DELETED, {"id": app_id, "user_id": user["id"]})
    return {"status": "deleted"}
