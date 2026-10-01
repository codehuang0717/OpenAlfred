"""Thin authenticated draft and immutable send-job routes."""
from fastapi import APIRouter, Depends, HTTPException

from db import email_drafts as store
from routers.auth import get_current_user
from schemas.email_workflow import DraftAdopt, DraftCreate, DraftResponse, DraftSend, DraftUpdate, SendJobResponse
from services.email_worker import notify

router = APIRouter(prefix="/api", tags=["email"])


async def _call(awaitable):
    try:
        return await awaitable
    except store.MailNotFound as error:
        raise HTTPException(404, str(error)) from error
    except store.MailConflict as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.get("/email-drafts", response_model=list[DraftResponse])
async def list_drafts(user: dict = Depends(get_current_user)):
    return await _call(store.list_drafts(user["id"]))


@router.post("/email-drafts", response_model=DraftResponse)
async def create_draft(req: DraftCreate, user: dict = Depends(get_current_user)):
    draft = await _call(store.create_draft(user["id"], req.model_dump(exclude={"client_key"}), req.client_key))
    await notify(user["id"], draft["id"])
    return draft


@router.get("/email-drafts/{draft_id}", response_model=DraftResponse)
async def get_draft(draft_id: str, user: dict = Depends(get_current_user)):
    return await _call(store.get_draft(user["id"], draft_id))


@router.patch("/email-drafts/{draft_id}", response_model=DraftResponse)
async def save_draft(draft_id: str, req: DraftUpdate, user: dict = Depends(get_current_user)):
    draft = await _call(store.update_draft(user["id"], draft_id, req.revision, req.model_dump(exclude={"revision"})))
    await notify(user["id"], draft_id)
    return draft


@router.post("/email-drafts/{draft_id}/send", response_model=SendJobResponse, status_code=202)
async def send_draft(draft_id: str, req: DraftSend, user: dict = Depends(get_current_user)):
    job = await _call(store.enqueue_send(user["id"], draft_id, req.revision, req.idempotency_key))
    await notify(user["id"], draft_id)
    return job


@router.post("/email-drafts/{draft_id}/adopt", response_model=DraftResponse)
async def adopt_draft(draft_id: str, req: DraftAdopt, user: dict = Depends(get_current_user)):
    draft = await _call(store.adopt_proposal(user["id"], draft_id, req.revision))
    await notify(user["id"], draft["id"])
    return draft


@router.get("/email-send-jobs/{job_id}", response_model=SendJobResponse)
async def get_job(job_id: str, user: dict = Depends(get_current_user)):
    return await _call(store.get_job(user["id"], job_id))


@router.get("/email-send-jobs", response_model=list[SendJobResponse])
async def list_jobs(user: dict = Depends(get_current_user)):
    return await _call(store.list_jobs(user["id"]))


@router.post("/email-send-jobs/{job_id}/cancel", response_model=SendJobResponse)
async def cancel_job(job_id: str, user: dict = Depends(get_current_user)):
    job = await _call(store.cancel_job(user["id"], job_id))
    await notify(user["id"], job["draft_id"])
    return job
