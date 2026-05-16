import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from routers.auth import get_current_user
from db.rag import get_documents, get_document_by_id
from rag.ingestion import ingest_text, ingest_markdown_file
from rag.retriever import search
from rag.store import delete_document as delete_document_full
from rag.image_handler import IMAGES_DIR

logger = logging.getLogger("rag-router")

router = APIRouter(prefix="/api/rag", tags=["rag"])

# Image serving — public, no auth needed
images_router = APIRouter(prefix="/api/images", tags=["images"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class IngestTextRequest(BaseModel):
    title: str
    content: str


# ─── Document CRUD ──────────────────────────────────────────

@router.get("/documents")
async def list_documents(user: dict = Depends(get_current_user)):
    return await get_documents(user["id"])


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str, user: dict = Depends(get_current_user)):
    doc = await get_document_by_id(doc_id)
    if not doc or doc["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/documents/{doc_id}")
async def remove_document(doc_id: str, user: dict = Depends(get_current_user)):
    doc = await get_document_by_id(doc_id)
    if not doc or doc["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Document not found")
    await delete_document_full(user["id"], doc_id)
    return {"status": "deleted"}


# ─── Search ─────────────────────────────────────────────────

@router.post("/search")
async def search_docs(req: SearchRequest, user: dict = Depends(get_current_user)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")
    results = search(user["id"], req.query, req.top_k)
    return {"query": req.query, "results": results}


# ─── Upload ─────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(
    user: dict = Depends(get_current_user),
    file: UploadFile = File(...),
    title: str = Form(""),
    source_dir: str = Form(""),
):
    """Upload a file.

    For .md files, if source_dir is provided, images referenced by
    relative paths will be copied from source_dir to the local images store.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = (file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "txt")

    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("utf-8", errors="replace")

    if ext == "md":
        doc = await ingest_markdown_file(
            user_id=user["id"],
            filename=file.filename,
            content=text,
            title=title or file.filename,
            source_dir=source_dir or None,
        )
    else:
        doc = await ingest_text(user["id"], text, title or file.filename)
    return doc


@router.post("/ingest-text")
async def ingest_text_api(req: IngestTextRequest, user: dict = Depends(get_current_user)):
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content is empty")
    doc = await ingest_text(user["id"], req.content, req.title)
    return doc


# ─── Image Serving (public) ─────────────────────────────────

@images_router.get("/{doc_id}/{filename}")
async def serve_image(doc_id: str, filename: str):
    path = IMAGES_DIR / doc_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path))
