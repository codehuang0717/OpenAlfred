import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from routers.auth import get_current_user
from db.rag import get_documents, get_document_by_id
from rag.ingestion import ingest_text, ingest_markdown_file, ingest_file
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


class IngestPathRequest(BaseModel):
    filepath: str
    title: str = ""


@router.post("/ingest-path")
async def ingest_by_path(req: IngestPathRequest, user: dict = Depends(get_current_user)):
    """Ingest a file from a local path. Auto-resolves source_dir for images."""
    if not req.filepath.strip():
        raise HTTPException(status_code=400, detail="Filepath is empty")

    # Dedup: check if this user already ingested this exact filepath
    existing = await get_documents(user["id"])
    import os
    target_path = os.path.abspath(req.filepath.strip())
    for doc in existing:
        # Compare by filename + title (title is stem from path)
        from pathlib import Path
        expected_title = req.title or Path(target_path).stem
        if doc["title"] == expected_title and doc["filename"].endswith(Path(target_path).suffix):
            # Already exists — skip duplicate
            logger.info("Duplicate ingest skipped: %s (existing doc_id=%s)", target_path, doc["id"])
            return doc

    try:
        doc = await ingest_file(user["id"], req.filepath, req.title)
        return doc
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/select-file")
async def select_file_via_dialog(user: dict = Depends(get_current_user)):
    """Open a native Windows file dialog to let the user select a file on their disk."""
    import subprocess
    # PowerShell script to open file dialog and output selected file path
    ps_script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$d = New-Object System.Windows.Forms.OpenFileDialog; "
        "$d.Filter = 'Markdown Files (*.md)|*.md|All Files (*.*)|*.*'; "
        "$d.Title = 'Select Markdown File for Knowledge Base'; "
        "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.FileName }"
    )
    try:
        res = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True,
            text=True,
            check=True,
            creationflags=0x08000000  # CREATE_NO_WINDOW: prevent flashing terminal window
        )
        filepath = res.stdout.strip()
        return {"filepath": filepath}
    except Exception as e:
        logger.error("Failed to open file dialog: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to open file dialog: {e}")


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
