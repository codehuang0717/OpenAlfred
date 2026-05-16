import os
import shutil
import chromadb
from core.config import config
from db.rag import add_document, delete_document as db_delete_document
from rag.embedding import embed_texts
from rag.image_handler import IMAGES_DIR
from typing import Any
from utils.logger import get_logger

logger = get_logger("rag.store")

_chroma_client: Any = None


def _get_client() -> Any:
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(config.RAG_CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=config.RAG_CHROMA_DIR,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        logger.debug("ChromaDB client initialized. path=%s", config.RAG_CHROMA_DIR)
    return _chroma_client


def _get_collection(user_id: str):
    client = _get_client()
    name = f"rag_{user_id}"
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug(
        "Got collection name=%s count=%d", name, collection.count(),
    )
    return collection


async def store_document(
    user_id: str,
    filename: str,
    title: str,
    file_type: str,
    chunks: list,  # list[str] for plain text, list[dict] for markdown
) -> dict:
    """Store document metadata in SQLite and chunks+embeddings in ChromaDB.

    chunks can be:
      - list[str] for plain text files
      - list[dict] for markdown: {text, heading, images}
    """
    collection = _get_collection(user_id)

    if isinstance(chunks[0], dict):
        texts = [c["text"] for c in chunks]
    else:
        texts = chunks

    logger.debug("Generating embeddings for %d chunks...", len(texts))
    embeddings = embed_texts(texts)
    logger.debug("Generated %d embeddings", len(embeddings))

    doc = None
    try:
        doc = await add_document(
            user_id=user_id,
            filename=filename,
            title=title or filename,
            file_type=file_type,
            chunk_count=len(chunks),
        )
        logger.debug("Document metadata saved. doc_id=%s", doc["id"])

        ids = [f"{doc['id']}_{i}" for i in range(len(chunks))]
        chroma_metadatas = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                import re
                img_count = len(re.findall(r'\{"_img"\s*:\s*\{[^}]+}\s*}', chunk["text"]))
                meta = {
                    "document_id": doc["id"],
                    "chunk_index": i,
                    "filename": filename,
                    "heading": chunk.get("heading", ""),
                    "image_count": img_count,
                }
            else:
                meta = {
                    "document_id": doc["id"],
                    "chunk_index": i,
                    "filename": filename,
                    "heading": "",
                    "image_count": 0,
                }
            chroma_metadatas.append(meta)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=chroma_metadatas,
        )
        logger.info(
            "Stored document %s: title=%r file_type=%s chunks=%d",
            doc["id"], doc["title"], file_type, len(chunks),
        )
        return doc
    except Exception:
        if doc:
            logger.warning(
                "Rolling back document metadata after ChromaDB failure. doc_id=%s", doc["id"],
            )
            await db_delete_document(doc["id"])
        raise


async def delete_document(user_id: str, doc_id: str) -> bool:
    """Delete document from SQLite, ChromaDB, and image files."""
    logger.debug("Deleting document. doc_id=%s user_id=%s", doc_id, user_id)
    deleted = await db_delete_document(doc_id)
    if not deleted:
        logger.debug("Document not found in SQLite. doc_id=%s", doc_id)
        return False
    try:
        collection = _get_collection(user_id)
        results = collection.get(where={"document_id": doc_id})
        vector_count = len(results["ids"]) if results and results["ids"] else 0
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
        logger.info(
            "Deleted document %s: sqlite_ok=True vectors_deleted=%d", doc_id, vector_count,
        )
    except Exception:
        logger.warning(
            "Failed to delete vectors for document %s", doc_id, exc_info=True,
        )
    # Clean up image directory
    img_dir = IMAGES_DIR / doc_id
    if img_dir.exists():
        shutil.rmtree(img_dir)
        logger.debug("Deleted image directory: %s", img_dir)
    return True
