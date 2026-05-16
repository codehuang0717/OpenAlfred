import uuid
from datetime import datetime, timezone
from db.connection import get_db


async def add_document(
    user_id: str,
    filename: str,
    title: str = "",
    file_type: str = "txt",
    chunk_count: int = 0,
) -> dict:
    doc_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    async with get_db() as db:
        await db.execute(
            """INSERT INTO documents (id, user_id, filename, title, file_type, chunk_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, user_id, filename, title, file_type, chunk_count, now),
        )
        await db.commit()
    return {
        "id": doc_id,
        "user_id": user_id,
        "filename": filename,
        "title": title,
        "file_type": file_type,
        "chunk_count": chunk_count,
        "created_at": now,
    }


async def get_documents(user_id: str) -> list[dict]:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, user_id, filename, title, file_type, chunk_count, created_at "
            "FROM documents WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def get_document_by_id(doc_id: str) -> dict | None:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, user_id, filename, title, file_type, chunk_count, created_at "
            "FROM documents WHERE id = ?",
            (doc_id,),
        )
        row = await cursor.fetchone()
    return dict(row) if row else None


async def delete_document(doc_id: str) -> bool:
    async with get_db() as db:
        cursor = await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await db.commit()
    return cursor.rowcount > 0


async def update_chunk_count(doc_id: str, chunk_count: int):
    async with get_db() as db:
        await db.execute(
            "UPDATE documents SET chunk_count = ? WHERE id = ?",
            (chunk_count, doc_id),
        )
        await db.commit()


async def add_image_lookup(document_id: str, url: str, alt: str, filename: str) -> int:
    """Insert an image record and return its integer id."""
    async with get_db() as db:
        cursor = await db.execute(
            "INSERT INTO image_lookup (document_id, url, alt, filename) VALUES (?, ?, ?, ?)",
            (document_id, url, alt, filename),
        )
        await db.commit()
        return cursor.lastrowid


async def get_images_for_doc(document_id: str) -> list[dict]:
    """Get all image records for a document."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, document_id, url, alt, filename, created_at "
            "FROM image_lookup WHERE document_id = ? ORDER BY id",
            (document_id,),
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def get_image_by_id(img_id: int) -> dict | None:
    """Get a single image record by id."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, document_id, url, alt, filename, created_at "
            "FROM image_lookup WHERE id = ?",
            (img_id,),
        )
        row = await cursor.fetchone()
    return dict(row) if row else None


async def delete_images_for_doc(document_id: str):
    """Delete all image records for a document."""
    async with get_db() as db:
        await db.execute("DELETE FROM image_lookup WHERE document_id = ?", (document_id,))
        await db.commit()
