from datetime import datetime, timezone
from pathlib import Path
from rag.chunker import chunk_text, chunk_sections, is_markdown
from rag.md_parser import parse_markdown
from rag.store import store_document
from utils.logger import get_logger

logger = get_logger("rag.ingestion")

SUPPORTED_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".json": "application/json",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".csv": "text/csv",
    ".html": "text/html",
    ".css": "text/css",
}


def _read_text_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    logger.debug("Read text file %s: %d chars", filepath, len(content))
    return content


def _read_pdf(filepath: str) -> str:
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        logger.debug("Read PDF %s: %d pages", filepath, len(pages))
        return "\n\n".join(p.page_content for p in pages)
    except ImportError:
        raise ImportError(
            "PDF support requires pypdf. Install with: pip install pypdf"
        )


async def ingest_file(user_id: str, filepath: str, title: str = "", progress_callback = None) -> dict:
    """Ingest a single file into the RAG knowledge base.
    For .md files, auto-detects source directory for image resolution.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = path.suffix.lower()
    logger.info("Ingesting file: %s (type=%s size=%d)", path.name, ext, path.stat().st_size)

    if ext in (".pdf", ".docx"):
        if ext == ".pdf":
            text = _read_pdf(filepath)
            file_type = "pdf"
        elif ext == ".docx":
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(filepath)
                docs = loader.load()
                text = "\n\n".join(d.page_content for d in docs)
                file_type = "docx"
            except ImportError:
                raise ImportError(
                    "DOCX support requires docx2txt. Install with: pip install docx2txt"
                )
        if progress_callback: progress_callback("embedding", 50)
        doc = await ingest_text(user_id, text, title or path.stem)
        if progress_callback: progress_callback("done", 100)
        return doc

    if ext == ".md":
        content = _read_text_file(filepath)
        doc = await ingest_markdown_file(
            user_id=user_id,
            filename=path.name,
            content=content,
            title=title or path.stem,
            source_dir=str(path.parent),
            progress_callback=progress_callback,
        )
        return doc

    # Plain text files
    text = _read_text_file(filepath)
    if progress_callback: progress_callback("embedding", 50)
    doc = await ingest_text(user_id, text, title or path.stem)
    if progress_callback: progress_callback("done", 100)
    return doc


async def ingest_markdown_file(
    user_id: str,
    filename: str,
    content: str,
    title: str,
    source_dir: str | None = None,
    progress_callback = None,
) -> dict:
    """Ingest a markdown file with image handling.

    1. Parse sections by heading
    2. Copy images to local store (if source_dir provided)
    3. Chunk by sections (keeping text+images together)
    4. Store in ChromaDB + SQLite
    """
    logger.info("Ingesting markdown: title=%r chars=%d source_dir=%s", title, len(content), source_dir)
    if progress_callback: progress_callback("parsing", 20)

    # Parse into sections
    sections = parse_markdown(content)
    logger.debug("Parsed %d sections", len(sections))

    # Generate doc_id early for image dir
    import uuid
    doc_id = str(uuid.uuid4())

    # Process images if source_dir provided
    if source_dir:
        if progress_callback: progress_callback("images", 40)
        from rag.image_handler import process_section_images

        for sec in sections:
            if sec.images:
                sec.text = await process_section_images(sec.text, source_dir, doc_id)
                logger.debug("Section '%s': processed %d images", sec.heading, len(sec.images))

        logger.info("Image processing complete for document %s", doc_id)
    else:
        logger.debug("No source_dir provided — skipping image copy")

    # Chunk by sections
    chunks = chunk_sections(sections)
    if not chunks:
        raise ValueError("Markdown produced no chunks (empty file?)")

    # Override doc_id — store_document will use our pre-generated one
    if progress_callback: progress_callback("embedding", 70)
    doc = await _store_with_id(
        user_id=user_id,
        doc_id=doc_id,
        filename=filename,
        title=title,
        file_type="md",
        chunks=chunks,
        progress_callback=progress_callback,
    )
    logger.info(
        "Ingested markdown: title=%r sections=%d chunks=%d doc_id=%s",
        title, len(sections), len(chunks), doc["id"],
    )
    if progress_callback: progress_callback("done", 100)
    return doc


async def _store_with_id(
    user_id: str,
    doc_id: str,
    filename: str,
    title: str,
    file_type: str,
    chunks: list,
    progress_callback = None,
) -> dict:
    """Store document with a pre-generated ID (needed for image path matching)."""
    from rag.embedding import embed_texts
    from rag.store import _get_collection

    collection = _get_collection(user_id)
    texts = [c["text"] if isinstance(c, dict) else c for c in chunks]
    embeddings = embed_texts(texts)

    now = datetime.now(timezone.utc).isoformat()

    # Manually insert with pre-generated ID
    from db.connection import get_db
    async with get_db() as db:
        await db.execute(
            """INSERT INTO documents (id, user_id, filename, title, file_type, chunk_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, user_id, filename, title, file_type, len(chunks), now),
        )
        await db.commit()
    logger.debug("Document metadata saved with provided id. doc_id=%s", doc_id)

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    chroma_metadatas = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            # Count JSON image placeholders in text
            import re
            img_count = len(re.findall(r'\{"_img"\s*:\s*\{[^}]+}\s*}', chunk["text"]))
            meta = {
                "document_id": doc_id,
                "chunk_index": i,
                "filename": filename,
                "heading": chunk.get("heading", ""),
                "image_count": img_count,
            }
        else:
            meta = {
                "document_id": doc_id,
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
    logger.info("Stored document %s: title=%r chunks=%d", doc_id, title, len(chunks))

    return {
        "id": doc_id,
        "user_id": user_id,
        "filename": filename,
        "title": title,
        "file_type": file_type,
        "chunk_count": len(chunks),
        "created_at": now,
    }


async def ingest_text(user_id: str, text: str, title: str) -> dict:
    """Ingest raw text into the RAG knowledge base."""
    if not text.strip():
        raise ValueError("Text is empty")

    logger.info("Ingesting text: title=%r chars=%d", title, len(text))
    chunks = chunk_text(text)
    doc = await store_document(
        user_id=user_id,
        filename=f"{title}.txt",
        title=title,
        file_type="txt",
        chunks=chunks,
    )
    logger.info(
        "Ingested text: title=%r chunks=%d doc_id=%s",
        title, len(chunks), doc["id"],
    )
    return doc
