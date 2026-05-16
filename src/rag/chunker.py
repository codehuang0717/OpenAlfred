"""
Smart chunker — for markdown files, chunks by heading sections
to keep text and images together. Falls back to RecursiveCharacterTextSplitter
for non-markdown files or overlong sections.
"""

from core.config import config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import get_logger

logger = get_logger("rag.chunker")


def chunk_sections(sections: list) -> list[dict]:
    """Convert parsed Sections into chunks with metadata.

    Each section becomes one or more chunks (split if overlong).
    Returns list of dicts: {text, heading, images}

    Args:
        sections: list of Section dataclasses from md_parser
    """
    chunks = []
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.RAG_CHUNK_SIZE,
        chunk_overlap=config.RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", "{", " ", ""],
    )

    for sec in sections:
        heading = sec.heading
        images = [{"alt": img.alt, "filename": img.filename} for img in sec.images]

        # Build chunk text: heading + body
        if heading:
            prefix = f"# {heading}" if sec.heading_level == 1 else f"## {heading}"
            full_text = f"{prefix}\n\n{sec.text}"
        else:
            full_text = sec.text

        if not full_text.strip():
            continue

        # If short enough, one chunk
        if len(full_text) <= config.RAG_CHUNK_SIZE:
            chunks.append({
                "text": full_text,
                "heading": heading,
                "images": images,
            })
        else:
            # For long sections, split the body while keeping heading as prefix
            if heading:
                head_prefix = f"# {heading}\n\n" if sec.heading_level == 1 else f"## {heading}\n\n"
                sub_chunks = fallback_splitter.split_text(sec.text)
                for sc in sub_chunks:
                    if sc.strip():
                        chunks.append({
                            "text": head_prefix + sc,
                            "heading": heading,
                            "images": images,  # same images associated with all sub-chunks
                        })
            else:
                sub_chunks = fallback_splitter.split_text(full_text)
                for sc in sub_chunks:
                    if sc.strip():
                        chunks.append({
                            "text": sc,
                            "heading": "",
                            "images": images,
                        })

    logger.debug(
        "Section chunking: sections=%d chunks=%d avg_len=%d",
        len(sections), len(chunks),
        sum(len(c["text"]) for c in chunks) // len(chunks) if chunks else 0,
    )
    return chunks


def is_markdown(filename: str) -> bool:
    return filename.lower().endswith((".md", ".markdown"))


def chunk_text(text: str, filename: str = "") -> list[str]:
    """Simple text chunking for non-markdown files. Returns plain text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.RAG_CHUNK_SIZE,
        chunk_overlap=config.RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    logger.debug(
        "Text chunking: input_chars=%d output_chunks=%d avg_chunk_len=%d",
        len(text), len(chunks), avg_len,
    )
    return chunks
