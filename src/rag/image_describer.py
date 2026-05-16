"""
Image describer — uses Gemini multimodal to generate text descriptions
of images, with file-hash-based cache to avoid re-describing.
"""

import os
import hashlib
from pathlib import Path
from core.config import config
from utils.logger import get_logger

logger = get_logger("rag.image_describer")

CACHE_DIR = config.PROJECT_ROOT / "data" / "descriptions"

DESCRIBE_PROMPT = (
    "Please describe this image in detail, in Chinese. "
    "Focus on: what this image shows (diagram/screenshot/photo), "
    "the key information, text content, and concepts presented. "
    "Keep the description within 200 characters. "
    "Only output the description text, nothing else."
)


def _hash_file(filepath: str) -> str:
    """SHA256 hash of file content for cache key."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(file_hash: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{file_hash[:16]}.txt"


def _read_cache(file_hash: str) -> str | None:
    path = _cache_path(file_hash)
    if path.exists():
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        # Handle dirty cache from gemini-3-flash with extras/signature
        if raw.startswith("[{") and "extras" in raw:
            parsed = _extract_text_from_response(raw)
            if parsed:
                _write_cache(file_hash, parsed)
                return parsed
        return raw
    return None


def _extract_text_from_response(raw: str) -> str:
    """Extract clean text from a Gemini response that leaked extras metadata."""
    import ast
    try:
        blocks = ast.literal_eval(raw)
        if isinstance(blocks, list):
            parts = []
            for b in blocks:
                if isinstance(b, dict):
                    parts.append(b.get("text", ""))
            return " ".join(parts).strip()
    except Exception:
        return ""


def _write_cache(file_hash: str, description: str):
    path = _cache_path(file_hash)
    path.write_text(description, encoding="utf-8")


def describe_image(image_path: str, force: bool = False) -> str:
    """Generate a Chinese text description for an image using Gemini.

    Uses file-content hash for cache. Returns empty string on failure.
    """
    if not os.path.exists(image_path):
        logger.warning("Image not found for description: %s", image_path)
        return ""

    file_hash = _hash_file(image_path)

    if not force:
        cached = _read_cache(file_hash)
        if cached is not None:
            logger.debug("Using cached description for %s (hash=%s)", image_path, file_hash[:12])
            return cached

    logger.info("Describing image: %s (hash=%s)", Path(image_path).name, file_hash[:12])

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        import base64

        model = ChatGoogleGenerativeAI(
            model=config.GEMINI_CHAT_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                     ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
        mime_type = mime_map.get(ext, "image/png")

        msg = HumanMessage(content=[
            {"type": "text", "text": DESCRIBE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
        ])

        response = model.invoke([msg])
        raw = response.content
        if isinstance(raw, list):
            # Extract text from content blocks
            parts = []
            for block in raw:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                else:
                    parts.append(str(block))
            description = " ".join(parts).strip()
        else:
            description = str(raw).strip()

        if description:
            _write_cache(file_hash, description)
            logger.info("Image described: %s -> %.80s...", Path(image_path).name, description)
            return description
        else:
            logger.warning("Gemini returned empty description for %s", image_path)
            return ""

    except Exception as e:
        logger.warning("Failed to describe image %s: %s", image_path, e)
        return ""
