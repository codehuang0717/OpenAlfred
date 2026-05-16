"""
Image handler — copies images from source directory, generates
descriptions via multimodal LLM, and replaces markdown image
syntax with compact JSON placeholders for embedding.

    ![alt](path)  →  {"_img":{"i":1,"d":"description text..."}}

Image metadata is stored in the image_lookup SQLite table for
later resolution back to markdown at query time.
"""

import os
import re
import shutil
import json
from pathlib import Path
from core.config import config
from utils.logger import get_logger

logger = get_logger("rag.image_handler")

IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

IMAGES_DIR = config.PROJECT_ROOT / "data" / "images"


def ensure_images_dir(doc_id: str) -> Path:
    p = IMAGES_DIR / doc_id
    p.mkdir(parents=True, exist_ok=True)
    return p


async def process_section_images(
    text: str,
    source_dir: str,
    doc_id: str,
) -> str:
    """Process images in a section's markdown text.

    For each ![alt](path) found:
      1. Copy the image to data/images/{doc_id}/
      2. Generate a text description via multimodal LLM
      3. Insert into image_lookup table, get integer id
      4. Replace the markdown with {"_img":{"i":id,"d":"desc"}}

    Returns the rewritten text.
    """
    from rag.image_describer import describe_image
    from db.rag import add_image_lookup

    source_path = Path(source_dir)
    dest_dir = ensure_images_dir(doc_id)

    async def _replace(match):
        alt = match.group(1).strip()
        raw_path = match.group(2).strip()

        # Skip external URLs and data URIs
        if raw_path.startswith(("http://", "https://", "data:")):
            return match.group(0)

        # Decode URL-encoded path
        try:
            from urllib.parse import unquote
            raw_path = unquote(raw_path)
        except Exception:
            pass

        # Resolve relative to source dir
        src_file = (source_path / raw_path).resolve()
        if not src_file.exists():
            logger.debug("Image not found: %s", raw_path)
            return match.group(0)

        # Copy to dest
        filename = raw_path.replace("\\", "/").rsplit("/", 1)[-1]
        dest_file = dest_dir / filename
        if dest_file.exists():
            stem, ext = os.path.splitext(filename)
            filename = f"{stem}_{hash(raw_path) & 0xFFFF:04x}{ext}"
            dest_file = dest_dir / filename

        shutil.copy2(src_file, dest_file)
        serving_url = f"/api/images/{doc_id}/{filename}"

        # Describe the image
        description = describe_image(str(dest_file))

        # Insert lookup record
        img_id = await add_image_lookup(
            document_id=doc_id,
            url=serving_url,
            alt=alt,
            filename=filename,
        )

        placeholder = json.dumps(
            {"_img": {"i": img_id, "d": description}},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        logger.debug("Image %s -> id=%d", filename, img_id)
        return placeholder

    # Need to run async replacements — collect matches first
    matches = list(IMAGE_RE.finditer(text))
    if not matches:
        return text

    # Process matches in reverse order so indices don't shift
    result = text
    for m in reversed(matches):
        replacement = await _replace(m)
        result = result[:m.start()] + replacement + result[m.end():]

    return result
