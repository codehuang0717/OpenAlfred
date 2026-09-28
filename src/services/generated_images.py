"""Generate images and store them outside public static files, scoped by owner."""

import asyncio
import base64
import hashlib
from pathlib import Path
import re
from uuid import uuid4

from openai import AsyncOpenAI

from core.config import config
from utils.auth_utils import require_config_value, require_explicit_user_id


IMAGE_ID = re.compile(r"^[0-9a-f]{32}$")
IMAGE_URL = re.compile(r"^/api/generated-images/[0-9a-f]{32}$")
MAX_IMAGE_BYTES = 30 * 1024 * 1024


def image_path(user_id: str, image_id: str) -> Path:
    owner = require_explicit_user_id(user_id)
    if not IMAGE_ID.fullmatch(image_id):
        raise ValueError("Invalid image identifier")
    owner_dir = hashlib.sha256(owner.encode("utf-8")).hexdigest()
    return config.PROJECT_ROOT / "data" / "generated_images" / owner_dir / f"{image_id}.png"


def image_markdown(artifact: object) -> str:
    """Only turn our structured image artifact into visible chat content."""
    if not isinstance(artifact, dict) or artifact.get("type") != "generated_image":
        return ""
    url = artifact.get("url")
    if not isinstance(url, str) or not IMAGE_URL.fullmatch(url):
        return ""
    return f"![生成的图片]({url})"


def _save_image(user_id: str, encoded: str) -> dict:
    if len(encoded) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        raise ValueError("Generated image exceeds the storage limit")
    payload = base64.b64decode(encoded, validate=True)
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Image provider did not return the requested PNG format")
    image_id = uuid4().hex
    path = image_path(user_id, image_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)
    return {"type": "generated_image", "url": f"/api/generated-images/{image_id}"}


async def generate_image_for_user(user_id: str, prompt: str, size: str) -> dict:
    owner = require_explicit_user_id(user_id)
    if not prompt.strip() or len(prompt) > 16000:
        raise ValueError("Image prompt must contain 1 to 16000 characters")
    if size not in {"1024x1024", "1536x1024", "1024x1536", "auto"}:
        raise ValueError("Unsupported image size")
    key = require_config_value("OPENAI_API_KEY", config.OPENAI_API_KEY)
    # A timeout may follow a billable generation; do not automatically duplicate it.
    async with AsyncOpenAI(api_key=key, timeout=180.0, max_retries=0) as client:
        response = await client.images.generate(
            model=config.IMAGE_GENERATION_MODEL, prompt=prompt, n=1,
            size=size, quality="auto", output_format="png",
        )
    if not response.data or not response.data[0].b64_json:
        raise RuntimeError("Image provider returned no image")
    return await asyncio.to_thread(_save_image, owner, response.data[0].b64_json)
