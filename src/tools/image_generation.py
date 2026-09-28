"""Image generation exposed to the agent with a small, persistent UI artifact."""

from typing import Literal

from langchain.tools import ToolRuntime, tool

from services.generated_images import generate_image_for_user
from utils.auth_utils import require_runtime_user_id


@tool(response_format="content_and_artifact")
async def generate_image(
    runtime: ToolRuntime,
    prompt: str,
    size: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = "auto",
) -> tuple[str, dict]:
    """Generate one new image when the user asks to draw or create an image.

    Include the user's subjects, style, composition and desired text in prompt.
    The chat displays the result automatically. Do not invent image URLs or
    repeat the image as Markdown. This tool creates new images, not edits.
    """
    artifact = await generate_image_for_user(require_runtime_user_id(runtime), prompt, size)
    return "图片已生成并显示在本次聊天中。请简短回复，无需重复插入图片。", artifact


image_tools = [generate_image]
