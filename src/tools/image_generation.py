"""Image generation exposed to the agent with a small, persistent UI artifact."""

from typing import Literal
from openai import APIConnectionError, APITimeoutError
from services.tool_observations import observed, fields

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
    try:
        artifact = await generate_image_for_user(
            require_runtime_user_id(runtime), prompt, size
        )
    except (APIConnectionError, APITimeoutError):
        observed(
            None,
            "生成连接中断，远端可能已处理，请核实后再尝试",
            status="unknown",
            details=fields(说明="可能已产生生成费用；不会自动重试"),
        )
        raise
    return observed(
        ("图片已生成并显示在本次聊天中。请简短回复，无需重复插入图片。", artifact),
        "图片已生成，可在聊天中打开查看",
        details=fields(描述=prompt, 请求尺寸=size, 说明="实际图片以预览为准"),
    )


image_tools = [generate_image]
