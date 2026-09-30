"""Generate a bounded thread title without hiding provider failures."""

import asyncio

from langchain_core.messages import HumanMessage

from logic.prompts import TITLE_GENERATION_PROMPT
from services.llm import get_strict_model


async def generate_title(content: str | list) -> str:
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "file_text"}
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("首条消息没有可用于标题的文本")
    model = get_strict_model("mimo-title")
    result = await asyncio.wait_for(
        model.ainvoke(
            [HumanMessage(content=TITLE_GENERATION_PROMPT.format(message=content))],
            config={"callbacks": []}, max_tokens=256,
        ), timeout=30,
    )
    if not isinstance(result.content, str) or not result.content.strip():
        raise ValueError("标题模型返回空结果或非文本结果")
    title = result.content.strip().strip('"\'')[:20].strip()
    if not title:
        raise ValueError("标题模型未返回有效标题")
    return title
