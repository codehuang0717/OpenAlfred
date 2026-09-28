"""Write standalone browser mini-apps without executing generated code on the host."""

import asyncio
import json
import os
import re
import shutil
import subprocess
from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError
from pydantic import BaseModel, ConfigDict, Field

from core.config import config
from core.event_bus import EventType, event_bus
from db.user_apps import create_code_app_job, finish_code_app_job, set_code_app_job_stage
from services.llm import CEREBRAS_BASE_URL, DEEPSEEK_BASE_URL, MIMO_BASE_URL
from utils.auth_utils import require_explicit_user_id
from utils.logger import get_logger

logger = get_logger("code_apps")


class CodeAppRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=40)
    prompt: str = Field(min_length=10, max_length=2000)


class CodeAppSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    html: str = Field(min_length=1, max_length=30000)
    css: str = Field(max_length=30000)
    javascript: str = Field(max_length=30000)


SYSTEM_PROMPT = """你是 OpenAlfred 的小程序编码 Agent。根据用户需求编写可独立运行的小程序。
只返回 JSON 对象，严格包含 html、css、javascript 三个字符串字段，不要 Markdown。
html 是放入 body 的片段，javascript 是普通浏览器脚本，使用原生 DOM API。
所有内容必须自包含；不要使用外部资源、网络请求、包依赖、iframe、表单提交或主应用 API。
不要使用 localStorage、sessionStorage、IndexedDB 或 Cookie；状态仅在当前运行会话有效。
不要声称能够读取用户真实待办、邮件、记忆、位置或账号信息。
界面必须响应式：在约 320px 的右侧工具箱以及 800px 以上的放大窗口中都能自然重排，
不要用 320px 的固定宽度或 max-width 限制整个应用。外层布局使用流式宽度，宽屏可使用多列布局；
字体和控件在宽屏下保持原生清晰，不依赖 CSS transform/zoom 放大。
如果使用 canvas，应按容器尺寸和 devicePixelRatio 调整绘图缓冲区，并在 resize 时重绘。
提供清楚的空状态和可键盘操作的控件。"""


def validate_code_source(source: CodeAppSource) -> dict:
    """Fail closed on malformed candidates; Node parses JS but never runs it."""
    if re.search(r"<\s*/?\s*(script|iframe|object|embed|base|meta|link)\b", source.html, re.I):
        raise ValueError("HTML contains a forbidden element")
    if re.search(r"@import\b|url\s*\(", source.css, re.I):
        raise ValueError("CSS may not load external resources")
    for rule in re.finditer(r"([^{}]+)\{([^{}]*)\}", source.css):
        selector, declarations = rule.groups()
        if not re.search(
            r"(?:^|,)\s*(?:html|body|main|#(?:app|root)|\.(?:app|container|wrapper|shell))\s*(?:,|$)",
            selector.strip(), re.I,
        ):
            continue
        for cap in re.finditer(r"\bmax-width\s*:\s*(\d+)px\b", declarations, re.I):
            if int(cap.group(1)) <= 480:
                raise ValueError("Main app layout is capped to a narrow width; use a fluid container")
    if re.search(r"\b(localStorage|sessionStorage|indexedDB)\b", source.javascript, re.I):
        raise ValueError("Persistent browser storage is unavailable in this sandbox")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required for JavaScript syntax validation")
    result = subprocess.run(
        [node, "--check", "-"],
        input=source.javascript,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
        env={
            "SystemRoot": os.environ.get("SystemRoot", ""),
            "TEMP": os.environ.get("TEMP", ""),
            "TMP": os.environ.get("TMP", ""),
            "NODE_OPTIONS": "",
        },
    )
    if result.returncode:
        raise ValueError(f"JavaScript syntax check failed: {result.stderr[:500]}")
    return {"javascript_syntax": "passed", "runtime": "browser_sandbox", "data_access": "none"}


def create_codegen_model(selection: str):
    """Use exactly the chat-selected provider; never switch credentials or models."""
    providers = {
        "gpt-cloud": (config.CLOUD_CHAT_MODEL, config.OPENAI_API_KEY, None, "OPENAI_API_KEY"),
        "cerebras": (config.CEREBRAS_CHAT_MODEL, config.CEREBRAS_API_KEY, CEREBRAS_BASE_URL, "CEREBRAS_API_KEY"),
        "deepseek": (config.DEEPSEEK_FLASH_MODEL, config.DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, "DEEPSEEK_API_KEY"),
        "deepseek-pro": (config.DEEPSEEK_PRO_MODEL, config.DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, "DEEPSEEK_API_KEY"),
        "mimo": (config.MIMO_CHAT_MODEL, config.MIMO_API_KEY, MIMO_BASE_URL, "MIMO_API_KEY"),
        "mimo-v2.6": ("mimo-v2.6", config.MIMO_API_KEY, MIMO_BASE_URL, "MIMO_API_KEY"),
    }
    if selection in providers:
        model_name, api_key, base_url, key_name = providers[selection]
        if not api_key:
            raise RuntimeError(f"所选模型 {selection} 缺少 {key_name} 配置")
        kwargs = {"model": model_name, "api_key": api_key, "max_retries": 1}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    if selection == "gemini":
        if not config.GOOGLE_API_KEY:
            raise RuntimeError("所选模型 gemini 缺少 GOOGLE_API_KEY 配置")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_CHAT_MODEL, google_api_key=config.GOOGLE_API_KEY,
        )
    if selection == "gemma-local":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=config.LOCAL_MODEL_NAME, base_url=config.OLLAMA_BASE_URL)
    raise ValueError(f"不支持的小程序生成模型：{selection}")


async def write_code_source(
    request: CodeAppRequest, model, *, previous_source: dict | None = None,
    on_stage: Callable[[str], Awaitable[None]] | None = None,
) -> CodeAppSource:
    user_prompt = f"标题：{request.title}\n需求：{request.prompt}"
    if previous_source is not None:
        user_prompt += (
            "\n请保留现有功能、内容与交互，只修订为窄屏和宽屏都能自然重排的版本。"
            "不要仅放大旧画面。现有代码：\n"
            + json.dumps(previous_source, ensure_ascii=False)
        )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    last_error = ""
    for attempt in range(2):
        response = await model.ainvoke(messages)
        if on_stage is not None:
            await on_stage("validation")
        content = response.content
        if not isinstance(content, str):
            last_error = "Model returned non-text content"
        else:
            try:
                source = CodeAppSource.model_validate(json.loads(content))
                await asyncio.to_thread(validate_code_source, source)
                return source
            except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
                last_error = str(exc)[:500]
        if attempt == 0:
            messages.append(HumanMessage(content=f"上一版未通过验证：{last_error}。请修复并只返回 JSON。"))
            if on_stage is not None:
                await on_stage("model")
    raise ValueError(f"Code generation failed validation: {last_error}")


async def _complete_code_app_job(
    user_id: str, job: dict, request: CodeAppRequest, model_selection: str,
    *, previous_source: dict | None = None, model=None,
) -> dict:
    async def on_stage(stage: str) -> None:
        await set_code_app_job_stage(user_id, job["job_id"], stage)
        await event_bus.publish(EventType.USER_APP_UPDATED, {"id": job["app_id"], "user_id": user_id})

    try:
        if model is None:
            model = create_codegen_model(model_selection)
        source = await write_code_source(
            request, model, previous_source=previous_source, on_stage=on_stage,
        )
        validation = await asyncio.to_thread(validate_code_source, source)
        revision_id = await finish_code_app_job(
            user_id, job["job_id"], source=source.model_dump(),
            validation=validation,
        )
    except Exception as exc:
        logger.exception("User app generation failed for app %s using %s", job["app_id"], model_selection)
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            public_error = (
                f"无法连接所选模型服务（{model_selection}）。请检查该服务的网络或代理后重试；"
                "未切换模型，也未发布代码。"
            )
        elif isinstance(exc, ValueError):
            public_error = str(exc)[:500]
        elif isinstance(exc, RuntimeError) and str(exc).startswith("所选模型 "):
            public_error = str(exc)[:500]
        else:
            public_error = f"所选模型 {model_selection} 生成失败（{type(exc).__name__}）；未发布代码。"
        await finish_code_app_job(user_id, job["job_id"], error=public_error)
        await event_bus.publish(EventType.USER_APP_UPDATED, {"id": job["app_id"], "user_id": user_id})
        return {**job, "status": "failed", "error": public_error}
    await event_bus.publish(EventType.USER_APP_UPDATED, {"id": job["app_id"], "user_id": user_id})
    return {**job, "status": "ready", "revision_id": revision_id}


async def create_code_app(user_id: str, request: CodeAppRequest, model_selection: str) -> dict:
    user_id = require_explicit_user_id(user_id)
    if not shutil.which("node"):
        raise RuntimeError("Node.js is required for code generation validation")
    model = create_codegen_model(model_selection)
    job = await create_code_app_job(user_id, request.title, request.prompt, model_selection)
    await event_bus.publish(EventType.USER_APP_UPDATED, {"id": job["app_id"], "user_id": user_id})
    return await _complete_code_app_job(user_id, job, request, model_selection, model=model)


async def complete_code_app_revision(user_id: str, job: dict) -> dict:
    """Complete a queued responsive draft; keep any published revision untouched."""
    user_id = require_explicit_user_id(user_id)
    request = CodeAppRequest(title=job["title"], prompt=job["prompt"])
    await event_bus.publish(EventType.USER_APP_UPDATED, {"id": job["app_id"], "user_id": user_id})
    return await _complete_code_app_job(
        user_id, job, request, job["model"], previous_source=job["previous_source"],
    )
