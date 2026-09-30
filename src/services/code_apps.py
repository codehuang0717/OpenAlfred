"""Write standalone browser mini-apps without executing generated code on the host."""

import os
import re
import shutil
import subprocess
from html.parser import HTMLParser

from pydantic import BaseModel, ConfigDict, Field

from core.event_bus import EventType, event_bus
from db.user_apps import create_code_app_job
from services.llm import get_strict_model
from services.user_time import get_user_timezone
from utils.auth_utils import require_explicit_user_id


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
不要声称小程序运行时能够直接读取用户待办、邮件、记忆、位置或账号信息。
界面必须响应式：在约 320px 的右侧工具箱以及 800px 以上的放大窗口中都能自然重排，
不要用 320px 的固定宽度或 max-width 限制整个应用。外层布局使用流式宽度，宽屏可使用多列布局；
字体和控件在宽屏下保持原生清晰，不依赖 CSS transform/zoom 放大。
如果使用 canvas，应按容器尺寸和 devicePixelRatio 调整绘图缓冲区，并在 resize 时重绘。
提供清楚的空状态和可键盘操作的控件。"""


def validate_code_source(source: CodeAppSource) -> dict:
    """Fail closed on malformed candidates; Node parses JS but never runs it."""
    if re.search(r"<\s*/?\s*(script|iframe|object|embed|base|meta|link)\b", source.html, re.I):
        raise ValueError("HTML contains a forbidden element")
    _AppHtmlPolicy().feed(source.html)
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
    # Defense in depth, NOT proof of isolation against obfuscated malicious JS.
    forbidden_js = (
        r"\b(fetch|XMLHttpRequest|WebSocket|EventSource|Worker|SharedWorker|importScripts|eval)\s*\("
        r"|\bimport\s*\(|\.\s*(sendBeacon|open|cookie|innerHTML|outerHTML|insertAdjacentHTML)\b"
        r"|\bdocument\s*\.\s*(write|writeln)\s*\("
        r"|\b(?:window|document|globalThis)\s*\.\s*location\b"
        r"|\blocation\s*(?:\.|\[|=)|\.\s*href\s*="
        r"|\.\s*setAttribute\s*\(\s*['\"](?:href|action|formaction|srcdoc|on\w+)['\"]"
    )
    # JavaScript's lowercase `function(...)` is an ordinary anonymous function,
    # not the uppercase `Function(...)` dynamic-code constructor.
    if re.search(forbidden_js, source.javascript, re.I) or re.search(r"\bFunction\s*\(", source.javascript):
        raise ValueError("Network, navigation, dynamic execution and HTML injection APIs are not allowed; use DOM/textContent")
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
    return {"javascript_syntax": "passed", "runtime": "browser_sandbox", "data_access": "none",
            "browser_tests": "not_run"}


class _AppHtmlPolicy(HTMLParser):
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "form":
            raise ValueError("Forms are not allowed in a snapshot app")
        for name, value in attrs:
            if name.startswith("on") or name in {"action", "formaction", "srcdoc", "target", "ping"}:
                raise ValueError("Inline handlers and navigation attributes are not allowed")
            if name in {"href", "xlink:href"} and not (value or "").startswith("#"):
                raise ValueError("Only in-document fragment links are allowed")


def create_codegen_model(selection: str):
    """Reuse provider adapters, never the main agent's closable HTTP pools."""
    return get_strict_model(selection, isolated_http_clients=True)


async def coding_context(user_id: str, selection: str, run_config: dict | None = None) -> dict:
    """Snapshot trusted context; never persist caller credentials or whole chat history."""
    model = create_codegen_model(selection)
    try:
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None)
        if not model_name:
            raise ValueError("编码模型缺少具体模型名称")
        return {"timezone": await get_user_timezone(user_id, run_config),
                "model_name": model_name, "checkpoint_generation": 0,
                "source_thread_id": (run_config or {}).get("configurable", {}).get("thread_id")}
    finally:
        # Preflight owns its pools, separate from both main chat and the worker.
        if getattr(model, "root_async_client", None) is not None:
            await model.root_async_client.close()
        if getattr(model, "root_client", None) is not None:
            model.root_client.close()


async def create_code_app(
    user_id: str, request: CodeAppRequest, model_selection: str, *, run_config: dict | None = None,
) -> dict:
    user_id = require_explicit_user_id(user_id)
    if not shutil.which("node"):
        raise RuntimeError("Node.js is required for code generation validation")
    context = await coding_context(user_id, model_selection, run_config)
    job = await create_code_app_job(user_id, request.title, request.prompt, model_selection, context=context, queued=True)
    await event_bus.publish(EventType.USER_APP_UPDATED, {"id": job["app_id"], "user_id": user_id})
    return {**job, "status": "queued"}
