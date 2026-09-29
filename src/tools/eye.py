import httpx
from utils.logger import get_logger
from typing import Optional, List, Dict, Literal
from datetime import datetime, timedelta, timezone
from services.user_time import runtime_timezone
from utils.time_utils import localize_to_utc
from core.config import config
from services.screen_monitor import require_screen_access
from utils.auth_utils import require_runtime_user_id

logger = get_logger("eye-tool")

from rich.console import Console
from rich.panel import Panel

console = Console()

# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

async def _search_screenpipe(
    user_id: str,
    q: Optional[str] = None,
    content_type: str = "all",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    app_name: Optional[str] = None,
    window_name: Optional[str] = None,
    browser_url: Optional[str] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    speaker_name: Optional[str] = None,
    limit: int = 60,
    offset: int = 0,
    include_frames: bool = False,
    timeout: float = 10.0,
) -> dict:
    """Low-level search against the Screenpipe /search endpoint.

    Returns the parsed JSON response dict (with ``data`` and ``pagination`` keys)
    or an error dict with an ``error`` key.
    """
    await require_screen_access(user_id)
    params: Dict[str, str | int] = {
        "limit": limit,
        "offset": offset,
        "content_type": content_type,
        "include_frames": str(include_frames).lower(),
    }
    if q:
        params["q"] = q
    if start_time:
        params["start_time"] = start_time
    if end_time:
        params["end_time"] = end_time
    if app_name:
        params["app_name"] = app_name
    if window_name:
        params["window_name"] = window_name
    if browser_url:
        params["browser_url"] = browser_url
    if min_length is not None:
        params["min_length"] = min_length
    if max_length is not None:
        params["max_length"] = max_length
    if speaker_name:
        params["speaker_name"] = speaker_name

    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            resp = await client.get(
                f"{config.SCREENPIPE_URL}/search",
                params=params,
                timeout=timeout,
            )
            if resp.status_code != 200:
                logger.error(f"Screenpipe Search Error: {resp.status_code}")
                return {"error": f"Screenpipe returned {resp.status_code}: {resp.text[:300]}"}
            return resp.json()
    except httpx.TimeoutException:
        return {"error": "Screenpipe request timed out"}
    except Exception as e:
        logger.error(f"Error connecting to Screenpipe: {e}")
        return {"error": f"Error connecting to Screenpipe: {str(e)}"}


def _format_content_item(item: dict) -> str:
    """Format a single ContentItem into a readable one-line summary."""
    itype = item.get("type", "?")
    content = item.get("content", {})
    ts = content.get("timestamp", "?")
    app = content.get("app_name") or content.get("app") or "N/A"
    window = content.get("window_name", "")
    url = content.get("browser_url", "")
    text = content.get("text") or content.get("transcription") or content.get("text_content") or ""

    # Truncate long text
    text = text.strip()
    if len(text) > 300:
        text = text[:300] + "..."

    parts = [f"[{ts}]"]
    if itype:
        parts.append(f"[{itype}]")
    parts.append(f"[{app}]")
    if window and window != app:
        parts.append(f"({window})")
    if url:
        parts.append(f"{{{url}}}")
    parts.append(text if text else "(no text)")

    return " ".join(parts)


def _format_pagination(pagination: dict) -> str:
    """Format pagination info for LLM consumption."""
    total = pagination.get("total", 0)
    limit = pagination.get("limit", 0)
    offset = pagination.get("offset", 0)
    if total <= limit:
        return f"Returned {total} results."
    page_num = (offset // limit) + 1 if limit else 1
    total_pages = (total + limit - 1) // limit if limit else 1
    return f"Page {page_num}/{total_pages} — {offset+1}-{min(offset+limit, total)} of {total} total results."

# ──────────────────────────────────────────────
# Legacy convenience functions (kept for supervisor)
# ──────────────────────────────────────────────

async def get_enhanced_context(user_id: str, minutes: int = 10) -> str:
    """Read owner-scoped OCR; acquisition errors must not become model input."""
    start = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    resp = await _search_screenpipe(
        user_id=user_id, content_type="ocr", start_time=start, limit=60,
    )
    if "error" in resp:
        raise RuntimeError(resp["error"])
    items = resp.get("data", [])
    if not items:
        raise RuntimeError("Screenpipe 在指定时间内没有 OCR 数据，跳过智能分析")
    return "\n".join(_format_content_item(item) for item in items)


async def get_recent_ocr_text(user_id: str, minutes: int = 10) -> str:
    """Legacy wrapper for backward compatibility."""
    return await get_enhanced_context(user_id, minutes)


# ──────────────────────────────────────────────
# LangChain tool definitions
# ──────────────────────────────────────────────

from langchain.tools import ToolRuntime, tool


@tool
async def view_screen(
    runtime: ToolRuntime,
    mode: Literal["current", "history", "time_range"] = "current",
    query: str = "",
    start_time: str = "",
    end_time: str = "",
    app_name: str = "",
    content_type: Literal["all", "ocr", "audio", "input", "ui"] = "all",
    limit: int = 30,
) -> str:
    """View the user's screen content and audio activity captured by Screenpipe.

    Three query modes:
    - 'current' — what the user is doing RIGHT NOW (last 2-5 min). Fast and focused.
    - 'history' — full-text search across ALL captured content. Use when the user
      asks about a specific keyword, phrase, or topic (e.g., "what was I reading
      about React hooks?").
    - 'time_range' — query content within a specific time window. Use when the
      user asks about a specific date or time (e.g., "what was on my screen
      yesterday at 3pm?", "show me my activity between 2-4pm last Friday").

    Time format: ISO 8601 strings like '2026-05-06T14:00:00+08:00' or
    '2026-05-06T14:00:00Z'. The start_time and end_time params are only used
    with mode='time_range'.
    """
    try:
        user_id = require_runtime_user_id(runtime)
        if mode == "current":
            now_utc = datetime.now(timezone.utc)
            st = (now_utc - timedelta(minutes=5)).isoformat()
            resp = await _search_screenpipe(
                user_id=user_id,
                content_type=content_type,
                start_time=st,
                limit=limit,
            )

        elif mode == "time_range":
            user_timezone = runtime_timezone(runtime)
            if not start_time and not end_time:
                return "Error: time_range mode requires at least one of start_time or end_time."
            resp = await _search_screenpipe(
                user_id=user_id,
                q=query or None,
                content_type=content_type,
                start_time=localize_to_utc(start_time, user_timezone) if start_time else None,
                end_time=localize_to_utc(end_time, user_timezone) if end_time else None,
                app_name=app_name or None,
                limit=limit,
            )

        else:  # history — full-text search
            if not query:
                return "Error: history mode requires a query string."
            resp = await _search_screenpipe(
                user_id=user_id,
                q=query,
                content_type=content_type,
                limit=limit,
            )

        if "error" in resp:
            return f"Screenpipe error: {resp['error']}"

        items = resp.get("data", [])
        pagination = resp.get("pagination", {})

        if not items:
            return "No results found."

        lines = [_format_content_item(item) for item in items]
        lines.append("---")
        lines.append(_format_pagination(pagination))

        return "\n".join(lines)

    except Exception as e:
        return f"Error querying screen data: {e}"


@tool
async def search_screen_time(
    runtime: ToolRuntime,
    start_time: str,
    end_time: str = "",
    query: str = "",
    content_type: Literal["all", "ocr", "audio", "input", "ui"] = "all",
    app_name: str = "",
    limit: int = 40,
) -> str:
    """Search the user's screen/audio history for a SPECIFIC TIME RANGE.

    Use this when the user asks about a particular date or time window:
    - "What was I doing yesterday afternoon?"
    - "Show me my screen between 2pm and 4pm on May 3rd"
    - "What did I read about on Monday morning?"

    Parameters:
    - start_time: REQUIRED. ISO 8601 time string for the start of the window.
      Example: '2026-05-06T14:00:00+08:00' or '2026-05-06T06:00:00Z'.
    - end_time: Optional. ISO 8601 time string for the end of the window.
      If omitted, searches from start_time to 'now'.
    - query: Optional keyword to filter results by text content.
    - content_type: Filter by data type ('ocr', 'audio', 'input', 'ui', or 'all').
    - app_name: Optional. Filter results to a specific application (e.g., 'Chrome', 'VS Code').
    - limit: Max results to return (default 40, max 100).
    """
    try:
        limit = min(limit, 100)
        user_id = require_runtime_user_id(runtime)
        user_timezone = runtime_timezone(runtime)

        resp = await _search_screenpipe(
            user_id=user_id,
            q=query or None,
            content_type=content_type,
            start_time=localize_to_utc(start_time, user_timezone),
            end_time=localize_to_utc(end_time, user_timezone) if end_time else None,
            app_name=app_name or None,
            limit=limit,
        )

        if "error" in resp:
            return f"Screenpipe error: {resp['error']}"

        items = resp.get("data", [])
        pagination = resp.get("pagination", {})

        if not items:
            time_desc = f"between {start_time} and {end_time or 'now'}"
            return f"No screen/audio data found {time_desc}."

        lines = [f"Results for time range: {start_time} → {end_time or 'now'}"]
        if query:
            lines.append(f"Filtered by query: '{query}'")
        lines.append("---")

        for item in items:
            lines.append(_format_content_item(item))

        lines.append("---")
        lines.append(_format_pagination(pagination))

        return "\n".join(lines)

    except Exception as e:
        return f"Error querying screen data by time: {e}"


screen_tools = [view_screen, search_screen_time]
