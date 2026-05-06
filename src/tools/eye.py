import httpx
from utils.logger import get_logger
from typing import Optional, List, Dict, Literal
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from core.config import config

logger = get_logger("eye-tool")

from rich.console import Console
from rich.panel import Panel

console = Console()

# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

async def _search_screenpipe(
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
        async with httpx.AsyncClient() as client:
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


def _local_to_utc(time_str: str) -> str:
    """Convert a naive local-time string (per agent prompt convention) to
    an ISO 8601 UTC string that the Screenpipe API expects.

    Handles three input forms:
    - Naive: ``2026-05-06T14:00:00`` → interpreted as config.TIMEZONE, output UTC
    - With offset: ``2026-05-06T14:00:00+01:00`` → converted to UTC
    - Already UTC: ``2026-05-06T14:00:00Z`` → returned as-is
    """
    if not time_str:
        return time_str

    tz = ZoneInfo(config.TIMEZONE)

    # Already UTC
    if time_str.endswith("Z"):
        return time_str

    # Has explicit offset
    if "+" in time_str[10:] or time_str.count("-") > 1:
        try:
            dt = datetime.fromisoformat(time_str)
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass

    # Naive — interpret as local timezone
    try:
        dt_naive = datetime.fromisoformat(time_str)
        dt_local = dt_naive.replace(tzinfo=tz)
        return dt_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return time_str


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

async def get_enhanced_context(minutes: int = 10) -> str:
    """Fetch OCR, Audio, and UI context from Screenpipe for a comprehensive user activity view."""
    try:
        now_utc = datetime.now(timezone.utc)
        start_time_dt = now_utc - timedelta(minutes=minutes)
        start_time_str = start_time_dt.isoformat()

        resp = await _search_screenpipe(
            content_type="all",
            start_time=start_time_str,
            limit=60,
        )
        if "error" in resp:
            return resp["error"]

        items = resp.get("data", [])

        # Process items into structured context
        ocr_parts = []
        audio_parts = []
        input_events = 0
        apps_active = set()
        urls_active = set()

        for item in items:
            content_type = item.get("type", "").upper()
            content = item.get("content", {})

            if content_type == "OCR":
                text = content.get("text", "").strip()
                app = content.get("app_name", "Unknown")
                window = content.get("window_name", "Unknown")
                url = content.get("browser_url", "")

                if text:
                    ocr_parts.append(f"[{app} | {window}] {text[:200]}")
                    apps_active.add(app)
                    if url:
                        urls_active.add(url)

            elif content_type == "AUDIO":
                text = content.get("transcription", "").strip()
                if text:
                    audio_parts.append(text)

            elif content_type == "INPUT":
                input_events += 1

        # Activity Summary (optional)
        activity_summary = ""
        try:
            async with httpx.AsyncClient() as client:
                summary_resp = await client.get(
                    f"{config.SCREENPIPE_URL}/activity/get-activity-summary",
                    timeout=5.0,
                )
                if summary_resp.status_code == 200:
                    activity_data = summary_resp.json()
                    if isinstance(activity_data, list):
                        summary_items = [
                            f"{a.get('app_name')}: {a.get('duration')}s"
                            for a in activity_data[:5]
                        ]
                        activity_summary = "Recent Apps: " + ", ".join(summary_items)
        except Exception:
            pass

        # Construct Final Context String
        context_blocks = []

        if apps_active:
            context_blocks.append(f"Active Apps: {', '.join(apps_active)}")

        if urls_active:
            context_blocks.append(f"Active URLs: {', '.join(urls_active)}")

        context_blocks.append(
            f"Physical Activity: {'Active' if input_events > 0 else 'Idle'} "
            f"({input_events} input events detected)"
        )

        if activity_summary:
            context_blocks.append(f"Activity Summary: {activity_summary}")

        if ocr_parts:
            unique_ocr = list(dict.fromkeys(ocr_parts))[:15]
            context_blocks.append(
                "--- Screen Content (OCR) ---\n" + "\n".join(unique_ocr)
            )

        if audio_parts:
            unique_audio = list(dict.fromkeys(audio_parts))[:10]
            context_blocks.append(
                "--- Audio Transcripts (Meetings/Speech) ---\n"
                + "\n".join(unique_audio)
            )

        final_context = "\n\n".join(context_blocks)

        console.print(
            f"[cyan][eye-tool] Fetched enhanced context: "
            f"{len(ocr_parts)} OCR items, {len(audio_parts)} Audio items.[/cyan]"
        )

        return final_context if final_context else "No activity detected."

    except Exception as e:
        logger.error(f"Error connecting to Screenpipe: {e}")
        return f"Error connecting to Screenpipe: {str(e)}"


async def get_recent_ocr_text(minutes: int = 10) -> str:
    """Legacy wrapper for backward compatibility."""
    return await get_enhanced_context(minutes)


# ──────────────────────────────────────────────
# LangChain tool definitions
# ──────────────────────────────────────────────

from langchain.tools import tool


@tool
async def view_screen(
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
        if mode == "current":
            now_utc = datetime.now(timezone.utc)
            st = (now_utc - timedelta(minutes=5)).isoformat()
            resp = await _search_screenpipe(
                content_type=content_type,
                start_time=st,
                limit=limit,
            )

        elif mode == "time_range":
            if not start_time and not end_time:
                return "Error: time_range mode requires at least one of start_time or end_time."
            resp = await _search_screenpipe(
                q=query or None,
                content_type=content_type,
                start_time=_local_to_utc(start_time) if start_time else None,
                end_time=_local_to_utc(end_time) if end_time else None,
                app_name=app_name or None,
                limit=limit,
            )

        else:  # history — full-text search
            if not query:
                return "Error: history mode requires a query string."
            resp = await _search_screenpipe(
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

        resp = await _search_screenpipe(
            q=query or None,
            content_type=content_type,
            start_time=_local_to_utc(start_time),
            end_time=_local_to_utc(end_time) if end_time else None,
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
