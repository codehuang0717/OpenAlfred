from tools.memory import memTools
from tools.todos import todo_tools
from tools.reminder import reminder_tools
from tools.call_user import call_tools
from tools.eye import screen_tools
from tools.email_tools import email_tools
from tools.search import search_tools
from tools.screenshot import screenshot_tools

import asyncio
import logging

_import_logger = logging.getLogger("tools-init")

# Static built-in tools (always available, no async init required)
_BUILTIN_TOOLS: list = (
    todo_tools +
    reminder_tools +
    call_tools +
    memTools +
    email_tools +
    screen_tools +
    search_tools +
    screenshot_tools
)

_import_logger.debug(
    "[tools-init] Built-in tools loaded | count=%d | names=%s",
    len(_BUILTIN_TOOLS),
    [t.name for t in _BUILTIN_TOOLS],
)

# Mutable list: starts with built-in tools, MCP tools appended at startup
ALL_TOOLS = list(_BUILTIN_TOOLS)


async def ensure_tools_loaded():
    """Load MCP tools into ALL_TOOLS. Call once during app startup.

    Safe to call multiple times — subsequent calls are no-ops once MCP
    tools have been loaded.
    """
    from tools.mcp_client import init_mcp_client

    # Check if MCP tools already loaded (beyond built-in count)
    if len(ALL_TOOLS) > len(_BUILTIN_TOOLS):
        _import_logger.debug(
            "[tools-init] ensure_tools_loaded SKIP (already loaded) | total=%d",
            len(ALL_TOOLS),
        )
        return

    _import_logger.debug("[tools-init] ensure_tools_loaded ENTER (async path)")
    try:
        mcp_tools = await init_mcp_client()
        if mcp_tools:
            ALL_TOOLS.extend(mcp_tools)
            _import_logger.info(
                "[tools-init] MCP tools appended to ALL_TOOLS | "
                "mcp_count=%d | total_count=%d | mcp_names=%s",
                len(mcp_tools), len(ALL_TOOLS),
                [t.name for t in mcp_tools],
            )
        else:
            _import_logger.debug(
                "[tools-init] No MCP tools loaded (async) | total_count=%d",
                len(ALL_TOOLS),
            )
    except Exception as e:
        _import_logger.warning(
            "[tools-init] MCP tool loading failed (async) | err=%s | type=%s",
            e, type(e).__name__,
        )


def _load_mcp_tools_sync():
    """Best-effort synchronous MCP tool loading at import time.

    Only runs when NO event loop is already running (e.g., direct Python
    import, not under uvicorn/langgraph dev). When a loop IS running, MCP
    tools are loaded via ensure_tools_loaded() during app startup.
    """
    try:
        asyncio.get_running_loop()
        # A loop is already running — cannot create a new one.
        # defer to ensure_tools_loaded() called from lifespan.
        _import_logger.info(
            "[tools-init] Event loop already running, "
            "MCP tools deferred to async startup (ensure_tools_loaded)"
        )
        return
    except RuntimeError:
        pass  # No running loop, safe to create one

    _import_logger.debug("[tools-init] _load_mcp_tools_sync ENTER (sync path)")
    try:
        from tools.mcp_client import init_mcp_client

        _import_logger.debug("[tools-init] Creating fresh event loop for MCP init...")
        loop = asyncio.new_event_loop()
        try:
            mcp_tools = loop.run_until_complete(init_mcp_client())
            if mcp_tools:
                ALL_TOOLS.extend(mcp_tools)
                _import_logger.info(
                    "[tools-init] MCP tools appended to ALL_TOOLS | "
                    "mcp_count=%d | total_count=%d | mcp_names=%s",
                    len(mcp_tools), len(ALL_TOOLS),
                    [t.name for t in mcp_tools],
                )
            else:
                _import_logger.debug(
                    "[tools-init] No MCP tools loaded (sync) | total_count=%d",
                    len(ALL_TOOLS),
                )
        finally:
            loop.close()
            _import_logger.debug("[tools-init] Event loop closed")
    except Exception as e:
        _import_logger.warning(
            "[tools-init] MCP tool loading skipped (sync) | err=%s | type=%s",
            e, type(e).__name__,
        )


# Load MCP tools synchronously at import time if possible (no running loop).
# When a loop IS running (langgraph dev / uvicorn), MCP tools are loaded
# via ensure_tools_loaded() called from the app lifespan.
_load_mcp_tools_sync()

_import_logger.debug(
    "[tools-init] Module init complete | final_tool_count=%d",
    len(ALL_TOOLS),
)
