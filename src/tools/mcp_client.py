"""MCP Client Manager - Connects to external MCP servers and imports tools.

Uses langchain_mcp_adapters.MultiServerMCPClient to establish connections
and convert MCP tools to LangChain-compatible BaseTool instances.
"""

import json
import logging
from typing import Optional

from langchain_core.tools import BaseTool

from core.config import config

logger = logging.getLogger("mcp-client")

_mcp_client: Optional["MultiServerMCPClient"] = None
_mcp_tools: list[BaseTool] = []
_mcp_initialized: bool = False


def _parse_connections() -> dict | None:
    """Parse MCP_SERVERS_CONFIG from env. Returns None if empty or invalid."""
    raw = config.MCP_SERVERS_CONFIG
    logger.debug(
        "[mcp-client] _parse_connections | config_len=%d | raw_preview=%.100s",
        len(raw), raw if raw else "<empty>",
    )
    if not raw:
        logger.debug("[mcp-client] _parse_connections -> None (empty config)")
        return None
    try:
        parsed = json.loads(raw)
        logger.debug(
            "[mcp-client] _parse_connections OK | servers=%d | keys=%s",
            len(parsed), list(parsed.keys()),
        )
        return parsed
    except json.JSONDecodeError as e:
        logger.error(
            "[mcp-client] _parse_connections FAILED | err=%s | raw=%.200s",
            e, raw,
        )
        return None


async def init_mcp_client() -> list[BaseTool]:
    """Initialize MCP client connections and load tools. Idempotent.

    Returns:
        List of LangChain BaseTool instances from all configured MCP servers.
        Empty list if no servers configured or all connections fail.
    """
    global _mcp_client, _mcp_tools, _mcp_initialized

    logger.debug(
        "[mcp-client] init_mcp_client ENTER | initialized=%s | cached_tools=%d",
        _mcp_initialized, len(_mcp_tools),
    )

    if _mcp_initialized:
        logger.debug(
            "[mcp-client] init_mcp_client SKIP (already initialized) | tools=%d",
            len(_mcp_tools),
        )
        return _mcp_tools

    connections = _parse_connections()
    if not connections:
        logger.info("[mcp-client] No MCP server connections configured.")
        _mcp_initialized = True
        return []

    logger.debug(
        "[mcp-client] Connecting to %d MCP server(s): %s",
        len(connections), list(connections.keys()),
    )

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        for name, cfg in connections.items():
            transport = cfg.get("transport", "?")
            logger.debug(
                "[mcp-client] Creating connection | server=%s | transport=%s",
                name, transport,
            )

        _mcp_client = MultiServerMCPClient(
            connections=connections,
            tool_name_prefix=True,
        )

        logger.debug("[mcp-client] Calling get_tools()...")
        _mcp_tools = await _mcp_client.get_tools()

        logger.info(
            "[mcp-client] Loaded %d MCP tools from %d server(s): %s",
            len(_mcp_tools), len(connections),
            [t.name for t in _mcp_tools],
        )
        for tool in _mcp_tools:
            logger.debug(
                "[mcp-client] MCP tool | name=%s | desc=%.80s",
                tool.name, getattr(tool, "description", "?") or "?",
            )
    except Exception as e:
        logger.error(
            "[mcp-client] init_mcp_client FAILED | err=%s | type=%s",
            e, type(e).__name__,
        )
        _mcp_tools = []

    _mcp_initialized = True
    logger.debug(
        "[mcp-client] init_mcp_client EXIT | total_tools=%d",
        len(_mcp_tools),
    )
    return _mcp_tools


def get_mcp_tools() -> list[BaseTool]:
    """Return cached MCP tools (empty if not yet initialized)."""
    logger.debug(
        "[mcp-client] get_mcp_tools | count=%d | initialized=%s",
        len(_mcp_tools), _mcp_initialized,
    )
    return _mcp_tools
