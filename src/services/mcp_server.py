"""MCP Server for OpenAlfred - Exposes agent tools via Model Context Protocol.

Uses FastMCP to serve tools over SSE, streamable-HTTP, or stdio transport.
Other AI applications (Claude Desktop, Continue, etc.) can connect to this
server to use OpenAlfred's todo, reminder, memory, screen, and search tools.

Usage:
    python -m services.mcp_server              # SSE on 0.0.0.0:8100
    MCP_SERVER_TRANSPORT=stdio python -m services.mcp_server  # stdio mode
"""

import logging

from core.config import config

logger = logging.getLogger("mcp-server")


def _warn(msg: str):
    logger.warning("[mcp-server] %s", msg)
    print(f"[mcp-server] {msg}")


def _info(msg: str):
    logger.info("[mcp-server] %s", msg)
    print(f"[mcp-server] {msg}")


def create_mcp_server():
    """Create a FastMCP server with all built-in OpenAlfred tools registered.

    Returns:
        FastMCP instance with tools registered, or None if dependencies
        are missing.
    """
    logger.debug(
        "[mcp-server] create_mcp_server ENTER | name=%s | transport=%s",
        config.MCP_SERVER_NAME, config.MCP_SERVER_TRANSPORT,
    )

    # ── Check mcp package ──
    try:
        from mcp.server.fastmcp import FastMCP
        logger.debug("[mcp-server] FastMCP imported OK")
    except ImportError:
        _warn("mcp package not installed. Install with: pip install mcp")
        return None

    # ── Check langchain_mcp_adapters ──
    try:
        from langchain_mcp_adapters.tools import to_fastmcp
        logger.debug("[mcp-server] to_fastmcp imported OK")
    except ImportError:
        _warn(
            "langchain_mcp_adapters.tools.to_fastmcp not available. "
            "Update langchain-mcp-adapters to >= 0.2.1"
        )
        return None

    from tools import _BUILTIN_TOOLS

    logger.debug(
        "[mcp-server] Converting %d built-in tools to MCP format...",
        len(_BUILTIN_TOOLS),
    )

    mcp = FastMCP(
        name=config.MCP_SERVER_NAME,
        instructions=(
            "OpenAlfred AI Agent — tools for todos, reminders, user memory, "
            "screen capture, web search, browser automation, email, and "
            "outbound phone calls."
        ),
    )

    registered = 0
    skipped = 0
    for lc_tool in _BUILTIN_TOOLS:
        tool_name = lc_tool.name
        logger.debug("[mcp-server] Converting tool: %s", tool_name)
        try:
            fastmcp_tool = to_fastmcp(lc_tool)
            setattr(mcp, tool_name, fastmcp_tool)
            registered += 1
            logger.debug("[mcp-server]   -> REGISTERED: %s", tool_name)
        except Exception as e:
            skipped += 1
            logger.warning(
                "[mcp-server]   -> SKIPPED: %s | reason=%s",
                tool_name, e,
            )

    msg = (
        f"MCP Server '{config.MCP_SERVER_NAME}' ready: "
        f"{registered} tools registered, {skipped} skipped "
        f"(total built-in: {len(_BUILTIN_TOOLS)})"
    )
    logger.info("[mcp-server] %s", msg)
    print(f"[mcp-server] {msg}")
    return mcp


def run_mcp_server():
    """Run the MCP server (blocking). Transport is determined by config."""
    logger.debug("[mcp-server] run_mcp_server ENTER")
    mcp = create_mcp_server()
    if mcp is None:
        logger.error("[mcp-server] run_mcp_server ABORT (server creation failed)")
        return

    transport = config.MCP_SERVER_TRANSPORT
    host = config.MCP_SERVER_HOST
    port = config.MCP_SERVER_PORT
    logger.info(
        "[mcp-server] Starting | transport=%s | host=%s | port=%d",
        transport, host, port,
    )
    print(f"[mcp-server] Starting on transport={transport} host={host} port={port} ...")

    if transport == "stdio":
        logger.debug("[mcp-server] Running in stdio mode")
        mcp.run(transport="stdio")
    elif transport == "streamable-http":
        logger.debug(
            "[mcp-server] Running in streamable-http mode | %s:%d", host, port
        )
        mcp.run(
            transport="streamable-http",
            host=host,
            port=port,
        )
    else:
        # Default: SSE
        logger.debug("[mcp-server] Running in SSE mode | %s:%d", host, port)
        mcp.run(
            transport="sse",
            host=host,
            port=port,
        )


if __name__ == "__main__":
    import sys

    # Allow --help
    if "--help" in sys.argv or "-h" in sys.argv:
        print("OpenAlfred MCP Server")
        print()
        print("Usage: python -m services.mcp_server")
        print()
        print("Environment variables:")
        print("  MCP_SERVER_ENABLED    Enable MCP server (default: false)")
        print("  MCP_SERVER_TRANSPORT  stdio | sse | streamable-http")
        print("  MCP_SERVER_HOST       Bind host (default: 0.0.0.0)")
        print("  MCP_SERVER_PORT       Bind port (default: 8100)")
        print("  MCP_SERVER_NAME       Server name (default: OpenAlfred)")
        sys.exit(0)

    run_mcp_server()
