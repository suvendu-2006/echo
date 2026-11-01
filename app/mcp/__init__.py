"""Echo's MCP package (Phase 4) — the hands: external servers as tool sources.

Stage modules:
- ``manager.py``: the stdio MCP client (JSON-RPC 2.0 over subprocess pipes),
  the :class:`McpManager` that merges server tools into the harness tool
  desk, and the tolerant ``data/mcp_servers.json`` loader.
- Later phases plug additional MCP-powered features in here (Gmail OAuth +
  efficiency helpers live in their own stage modules).

Safety posture: every MCP tool registers as ``Danger.CONFIRM`` unless a
config entry explicitly and validly says ``"danger": "safe"`` — the
permission engine (H4) gates everything, matching rules like
``mcp__server__tool(*)``.
"""

from app.mcp.manager import (
    CLIENT_INFO,
    DEFAULT_SERVERS_PATH,
    PROTOCOL_VERSION,
    REQUEST_TIMEOUT_SECONDS,
    START_TIMEOUT_SECONDS,
    McpError,
    McpManager,
    McpServerConnection,
    McpToolError,
    McpTransport,
    StdioTransport,
    load_server_configs,
)

__all__ = [
    "CLIENT_INFO",
    "DEFAULT_SERVERS_PATH",
    "PROTOCOL_VERSION",
    "REQUEST_TIMEOUT_SECONDS",
    "START_TIMEOUT_SECONDS",
    "McpError",
    "McpManager",
    "McpServerConnection",
    "McpToolError",
    "McpTransport",
    "StdioTransport",
    "load_server_configs",
]
