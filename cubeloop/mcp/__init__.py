"""cubepi MCP tool loaders.

cubepi[mcp] extra required.
"""

from cubeloop.mcp.http_loader import load_mcp_tools_http
from cubeloop.mcp.stdio_loader import load_mcp_tools_stdio
from cubeloop.mcp.types import (
    MCPDiscoveryResult,
    MCPIcon,
    MCPServerInfo,
    MCPToolInfo,
)

__all__ = [
    "MCPDiscoveryResult",
    "MCPIcon",
    "MCPServerInfo",
    "MCPToolInfo",
    "load_mcp_tools_http",
    "load_mcp_tools_stdio",
]
