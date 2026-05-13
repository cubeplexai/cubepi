"""cubepi MCP tool loaders.

cubepi[mcp] extra required.
"""

from cubepi.mcp.http_loader import load_mcp_tools_http
from cubepi.mcp.stdio_loader import load_mcp_tools_stdio

__all__ = ["load_mcp_tools_http", "load_mcp_tools_stdio"]
