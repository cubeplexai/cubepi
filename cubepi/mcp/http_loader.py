"""HTTP/SSE transport MCP tool loader."""
from __future__ import annotations

from typing import Any

from cubepi.agent.types import AgentTool
from cubepi.mcp._adapter import make_mcp_agent_tool


async def load_mcp_tools_http(
    server_url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> list[AgentTool]:
    """Connect to an HTTP/SSE MCP server, discover tools, return AgentTools.

    Uses the `mcp` SDK's HTTP client. Each returned tool's execute method
    invokes tools/call against a fresh session — v1 simplicity, no pooling.
    """
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async def _call_remote(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        async with sse_client(server_url, headers=headers, timeout=timeout) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                resp = await session.call_tool(tool_name, args)
                return _serialize_call_tool_response(resp)

    async with sse_client(server_url, headers=headers, timeout=timeout) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
            tool_descs = tools_resp.tools

    return [
        make_mcp_agent_tool(
            name=desc.name,
            description=desc.description or "",
            input_schema=desc.inputSchema or {"type": "object", "properties": {}},
            call_remote=_call_remote,
        )
        for desc in tool_descs
    ]


def _serialize_call_tool_response(resp: Any) -> dict[str, Any]:
    """Normalize mcp SDK CallToolResult → dict for adapter."""
    content = []
    for c in (resp.content or []):
        if getattr(c, "type", None) == "text":
            content.append({"type": "text", "text": c.text})
    return {
        "content": content,
        "isError": bool(getattr(resp, "isError", False)),
    }
