"""HTTP/SSE transport MCP tool loader."""

from __future__ import annotations

import asyncio
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

    The transport's own timeout bounds the SSE connection; we additionally
    wrap initialize/list/call awaits in asyncio.wait_for so a server that
    accepts the connection but stalls on protocol messages still aborts.
    """
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async def _call_remote(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        async with sse_client(server_url, headers=headers, timeout=timeout) as streams:
            async with ClientSession(*streams) as session:
                await asyncio.wait_for(session.initialize(), timeout=timeout)
                resp = await asyncio.wait_for(
                    session.call_tool(tool_name, args), timeout=timeout
                )
                return _serialize_call_tool_response(resp)

    async with sse_client(server_url, headers=headers, timeout=timeout) as streams:
        async with ClientSession(*streams) as session:
            await asyncio.wait_for(session.initialize(), timeout=timeout)
            tools_resp = await asyncio.wait_for(session.list_tools(), timeout=timeout)
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
    """Normalize mcp SDK CallToolResult → dict for adapter.

    Preserves text and image content blocks plus the optional
    ``structuredContent`` field. Unknown block types are dropped (after
    being surfaced once the agent loop has a place to put them).
    """
    content: list[dict[str, Any]] = []
    for c in resp.content or []:
        ctype = getattr(c, "type", None)
        if ctype == "text":
            content.append({"type": "text", "text": c.text})
        elif ctype == "image":
            content.append(
                {
                    "type": "image",
                    "data": getattr(c, "data", ""),
                    "mimeType": getattr(c, "mimeType", "")
                    or getattr(c, "media_type", ""),
                }
            )
    out: dict[str, Any] = {
        "content": content,
        "isError": bool(getattr(resp, "isError", False)),
    }
    structured = getattr(resp, "structuredContent", None)
    if structured is not None:
        out["structuredContent"] = structured
    return out
