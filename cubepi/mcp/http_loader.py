"""HTTP/SSE transport MCP tool loader."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, Literal

from cubepi.agent.types import AgentTool
from cubepi.mcp._adapter import make_mcp_agent_tool

Transport = Literal["sse", "streamable_http"]


@asynccontextmanager
async def _open_session(
    server_url: str,
    *,
    headers: dict[str, str] | None,
    timeout: float,
    transport: Transport,
) -> AsyncIterator[Any]:
    """Open an MCP ClientSession over the requested transport.

    Normalises the two SDK transport client signatures: ``sse_client``
    yields a 2-tuple ``(read, write)`` while ``streamablehttp_client``
    yields a 3-tuple ``(read, write, get_session_id_callable)``. We drop
    the session-id callable and expose a single ``ClientSession`` to the
    caller.
    """
    from mcp import ClientSession

    if transport == "streamable_http":
        from mcp.client.streamable_http import streamablehttp_client

        # streamablehttp_client's timeout signature drifted across mcp SDK
        # versions: 1.8/1.9-era releases required ``timedelta`` and called
        # ``.total_seconds()`` internally, while ~1.10+ accepts ``float |
        # timedelta``. Passing a ``timedelta`` works on every version we
        # declare in our ``mcp>=1.0`` floor, so we always convert here.
        timeout_td = timedelta(seconds=timeout)
        async with streamablehttp_client(
            server_url,
            headers=headers,
            timeout=timeout_td,
            sse_read_timeout=timeout_td,
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(read_stream, write_stream) as session:
                yield session
        return

    if transport == "sse":
        from mcp.client.sse import sse_client

        async with sse_client(
            server_url,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=timeout,
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                yield session
        return

    raise ValueError(f"unsupported MCP transport '{transport}'")


async def load_mcp_tools_http(
    server_url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    transport: Transport = "sse",
) -> list[AgentTool]:
    """Connect to an HTTP MCP server, discover tools, return AgentTools.

    Uses the `mcp` SDK's HTTP client. ``transport`` picks the wire format:

    - ``"sse"`` (default) — legacy SSE-over-GET transport (``sse_client``).
    - ``"streamable_http"`` — newer SSE-over-POST transport
      (``streamablehttp_client``).

    Each returned tool's execute method invokes ``tools/call`` against a
    fresh session — v1 simplicity, no pooling. The session is opened over
    the same transport that was used for discovery, so the call path
    cannot accidentally regress to the wrong wire format.

    The transport's own timeout bounds the connection; we additionally
    wrap initialize/list/call awaits in ``asyncio.wait_for`` so a server
    that accepts the connection but stalls on protocol messages still
    aborts.
    """

    async def _call_remote(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        async with _open_session(
            server_url, headers=headers, timeout=timeout, transport=transport
        ) as session:
            await asyncio.wait_for(session.initialize(), timeout=timeout)
            resp = await asyncio.wait_for(
                session.call_tool(tool_name, args), timeout=timeout
            )
            return _serialize_call_tool_response(resp)

    async with _open_session(
        server_url, headers=headers, timeout=timeout, transport=transport
    ) as session:
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
