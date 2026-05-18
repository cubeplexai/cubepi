"""stdio transport MCP tool loader."""

from __future__ import annotations

import asyncio
from typing import Any

from cubepi.mcp._adapter import make_mcp_agent_tool
from cubepi.mcp.types import (
    MCPDiscoveryResult,
    server_info_from_init_result,
    tool_info_from_desc,
)


async def load_mcp_tools_stdio(
    command: str,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: float = 30.0,
) -> MCPDiscoveryResult:
    """Spawn a stdio MCP server subprocess and discover its tools + metadata.

    Returns a :class:`MCPDiscoveryResult` with the same shape as
    :func:`load_mcp_tools_http`: ``tools`` (executable AgentTools),
    ``server`` (Implementation info from ``initialize``), and ``tool_infos``
    (per-tool display metadata such as icons).

    Each returned tool's execute opens a fresh subprocess per call (v1
    simplicity, no process pooling).

    Args:
        command: executable to run (e.g. "npx" or sys.executable)
        args: argv for the server process
        env: environment variables (passed to subprocess)
        cwd: working directory for the subprocess
        timeout: per-call wall-clock timeout for initialize/list/call awaits.
            A hung server raises asyncio.TimeoutError instead of blocking forever.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env,
        cwd=cwd,
    )

    async def _call_remote(tool_name: str, args_dict: dict[str, Any]) -> dict[str, Any]:
        async with stdio_client(server_params) as streams:
            async with ClientSession(*streams) as session:
                await asyncio.wait_for(session.initialize(), timeout=timeout)
                resp = await asyncio.wait_for(
                    session.call_tool(tool_name, args_dict), timeout=timeout
                )
                return _serialize_call_tool_response(resp)

    async with stdio_client(server_params) as streams:
        async with ClientSession(*streams) as session:
            init_result = await asyncio.wait_for(session.initialize(), timeout=timeout)
            tools_resp = await asyncio.wait_for(session.list_tools(), timeout=timeout)
            tool_descs = tools_resp.tools

    # stdio has no network address; only protocol version is observable.
    protocol_version = getattr(init_result, "protocolVersion", None)
    if not isinstance(protocol_version, str):
        protocol_version = None

    tools = [
        make_mcp_agent_tool(
            name=desc.name,
            description=desc.description or "",
            input_schema=desc.inputSchema or {"type": "object", "properties": {}},
            call_remote=_call_remote,
            protocol_version=protocol_version,
        )
        for desc in tool_descs
    ]
    tool_infos = [tool_info_from_desc(desc) for desc in tool_descs]
    return MCPDiscoveryResult(
        tools=tools,
        server=server_info_from_init_result(init_result),
        tool_infos=tool_infos,
    )


def _serialize_call_tool_response(resp: Any) -> dict[str, Any]:
    """Normalize mcp SDK CallToolResult → dict for adapter.

    Mirrors http_loader._serialize_call_tool_response so both transports
    feed the adapter identically.
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
