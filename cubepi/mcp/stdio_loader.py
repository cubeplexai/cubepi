"""stdio transport MCP tool loader."""

from __future__ import annotations

from typing import Any

from cubepi.agent.types import AgentTool
from cubepi.mcp._adapter import make_mcp_agent_tool


async def load_mcp_tools_stdio(
    command: str,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: float = 30.0,
) -> list[AgentTool]:
    """Spawn a stdio MCP server subprocess, discover tools, return AgentTools.

    Each returned tool's execute opens a fresh subprocess per call (v1
    simplicity, no process pooling).

    Args:
        command: executable to run (e.g. "npx" or sys.executable)
        args: argv for the server process
        env: environment variables (passed to subprocess)
        cwd: working directory for the subprocess
        timeout: per-call timeout (not currently enforced strictly)
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
                await session.initialize()
                resp = await session.call_tool(tool_name, args_dict)
                return _serialize_call_tool_response(resp)

    async with stdio_client(server_params) as streams:
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
    content = []
    for c in resp.content or []:
        if getattr(c, "type", None) == "text":
            content.append({"type": "text", "text": c.text})
    return {
        "content": content,
        "isError": bool(getattr(resp, "isError", False)),
    }
