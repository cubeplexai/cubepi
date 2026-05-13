"""Minimal stdio MCP server for testing.

Uses the mcp SDK's server primitives. Advertises one 'echo' tool.

Run as: python -m tests.mcp._fake_stdio_server
"""

import asyncio


async def main() -> None:
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    server = Server("fake-cubepi-test-server")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name="echo",
                description="Echo the input back",
                inputSchema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "echo":
            return [TextContent(type="text", text=arguments.get("text", ""))]
        raise ValueError(f"unknown tool: {name}")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="fake-cubepi-test-server",
                server_version="0.0.1",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
