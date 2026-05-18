"""stdio MCP loader integration tests (D2.3)."""

import sys

import pytest


def test_import_stdio_loader() -> None:
    from cubepi.mcp import load_mcp_tools_stdio

    assert callable(load_mcp_tools_stdio)


@pytest.mark.asyncio
async def test_stdio_loader_against_fake_server() -> None:
    """Spawn the fake stdio server, list tools, invoke 'echo'."""
    from cubepi.mcp import MCPDiscoveryResult, load_mcp_tools_stdio

    discovery = await load_mcp_tools_stdio(
        command=sys.executable,
        args=["-m", "tests.mcp._fake_stdio_server"],
    )
    assert isinstance(discovery, MCPDiscoveryResult)
    assert len(discovery.tools) == 1
    echo = discovery.tools[0]
    assert echo.name == "echo"

    # Server info is captured from the initialize handshake; the fake
    # server reports its name + version via Implementation.
    assert discovery.server is not None
    assert discovery.server.name

    args = echo.parameters(text="hello")
    result = await echo.execute("tc-stdio-1", args, signal=None, on_update=None)
    assert len(result.content) == 1
    from cubepi.providers.base import TextContent

    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "hello"
