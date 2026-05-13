"""HTTP MCP loader tests (D2.2).

Full E2E requires a running MCP test server. We provide a smoke import
test always, and a skip-unless-env E2E hook (CUBEPI_TEST_MCP_HTTP_URL).
"""

import os
import pytest


def test_import_http_loader() -> None:
    """Loader function is importable from the public module path."""
    from cubepi.mcp import load_mcp_tools_http

    assert callable(load_mcp_tools_http)


@pytest.mark.asyncio
async def test_load_mcp_tools_http_against_test_server() -> None:
    """End-to-end: connect to a real MCP test server, list + call a tool."""
    server_url = os.environ.get("CUBEPI_TEST_MCP_HTTP_URL")
    if not server_url:
        pytest.skip("Set CUBEPI_TEST_MCP_HTTP_URL to run this test")

    from cubepi.mcp import load_mcp_tools_http

    tools = await load_mcp_tools_http(server_url)
    assert len(tools) > 0
    first = tools[0]
    assert first.name
    assert first.description is not None
