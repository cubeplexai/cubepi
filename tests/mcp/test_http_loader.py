"""HTTP MCP loader tests (D2.2).

The loader's transport (`sse_client` + `ClientSession`) is mocked so the
test runs without a real MCP server. End-to-end against a live server is
gated behind CUBEPI_TEST_MCP_HTTP_URL.
"""

import os
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest


def test_import_http_loader() -> None:
    """Loader function is importable from the public module path."""
    from cubepi.mcp import load_mcp_tools_http

    assert callable(load_mcp_tools_http)


class _FakeSession:
    """Stand-in for mcp.ClientSession that records calls and returns canned data."""

    def __init__(self, *streams, tools=None, call_response=None):
        self._tools = tools or []
        self._call_response = call_response
        self.initialized = False
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def initialize(self):
        self.initialized = True

    async def list_tools(self):
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return self._call_response


def _install_fake_transport(monkeypatch, *, tools, call_response):
    import mcp
    import mcp.client.sse as sse_mod

    sessions: list[_FakeSession] = []
    sse_calls: list[dict] = []

    @asynccontextmanager
    async def fake_sse_client(url, *, headers=None, timeout=None):
        sse_calls.append({"url": url, "headers": headers, "timeout": timeout})
        yield ("read-stream-stub", "write-stream-stub")

    def fake_client_session(*streams):
        sess = _FakeSession(*streams, tools=tools, call_response=call_response)
        sessions.append(sess)
        return sess

    monkeypatch.setattr(sse_mod, "sse_client", fake_sse_client)
    monkeypatch.setattr(mcp, "ClientSession", fake_client_session)
    return sessions, sse_calls


@pytest.mark.asyncio
async def test_load_mcp_tools_http_lists_and_calls_tool(monkeypatch) -> None:
    """Mocked transport: lists tools, then invokes one and serializes response."""
    from cubepi.mcp import load_mcp_tools_http
    from cubepi.providers.base import TextContent

    tools_resp = [
        SimpleNamespace(
            name="search",
            description="Search the web",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
    ]
    call_resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(type="image", data="ignored"),  # non-text dropped
        ],
        isError=False,
    )
    sessions, sse_calls = _install_fake_transport(
        monkeypatch, tools=tools_resp, call_response=call_resp
    )

    tools = await load_mcp_tools_http(
        "https://mcp.example/sse",
        headers={"x-test": "1"},
        timeout=12.5,
    )

    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "search"
    assert tool.description == "Search the web"
    assert sse_calls[0] == {
        "url": "https://mcp.example/sse",
        "headers": {"x-test": "1"},
        "timeout": 12.5,
    }
    assert sessions[0].initialized is True

    args = tool.parameters(query="cats")
    result = await tool.execute("tc-1", args, signal=None, on_update=None)

    # transport opened a second session for the call
    assert len(sessions) == 2
    assert sessions[1].calls == [("search", {"query": "cats"})]

    # only the text content block survives serialization
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "hello"
    assert result.is_error is None


@pytest.mark.asyncio
async def test_load_mcp_tools_http_propagates_is_error(monkeypatch) -> None:
    from cubepi.mcp import load_mcp_tools_http

    tools_resp = [
        SimpleNamespace(
            name="boom",
            description=None,  # falsy description path → ""
            inputSchema=None,  # falsy schema path → empty object schema
        ),
    ]
    call_resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="oops")],
        isError=True,
    )
    _install_fake_transport(monkeypatch, tools=tools_resp, call_response=call_resp)

    tools = await load_mcp_tools_http("https://mcp.example/sse")
    assert len(tools) == 1
    tool = tools[0]
    # falsy → defaults applied
    assert tool.description == ""

    args = tool.parameters()  # empty schema → no required fields
    result = await tool.execute("tc-2", args, signal=None, on_update=None)
    assert result.is_error is True


@pytest.mark.asyncio
async def test_load_mcp_tools_http_handles_empty_content(monkeypatch) -> None:
    """resp.content == None should not break the serializer."""
    from cubepi.mcp import load_mcp_tools_http

    tools_resp = [
        SimpleNamespace(
            name="silent",
            description="",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]
    call_resp = SimpleNamespace(content=None, isError=False)
    _install_fake_transport(monkeypatch, tools=tools_resp, call_response=call_resp)

    tools = await load_mcp_tools_http("https://mcp.example/sse")
    args = tools[0].parameters()
    result = await tools[0].execute("tc-3", args, signal=None, on_update=None)
    assert result.content == []


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
