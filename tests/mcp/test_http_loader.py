"""HTTP MCP loader tests (D2.2).

The loader's transport (`sse_client` / `streamablehttp_client` +
`ClientSession`) is mocked so the test runs without a real MCP server.
End-to-end against a live server is gated behind
``CUBEPI_TEST_MCP_HTTP_URL``.
"""

import asyncio
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

    def __init__(self, *streams, tools=None, call_response=None, init_result=None):
        self._tools = tools or []
        self._call_response = call_response
        self._init_result = init_result or SimpleNamespace(
            serverInfo=SimpleNamespace(
                name="fake-server", version="0.0.0", icons=None, websiteUrl=None
            )
        )
        self.initialized = False
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def initialize(self):
        self.initialized = True
        return self._init_result

    async def list_tools(self):
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return self._call_response


def _install_fake_transport(monkeypatch, *, tools, call_response, init_result=None):
    """Patch both transport clients; return per-transport call recorders.

    The loader picks the transport at runtime, so we patch both
    ``mcp.client.sse.sse_client`` and
    ``mcp.client.streamable_http.streamablehttp_client``. Tests can assert
    on whichever recorder corresponds to the transport under test.
    """
    import mcp
    import mcp.client.sse as sse_mod
    import mcp.client.streamable_http as sh_mod

    sessions: list[_FakeSession] = []
    sse_calls: list[dict] = []
    sh_calls: list[dict] = []

    @asynccontextmanager
    async def fake_sse_client(
        url, *, headers=None, timeout=None, sse_read_timeout=None
    ):
        sse_calls.append(
            {
                "url": url,
                "headers": headers,
                "timeout": timeout,
                "sse_read_timeout": sse_read_timeout,
            }
        )
        yield ("read-stream-stub", "write-stream-stub")

    @asynccontextmanager
    async def fake_streamablehttp_client(
        url, *, headers=None, timeout=None, sse_read_timeout=None
    ):
        sh_calls.append(
            {
                "url": url,
                "headers": headers,
                "timeout": timeout,
                "sse_read_timeout": sse_read_timeout,
            }
        )
        yield ("read-stream-stub", "write-stream-stub", lambda: None)

    def fake_client_session(*streams):
        sess = _FakeSession(
            *streams,
            tools=tools,
            call_response=call_response,
            init_result=init_result,
        )
        sessions.append(sess)
        return sess

    monkeypatch.setattr(sse_mod, "sse_client", fake_sse_client)
    monkeypatch.setattr(sh_mod, "streamablehttp_client", fake_streamablehttp_client)
    monkeypatch.setattr(mcp, "ClientSession", fake_client_session)
    return sessions, sse_calls, sh_calls


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
            icons=None,
        ),
    ]
    call_resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(type="image", data="base64payload", mimeType="image/png"),
            SimpleNamespace(type="resource", uri="ignored"),  # unsupported → dropped
        ],
        isError=False,
    )
    sessions, sse_calls, sh_calls = _install_fake_transport(
        monkeypatch, tools=tools_resp, call_response=call_resp
    )

    result = await load_mcp_tools_http(
        "https://mcp.example/sse",
        headers={"x-test": "1"},
        timeout=12.5,
    )

    tools = result.tools
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "search"
    assert tool.description == "Search the web"
    # Default transport is sse — streamable_http must not be touched.
    assert sh_calls == []
    assert sse_calls[0] == {
        "url": "https://mcp.example/sse",
        "headers": {"x-test": "1"},
        "timeout": 12.5,
        "sse_read_timeout": 12.5,
    }
    assert sessions[0].initialized is True

    args = tool.parameters(query="cats")
    result = await tool.execute("tc-1", args, signal=None, on_update=None)

    # transport opened a second session for the call
    assert len(sessions) == 2
    assert sessions[1].calls == [("search", {"query": "cats"})]

    # text + image preserved; unsupported "resource" dropped
    from cubepi.providers.base import ImageContent

    assert len(result.content) == 2
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "hello"
    assert isinstance(result.content[1], ImageContent)
    assert result.content[1].source == "base64payload"
    assert result.content[1].media_type == "image/png"
    assert result.is_error is None


@pytest.mark.asyncio
async def test_load_mcp_tools_http_propagates_is_error(monkeypatch) -> None:
    from cubepi.mcp import load_mcp_tools_http

    tools_resp = [
        SimpleNamespace(
            name="boom",
            description=None,  # falsy description path → ""
            inputSchema=None,  # falsy schema path → empty object schema
            icons=None,
        ),
    ]
    call_resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="oops")],
        isError=True,
    )
    _install_fake_transport(monkeypatch, tools=tools_resp, call_response=call_resp)

    result = await load_mcp_tools_http("https://mcp.example/sse")
    tools = result.tools
    assert len(tools) == 1
    tool = tools[0]
    # falsy → defaults applied
    assert tool.description == ""

    args = tool.parameters()  # empty schema → no required fields
    result = await tool.execute("tc-2", args, signal=None, on_update=None)
    assert result.is_error is True


@pytest.mark.asyncio
async def test_load_mcp_tools_http_preserves_structured_content(monkeypatch) -> None:
    """structuredContent flows through to AgentToolResult.details."""
    from cubepi.mcp import load_mcp_tools_http

    tools_resp = [
        SimpleNamespace(
            name="weather",
            description="",
            inputSchema={"type": "object", "properties": {}},
            icons=None,
        ),
    ]
    call_resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="73F")],
        isError=False,
        structuredContent={"temp_f": 73, "conditions": "clear"},
    )
    _install_fake_transport(monkeypatch, tools=tools_resp, call_response=call_resp)

    discovery = await load_mcp_tools_http("https://mcp.example/sse")
    args = discovery.tools[0].parameters()
    result = await discovery.tools[0].execute(
        "tc-sc", args, signal=None, on_update=None
    )
    assert result.details["structuredContent"] == {"temp_f": 73, "conditions": "clear"}


@pytest.mark.asyncio
async def test_load_mcp_tools_http_handles_empty_content(monkeypatch) -> None:
    """resp.content == None should not break the serializer."""
    from cubepi.mcp import load_mcp_tools_http

    tools_resp = [
        SimpleNamespace(
            name="silent",
            description="",
            inputSchema={"type": "object", "properties": {}},
            icons=None,
        ),
    ]
    call_resp = SimpleNamespace(content=None, isError=False)
    _install_fake_transport(monkeypatch, tools=tools_resp, call_response=call_resp)

    discovery = await load_mcp_tools_http("https://mcp.example/sse")
    args = discovery.tools[0].parameters()
    result = await discovery.tools[0].execute("tc-3", args, signal=None, on_update=None)
    assert result.content == []


@pytest.mark.asyncio
async def test_load_mcp_tools_http_streamable_transport(monkeypatch) -> None:
    """transport="streamable_http" dispatches to streamablehttp_client.

    Regression gate: cubepi previously hard-coded ``sse_client`` for every
    URL. A streamable_http server accepts the SSE GET but never pushes
    the legacy ``endpoint`` event, so the agent would hang in
    ``session.initialize()`` until the per-op timeout fired. Now both
    discovery and per-tool ``call_tool`` must go through the matching
    streamable_http client.
    """
    from cubepi.mcp import load_mcp_tools_http

    tools_resp = [
        SimpleNamespace(
            name="search",
            description="",
            inputSchema={"type": "object", "properties": {}},
            icons=None,
        ),
    ]
    call_resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")], isError=False
    )
    sessions, sse_calls, sh_calls = _install_fake_transport(
        monkeypatch, tools=tools_resp, call_response=call_resp
    )

    discovery = await load_mcp_tools_http(
        "https://mcp.example/mcp",
        headers={"authorization": "Bearer x"},
        timeout=7.5,
        transport="streamable_http",
    )
    tools = discovery.tools

    # Discovery used streamable_http (sse must not be touched).
    assert sse_calls == []
    # streamablehttp_client expects ``timedelta`` on older mcp SDKs (1.8/1.9
    # era) and ``float | timedelta`` on newer ones; we always pass a
    # ``timedelta`` so the loader works on every release in our ``mcp>=1.0``
    # dependency range.
    from datetime import timedelta

    assert sh_calls[0] == {
        "url": "https://mcp.example/mcp",
        "headers": {"authorization": "Bearer x"},
        "timeout": timedelta(seconds=7.5),
        "sse_read_timeout": timedelta(seconds=7.5),
    }
    assert len(sessions) == 1

    # Per-tool call_tool must reuse the same transport so the call path
    # cannot regress to the wrong wire format.
    args = tools[0].parameters()
    await tools[0].execute("tc-sh", args, signal=None, on_update=None)
    assert sse_calls == []
    assert len(sh_calls) == 2  # one for list_tools, one for call_tool


@pytest.mark.asyncio
async def test_load_mcp_tools_http_rejects_unknown_transport() -> None:
    """Unknown transport raises ValueError; we do not silently fall back."""
    from cubepi.mcp import load_mcp_tools_http

    with pytest.raises(ValueError, match="unsupported MCP transport"):
        await load_mcp_tools_http(
            "https://mcp.example/sse",
            transport="websocket",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_load_mcp_tools_http_initialize_timeout(monkeypatch) -> None:
    """A session that hangs on initialize must raise TimeoutError, not block."""
    from cubepi.mcp import load_mcp_tools_http

    class _HangingSession:
        def __init__(self, *streams):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def initialize(self):
            await asyncio.sleep(10)  # forever, from the test's perspective

        async def list_tools(self):  # pragma: no cover - never reached
            return SimpleNamespace(tools=[])

        async def call_tool(self, *a, **k):  # pragma: no cover - never reached
            return SimpleNamespace(content=[], isError=False)

    import mcp
    import mcp.client.sse as sse_mod

    @asynccontextmanager
    async def fake_sse_client(
        url, *, headers=None, timeout=None, sse_read_timeout=None
    ):
        yield ("r", "w")

    monkeypatch.setattr(sse_mod, "sse_client", fake_sse_client)
    monkeypatch.setattr(mcp, "ClientSession", _HangingSession)

    with pytest.raises(asyncio.TimeoutError):
        await load_mcp_tools_http("https://mcp.example/sse", timeout=0.05)


@pytest.mark.asyncio
async def test_load_mcp_tools_http_against_test_server() -> None:
    """End-to-end: connect to a real MCP test server, list + call a tool."""
    server_url = os.environ.get("CUBEPI_TEST_MCP_HTTP_URL")
    if not server_url:
        pytest.skip("Set CUBEPI_TEST_MCP_HTTP_URL to run this test")

    from cubepi.mcp import load_mcp_tools_http

    discovery = await load_mcp_tools_http(server_url)
    assert len(discovery.tools) > 0
    first = discovery.tools[0]
    assert first.name
    assert first.description is not None


# ---------------------------------------------------------------------------
# New: server + tool icon metadata capture (MCP spec 2025-11-25)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_mcp_tools_http_captures_server_info_from_initialize(
    monkeypatch,
) -> None:
    """Implementation.icons + websiteUrl + name flow into MCPServerInfo."""
    from cubepi.mcp import MCPDiscoveryResult, MCPIcon, load_mcp_tools_http

    init_result = SimpleNamespace(
        serverInfo=SimpleNamespace(
            name="Linear",
            version="1.4.2",
            websiteUrl="https://linear.app",
            icons=[
                SimpleNamespace(
                    src="https://linear.app/favicon.svg", mimeType="image/svg+xml"
                ),
                SimpleNamespace(
                    src="data:image/png;base64,zzz",
                    mimeType="image/png",
                    sizes=["48x48"],
                ),
            ],
        )
    )
    tools_resp = [
        SimpleNamespace(
            name="create_issue",
            description="Create a Linear issue",
            inputSchema={"type": "object", "properties": {}},
            icons=None,
        ),
    ]
    call_resp = SimpleNamespace(content=[], isError=False)
    _install_fake_transport(
        monkeypatch, tools=tools_resp, call_response=call_resp, init_result=init_result
    )

    discovery = await load_mcp_tools_http("https://mcp.example/sse")
    assert isinstance(discovery, MCPDiscoveryResult)
    assert discovery.server is not None
    assert discovery.server.name == "Linear"
    assert discovery.server.version == "1.4.2"
    assert discovery.server.website_url == "https://linear.app"
    assert discovery.server.icons == (
        MCPIcon(src="https://linear.app/favicon.svg", mime_type="image/svg+xml"),
        MCPIcon(
            src="data:image/png;base64,zzz", mime_type="image/png", sizes=("48x48",)
        ),
    )


@pytest.mark.asyncio
async def test_load_mcp_tools_http_captures_per_tool_icons(monkeypatch) -> None:
    """Per-tool ``Tool.icons`` flow into ``tool_infos`` keyed by tool name."""
    from cubepi.mcp import MCPIcon, load_mcp_tools_http

    tools_resp = [
        SimpleNamespace(
            name="search",
            description="",
            inputSchema={"type": "object", "properties": {}},
            icons=[
                SimpleNamespace(
                    src="data:image/svg+xml;base64,abc", mimeType=None, sizes=None
                )
            ],
        ),
        SimpleNamespace(
            name="fetch",
            description="",
            inputSchema={"type": "object", "properties": {}},
            icons=None,
        ),
    ]
    call_resp = SimpleNamespace(content=[], isError=False)
    _install_fake_transport(monkeypatch, tools=tools_resp, call_response=call_resp)

    discovery = await load_mcp_tools_http("https://mcp.example/sse")
    by_name = {ti.name: ti for ti in discovery.tool_infos}
    assert by_name["search"].icons == (MCPIcon(src="data:image/svg+xml;base64,abc"),)
    assert by_name["fetch"].icons == ()


@pytest.mark.asyncio
async def test_load_mcp_tools_http_handles_missing_server_icons(monkeypatch) -> None:
    """Server without icons / websiteUrl yields MCPServerInfo with empty defaults."""
    from cubepi.mcp import load_mcp_tools_http

    init_result = SimpleNamespace(
        serverInfo=SimpleNamespace(
            name="bare-server", version="0.1.0", icons=None, websiteUrl=None
        )
    )
    tools_resp = [
        SimpleNamespace(
            name="echo",
            description="",
            inputSchema={"type": "object", "properties": {}},
            icons=None,
        ),
    ]
    call_resp = SimpleNamespace(content=[], isError=False)
    _install_fake_transport(
        monkeypatch, tools=tools_resp, call_response=call_resp, init_result=init_result
    )

    discovery = await load_mcp_tools_http("https://mcp.example/sse")
    assert discovery.server is not None
    assert discovery.server.icons == ()
    assert discovery.server.website_url is None
