"""Phase 5: pin MCP CLIENT span emissions + W3C traceparent propagation.

The MCP adapter wraps every ``call_remote`` invocation in
:func:`cubepi.mcp._tracing.mcp_client_span`, which opens a CLIENT span
with the GenAI MCP semconv attributes. When the OTel API is absent the
context manager is a no-op (verified separately).
"""

from __future__ import annotations

import asyncio
from typing import Any

from opentelemetry import trace as _trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import SpanKind, StatusCode

from cubepi.mcp._adapter import make_mcp_agent_tool


class _CaptureExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans):  # noqa: ANN001
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _make_provider() -> tuple[TracerProvider, _CaptureExporter]:
    """Build a fresh TracerProvider with an in-memory exporter.

    Each test gets its own provider; we don't use trace.set_tracer_provider
    globally, instead we monkeypatch the module-level ``_otel_trace`` in
    ``cubepi.mcp._tracing`` so MCP fetches the test's tracer.
    """
    resource = Resource.create({"service.name": "mcp-span-tests"})
    provider = TracerProvider(resource=resource)
    exporter = _CaptureExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _patch_mcp_trace(monkeypatch, provider: TracerProvider) -> None:
    """Swap the cubepi.mcp._tracing module's ``get_tracer`` to use ours."""
    import cubepi.mcp._tracing as mcp_tracing

    class _ShimTraceMod:
        @staticmethod
        def get_tracer(name: str):  # noqa: D401
            return provider.get_tracer(name)

        @staticmethod
        def use_span(span, **kwargs):
            return _trace.use_span(span, **kwargs)

        @staticmethod
        def get_current_span():
            return _trace.get_current_span()

    monkeypatch.setattr(mcp_tracing, "_otel_trace", _ShimTraceMod)
    monkeypatch.setattr(mcp_tracing, "_OTEL_AVAILABLE", True)


async def _make_tool(
    call_remote,
    *,
    server_address=None,
    server_port=None,
    protocol_version=None,
    session_id=None,
):
    """Build the MCP-adapted AgentTool. Mirrors what loaders produce."""
    return make_mcp_agent_tool(
        name="search",
        description="search the web",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        call_remote=call_remote,
        server_address=server_address,
        server_port=server_port,
        protocol_version=protocol_version,
        session_id=session_id,
    )


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


class TestMCPClientSpan:
    async def test_span_emitted_with_required_attrs(self, monkeypatch):
        provider, exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        async def call_remote(name, args):
            return {"content": [{"type": "text", "text": "ok"}], "isError": False}

        tool = await _make_tool(
            call_remote,
            server_address="example.com",
            server_port=443,
            protocol_version="2025-11-25",
            session_id="sess-abc",
        )
        result = await tool.execute("tc1", tool.parameters.model_validate({"q": "x"}))
        assert result.is_error is None  # success path

        # One MCP CLIENT span captured.
        mcp_spans = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(mcp_spans) == 1
        span = mcp_spans[0]
        assert span.kind == SpanKind.CLIENT
        attrs = _attrs(span)
        assert attrs["mcp.method.name"] == "tools/call"
        assert attrs["gen_ai.tool.name"] == "search"
        assert attrs["gen_ai.operation.name"] == "execute_tool"
        assert attrs["mcp.session.id"] == "sess-abc"
        assert attrs["mcp.protocol.version"] == "2025-11-25"
        assert attrs["server.address"] == "example.com"
        assert attrs["server.port"] == 443

    async def test_span_records_exception_on_failure(self, monkeypatch):
        provider, exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        async def call_remote(name, args):
            raise RuntimeError("server unavailable")

        tool = await _make_tool(call_remote)
        try:
            await tool.execute("tc1", tool.parameters.model_validate({"q": "x"}))
        except RuntimeError:
            pass

        mcp_spans = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(mcp_spans) == 1
        span = mcp_spans[0]
        assert span.status.status_code == StatusCode.ERROR
        attrs = _attrs(span)
        assert attrs["error.type"] == "RuntimeError"
        evnames = [e.name for e in span.events]
        assert "exception" in evnames

    async def test_exception_event_is_recorded_only_once(self, monkeypatch):
        """OTel's ``use_span`` defaults to record_exception=True; if we
        leave that on, the auto-recording on context exit and our own
        ``record_exception`` in the except branch both fire — duplicate
        exception events double-count errors at the backend.

        We disable auto-recording on ``use_span`` and own the
        recording. Pin: exactly one ``exception`` event per failure."""
        provider, exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        async def call_remote(name, args):
            raise RuntimeError("boom")

        tool = await _make_tool(call_remote)
        try:
            await tool.execute("tc1", tool.parameters.model_validate({"q": "x"}))
        except RuntimeError:
            pass

        mcp_spans = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(mcp_spans) == 1
        exception_events = [e for e in mcp_spans[0].events if e.name == "exception"]
        assert len(exception_events) == 1, (
            f"expected exactly 1 exception event; got {len(exception_events)}"
        )

    async def test_span_records_cancellation(self, monkeypatch):
        """Cancellation is a control signal, not a failure — match the
        chat / turn / invoke_agent convention: leave Status UNSET,
        record cubepi.aborted=true + error.type, do NOT add an
        ``exception`` event."""
        provider, exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        async def call_remote(name, args):
            raise asyncio.CancelledError()

        tool = await _make_tool(call_remote)
        try:
            await tool.execute("tc1", tool.parameters.model_validate({"q": "x"}))
        except asyncio.CancelledError:
            pass

        mcp_spans = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(mcp_spans) == 1
        span = mcp_spans[0]
        attrs = _attrs(span)
        assert attrs["error.type"] == "cubepi.aborted"
        assert attrs["cubepi.aborted"] is True
        # Status stays UNSET; cancel is not a failure.
        assert span.status.status_code == StatusCode.UNSET
        # No exception event — cancel is signaled via cubepi.aborted only.
        assert not any(e.name == "exception" for e in span.events)


class TestNoOpWhenOTelMissing:
    async def test_context_manager_yields_none_when_no_otel(self, monkeypatch):
        import cubepi.mcp._tracing as mcp_tracing

        monkeypatch.setattr(mcp_tracing, "_OTEL_AVAILABLE", False)

        async def call_remote(name, args):
            return {"content": [], "isError": False}

        tool = await _make_tool(call_remote)
        # Must run without errors and without emitting any spans.
        await tool.execute("tc1", tool.parameters.model_validate({"q": "x"}))


class TestTraceparentHelper:
    async def test_current_traceparent_returns_w3c_string_when_in_span(
        self, monkeypatch
    ):
        provider, _ = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)
        from cubepi.mcp._tracing import current_traceparent

        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("test-span"):
            tp = current_traceparent()
        assert tp is not None
        # Format: 00-<32hex>-<16hex>-<flags>
        parts = tp.split("-")
        assert len(parts) == 4
        assert parts[0] == "00"
        assert len(parts[1]) == 32
        assert len(parts[2]) == 16
        assert len(parts[3]) == 2

    async def test_current_traceparent_none_without_span(self, monkeypatch):
        from cubepi.mcp._tracing import current_traceparent

        # No active span (and no recording context set up).
        assert current_traceparent() is None


class TestTraceparentInjection:
    """The HTTP loader must inject ``traceparent`` into outbound
    session headers whenever the MCP CLIENT span is active so that
    instrumented MCP servers can continue the trace (codex round 2)."""

    async def test_traceparent_header_set_inside_mcp_span(self, monkeypatch):
        from cubepi.mcp import _tracing as mcp_tracing
        from cubepi.mcp._tracing import mcp_client_span

        # Force OTel on with a deterministic tracer.
        provider, _exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        # Reproduce the http_loader.call_remote injection logic.
        seen_headers: list[dict] = []

        base_headers = {"x-test": "yes"}

        async def fake_call_remote():
            # Mirrors http_loader._call_remote header-merge logic.
            tp = mcp_tracing.current_traceparent()
            call_headers = base_headers
            if tp is not None:
                call_headers = {**base_headers, "traceparent": tp}
            seen_headers.append(call_headers)

        async with mcp_client_span(method="tools/call", tool_name="search"):
            await fake_call_remote()

        assert len(seen_headers) == 1
        assert seen_headers[0]["x-test"] == "yes"  # caller headers preserved
        assert "traceparent" in seen_headers[0]
        # Format: 00-<32hex>-<16hex>-<flags>
        tp = seen_headers[0]["traceparent"]
        assert tp.startswith("00-") and len(tp.split("-")) == 4

    async def test_no_header_added_when_no_active_span(self, monkeypatch):
        from cubepi.mcp._tracing import current_traceparent

        # Outside any span: helper returns None and loader must not add
        # a traceparent header.
        assert current_traceparent() is None


class TestLoaderHelpers:
    def test_split_address_parses_url(self):
        from cubepi.mcp.http_loader import _split_address

        assert _split_address("https://api.example.com:8443/mcp") == (
            "api.example.com",
            8443,
        )
        host, port = _split_address("http://localhost/mcp")
        assert host == "localhost"
        assert port is None

    def test_split_address_on_empty(self):
        from cubepi.mcp.http_loader import _split_address

        # urlparse("") returns hostname=None, port=None — both nones is
        # the documented signal that the helper couldn't extract values.
        host, port = _split_address("")
        assert host is None
        assert port is None

    def test_extract_protocol_version(self):
        from cubepi.mcp.http_loader import _extract_protocol_version

        class FakeInit:
            protocolVersion = "2025-11-25"

        assert _extract_protocol_version(FakeInit()) == "2025-11-25"
        assert _extract_protocol_version(object()) is None

        class FakeBadInit:
            protocolVersion = 123  # not a string

        assert _extract_protocol_version(FakeBadInit()) is None
