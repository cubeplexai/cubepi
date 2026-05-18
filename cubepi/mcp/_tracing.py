"""MCP-side OTel CLIENT span instrumentation.

This module is intentionally local to ``cubepi/mcp`` so the core MCP
adapter has zero hard dependency on ``opentelemetry`` — if the OTel API
is not installed, every public symbol becomes a no-op pass-through. When
``cubepi[tracing]`` is installed, MCP ``tools/call`` invocations
automatically emit CLIENT spans per the OTel GenAI MCP semconv (§14 of
the tracing design spec).

Span emitted per call::

    tools/call <tool_name>           [CLIENT]
        mcp.method.name = "tools/call"
        gen_ai.tool.name = <tool_name>
        gen_ai.operation.name = "execute_tool"
        mcp.session.id = <if provided>
        mcp.protocol.version = <if provided>
        server.address = <if provided>
        server.port = <if provided>
        error.type = <on failure>

The W3C ``traceparent`` for downstream-server propagation is exposed
via :func:`current_traceparent`; the HTTP loader injects it into the
session's HTTP headers so an instrumented MCP server can continue the
trace. The MCP Python SDK does not expose the JSON-RPC ``params._meta``
slot directly, so we use HTTP headers as the practical wire location —
W3C trace-context spec §3 permits either.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    pass


try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import SpanKind, Status, StatusCode

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only without the extra.
    _OTEL_AVAILABLE = False
    _otel_trace = None  # type: ignore[assignment]
    SpanKind = None  # type: ignore[assignment]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]


# Public attribute name constants — duplicated from
# cubepi.tracing.schema so this module can be imported without the
# tracing extra installed. Keep in sync.
_MCP_METHOD_NAME = "mcp.method.name"
_MCP_SESSION_ID = "mcp.session.id"
_MCP_PROTOCOL_VERSION = "mcp.protocol.version"
_GEN_AI_TOOL_NAME = "gen_ai.tool.name"
_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
_SERVER_ADDRESS = "server.address"
_SERVER_PORT = "server.port"
_ERROR_TYPE = "error.type"
_SCOPE_NAME = "cubepi.mcp"


@asynccontextmanager
async def mcp_client_span(
    *,
    method: str = "tools/call",
    tool_name: str | None = None,
    session_id: str | None = None,
    protocol_version: str | None = None,
    server_address: str | None = None,
    server_port: int | None = None,
) -> AsyncIterator[Any]:
    """Open an OTel CLIENT span around an MCP RPC.

    When the OTel API is not installed (``cubepi[tracing]`` not
    selected) this yields ``None`` and the body runs unwrapped — the
    caller pays no overhead and has no observability impact.
    """
    if not _OTEL_AVAILABLE:
        yield None
        return

    tracer = _otel_trace.get_tracer(_SCOPE_NAME)
    span_name = f"{method} {tool_name}" if tool_name else method
    attrs: dict[str, Any] = {
        _MCP_METHOD_NAME: method,
        _GEN_AI_OPERATION_NAME: "execute_tool",
    }
    if tool_name is not None:
        attrs[_GEN_AI_TOOL_NAME] = tool_name
    if session_id is not None:
        attrs[_MCP_SESSION_ID] = session_id
    if protocol_version is not None:
        attrs[_MCP_PROTOCOL_VERSION] = protocol_version
    if server_address is not None:
        attrs[_SERVER_ADDRESS] = server_address
    if server_port is not None:
        attrs[_SERVER_PORT] = server_port

    span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes=attrs)
    try:
        # Disable use_span's default record_exception / set_status_on_exception
        # so we are the single source of the exception event and ERROR
        # status — otherwise OTel would auto-record on context exit AND
        # this ``except`` block would record again, double-counting.
        with _otel_trace.use_span(
            span,
            record_exception=False,
            set_status_on_exception=False,
        ):
            yield span
    except BaseException as exc:
        try:
            span.set_status(Status(StatusCode.ERROR, str(exc)[:256]))
            span.set_attribute(_ERROR_TYPE, _error_type_for(exc))
            span.record_exception(exc)
        finally:
            span.end()
        raise
    else:
        span.end()


def current_traceparent() -> str | None:
    """Return a W3C ``traceparent`` string for the current span context,
    or ``None`` when there is no active recording span (or OTel is not
    installed).

    Used by the HTTP loader to inject the header on outgoing MCP
    requests so an instrumented server can continue the trace.
    """
    if not _OTEL_AVAILABLE:
        return None
    span = _otel_trace.get_current_span()
    ctx = span.get_span_context()
    if not getattr(ctx, "is_valid", False):
        return None
    trace_id = ctx.trace_id
    span_id = ctx.span_id
    if not trace_id or not span_id:
        return None
    # W3C trace context §3.2: traceparent = "00-<32hex>-<16hex>-<flags>"
    flags = int(getattr(ctx, "trace_flags", 0))
    return f"00-{trace_id:032x}-{span_id:016x}-{flags:02x}"


def _error_type_for(exc: BaseException) -> str:
    """Local error.type derivation. Mirrors cubepi.tracing.errors but
    re-implemented here so the MCP module has no hard tracing dep."""
    import asyncio

    if isinstance(exc, asyncio.CancelledError):
        return "cubepi.aborted"
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "connection_error"
    cls = type(exc)
    if cls.__module__ in {"builtins", "__main__"}:
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"
