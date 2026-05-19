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


# When :class:`cubepi.tracing.Tracer` is attached to an agent it
# pushes its private :class:`TracerProvider` onto this stack. Without
# any registration, ``_otel_trace.get_tracer`` falls back to the OTel
# global default — which is a no-op provider unless the caller also
# did ``set_tracer_provider`` themselves.
#
# Using a stack rather than a single slot lets one Tracer attach to
# multiple agents (each attach pushes a token; each detach pops just
# its own entry) and supports detaches in any order without clearing
# routing for the still-attached agents.
_provider_stack: list[tuple[object, Any]] = []


def register_provider(provider: Any) -> object:
    """Push ``provider`` onto the routing stack as the preferred source
    for MCP spans. Returns an opaque token that
    :func:`unregister_provider` uses to remove this exact entry.

    Called by :meth:`cubepi.tracing.Tracer.attach`.
    """
    token = object()
    _provider_stack.append((token, provider))
    return token


def unregister_provider(token: object | None = None) -> None:
    """Remove a previously-registered provider.

    ``token`` is the value returned from :func:`register_provider`.
    When ``None`` (legacy callers) the most recent entry is popped.
    Out-of-order detaches affect only their own registration; siblings
    remain.
    """
    if not _provider_stack:
        return
    if token is None:
        _provider_stack.pop()
        return
    for i, (t, _p) in enumerate(_provider_stack):
        if t is token:
            _provider_stack.pop(i)
            return


def _get_tracer(scope_name: str) -> Any:
    """Resolve the tracer to use for emitting an MCP span.

    Prefers the most recently-registered provider over OTel's global
    default (which is a no-op unless the user separately called
    ``set_tracer_provider``).
    """
    if _provider_stack:
        return _provider_stack[-1][1].get_tracer(scope_name)
    return _otel_trace.get_tracer(scope_name)


# When the cubepi Recorder opens an ``execute_tool`` span, it registers
# (span, owning_provider) here keyed by ``tool_call_id``. The MCP
# adapter looks up the parent span by tool_call_id and passes it as an
# explicit context to :func:`mcp_client_span`, so the MCP CLIENT span
# becomes a child of the agent's ``execute_tool`` span rather than
# starting an orphan trace under the OTel "current span" (which the
# recorder doesn't bother making current — see docs/specs/2026-05-18-
# cubepi-tracing-design.md §9).
#
# We store the owning provider alongside the span so that when multiple
# ``Tracer`` instances are attached to different agents in the same
# process, each MCP CLIENT span is exported through *its parent's*
# Tracer (matching trace IDs), not whichever Tracer's registration is
# at the top of the LIFO ``_provider_stack`` (codex round-7 review on
# PR #86). The stack is now only consulted when there is no parent —
# e.g., a bare ``mcp_client_span`` outside an agent's tool execution.
_active_tool_spans: dict[str, tuple[Any, Any]] = {}


def register_tool_span(
    tool_call_id: str,
    span: Any,
    provider: Any = None,
) -> None:
    _active_tool_spans[tool_call_id] = (span, provider)


def unregister_tool_span(tool_call_id: str) -> None:
    _active_tool_spans.pop(tool_call_id, None)


def _get_tool_span_entry(tool_call_id: str | None) -> tuple[Any, Any] | None:
    if tool_call_id is None:
        return None
    return _active_tool_spans.get(tool_call_id)


def _current_span_via_registered() -> Any:
    """Return the current span — preferring the registered provider's
    context. Used by :func:`current_traceparent`.

    The OTel context is process-global and shared across providers, so
    ``get_current_span()`` returns the right answer whether or not we
    use the registered provider. We expose this indirection so tests
    can monkeypatch a single helper.
    """
    return _otel_trace.get_current_span()


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
    parent_tool_call_id: str | None = None,
) -> AsyncIterator[Any]:
    """Open an OTel CLIENT span around an MCP RPC.

    When the OTel API is not installed (``cubepi[tracing]`` not
    selected) this yields ``None`` and the body runs unwrapped — the
    caller pays no overhead and has no observability impact.
    """
    if not _OTEL_AVAILABLE:
        yield None
        return

    span_name = f"{method} {tool_name}" if tool_name else method
    # Resolve explicit parent + owning provider: if the caller passed a
    # tool_call_id and the cubepi recorder has an active
    # ``execute_tool`` span for it, make the MCP CLIENT span its child
    # rather than an orphan trace, AND route it through the parent's
    # owning provider so trace_ids and exporter destination stay
    # consistent (codex round-7 fix). Fall back to the registered-
    # provider stack only when there is no parent.
    entry = _get_tool_span_entry(parent_tool_call_id)
    if entry is not None:
        parent_span, parent_provider = entry
        parent_context = _otel_trace.set_span_in_context(parent_span)
        tracer = (
            parent_provider.get_tracer(_SCOPE_NAME)
            if parent_provider is not None
            else _get_tracer(_SCOPE_NAME)
        )
    else:
        parent_context = None
        tracer = _get_tracer(_SCOPE_NAME)
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

    span = tracer.start_span(
        span_name,
        kind=SpanKind.CLIENT,
        attributes=attrs,
        context=parent_context,
    )
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
            error_type = _error_type_for(exc)
            span.set_attribute(_ERROR_TYPE, error_type)
            # Cancellation is a control signal, not a failure — match the
            # convention from the chat / turn / invoke_agent spans: leave
            # Status UNSET and mark cubepi.aborted=true, do NOT record an
            # exception event.
            if error_type == "cubepi.aborted":
                span.set_attribute("cubepi.aborted", True)
            else:
                span.set_status(Status(StatusCode.ERROR, str(exc)[:256]))
                span.record_exception(exc)
        finally:
            span.end()
        raise
    else:
        span.end()


def mark_span_mcp_error(span: Any, message: str) -> None:
    """Mark an MCP CLIENT span as a protocol-level failure.

    An MCP server can return a normal ``tools/call`` JSON-RPC response
    with ``isError: true`` — the wire call succeeds but the tool
    reports failure. Without this helper the CLIENT span would close
    with UNSET status, hiding the failure in trace dashboards.

    Pass the span yielded by :func:`mcp_client_span` (which may be
    ``None`` when OTel isn't installed); no-op on None.
    """
    if span is None or not _OTEL_AVAILABLE:
        return
    span.set_status(Status(StatusCode.ERROR, message[:256]))
    span.set_attribute(_ERROR_TYPE, "mcp.is_error")


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
