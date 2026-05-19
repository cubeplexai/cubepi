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

import contextvars
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


# When the cubepi Recorder opens an ``execute_tool`` span, it publishes
# ``(span, owning_provider)`` here so an MCP tool call running inside
# the AgentTool body can make its CLIENT span a child of this span
# (rather than starting an orphan root trace — recorder doesn't bother
# installing ``execute_tool`` as the OTel current span; see
# docs/specs/2026-05-18-cubepi-tracing-design.md §9), and route the
# CLIENT span through the parent's owning provider so trace_ids and
# exporter destination stay consistent (codex round-7).
#
# Lookup uses a per-task ``ContextVar`` holding an opaque handle, with
# the actual ``(span, provider)`` payload stored in a module-level
# ``_active_entries`` dict. The dict is the source of truth for
# cleanup; the contextvar is the per-task pointer.
#
# Why both: tool-call ids are provider-local (Faux / OpenAI-style mint
# ``tc1`` / ``call_...`` per conversation), so a global dict keyed by
# ``tool_call_id`` lets concurrent agents overwrite each other (codex
# round-8). ContextVars scope per asyncio task and copy on
# ``create_task``, so each agent run — and each spawned tool task
# within it — sees only its own parent. BUT the cubepi agent loop
# emits ``ToolExecutionStartEvent`` in the parent task and (in
# ``parallel`` tool mode) ``ToolExecutionEndEvent`` from the per-tool
# *child* task it spawns. A ``Token`` produced in the parent task
# cannot be ``reset`` in a child task — ``ContextVar.reset`` raises
# ``ValueError`` (codex round-9). The dict-handle indirection lets
# ``unregister_tool_span`` clean up the payload unconditionally (so
# any later lookup through the stale contextvar returns ``None``),
# while the contextvar reset is best-effort and skipped silently when
# the calling task differs from the registering one.
_active_entries: dict[object, tuple[Any, Any]] = {}
_current_handle: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "_cubepi_mcp_current_tool_handle", default=None
)


def register_tool_span(
    tool_call_id: str,
    span: Any,
    provider: Any = None,
) -> tuple[object, contextvars.Token[object | None]]:
    """Publish ``span`` (and its owning ``provider``) as the current
    ``execute_tool`` parent for the calling task.

    Returns a ``(handle, cv_token)`` tuple — pass it to
    :func:`unregister_tool_span` on tool-exec end. The handle is used
    for dict cleanup (safe across tasks); the cv_token is used for
    contextvar reset (best-effort, only valid in the registering
    task).

    ``tool_call_id`` is accepted for API symmetry / future debug attrs
    but is not used as a lookup key — see module-level comment.
    """
    del tool_call_id
    handle = object()
    _active_entries[handle] = (span, provider)
    cv_token = _current_handle.set(handle)
    return (handle, cv_token)


def unregister_tool_span(
    token: tuple[object, contextvars.Token[object | None]] | None,
) -> None:
    """Clean up a previously-registered entry.

    ``token`` is the value returned from :func:`register_tool_span`,
    or ``None`` (no-op, for callers that defensively unregister when
    the corresponding register failed).

    The dict entry is always removed — this is the safety net for
    cross-task ``unregister`` (parallel tool mode emits
    ``ToolExecutionEndEvent`` from the child task even though
    ``register`` ran in the parent). The contextvar reset is
    attempted but silently skipped if the calling task is not the
    registering task; the next ``register`` in the parent task will
    overwrite the stale handle.
    """
    if token is None:
        return
    handle, cv_token = token
    _active_entries.pop(handle, None)
    try:
        _current_handle.reset(cv_token)
    except (ValueError, LookupError):
        pass


def _get_tool_span_entry() -> tuple[Any, Any] | None:
    """Return the (span, provider) entry for the current task, or
    ``None`` when no live ``execute_tool`` is in scope.

    A stale contextvar handle (pointing at an already-cleaned-up dict
    entry) returns ``None`` — exactly the same outcome as no parent at
    all, so the MCP CLIENT span starts a new root rather than
    parenting under an already-ended span.
    """
    handle = _current_handle.get()
    if handle is None:
        return None
    return _active_entries.get(handle)


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
    # Resolve explicit parent + owning provider from the per-task
    # contextvar published by the cubepi recorder on
    # ``_on_tool_exec_start``. If present, the MCP CLIENT span becomes a
    # child of the agent's execute_tool span AND is exported through
    # the parent's owning provider so trace_ids and exporter
    # destination stay consistent across concurrent Tracers (codex
    # round-7 + round-8). When no parent is active, fall back to the
    # registered-provider stack — used by bare ``mcp_client_span``
    # calls outside an agent run.
    #
    # ``parent_tool_call_id`` is accepted for API symmetry / future
    # debug attrs; the lookup is purely contextvar-scoped now.
    del parent_tool_call_id
    entry = _get_tool_span_entry()
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
