"""The ``trace`` best-effort scope helper.

``trace(tracer, agent)`` is a context manager that attaches the tracer on
enter and detaches + flushes on exit, swallowing every tracing fault so that
tracing can never break or fail the wrapped work. ``tracer=None`` is a no-op.
These tests pin that contract: attach/detach lifecycle on the happy path, and
silent fallback on attach / flush failures.
"""

from __future__ import annotations

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from cubepi.agent.agent import Agent
from cubepi.providers.base import Model
from cubepi.providers.faux import FauxProvider
from cubepi.tracing import Tracer, trace

MODEL = Model(id="faux-1", provider="faux")


class _NullExporter(SpanExporter):
    def export(self, spans):  # noqa: ANN001
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _build() -> tuple[Agent, FauxProvider, Tracer]:
    provider = FauxProvider()
    agent = Agent(provider=provider, model=MODEL, system_prompt="t")
    tracer = Tracer(
        service_name="t",
        agent_name="t",
        exporters=[_NullExporter()],
        atexit_flush=False,
    )
    return agent, provider, tracer


async def test_trace_none_is_noop():
    agent, _provider, tracer = _build()
    try:
        ran = False
        async with trace(None, agent):
            ran = True
        assert ran
        assert agent._listeners == [], (
            "no listeners should be attached when tracer is None"
        )
    finally:
        await tracer.shutdown()


async def test_trace_attaches_within_scope_and_detaches_after():
    agent, provider, tracer = _build()
    try:
        assert agent._listeners == []
        async with trace(tracer, agent):
            assert len(agent._listeners) == 1, (
                "recorder should be attached inside the scope"
            )
            assert len(provider._request_listeners) == 1
            assert len(provider._chunk_listeners) == 1
            assert len(provider._response_listeners) == 1
        assert agent._listeners == [], "recorder should be detached after the scope"
        assert provider._request_listeners == []
        assert provider._chunk_listeners == []
        assert provider._response_listeners == []
    finally:
        await tracer.shutdown()


async def test_trace_swallows_attach_failure(monkeypatch):
    agent, _provider, tracer = _build()
    try:

        def _boom(_listener):  # noqa: ANN202
            raise RuntimeError("attach boom")

        monkeypatch.setattr(agent, "subscribe", _boom)

        ran = False
        async with trace(tracer, agent):  # must not raise
            ran = True
        assert ran, "body must run even when attach fails"
    finally:
        await tracer.shutdown()


async def test_trace_swallows_flush_failure(monkeypatch):
    agent, provider, tracer = _build()
    try:

        async def _boom_flush(*_args, **_kwargs):  # noqa: ANN202
            raise RuntimeError("flush boom")

        monkeypatch.setattr(tracer, "force_flush", _boom_flush)

        ran = False
        async with trace(tracer, agent):
            ran = True
        assert ran, "body must run even when flush fails on exit"
        # Synchronous detach still ran before the flush failure.
        assert agent._listeners == []
        assert provider._request_listeners == []
    finally:
        # Restore the real force_flush so shutdown() can clean up.
        monkeypatch.undo()
        await tracer.shutdown()
