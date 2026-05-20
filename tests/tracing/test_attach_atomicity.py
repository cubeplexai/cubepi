"""Attach atomicity: a failed ``Tracer.attach`` / ``Recorder.attach`` must
leave no dangling listener subscriptions behind.

``attach`` wires several subscriptions (agent event listener + three provider
listeners, then an MCP provider registration). If any step raises partway, the
subscriptions registered before it would otherwise leak — the caller never
receives a ``detach`` callable, so it cannot clean them up. These tests pin the
contract: attach is all-or-nothing.
"""

from __future__ import annotations

import pytest

from cubepi.agent.agent import Agent
from cubepi.providers.base import Model
from cubepi.providers.faux import FauxProvider
from cubepi.tracing import Tracer

MODEL = Model(id="faux-1", provider="faux")


def _build_unattached() -> tuple[Agent, FauxProvider, Tracer]:
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class _NullExporter(SpanExporter):
        def export(self, spans):  # noqa: ANN001
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            return True

    provider = FauxProvider()
    agent = Agent(provider=provider, model=MODEL, system_prompt="t")
    tracer = Tracer(
        service_name="t",
        agent_name="t",
        exporters=[_NullExporter()],
        atexit_flush=False,
    )
    return agent, provider, tracer


def _assert_no_listeners(agent: Agent, provider: FauxProvider) -> None:
    assert agent._listeners == [], "agent event listener leaked after failed attach"
    assert provider._request_listeners == [], "provider request listener leaked"
    assert provider._chunk_listeners == [], "provider chunk listener leaked"
    assert provider._response_listeners == [], "provider response listener leaked"


async def test_recorder_attach_unwinds_on_partial_provider_subscription_failure():
    agent, provider, tracer = _build_unattached()
    try:
        # subscribe_request succeeds, subscribe_chunk blows up midway.
        def _boom(cb):  # noqa: ANN001, ANN202
            raise RuntimeError("subscribe_chunk failed")

        provider.subscribe_chunk = _boom  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="subscribe_chunk failed"):
            tracer.attach(agent)

        _assert_no_listeners(agent, provider)
    finally:
        await tracer.shutdown()


async def test_tracer_attach_unwinds_recorder_on_mcp_register_failure(monkeypatch):
    agent, provider, tracer = _build_unattached()
    try:
        import cubepi.mcp._tracing as mcp_tracing

        def _boom(_provider):  # noqa: ANN001, ANN202
            raise RuntimeError("register_provider failed")

        monkeypatch.setattr(mcp_tracing, "register_provider", _boom)

        with pytest.raises(RuntimeError, match="register_provider failed"):
            tracer.attach(agent)

        _assert_no_listeners(agent, provider)
    finally:
        await tracer.shutdown()
