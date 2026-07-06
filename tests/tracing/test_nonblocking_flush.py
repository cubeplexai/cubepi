"""Non-blocking span flush: ``force_flush`` off the loop, background trace mode.

The provider's ``force_flush`` is synchronous and blocks its calling
thread until every processor drains its queue through
``exporter.export()`` — seconds when an OTLP collector is remote or
backlogged. Two invariants pinned here:

1. ``Tracer.force_flush`` runs that sync flush in a worker thread, so
   awaiting it never stalls the event loop (a stalled loop freezes
   every concurrent request in the host process).
2. ``trace(tracer, agent, flush="background")`` exits the block without
   waiting for export, while ``await tracer.shutdown()`` still settles
   any in-flight background flush so spans are not lost on clean
   shutdown.
"""

from __future__ import annotations

import asyncio
import threading
import time

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from cubepi.agent.agent import Agent
from cubepi.providers.base import Model
from cubepi.providers.faux import FauxProvider
from cubepi.tracing import Tracer, trace

MODEL = Model(id="faux-1", provider_id="faux")


class _SlowExporter(SpanExporter):
    """Sleeps in ``export`` to simulate a slow/backlogged collector."""

    def __init__(self, export_seconds: float = 0.0) -> None:
        self.export_seconds = export_seconds
        self.export_count = 0

    def export(self, spans):  # noqa: ANN001
        self.export_count += 1
        if self.export_seconds:
            time.sleep(self.export_seconds)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _build(exporter: SpanExporter) -> tuple[Agent, Tracer]:
    provider = FauxProvider(provider_id="faux")
    agent = Agent(model=provider.model(MODEL.id), system_prompt="t")
    tracer = Tracer(
        service_name="t",
        agent_name="t",
        exporters=[exporter],
        atexit_flush=False,
    )
    return agent, tracer


def _queue_span(tracer: Tracer) -> None:
    """Put one ended span into the BatchSpanProcessor queue so
    ``force_flush`` actually has something to export."""
    tracer._otel_tracer.start_span("flush-probe").end()


async def test_force_flush_calls_provider_off_the_loop_thread():
    exporter = _SlowExporter()
    _agent, tracer = _build(exporter)
    loop_thread = threading.current_thread().name
    flush_threads: list[str] = []
    inner = tracer._provider.force_flush

    def _spy(timeout_millis: int = 30_000) -> bool:
        flush_threads.append(threading.current_thread().name)
        return inner(timeout_millis=timeout_millis)

    tracer._provider.force_flush = _spy  # type: ignore[method-assign]
    try:
        assert await tracer.force_flush() is True
        assert flush_threads, "provider.force_flush must have run"
        assert flush_threads[0] != loop_thread, (
            "sync provider flush must run in a worker thread, not on the loop"
        )
    finally:
        tracer._provider.force_flush = inner  # type: ignore[method-assign]
        await tracer.shutdown()


async def test_force_flush_does_not_stall_concurrent_tasks():
    # With a span queued against a 0.5s exporter, force_flush blocks its
    # calling thread for >=0.5s. On the loop thread that would freeze the
    # ticker below; off-loop it keeps ticking. Generous threshold to
    # avoid CI flakes.
    exporter = _SlowExporter(export_seconds=0.5)
    _agent, tracer = _build(exporter)
    _queue_span(tracer)
    ticks = 0

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticker = asyncio.create_task(_ticker())
    try:
        await tracer.force_flush()
        assert exporter.export_count >= 1, "flush must have exported the queued span"
        assert ticks >= 10, (
            f"event loop stalled during flush (ticks={ticks}); "
            "provider.force_flush must not run on the loop thread"
        )
    finally:
        ticker.cancel()
        await tracer.shutdown()


async def test_meter_force_flush_calls_provider_off_the_loop_thread():
    # Meter mirrors Tracer: its provider force_flush is synchronous and
    # must run in a worker thread, never on the loop.
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.resources import Resource

    from cubepi.tracing import Meter
    from cubepi.tracing.schema import SCHEMA_URL

    reader = InMemoryMetricReader()
    resource = Resource.create({"service.name": "test"}, schema_url=SCHEMA_URL)
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    meter = Meter.__new__(Meter)
    meter._provider = provider  # type: ignore[attr-defined]
    meter._shutdown = False  # type: ignore[attr-defined]

    loop_thread = threading.current_thread().name
    flush_threads: list[str] = []
    inner = provider.force_flush

    def _spy(timeout_millis: int = 30_000) -> bool:
        flush_threads.append(threading.current_thread().name)
        return inner(timeout_millis=timeout_millis)

    provider.force_flush = _spy  # type: ignore[method-assign]
    try:
        assert await meter.force_flush() is True
        assert flush_threads, "provider.force_flush must have run"
        assert flush_threads[0] != loop_thread, (
            "sync metric flush must run in a worker thread, not on the loop"
        )
    finally:
        provider.force_flush = inner  # type: ignore[method-assign]
        provider.shutdown()


async def test_trace_background_exits_before_export_completes():
    exporter = _SlowExporter(export_seconds=0.5)
    agent, tracer = _build(exporter)
    try:
        t0 = time.monotonic()
        async with trace(tracer, agent, flush="background"):
            _queue_span(tracer)
        exited_after = time.monotonic() - t0
        # Give the background task a tick to start without waiting for it.
        await asyncio.sleep(0)
        assert exited_after < 0.3, (
            f"background mode must not gate block exit on export "
            f"(waited {exited_after:.2f}s against a 0.5s exporter)"
        )
        # Detach is still synchronous: listeners are gone at exit.
        assert agent._listeners == []
        # The flush is tracked, not dropped.
        assert tracer._pending_flushes, "background flush task must be tracked"
    finally:
        await tracer.shutdown()
    assert exporter.export_count >= 1, "shutdown must settle the background flush"
    assert not tracer._pending_flushes


async def test_trace_background_flush_failure_is_swallowed_and_logged(caplog):
    exporter = _SlowExporter()
    agent, tracer = _build(exporter)

    async def _boom_flush(*_args, **_kwargs):  # noqa: ANN202
        raise RuntimeError("flush boom")

    tracer.force_flush = _boom_flush  # type: ignore[method-assign]
    try:
        async with trace(tracer, agent, flush="background"):
            pass
        # Let the supervisor observe the failure.
        for _ in range(10):
            if not tracer._pending_flushes:
                break
            await asyncio.sleep(0.01)
        assert not tracer._pending_flushes
        assert any(
            "background flush failed" in r.getMessage() for r in caplog.records
        ), "background flush failures must be logged, not silently dropped"
    finally:
        del tracer.force_flush  # restore the real method for shutdown
        await tracer.shutdown()


async def test_trace_default_still_awaits_flush():
    # Regression pin: default mode blocks until export is done, so callers
    # relying on "block exit == spans persisted" keep that guarantee.
    exporter = _SlowExporter(export_seconds=0.2)
    agent, tracer = _build(exporter)
    try:
        t0 = time.monotonic()
        async with trace(tracer, agent):
            _queue_span(tracer)
        assert time.monotonic() - t0 >= 0.2
        assert exporter.export_count >= 1
    finally:
        await tracer.shutdown()
