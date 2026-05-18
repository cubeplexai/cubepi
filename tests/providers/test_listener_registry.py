"""Pin the persistent Provider listener registry contract.

These tests don't depend on any real LLM provider — they use FauxProvider
which inherits from BaseProvider with the same listener wiring as the
production providers.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from cubepi.providers.base import Model, StreamEvent, StreamOptions, UserMessage
from cubepi.providers.faux import FauxProvider, faux_assistant_message


MODEL = Model(id="faux-1", provider="faux")


async def _drain(stream) -> None:
    """Consume every event from a stream until done; return final result."""
    async for _ in stream:
        pass
    return await stream.result()


async def _run_once(provider: FauxProvider, response: str = "hello") -> None:
    provider.append_responses([faux_assistant_message(response)])
    ms = await provider.stream(MODEL, [UserMessage(content=[])])
    await _drain(ms)


class TestSubscribeAndFire:
    async def test_each_listener_type_fires(self):
        provider = FauxProvider()
        req_seen: list[tuple] = []
        chunk_seen: list[tuple] = []
        resp_seen: list[tuple] = []

        provider.subscribe_request(
            lambda payload, model: req_seen.append((payload, model))
        )
        provider.subscribe_chunk(lambda event, model: chunk_seen.append((event, model)))
        provider.subscribe_response(
            lambda body, model, exc: resp_seen.append((body, model, exc))
        )

        await _run_once(provider, "hi")

        assert len(req_seen) == 1
        payload, model = req_seen[0]
        assert isinstance(payload, dict)
        assert payload["model"] == MODEL.id
        assert model is MODEL

        # At least start, text_start, one delta, text_end, done.
        assert len(chunk_seen) >= 4
        assert all(isinstance(ev, StreamEvent) for ev, _ in chunk_seen)
        assert all(m is MODEL for _, m in chunk_seen)
        assert chunk_seen[0][0].type == "start"
        assert chunk_seen[-1][0].type == "done"

        assert len(resp_seen) == 1
        body, model, exc = resp_seen[0]
        assert exc is None
        assert model is MODEL
        assert body is not None
        # Faux body is deterministic — pin schema:
        assert body["id"] == "faux-1"
        assert body["model"] == MODEL.id
        assert body["role"] == "assistant"
        assert body["content"] == [{"type": "text", "text": "hi"}]
        assert body["stop_reason"] == "stop"


class TestDetach:
    async def test_detach_stops_invocations(self):
        provider = FauxProvider()
        seen: list = []
        detach = provider.subscribe_request(lambda payload, model: seen.append(payload))

        await _run_once(provider)
        assert len(seen) == 1

        detach()
        await _run_once(provider)
        assert len(seen) == 1


class TestMultipleSubscribers:
    async def test_registration_order_preserved(self):
        provider = FauxProvider()
        order: list[str] = []
        provider.subscribe_request(lambda p, m: order.append("first"))
        provider.subscribe_request(lambda p, m: order.append("second"))
        provider.subscribe_request(lambda p, m: order.append("third"))

        await _run_once(provider)
        assert order == ["first", "second", "third"]


class TestExceptionIsolation:
    async def test_raising_listener_does_not_crash_stream(self):
        provider = FauxProvider()

        def bad(payload, model):
            raise RuntimeError("listener bomb")

        seen: list = []
        provider.subscribe_request(bad)
        provider.subscribe_request(lambda p, m: seen.append("after-bomb"))

        # Stream must complete normally.
        provider.append_responses([faux_assistant_message("ok")])
        ms = await provider.stream(MODEL, [UserMessage(content=[])])
        result = await _drain(ms)
        assert result.stop_reason == "stop"

        # Second listener still fires despite the first raising.
        assert seen == ["after-bomb"]


class TestResponseListenerExactlyOnce:
    async def test_normal_completion(self):
        provider = FauxProvider()
        seen: list[tuple] = []
        provider.subscribe_response(lambda body, model, exc: seen.append((body, exc)))
        await _run_once(provider, "done")
        assert len(seen) == 1
        body, exc = seen[0]
        assert exc is None
        assert body is not None
        assert body["stop_reason"] == "stop"

    async def test_exception_path(self):
        provider = FauxProvider()
        seen: list[tuple] = []
        provider.subscribe_response(lambda body, model, exc: seen.append((body, exc)))

        async def boom(messages, model, system_prompt, tools):
            raise RuntimeError("provider boom")

        provider.append_responses([boom])
        ms = await provider.stream(MODEL, [UserMessage(content=[])])
        await _drain(ms)

        assert len(seen) == 1
        body, exc = seen[0]
        assert isinstance(exc, RuntimeError)
        assert "provider boom" in str(exc)

    async def test_cancellation_path(self):
        """When the producer task is cancelled mid-stream, subscribe_response
        must still fire exactly once with a CancelledError. (A cancel issued
        before the producer task has begun running is a no-op per asyncio
        semantics — the coroutine body never executes — and is not the
        observability contract we're guaranteeing.)"""
        provider = FauxProvider(tokens_per_second=10.0)  # slow chunking
        seen: list[tuple] = []
        provider.subscribe_response(
            lambda body, model, exc: seen.append((body, type(exc) if exc else None))
        )

        # Long content so the producer is mid-stream when we cancel.
        provider.append_responses([faux_assistant_message("x" * 400)])
        ms = await provider.stream(MODEL, [UserMessage(content=[])])

        # Let the producer reach at least one await before cancelling.
        # Without this, cancel() races the task scheduler — see asyncio
        # semantics: a cancel before the first await skips the body.
        await asyncio.sleep(0.05)

        assert ms._producer_task is not None
        ms._producer_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ms._producer_task

        # Listener must have fired exactly once and seen CancelledError.
        assert len(seen) == 1
        body, exc_type = seen[0]
        assert exc_type is asyncio.CancelledError


class TestPayloadOrdering:
    async def test_on_payload_mutation_visible_to_request_listener(self):
        """StreamOptions.on_payload mutates the payload; the persistent
        subscribe_request listener fires AFTER that mutation, so it sees
        the final wire payload."""
        provider = FauxProvider()

        async def mutator(payload, model):
            new = dict(payload)
            new["mutated_by_on_payload"] = True
            return new

        seen: list[dict] = []
        provider.subscribe_request(lambda payload, model: seen.append(payload))

        provider.append_responses([faux_assistant_message("ok")])
        opts = StreamOptions(on_payload=mutator)
        ms = await provider.stream(MODEL, [UserMessage(content=[])], options=opts)
        await _drain(ms)

        assert len(seen) == 1
        assert seen[0].get("mutated_by_on_payload") is True


class TestConcurrentStreams:
    async def test_concurrent_streams_share_listeners(self):
        provider = FauxProvider()
        responses_seen: list[tuple] = []
        provider.subscribe_response(
            lambda body, model, exc: responses_seen.append((body["id"], model.id))
        )

        provider.append_responses(
            [
                faux_assistant_message("a"),
                faux_assistant_message("b"),
            ]
        )
        model_a = Model(id="faux-a", provider="faux")
        model_b = Model(id="faux-b", provider="faux")

        ms_a, ms_b = await asyncio.gather(
            provider.stream(model_a, [UserMessage(content=[])]),
            provider.stream(model_b, [UserMessage(content=[])]),
        )
        await asyncio.gather(_drain(ms_a), _drain(ms_b))

        assert len(responses_seen) == 2
        ids = {r[0] for r in responses_seen}
        # seq counter increments per call: faux-1, faux-2.
        assert ids == {"faux-1", "faux-2"}
        models = {r[1] for r in responses_seen}
        assert models == {"faux-a", "faux-b"}


class TestSelfDetachMidIteration:
    async def test_listener_can_detach_itself_during_iteration(self):
        """A listener detaching itself mid-stream must not skip the next
        listener (snapshot semantics via tuple(listeners) in _fire_listeners).
        """
        provider = FauxProvider()
        fires = {"first": 0, "second": 0}

        detach_holder: list = [None]

        def first(event, model):
            fires["first"] += 1
            # After the first invocation, detach self.
            if fires["first"] == 1 and detach_holder[0] is not None:
                detach_holder[0]()

        def second(event, model):
            fires["second"] += 1

        detach_holder[0] = provider.subscribe_chunk(first)
        provider.subscribe_chunk(second)

        await _run_once(provider, "abc")

        # First fired exactly once (it detached itself); second fired
        # on the same first chunk AND subsequent chunks.
        assert fires["first"] == 1
        assert fires["second"] >= 2


class TestMidStreamSubscription:
    async def test_listener_subscribed_inside_a_listener_fires_on_next_stream(self):
        """A listener registered while another listener is mid-execution
        starts firing on the NEXT stream call, not retroactively on the
        same one."""
        provider = FauxProvider()
        late: list[int] = []
        first_seen: list[int] = []

        def first(body, model, exc):
            first_seen.append(1)
            provider.subscribe_response(lambda body, model, exc: late.append(1))

        provider.subscribe_response(first)

        await _run_once(provider, "first")
        assert first_seen == [1]
        assert late == []  # Not retroactive.

        await _run_once(provider, "second")
        # Both `first` (still subscribed) and the late listener fire on the
        # second stream.
        assert first_seen == [1, 1]
        assert len(late) == 1


class TestSlowListenerBlocks:
    async def test_slow_async_listener_serializes_stream(self):
        """Listeners run inline in the producer coroutine; a slow listener
        delays subsequent chunks. This documents the contract."""
        provider = FauxProvider()
        per_chunk_sleep = 0.02
        n_chunks = 0

        async def slow(event, model):
            nonlocal n_chunks
            n_chunks += 1
            await asyncio.sleep(per_chunk_sleep)

        provider.subscribe_chunk(slow)
        provider.append_responses([faux_assistant_message("hello there friend")])

        start = time.monotonic()
        ms = await provider.stream(MODEL, [UserMessage(content=[])])
        await _drain(ms)
        elapsed = time.monotonic() - start

        # We expect at least n_chunks * per_chunk_sleep elapsed wall time.
        # n_chunks should be >= 4 (start, text_start, deltas..., text_end, done).
        assert n_chunks >= 4
        assert elapsed >= n_chunks * per_chunk_sleep * 0.8  # 20% leeway


class TestAsyncListeners:
    async def test_async_request_listener_awaited(self):
        provider = FauxProvider()
        seen: list = []

        async def cb(payload, model):
            await asyncio.sleep(0)
            seen.append(payload)

        provider.subscribe_request(cb)
        await _run_once(provider, "ok")
        assert len(seen) == 1

    async def test_async_response_listener_runs_through_normal_completion(self):
        """In the normal-completion path, async response listeners are
        scheduled as detached tasks by _fire_listeners_sync. Verify the
        coroutine body actually runs."""
        provider = FauxProvider()
        ran = asyncio.Event()

        async def cb(body, model, exc):
            await asyncio.sleep(0)
            ran.set()

        provider.subscribe_response(cb)
        await _run_once(provider, "ok")
        # Give the detached task one tick to complete.
        await asyncio.wait_for(ran.wait(), timeout=1.0)
        assert ran.is_set()

    async def test_async_response_listener_exception_does_not_bubble(self):
        """Per the _fire_listeners_sync contract, async listener exceptions
        are wrapped in _safe_run_coroutine and swallowed. This must not
        surface as an asyncio unhandled-task-exception warning."""
        provider = FauxProvider()
        bombed = asyncio.Event()

        async def bomb(body, model, exc):
            bombed.set()
            raise RuntimeError("async listener bomb")

        provider.subscribe_response(bomb)
        # Should complete normally despite the listener exception.
        await _run_once(provider, "ok")
        # Wait for the detached task to actually run.
        await asyncio.wait_for(bombed.wait(), timeout=1.0)
        # Yield once more so the inner coroutine has a chance to raise
        # and be caught by _safe_run_coroutine.
        await asyncio.sleep(0.01)


class TestBaseProvider:
    """Cover BaseProvider sentinels not exercised by Faux."""

    async def test_stream_raises_not_implemented(self):
        from cubepi.providers.base import BaseProvider

        class Empty(BaseProvider):
            pass

        with pytest.raises(NotImplementedError):
            await Empty().stream(MODEL, [UserMessage(content=[])])

    def test_detach_is_idempotent(self):
        provider = FauxProvider()
        detach = provider.subscribe_request(lambda p, m: None)
        detach()
        # Calling detach again must not raise even though the listener is
        # already gone — covers the ValueError swallow in _detach.
        detach()
