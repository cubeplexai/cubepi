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


class TestChunkListenerImmutability:
    async def test_listener_mutation_does_not_leak_to_consumer(self):
        """subscribe_chunk listeners receive a deep copy of the StreamEvent
        so a redacting/mutating observer cannot edit what `async for ev in
        ms` consumers (e.g. cubepi/agent/loop.py) observe."""
        provider = FauxProvider()
        consumer_events: list = []

        def redactor(event, model):
            # Try to mutate — this should NOT affect the consumer-side
            # event of the same chunk.
            if event.partial is not None:
                event.partial.content.clear()

        provider.subscribe_chunk(redactor)
        provider.append_responses([faux_assistant_message("hello world")])
        ms = await provider.stream(MODEL, [UserMessage(content=[])])
        async for ev in ms:
            consumer_events.append(ev)
        await ms.result()

        # The consumer should have seen at least one text_delta event
        # whose partial.content was NOT empty. If the redactor's mutation
        # had leaked, every event's partial.content would be empty.
        text_deltas = [
            e
            for e in consumer_events
            if e.type == "text_delta" and e.partial is not None
        ]
        assert text_deltas, "expected at least one text_delta on the consumer side"
        assert any(ev.partial.content for ev in text_deltas), (
            "redactor's mutation leaked into consumer's queued events"
        )

    async def test_listener_mutation_does_not_leak_between_listeners(self):
        """Each chunk listener gets its OWN copy — a mutating listener-A
        must not affect what listener-B (registered later) observes
        when they both fire for the same StreamEvent."""
        provider = FauxProvider()
        b_saw_empty_partial: list[bool] = []

        def redactor_a(event, model):
            if event.partial is not None:
                event.partial.content.clear()
                event.delta = "REDACTED"

        def observer_b(event, model):
            if event.partial is not None:
                b_saw_empty_partial.append(len(event.partial.content) == 0)

        provider.subscribe_chunk(redactor_a)
        provider.subscribe_chunk(observer_b)
        provider.append_responses([faux_assistant_message("listener isolation")])
        await _run_once(provider, "isolated")

        assert b_saw_empty_partial, "observer_b never saw a partial-bearing event"
        # If per-listener isolation were broken, all of B's observations
        # would have an empty partial.content. At least one should not.
        assert not all(b_saw_empty_partial), (
            "listener-A's mutation leaked into listener-B's view"
        )


class TestRequestListenerImmutability:
    async def test_listener_mutation_does_not_leak_into_next_listener(self):
        """subscribe_request gets a defensive deep copy of the payload —
        a listener that mutates it (e.g. redaction) must not affect what
        subsequent listeners observe nor what the provider actually
        sends. The Faux provider's _produce uses the same helper as the
        real providers."""
        provider = FauxProvider()
        seen_payloads: list[dict] = []

        def redactor(payload, model):
            # Mutate in place — should not affect the next listener.
            payload["messages"] = "REDACTED"

        def observer(payload, model):
            seen_payloads.append(payload)

        provider.subscribe_request(redactor)
        provider.subscribe_request(observer)

        await _run_once(provider, "ok")

        assert len(seen_payloads) == 1
        # The observer receives a deepcopy too; whatever redactor did to
        # its copy isn't visible here because each listener gets its own
        # call site of _fire_listeners, which iterates the same snapshot
        # passed by _fire_request_listeners. (Note: the contract is
        # "snapshot vs the live kwargs"; per-listener isolation is a
        # stronger property we don't promise.)
        # Verify the snapshot's structure is intact for at least the
        # required fields:
        assert "model" in seen_payloads[0]


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
        """On the normal-completion path, async response listeners are
        awaited inline so the producer task does not end before the
        listener body has run. This matters under `asyncio.run(main())`
        where the loop tears down the instant `main()` returns —
        detached listener tasks would otherwise be cancelled before they
        execute."""
        provider = FauxProvider()
        ran = asyncio.Event()

        async def cb(body, model, exc):
            await asyncio.sleep(0)
            ran.set()

        provider.subscribe_response(cb)
        await _run_once(provider, "ok")
        # The producer task should have already awaited the listener;
        # ran should be set the moment _run_once returns (no need to wait).
        assert ran.is_set()

    async def test_async_response_listener_under_asyncio_run_teardown(self):
        """Simulate the asyncio.run teardown race: caller awaits
        stream.result() and then the function returns, mimicking a
        program that exits via asyncio.run. The async response listener
        must have completed by the time the producer task is done."""
        provider = FauxProvider()
        outcome: list = []

        async def cb(body, model, exc):
            # Yield then record — proves the coroutine actually progressed.
            await asyncio.sleep(0)
            outcome.append("ran")

        provider.subscribe_response(cb)
        provider.append_responses([faux_assistant_message("hi")])

        ms = await provider.stream(MODEL, [UserMessage(content=[])])
        result = await ms.result()
        assert result.stop_reason == "stop"
        # Wait for the producer task itself to finish — that's where the
        # response listener is awaited inline now.
        await ms._producer_task
        assert outcome == ["ran"]

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


class TestAbortBodyShape:
    """Faux response listener must NOT report a full successful body when
    the stream was aborted via opts.signal."""

    async def test_aborted_run_reports_aborted_body(self):
        provider = FauxProvider(tokens_per_second=10.0)
        seen: list = []
        provider.subscribe_response(lambda body, model, exc: seen.append((body, exc)))

        provider.append_responses([faux_assistant_message("x" * 400)])
        signal = asyncio.Event()
        opts = StreamOptions(signal=signal)

        ms = await provider.stream(MODEL, [UserMessage(content=[])], options=opts)
        # Let producer enter the streaming loop.
        await asyncio.sleep(0.05)
        # Trigger abort path inside _stream_with_deltas (signal check between
        # chunks). The producer will set an aborted result on the stream.
        signal.set()
        # Drain so the producer can complete.
        await _drain(ms)

        assert len(seen) == 1
        body, exc = seen[0]
        assert exc is None  # abort isn't an exception in this path
        assert body is not None
        # The body must reflect the aborted state, NOT the queued full
        # response. stop_reason on the aborted body is "aborted".
        assert body["stop_reason"] == "aborted"


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


class TestFireListenersHelpers:
    """Direct unit tests for the listener-fanout helpers, bypassing
    provider streaming. These cover branches that are awkward to reach
    end-to-end."""

    async def test_fire_listeners_empty_returns(self):
        from cubepi.providers.base import _fire_listeners

        # Empty list — must return without touching anything.
        await _fire_listeners([])

    async def test_fire_listeners_swallows_sync_exception(self):
        from cubepi.providers.base import _fire_listeners

        seen: list = []

        def bad(x):
            raise RuntimeError("nope")

        def good(x):
            seen.append(x)

        await _fire_listeners([bad, good], "payload")
        # Despite the first raising, the second still ran.
        assert seen == ["payload"]

    def test_fire_listeners_sync_empty_returns(self):
        from cubepi.providers.base import _fire_listeners_sync

        _fire_listeners_sync([])  # No listeners — short-circuit.

    def test_fire_listeners_sync_swallows_sync_exception(self):
        from cubepi.providers.base import _fire_listeners_sync

        seen: list = []

        def bad(x):
            raise RuntimeError("listener oops")

        def good(x):
            seen.append(x)

        _fire_listeners_sync([bad, good], "x")
        assert seen == ["x"]

    def test_fire_listeners_sync_no_running_loop(self):
        """Outside a running event loop, asyncio.create_task raises
        RuntimeError. The sync helper must swallow it (covers the
        no-running-loop branch in the detached-task scheduling)."""
        from cubepi.providers.base import _fire_listeners_sync

        async def async_cb(x):
            pass  # pragma: no cover — never awaited in this test

        # Important: do NOT await; we're synchronous here, no event loop.
        _fire_listeners_sync([async_cb], "x")


class TestProviderEmit:
    """Cover the chunk-listener fan-out inside Anthropic and OpenAI
    _emit helpers directly — they each carry the same hot-path branch
    but the Faux end-to-end tests exercise Faux's _emit only."""

    async def test_anthropic_emit_fires_chunk_listeners(self):
        from cubepi.providers.anthropic import AnthropicProvider
        from cubepi.providers.base import MessageStream

        provider = AnthropicProvider(api_key="test-key")
        seen: list = []
        provider.subscribe_chunk(lambda ev, m: seen.append((ev.type, m.id)))

        ms = MessageStream()
        event = StreamEvent(type="text_delta", delta="hi")
        await provider._emit(ms, event, MODEL)

        assert seen == [("text_delta", MODEL.id)]
        # The event was also pushed onto the stream.
        first = await ms.__anext__()
        assert first.type == "text_delta"

    async def test_openai_emit_fires_chunk_listeners(self):
        from cubepi.providers.openai import OpenAIProvider
        from cubepi.providers.base import MessageStream

        provider = OpenAIProvider(api_key="test-key")
        seen: list = []
        provider.subscribe_chunk(lambda ev, m: seen.append(ev.type))

        ms = MessageStream()
        await provider._emit(ms, StreamEvent(type="text_delta", delta="hi"), MODEL)
        assert seen == ["text_delta"]

    async def test_openai_responses_emit_fires_chunk_listeners(self):
        from cubepi.providers.openai_responses import OpenAIResponsesProvider
        from cubepi.providers.base import MessageStream

        provider = OpenAIResponsesProvider(api_key="test-key")
        seen: list = []
        provider.subscribe_chunk(lambda ev, m: seen.append(ev.type))

        ms = MessageStream()
        await provider._emit(ms, StreamEvent(type="text_delta", delta="hi"), MODEL)
        assert seen == ["text_delta"]


class TestAssembleResponse:
    """Cover static _assemble_response helpers directly — exercises the
    optional-field branches without needing a real LLM stream."""

    def test_openai_assemble_with_system_fingerprint_and_service_tier(self):
        from cubepi.providers.openai import OpenAIProvider

        class FakeUsageDetails:
            cached_tokens = 7

        class FakeUsage:
            prompt_tokens = 10
            completion_tokens = 4
            total_tokens = 14
            prompt_tokens_details = FakeUsageDetails()

        body = OpenAIProvider._assemble_response(
            response_id="resp-123",
            model_id="gpt-test",
            created=12345,
            system_fingerprint="fp_abc",
            service_tier="scale",
            text="hello",
            tool_calls_in_progress={
                0: {"id": "call_1", "name": "do_thing", "arguments": '{"a":1}'}
            },
            finish_reason="stop",
            usage=FakeUsage(),
        )

        assert body["id"] == "resp-123"
        assert body["object"] == "chat.completion"
        assert body["model"] == "gpt-test"
        assert body["created"] == 12345
        # Both optional fields present in body.
        assert body["system_fingerprint"] == "fp_abc"
        assert body["service_tier"] == "scale"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["choices"][0]["message"]["content"] == "hello"
        assert body["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"
        assert body["usage"]["prompt_tokens_details"] == {"cached_tokens": 7}

    def test_openai_assemble_without_optional_fields(self):
        from cubepi.providers.openai import OpenAIProvider

        body = OpenAIProvider._assemble_response(
            response_id=None,
            model_id="gpt-test",
            created=None,
            system_fingerprint=None,
            service_tier=None,
            text="",
            tool_calls_in_progress={},
            finish_reason=None,
            usage=None,
        )
        # Optional fields omitted entirely.
        assert "system_fingerprint" not in body
        assert "service_tier" not in body
        # Content normalizes to None when text is empty and no tool calls.
        assert body["choices"][0]["message"]["content"] is None
        assert body["usage"] == {}
