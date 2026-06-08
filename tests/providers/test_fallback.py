from __future__ import annotations

from typing import Any

import pytest

from cubepi.errors import (
    ContextLengthExceeded,
    ProviderAuthFailed,
    ProviderBadRequest,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
)
from cubepi.providers.base import (
    AssistantMessage,
    BaseProvider,
    BoundModel,
    Message,
    MessageStream,
    Model,
    StreamEvent,
    StreamOptions,
    TextContent,
    ThinkingBudgets,
    ThinkingLevel,
    ToolDefinition,
    UserMessage,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message
from cubepi.providers.fallback import DEFAULT_TRIGGER_ERRORS, FallbackBoundModel


class _RaisingProvider(BaseProvider):
    """Provider that raises a given exception unconditionally from stream() and generate()."""

    def __init__(self, error: ProviderError) -> None:
        super().__init__(provider_id=error.provider or "raising")
        self._error = error

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        raise self._error

    async def generate(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking: ThinkingLevel | None = None,
        thinking_budgets: ThinkingBudgets | None = None,
    ) -> AssistantMessage:
        raise self._error


def _faux(provider_id: str = "faux", response: str | None = None) -> BoundModel:
    p = FauxProvider(provider_id=provider_id)
    if response is not None:
        p.set_responses([faux_assistant_message(response)])
    return p.model("model-1")


def _raising(error: ProviderError, model_id: str = "model-1") -> BoundModel:
    p = _RaisingProvider(error)
    return BoundModel(provider=p, spec=Model(id=model_id, provider_id=p.provider_id))


def _messages() -> list[Message]:
    return [UserMessage(content=[TextContent(text="hi")])]


# ---------------------------------------------------------------------------
# DEFAULT_TRIGGER_ERRORS tests
# ---------------------------------------------------------------------------


def test_default_trigger_errors_composition() -> None:
    """DEFAULT_TRIGGER_ERRORS contains the right three error types."""
    assert RateLimited in DEFAULT_TRIGGER_ERRORS
    assert ProviderUnavailable in DEFAULT_TRIGGER_ERRORS
    assert ContextLengthExceeded in DEFAULT_TRIGGER_ERRORS
    assert ProviderAuthFailed not in DEFAULT_TRIGGER_ERRORS
    assert ProviderBadRequest not in DEFAULT_TRIGGER_ERRORS


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_primary_succeeds() -> None:
    """Primary succeeds — returns its stream, no failover."""
    primary = _faux("primary", "hello")
    fallback = _faux("fallback", "world")
    fbm = FallbackBoundModel(chain=(primary, fallback))

    stream = await fbm.stream(_messages())
    events = [ev.type async for ev in stream]
    result = await stream.result()

    assert "done" in events
    assert result.provider_id == "primary"
    # fallback provider was never used
    assert fallback.provider.call_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stream_primary_raises_trigger_error_fallback_succeeds() -> None:
    """Primary raises RateLimited → failover to second model, on_failover called."""
    rate_err = RateLimited("429", provider="primary", model="model-1")
    primary = _raising(rate_err)
    fallback = _faux("fallback", "ok")

    failover_calls: list[tuple[BoundModel, BoundModel | None, Any]] = []

    async def _cb(failed: BoundModel, nxt: BoundModel | None, err: Any) -> None:
        failover_calls.append((failed, nxt, err))

    fbm = FallbackBoundModel(chain=(primary, fallback), on_failover=_cb)

    stream = await fbm.stream(_messages())
    result = await stream.result()

    assert result.provider_id == "fallback"
    assert len(failover_calls) == 1
    assert failover_calls[0][0] is primary
    assert failover_calls[0][1] is fallback
    assert isinstance(failover_calls[0][2], RateLimited)


@pytest.mark.asyncio
async def test_stream_primary_raises_non_trigger_error_reraises() -> None:
    """Primary raises ProviderBadRequest (not in trigger_errors) → re-raised, fallback not tried."""
    bad_req = ProviderBadRequest("400", provider="primary", model="model-1")
    primary = _raising(bad_req)
    fallback = _faux("fallback", "ok")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    with pytest.raises(ProviderBadRequest):
        await fbm.stream(_messages())

    assert fallback.provider.call_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stream_primary_first_event_error_fallback_succeeds() -> None:
    """Primary emits error as first StreamEvent → fallback to second model."""
    # FauxProvider with no responses queued emits StreamEvent(type="error") as first event.
    primary_prov = FauxProvider(provider_id="primary")
    primary = primary_prov.model("model-1")
    fallback = _faux("fallback", "rescued")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    stream = await fbm.stream(_messages())
    result = await stream.result()

    assert result.provider_id == "fallback"


@pytest.mark.asyncio
async def test_stream_all_exhausted_raises_provider_unavailable() -> None:
    """All models in chain fail → raises ProviderUnavailable."""
    err = RateLimited("429", provider="p", model="m")
    fbm = FallbackBoundModel(
        chain=(_raising(err, "m1"), _raising(err, "m2"), _raising(err, "m3"))
    )

    with pytest.raises(ProviderUnavailable, match="all providers exhausted"):
        await fbm.stream(_messages())


@pytest.mark.asyncio
async def test_stream_on_failover_callback_raises_is_swallowed() -> None:
    """on_failover callback that raises must not abort the failover."""
    rate_err = RateLimited("429", provider="primary", model="model-1")
    primary = _raising(rate_err)
    fallback = _faux("fallback", "ok")

    async def _bad_cb(failed: BoundModel, nxt: BoundModel | None, err: Any) -> None:
        raise RuntimeError("callback is broken")

    fbm = FallbackBoundModel(chain=(primary, fallback), on_failover=_bad_cb)

    stream = await fbm.stream(_messages())
    result = await stream.result()

    assert result.provider_id == "fallback"


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_primary_raises_trigger_error_fallback_succeeds() -> None:
    """generate() — primary raises RateLimited, fallback returns AssistantMessage."""
    rate_err = RateLimited("429", provider="primary", model="model-1")
    primary = _raising(rate_err)
    fallback = _faux("fallback", "generated")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    result = await fbm.generate(_messages())

    assert result.provider_id == "fallback"


# ---------------------------------------------------------------------------
# Custom trigger_errors tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_trigger_errors_includes_auth_failed() -> None:
    """Custom trigger_errors that includes ProviderAuthFailed → auth failure triggers failover."""
    auth_err = ProviderAuthFailed("401", provider="primary", model="model-1")
    primary = _raising(auth_err)
    fallback = _faux("fallback", "ok")

    fbm = FallbackBoundModel(
        chain=(primary, fallback),
        trigger_errors=frozenset({ProviderAuthFailed}),
    )

    stream = await fbm.stream(_messages())
    result = await stream.result()

    assert result.provider_id == "fallback"


# ---------------------------------------------------------------------------
# Coverage gap tests
# ---------------------------------------------------------------------------


def test_provider_and_spec_properties() -> None:
    """provider and spec proxy chain[0]."""
    primary = _faux("primary", "hello")
    fallback = _faux("fallback", "world")
    fbm = FallbackBoundModel(chain=(primary, fallback))

    assert fbm.provider is primary.provider
    assert fbm.spec is primary.spec


class _EmptyStreamProvider(BaseProvider):
    """Returns a stream that immediately terminates with no events (StopAsyncIteration path)."""

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        ms = MessageStream()

        async def _produce() -> None:
            raise RuntimeError("empty stream — no events emitted")

        ms.attach_task(__import__("asyncio").create_task(_produce()))
        return ms


@pytest.mark.asyncio
async def test_stream_empty_stream_triggers_failover() -> None:
    """Stream that terminates before emitting any event → StopAsyncIteration path → failover."""
    primary_prov = _EmptyStreamProvider()
    primary_prov.provider_id = "empty"
    primary = BoundModel(provider=primary_prov, spec=Model(id="m", provider_id="empty"))
    fallback = _faux("fallback", "recovered")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    stream = await fbm.stream(_messages())
    result = await stream.result()

    assert result.provider_id == "fallback"


class _MidStreamErrorProvider(BaseProvider):
    """Emits one start event then the producer task raises — exercises _forward's except handler."""

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        ms = MessageStream()

        async def _produce() -> None:
            ms.push(StreamEvent(type="start"))
            raise RuntimeError("mid-stream failure")

        ms.attach_task(__import__("asyncio").create_task(_produce()))
        return ms


@pytest.mark.asyncio
async def test_stream_mid_stream_error_is_forwarded() -> None:
    """Error after the first non-error event is forwarded as-is (no retry)."""
    prov = _MidStreamErrorProvider()
    prov.provider_id = "midstream"
    bound = BoundModel(provider=prov, spec=Model(id="m", provider_id="midstream"))
    fbm = FallbackBoundModel(chain=(bound,))

    stream = await fbm.stream(_messages())
    events = [ev.type async for ev in stream]
    result = await stream.result()

    assert "start" in events
    assert "error" in events
    assert result.stop_reason == "error"


@pytest.mark.asyncio
async def test_generate_non_trigger_error_reraises() -> None:
    """generate() — ProviderBadRequest (not in trigger_errors) re-raised immediately."""
    bad_req = ProviderBadRequest("400", provider="primary", model="model-1")
    primary = _raising(bad_req)
    fallback = _faux("fallback", "ok")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    with pytest.raises(ProviderBadRequest):
        await fbm.generate(_messages())


@pytest.mark.asyncio
async def test_generate_all_exhausted_raises_provider_unavailable() -> None:
    """generate() — all models fail → raises ProviderUnavailable."""
    err = RateLimited("429", provider="p", model="m")
    fbm = FallbackBoundModel(chain=(_raising(err, "m1"), _raising(err, "m2")))

    with pytest.raises(ProviderUnavailable, match="all providers exhausted"):
        await fbm.generate(_messages())


@pytest.mark.asyncio
async def test_generate_error_assistant_message_triggers_failover() -> None:
    """generate() — primary returns AssistantMessage(stop_reason="error") → failover."""
    # FauxProvider with no queued responses returns an error AssistantMessage.
    primary_prov = FauxProvider(provider_id="primary")
    primary = primary_prov.model("model-1")
    fallback = _faux("fallback", "generated via fallback")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    result = await fbm.generate(_messages())

    assert result.provider_id == "fallback"
    assert result.stop_reason != "error"
