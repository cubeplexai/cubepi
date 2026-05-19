"""OpenAIResponsesProvider — capability descriptor wiring.

Mirrors tests/providers/test_openai_capability.py but adapted for the
Responses API surface (client.responses.create + max_output_tokens).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cubepi.providers.base import (
    Model,
    StreamOptions,
    TextContent,
    ThinkingLevel,
    UserMessage,
)
from cubepi.providers.capability import (
    CapabilityDescriptor,
    ReasoningLevelSpec,
    TemperatureSpec,
)
from cubepi.providers.openai_responses import OpenAIResponsesProvider


def _model(id: str = "gpt-5-test", *, reasoning: bool = False, **kw) -> Model:
    return Model(
        id=id,
        provider="test",
        api="openai-responses",
        reasoning=reasoning,
        context_window=kw.get("context_window", 200_000),
        max_tokens=kw.get("max_tokens", 16384),
        temperature=kw.get("temperature", 1.0),
    )


async def _async_iter(events):
    for event in events:
        yield event


async def _capture_payload_responses(
    provider: OpenAIResponsesProvider,
    model: Model,
    *,
    on_payload=None,
    thinking: ThinkingLevel = "off",
) -> dict:
    """Run stream and return final wire kwargs via request_listeners."""
    captured: dict = {}

    async def listener(kwargs: dict, m: Model) -> None:
        captured.update(kwargs)

    provider._request_listeners.append(listener)

    # Empty event iterator — we only need request kwargs to be captured.
    mock_responses = MagicMock()
    mock_responses.create = AsyncMock(return_value=_async_iter([]))
    provider._client.responses = mock_responses  # type: ignore[assignment]

    stream = await provider.stream(
        model=model,
        messages=[UserMessage(content=[TextContent(text="hi")])],
        options=StreamOptions(thinking=thinking, on_payload=on_payload),
    )
    async for _ in stream:
        pass
    return captured


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


def test_provider_accepts_capability_kwarg():
    cap = CapabilityDescriptor(temperature=TemperatureSpec(mode="ignored"))
    p = OpenAIResponsesProvider(api_key="x", capability=cap)
    assert p._capability is cap


def test_provider_accepts_model_overrides():
    cap = CapabilityDescriptor()
    overrides = {
        "gpt-5": CapabilityDescriptor(reasoning_off_payload={"reasoning": {"effort": "low"}})
    }
    p = OpenAIResponsesProvider(
        api_key="x",
        capability=cap,
        model_capability_overrides=overrides,
    )
    assert p._model_overrides == overrides


def test_resolve_capability_uses_override_when_present():
    base = CapabilityDescriptor()
    override = CapabilityDescriptor(reasoning_off_payload={"reasoning": {"effort": "low"}})
    p = OpenAIResponsesProvider(
        api_key="x",
        capability=base,
        model_capability_overrides={"gpt-5": override},
    )
    assert p._resolve_capability("gpt-5") is override
    assert p._resolve_capability("o3") is base


def test_capability_default_when_kwarg_none():
    p = OpenAIResponsesProvider(api_key="x")
    assert isinstance(p._capability, CapabilityDescriptor)
    assert p._capability.reasoning_off_payload == {}
    assert p._cap_active is False


def test_cap_active_when_capability_passed():
    p = OpenAIResponsesProvider(api_key="x", capability=CapabilityDescriptor())
    assert p._cap_active is True


def test_cap_active_when_only_overrides_passed():
    p = OpenAIResponsesProvider(
        api_key="x",
        model_capability_overrides={"m": CapabilityDescriptor()},
    )
    assert p._cap_active is True


# ---------------------------------------------------------------------------
# Legacy path (no capability) — _THINKING_TO_EFFORT still fires.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_no_capability_uses_internal_effort_map():
    """When _cap_active=False, the existing _THINKING_TO_EFFORT writes effort
    on reasoning models. This preserves wire bytes for legacy callers."""
    p = OpenAIResponsesProvider(api_key="x")
    payload = await _capture_payload_responses(
        p, _model(reasoning=True), thinking="medium"
    )
    assert payload["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert "reasoning.encrypted_content" in payload["include"]


@pytest.mark.asyncio
async def test_legacy_no_capability_off_thinking_skips_reasoning():
    """Regression guard: thinking='off' -> no reasoning param sent."""
    p = OpenAIResponsesProvider(api_key="x")
    payload = await _capture_payload_responses(
        p, _model(reasoning=True), thinking="off"
    )
    assert "reasoning" not in payload


# ---------------------------------------------------------------------------
# Capability-driven path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capability_reasoning_level_overrides_internal_map():
    """When capability.reasoning_level is set, it takes precedence over the
    legacy map (different effort value confirms it)."""
    cap = CapabilityDescriptor(
        reasoning_on_payload={"reasoning": {"summary": "auto"}},
        reasoning_level=ReasoningLevelSpec(
            path="reasoning.effort",
            kind="effort",
            level_to_effort={"medium": "minimal"},
        ),
    )
    p = OpenAIResponsesProvider(api_key="x", capability=cap)
    payload = await _capture_payload_responses(
        p, _model(reasoning=True), thinking="medium"
    )
    assert payload["reasoning"]["effort"] == "minimal"


@pytest.mark.asyncio
async def test_capability_temperature_ignored_strips():
    cap = CapabilityDescriptor(temperature=TemperatureSpec(mode="ignored"))
    p = OpenAIResponsesProvider(api_key="x", capability=cap)
    payload = await _capture_payload_responses(p, _model(), thinking="off")
    assert "temperature" not in payload


@pytest.mark.asyncio
async def test_capability_temperature_fixed_overwrites():
    cap = CapabilityDescriptor(
        temperature=TemperatureSpec(mode="fixed", fixed_value=0.0)
    )
    p = OpenAIResponsesProvider(api_key="x", capability=cap)
    payload = await _capture_payload_responses(p, _model(temperature=0.7))
    assert payload["temperature"] == 0.0


@pytest.mark.asyncio
async def test_capability_injects_max_output_tokens():
    """Responses API uses max_output_tokens natively; capability path injects
    it via setdefault from model.max_tokens (same pattern as OpenAIProvider)."""
    cap = CapabilityDescriptor()
    p = OpenAIResponsesProvider(api_key="x", capability=cap)
    payload = await _capture_payload_responses(p, _model(max_tokens=2048))
    assert payload["max_output_tokens"] == 2048


@pytest.mark.asyncio
async def test_capability_reasoning_off_payload_merged():
    cap = CapabilityDescriptor(
        reasoning_off_payload={"reasoning": {"effort": "minimal"}},
    )
    p = OpenAIResponsesProvider(api_key="x", capability=cap)
    payload = await _capture_payload_responses(
        p, _model(reasoning=True), thinking="off"
    )
    assert payload["reasoning"] == {"effort": "minimal"}


@pytest.mark.asyncio
async def test_capability_reasoning_on_payload_merged():
    cap = CapabilityDescriptor(
        reasoning_on_payload={"reasoning": {"summary": "auto"}},
    )
    p = OpenAIResponsesProvider(api_key="x", capability=cap)
    payload = await _capture_payload_responses(
        p, _model(reasoning=True), thinking="medium"
    )
    assert payload["reasoning"] == {"summary": "auto"}
