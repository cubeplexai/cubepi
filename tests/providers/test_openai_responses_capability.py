"""OpenAIResponsesProvider — capability descriptor wiring.

Mirrors tests/providers/test_openai_capability.py but adapted for the
Responses API surface (client.responses.create + max_output_tokens).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cubepi.providers.base import (
    Model,
    ReasoningControl,
    StreamOptions,
    TextContent,
    UserMessage,
)
from cubepi.providers.capability import (
    CapabilityDescriptor,
    ReasoningCapability,
    TemperatureSpec,
)
from cubepi.providers.openai_responses import OpenAIResponsesProvider


def _model(id: str = "gpt-5-test", *, reasoning: bool = False, **kw) -> Model:
    return Model(
        id=id,
        provider_id="test",
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
    reasoning: ReasoningControl | None = None,
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
        options=StreamOptions(
            reasoning=reasoning or ReasoningControl(),
            on_payload=on_payload,
        ),
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
        "gpt-5": CapabilityDescriptor(
            reasoning=ReasoningCapability(effort_path="reasoning.effort")
        )
    }
    p = OpenAIResponsesProvider(
        api_key="x",
        capability=cap,
        model_capability_overrides=overrides,
    )
    assert p._model_overrides == overrides


def test_resolve_capability_uses_override_when_present():
    base = CapabilityDescriptor()
    override = CapabilityDescriptor(
        reasoning=ReasoningCapability(effort_path="reasoning.effort")
    )
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
    assert p._capability.reasoning is not None
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
# Default profile path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_profile_writes_reasoning_effort():
    p = OpenAIResponsesProvider(api_key="x")
    payload = await _capture_payload_responses(
        p,
        _model(reasoning=True),
        reasoning=ReasoningControl(mode="on", effort="medium", summary="auto"),
    )
    assert payload["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert "reasoning.encrypted_content" in payload["include"]


@pytest.mark.asyncio
async def test_default_profile_off_writes_minimal_reasoning():
    p = OpenAIResponsesProvider(api_key="x")
    payload = await _capture_payload_responses(
        p,
        _model(reasoning=True),
        reasoning=ReasoningControl(mode="off", effort="minimal", summary="none"),
    )
    assert payload["reasoning"] == {"effort": "minimal"}


# ---------------------------------------------------------------------------
# Capability-driven path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capability_reasoning_effort_overrides_profile_value():
    cap = CapabilityDescriptor(
        reasoning=ReasoningCapability(
            summary_path="reasoning.summary",
            summary_values={"auto": "auto"},
            effort_path="reasoning.effort",
            effort_values={"medium": "minimal"},
        ),
    )
    p = OpenAIResponsesProvider(api_key="x", capability=cap)
    payload = await _capture_payload_responses(
        p,
        _model(reasoning=True),
        reasoning=ReasoningControl(mode="on", effort="medium", summary="auto"),
    )
    assert payload["reasoning"]["effort"] == "minimal"


@pytest.mark.asyncio
async def test_capability_temperature_ignored_strips():
    cap = CapabilityDescriptor(temperature=TemperatureSpec(mode="ignored"))
    p = OpenAIResponsesProvider(api_key="x", capability=cap)
    payload = await _capture_payload_responses(p, _model())
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
async def test_capability_reasoning_mode_off_payload_merged():
    cap = CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={"off": {"reasoning": {"effort": "minimal"}}},
        )
    )
    p = OpenAIResponsesProvider(api_key="x", capability=cap)
    payload = await _capture_payload_responses(
        p,
        _model(reasoning=True),
        reasoning=ReasoningControl(mode="off"),
    )
    assert payload["reasoning"] == {"effort": "minimal"}


@pytest.mark.asyncio
async def test_capability_reasoning_mode_on_payload_merged():
    cap = CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={"on": {"reasoning": {"summary": "auto"}}},
        )
    )
    p = OpenAIResponsesProvider(api_key="x", capability=cap)
    payload = await _capture_payload_responses(
        p,
        _model(reasoning=True),
        reasoning=ReasoningControl(mode="on"),
    )
    assert payload["reasoning"] == {"summary": "auto"}


@pytest.mark.asyncio
async def test_capability_does_not_set_temperature_on_reasoning_model():
    """Reasoning models reject temperature — capability path must skip the setdefault."""
    cap = CapabilityDescriptor()  # default TemperatureSpec(mode="free")
    p = OpenAIResponsesProvider(api_key="x", capability=cap)
    # Model with reasoning=True
    m = Model(
        id="o3-test",
        provider_id="test",
        context_window=200_000,
        max_tokens=16384,
        temperature=1.0,
        reasoning=True,
    )
    payload = await _capture_payload_responses(p, m)
    assert "temperature" not in payload


@pytest.mark.asyncio
async def test_capability_on_payload_overrides_max_output_tokens():
    """on_payload runs before capability; capability uses setdefault → caller wins."""
    cap = CapabilityDescriptor()
    p = OpenAIResponsesProvider(api_key="x", capability=cap)

    async def set_max(kwargs, model):
        kwargs["max_output_tokens"] = 1234
        return kwargs

    payload = await _capture_payload_responses(p, _model(), on_payload=set_max)
    assert payload["max_output_tokens"] == 1234


@pytest.mark.asyncio
async def test_default_profile_preserves_on_payload_max_output_tokens():
    p = OpenAIResponsesProvider(api_key="x")

    async def set_max(kwargs, model):
        kwargs["max_output_tokens"] = 999
        return kwargs

    payload = await _capture_payload_responses(p, _model(), on_payload=set_max)
    assert payload["max_output_tokens"] == 999


@pytest.mark.asyncio
async def test_capability_skips_reasoning_for_non_reasoning_model():
    """Capability path must not write reasoning fields when model.reasoning=False."""
    cap = CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={"on": {"reasoning": {"effort": "low"}}},
            effort_path="reasoning.effort",
            effort_values={"medium": "medium"},
        ),
    )
    p = OpenAIResponsesProvider(api_key="x", capability=cap)
    m = Model(
        id="gpt-4o-test",
        provider_id="test",
        context_window=128000,
        max_tokens=16384,
        temperature=1.0,
        reasoning=False,  # non-reasoning
    )
    payload = await _capture_payload_responses(
        p,
        m,
        reasoning=ReasoningControl(mode="on", effort="medium"),
    )
    assert "reasoning" not in payload
