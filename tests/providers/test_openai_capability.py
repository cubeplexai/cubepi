"""OpenAIProvider — capability descriptor wiring.

Each test constructs a provider with a known capability, then asserts on
the provider's internal state. Stream-time assertions come in Task 6.
"""

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
from cubepi.providers.openai import OpenAIProvider


def test_provider_accepts_capability_kwarg():
    cap = CapabilityDescriptor(temperature=TemperatureSpec(mode="ignored"))
    p = OpenAIProvider(api_key="x", base_url="http://example", capability=cap)
    assert p._capability is cap


def test_provider_accepts_model_overrides():
    cap = CapabilityDescriptor()
    overrides = {
        "deepseek-r1": CapabilityDescriptor(
            reasoning_off_payload={"reasoning": {"effort": "low"}}
        )
    }
    p = OpenAIProvider(
        api_key="x",
        base_url="http://example",
        capability=cap,
        model_capability_overrides=overrides,
    )
    assert p._model_overrides == overrides


def test_resolve_capability_uses_override_when_present():
    base = CapabilityDescriptor()
    override = CapabilityDescriptor(
        reasoning_off_payload={"reasoning": {"effort": "low"}}
    )
    p = OpenAIProvider(
        api_key="x",
        base_url="http://example",
        capability=base,
        model_capability_overrides={"deepseek-r1": override},
    )
    assert p._resolve_capability("deepseek-r1") is override
    assert p._resolve_capability("llama-3") is base


def test_capability_default_when_kwarg_none():
    p = OpenAIProvider(api_key="x", base_url="http://example")
    # No capability passed -> legacy no-op default, _cap_active=False
    assert isinstance(p._capability, CapabilityDescriptor)
    assert p._capability.reasoning_off_payload == {}
    assert p._cap_active is False


def test_cap_active_when_capability_passed():
    p = OpenAIProvider(
        api_key="x", base_url="http://example", capability=CapabilityDescriptor()
    )
    assert p._cap_active is True


def test_cap_active_when_only_overrides_passed():
    p = OpenAIProvider(
        api_key="x",
        base_url="http://example",
        model_capability_overrides={"m": CapabilityDescriptor()},
    )
    assert p._cap_active is True


def test_resolve_capability_returns_default_when_inactive():
    """When no capability kwarg was passed, _resolve_capability still returns
    a safe-default CapabilityDescriptor (not used in practice — gated by _cap_active —
    but the API should be predictable)."""
    p = OpenAIProvider(api_key="x", base_url="http://example")
    result = p._resolve_capability("any-model")
    assert isinstance(result, CapabilityDescriptor)
    assert result.reasoning_off_payload == {}


async def _capture_payload_openai(
    provider: OpenAIProvider,
    model: Model,
    *,
    on_payload=None,
    thinking: ThinkingLevel = "off",
) -> dict:
    """Run a stream with a fake openai client and return the kwargs sent.

    on_payload: optional caller-side payload mutator passed via StreamOptions.
    thinking: ThinkingLevel value for StreamOptions; defaults to "off".
    """
    captured: dict = {}

    async def listener(kwargs: dict, m: Model) -> None:
        captured.update(kwargs)

    provider._request_listeners.append(listener)

    class _FakeResponse:
        response = None

        def __aiter__(self):
            async def gen():
                return
                yield

            return gen()

    async def fake_create(**kw):
        return _FakeResponse()

    provider._client.chat.completions.create = fake_create  # type: ignore[assignment]
    stream = await provider.stream(
        model=model,
        messages=[UserMessage(content=[TextContent(text="hi")])],
        options=StreamOptions(thinking=thinking, on_payload=on_payload),
    )
    async for _ in stream:
        pass
    return captured


def _model(id: str = "test-model", **kw) -> Model:
    return Model(
        id=id,
        provider="test",
        context_window=kw.get("context_window", 32000),
        max_tokens=kw.get("max_tokens", 4096),
        temperature=kw.get("temperature", 0.7),
    )


@pytest.mark.asyncio
async def test_temperature_ignored_strips_field():
    cap = CapabilityDescriptor(temperature=TemperatureSpec(mode="ignored"))
    p = OpenAIProvider(api_key="x", base_url="http://e", capability=cap)
    payload = await _capture_payload_openai(p, _model())
    assert "temperature" not in payload


@pytest.mark.asyncio
async def test_temperature_fixed_overwrites():
    cap = CapabilityDescriptor(
        temperature=TemperatureSpec(mode="fixed", fixed_value=0.0)
    )
    p = OpenAIProvider(api_key="x", base_url="http://e", capability=cap)
    payload = await _capture_payload_openai(p, _model(temperature=0.7))
    assert payload["temperature"] == 0.0


@pytest.mark.asyncio
async def test_max_tokens_field_renamed():
    cap = CapabilityDescriptor(max_tokens_field="max_completion_tokens")
    p = OpenAIProvider(api_key="x", base_url="http://e", capability=cap)
    payload = await _capture_payload_openai(p, _model())
    assert "max_completion_tokens" in payload
    assert "max_tokens" not in payload


@pytest.mark.asyncio
async def test_legacy_no_capability_does_not_inject_temperature_or_max_tokens():
    """Regression guard: no capability passed -> wire bytes identical to today."""
    p = OpenAIProvider(api_key="x", base_url="http://e")  # no capability
    payload = await _capture_payload_openai(p, _model())
    assert "temperature" not in payload
    assert "max_tokens" not in payload


@pytest.mark.asyncio
async def test_temperature_free_preserves_caller_value_via_on_payload():
    """setdefault must not overwrite a temperature the caller set via on_payload.
    Spec §3.5: capability-active path uses setdefault so caller wins."""
    cap = CapabilityDescriptor(
        temperature=TemperatureSpec(mode="free", min=0.0, max=2.0)
    )
    p = OpenAIProvider(api_key="x", base_url="http://e", capability=cap)

    async def set_caller_temp(kwargs, model):
        kwargs["temperature"] = 0.3
        return kwargs

    payload = await _capture_payload_openai(p, _model(), on_payload=set_caller_temp)
    assert payload["temperature"] == 0.3


@pytest.mark.asyncio
async def test_reasoning_off_payload_merged_qwen():
    cap = CapabilityDescriptor(
        reasoning_off_payload={"extra_body": {"enable_thinking": False}},
        reasoning_on_payload={"extra_body": {"enable_thinking": True}},
    )
    p = OpenAIProvider(api_key="x", base_url="http://e", capability=cap)
    payload = await _capture_payload_openai(p, _model())  # default thinking="off"
    assert payload["extra_body"]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_reasoning_on_payload_merged_qwen():
    cap = CapabilityDescriptor(
        reasoning_off_payload={"extra_body": {"enable_thinking": False}},
        reasoning_on_payload={"extra_body": {"enable_thinking": True}},
    )
    p = OpenAIProvider(api_key="x", base_url="http://e", capability=cap)
    payload = await _capture_payload_openai(p, _model(), thinking="medium")
    assert payload["extra_body"]["enable_thinking"] is True


@pytest.mark.asyncio
async def test_reasoning_level_effort_written():
    cap = CapabilityDescriptor(
        reasoning_level=ReasoningLevelSpec(
            path="reasoning_effort",
            kind="effort",
            level_to_effort={"low": "low", "medium": "medium", "high": "high"},
        ),
    )
    p = OpenAIProvider(api_key="x", base_url="http://e", capability=cap)
    payload = await _capture_payload_openai(p, _model(), thinking="medium")
    assert payload["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_model_override_wins_for_reasoning():
    base = CapabilityDescriptor()
    override = CapabilityDescriptor(
        reasoning_on_payload={"reasoning": {"effort": "low"}},
    )
    p = OpenAIProvider(
        api_key="x",
        base_url="http://e",
        capability=base,
        model_capability_overrides={"deepseek-r1": override},
    )
    payload = await _capture_payload_openai(p, _model("deepseek-r1"), thinking="medium")
    assert payload["reasoning"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_legacy_no_capability_does_not_merge_reasoning_payload():
    """Regression: no capability -> no reasoning_off/on payload write."""
    p = OpenAIProvider(api_key="x", base_url="http://e")  # legacy
    payload = await _capture_payload_openai(p, _model(), thinking="off")
    assert "extra_body" not in payload
    assert "reasoning_effort" not in payload


@pytest.mark.asyncio
async def test_reasoning_on_payload_and_level_both_applied():
    """Anthropic-style combined case: on_payload writes the thinking block,
    then reasoning_level writes budget_tokens into the same dict. Order
    matters — the merge must run before the level write so the dict exists."""
    cap = CapabilityDescriptor(
        reasoning_on_payload={"thinking": {"type": "enabled"}},
        reasoning_level=ReasoningLevelSpec(
            path="thinking.budget_tokens",
            kind="int_budget",
            level_budgets={"medium": 8192},
        ),
    )
    p = OpenAIProvider(api_key="x", base_url="http://e", capability=cap)
    payload = await _capture_payload_openai(p, _model(), thinking="medium")
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 8192}


@pytest.mark.asyncio
async def test_max_tokens_field_renamed_preserves_on_payload_value():
    """When on_payload sets max_tokens explicitly, the value survives the rename.
    Mirrors test_temperature_free_preserves_caller_value_via_on_payload but for
    max_tokens / max_completion_tokens."""
    cap = CapabilityDescriptor(max_tokens_field="max_completion_tokens")
    p = OpenAIProvider(api_key="x", base_url="http://e", capability=cap)

    async def set_caller_max(kwargs, model):
        kwargs["max_tokens"] = 1234
        return kwargs

    payload = await _capture_payload_openai(p, _model(), on_payload=set_caller_max)
    assert payload["max_completion_tokens"] == 1234
    assert "max_tokens" not in payload
