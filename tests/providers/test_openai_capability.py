"""OpenAIProvider — capability descriptor wiring.

Each test constructs a provider with a known capability, then asserts on
the provider's internal state. Stream-time assertions come in Task 6.
"""

import pytest

from cubepi.providers.capability import CapabilityDescriptor, TemperatureSpec
from cubepi.providers.openai import OpenAIProvider


def test_provider_accepts_capability_kwarg():
    cap = CapabilityDescriptor(temperature=TemperatureSpec(mode="ignored"))
    p = OpenAIProvider(api_key="x", base_url="http://example", capability=cap)
    assert p._capability is cap


def test_provider_accepts_model_overrides():
    cap = CapabilityDescriptor()
    overrides = {"deepseek-r1": CapabilityDescriptor(reasoning_off_payload={"reasoning": {"effort": "low"}})}
    p = OpenAIProvider(
        api_key="x",
        base_url="http://example",
        capability=cap,
        model_capability_overrides=overrides,
    )
    assert p._model_overrides == overrides


def test_resolve_capability_uses_override_when_present():
    base = CapabilityDescriptor()
    override = CapabilityDescriptor(reasoning_off_payload={"reasoning": {"effort": "low"}})
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
    p = OpenAIProvider(api_key="x", base_url="http://example", capability=CapabilityDescriptor())
    assert p._cap_active is True


def test_cap_active_when_only_overrides_passed():
    p = OpenAIProvider(
        api_key="x", base_url="http://example",
        model_capability_overrides={"m": CapabilityDescriptor()},
    )
    assert p._cap_active is True
