from cubepi.providers.capability import (
    CapabilityDescriptor,
    ReasoningLevelSpec,
    TemperatureSpec,
)


def test_descriptor_defaults_are_legacy_safe():
    """Empty descriptor must encode 'no-op' so existing callers behave the same."""
    cap = CapabilityDescriptor()
    assert cap.reasoning_off_payload == {}
    assert cap.reasoning_on_payload == {}
    assert cap.reasoning_level is None
    assert cap.temperature.mode == "free"
    assert cap.temperature.min == 0.0
    assert cap.temperature.max == 2.0
    assert cap.max_tokens_field == "max_tokens"
    assert cap.supports_tools is True
    assert cap.supports_images is False
    assert cap.supports_streaming is True


def test_temperature_fixed_requires_value():
    """mode=fixed without fixed_value must fail validation."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TemperatureSpec(mode="fixed")


def test_reasoning_level_int_budget_requires_map():
    """kind=int_budget without level_budgets must fail."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ReasoningLevelSpec(path="thinking.budget_tokens", kind="int_budget")


def test_reasoning_level_effort_requires_map():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ReasoningLevelSpec(path="reasoning_effort", kind="effort")


def test_reasoning_level_enum_requires_map():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ReasoningLevelSpec(path="extra_body.thinking.type", kind="enum")
