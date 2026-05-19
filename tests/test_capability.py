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
    assert cap.temperature.default == 1.0
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


def test_temperature_default_must_be_within_range():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TemperatureSpec(min=0.0, max=1.0, default=2.0)


from cubepi.providers.capability import merge_capability_payload


def test_merge_empty_patch_is_noop():
    kwargs = {"a": 1, "extra_body": {"b": 2}}
    merge_capability_payload(kwargs, {})
    assert kwargs == {"a": 1, "extra_body": {"b": 2}}


def test_merge_adds_new_top_level_keys():
    kwargs = {"a": 1}
    merge_capability_payload(kwargs, {"reasoning_effort": "low"})
    assert kwargs == {"a": 1, "reasoning_effort": "low"}


def test_merge_recurses_into_nested_dicts():
    kwargs: dict = {"extra_body": {"existing": True}}
    merge_capability_payload(kwargs, {"extra_body": {"enable_thinking": False}})
    assert kwargs == {"extra_body": {"existing": True, "enable_thinking": False}}


def test_merge_capability_wins_on_leaf_collision():
    kwargs = {"extra_body": {"enable_thinking": True}}
    merge_capability_payload(kwargs, {"extra_body": {"enable_thinking": False}})
    assert kwargs == {"extra_body": {"enable_thinking": False}}


def test_merge_arrays_are_atomic_capability_wins():
    """Arrays at colliding keys are replaced, not unioned."""
    kwargs = {"stop": ["\n", "."]}
    merge_capability_payload(kwargs, {"stop": ["END"]})
    assert kwargs == {"stop": ["END"]}


def test_merge_does_not_mutate_patch():
    """The patch dict the caller passes in must be left untouched."""
    patch = {"extra_body": {"enable_thinking": False}}
    kwargs: dict = {}
    merge_capability_payload(kwargs, patch)
    kwargs["extra_body"]["enable_thinking"] = True
    assert patch == {"extra_body": {"enable_thinking": False}}
