from cubepi.providers.capability import (
    CapabilityDescriptor,
    ReasoningLevelSpec,
    TemperatureSpec,
    apply_temperature,
    merge_capability_payload,
    write_reasoning_level,
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


def test_merge_patch_dict_overwrites_scalar():
    """When patch has a dict at a key where kwargs has a scalar, capability wins —
    the scalar is replaced with the dict (consistent with rule 3)."""
    kwargs = {"x": 5}
    merge_capability_payload(kwargs, {"x": {"y": 1}})
    assert kwargs == {"x": {"y": 1}}


def test_merge_none_patch_value_overwrites():
    """None in patch overwrites a nested dict in kwargs (capability wins, rule 3)."""
    kwargs: dict = {"x": {"nested": True}}
    merge_capability_payload(kwargs, {"x": None})
    assert kwargs["x"] is None


def test_apply_temperature_free_passes_through():
    kwargs = {"temperature": 0.7}
    apply_temperature(kwargs, TemperatureSpec(mode="free"))
    assert kwargs == {"temperature": 0.7}


def test_apply_temperature_free_clamps_above_max():
    kwargs = {"temperature": 5.0}
    apply_temperature(kwargs, TemperatureSpec(mode="free", min=0, max=2))
    assert kwargs == {"temperature": 2.0}


def test_apply_temperature_free_clamps_below_min():
    kwargs = {"temperature": -1.0}
    apply_temperature(kwargs, TemperatureSpec(mode="free", min=0, max=2))
    assert kwargs == {"temperature": 0.0}


def test_apply_temperature_ignored_strips():
    kwargs = {"temperature": 0.7}
    apply_temperature(kwargs, TemperatureSpec(mode="ignored"))
    assert "temperature" not in kwargs


def test_apply_temperature_fixed_overwrites():
    kwargs = {"temperature": 0.7}
    apply_temperature(kwargs, TemperatureSpec(mode="fixed", fixed_value=0.0))
    assert kwargs == {"temperature": 0.0}


def test_apply_temperature_fixed_sets_when_absent():
    kwargs: dict = {}
    apply_temperature(kwargs, TemperatureSpec(mode="fixed", fixed_value=0.0))
    assert kwargs == {"temperature": 0.0}


def test_apply_temperature_free_no_op_when_absent():
    kwargs: dict = {}
    apply_temperature(kwargs, TemperatureSpec(mode="free"))
    assert kwargs == {}


def test_apply_temperature_free_clamp_inclusive_at_min():
    """Value exactly at min stays unchanged (inclusive clamp)."""
    kwargs = {"temperature": 0.0}
    apply_temperature(kwargs, TemperatureSpec(mode="free", min=0.0, max=2.0))
    assert kwargs == {"temperature": 0.0}


def test_apply_temperature_free_clamp_inclusive_at_max():
    """Value exactly at max stays unchanged (inclusive clamp)."""
    kwargs = {"temperature": 2.0}
    apply_temperature(kwargs, TemperatureSpec(mode="free", min=0.0, max=2.0))
    assert kwargs == {"temperature": 2.0}


def test_int_budget_writes_top_level_path():
    kwargs: dict = {}
    spec = ReasoningLevelSpec(
        path="thinking.budget_tokens",
        kind="int_budget",
        level_budgets={"off": 0, "low": 4000, "medium": 10000, "high": 32000},
    )
    write_reasoning_level(kwargs, spec, "medium")
    assert kwargs == {"thinking": {"budget_tokens": 10000}}


def test_int_budget_skips_when_level_absent_in_map():
    kwargs: dict = {}
    spec = ReasoningLevelSpec(
        path="thinking.budget_tokens",
        kind="int_budget",
        level_budgets={"low": 4000},
    )
    write_reasoning_level(kwargs, spec, "xhigh")
    assert kwargs == {}


def test_effort_writes_string():
    kwargs: dict = {}
    spec = ReasoningLevelSpec(
        path="reasoning_effort",
        kind="effort",
        level_to_effort={"low": "low", "medium": "medium", "high": "high"},
    )
    write_reasoning_level(kwargs, spec, "high")
    assert kwargs == {"reasoning_effort": "high"}


def test_enum_writes_nested_extra_body():
    kwargs: dict = {}
    spec = ReasoningLevelSpec(
        path="extra_body.thinking.type",
        kind="enum",
        level_to_enum={"off": "disabled", "low": "enabled", "medium": "enabled", "high": "enabled"},
    )
    write_reasoning_level(kwargs, spec, "medium")
    assert kwargs == {"extra_body": {"thinking": {"type": "enabled"}}}


def test_writes_into_existing_nested_dict():
    kwargs: dict = {"extra_body": {"other": True}}
    spec = ReasoningLevelSpec(
        path="extra_body.thinking.type",
        kind="enum",
        level_to_enum={"off": "disabled", "medium": "enabled"},
    )
    write_reasoning_level(kwargs, spec, "off")
    assert kwargs == {"extra_body": {"other": True, "thinking": {"type": "disabled"}}}
