import pytest
from pydantic import ValidationError

from cubepi.providers.base import Model, ReasoningControl
from cubepi.providers.capability import (
    CapabilityDescriptor,
    ReasoningCapability,
    TemperatureSpec,
    apply_reasoning_control,
    apply_temperature,
    lint_capability,
    merge_capability_payload,
)


def test_descriptor_defaults_are_safe_noop():
    cap = CapabilityDescriptor()
    assert cap.reasoning is None
    assert cap.temperature.mode == "free"
    assert cap.max_tokens_field == "max_tokens"
    assert cap.supports_tools is True
    assert cap.supports_images is False
    assert cap.supports_streaming is True


def test_temperature_fixed_requires_value():
    with pytest.raises(ValidationError):
        TemperatureSpec(mode="fixed")


def test_temperature_default_must_be_within_range():
    with pytest.raises(ValidationError):
        TemperatureSpec(min=0.0, max=1.0, default=2.0)


def test_merge_recurses_and_capability_wins():
    kwargs: dict = {"extra_body": {"existing": True}, "stop": ["\n"]}
    patch = {"extra_body": {"enable_thinking": False}, "stop": ["END"]}

    merge_capability_payload(kwargs, patch)

    assert kwargs == {
        "extra_body": {"existing": True, "enable_thinking": False},
        "stop": ["END"],
    }
    kwargs["extra_body"]["enable_thinking"] = True
    assert patch == {"extra_body": {"enable_thinking": False}, "stop": ["END"]}


def test_apply_temperature_modes():
    kwargs = {"temperature": 5.0}
    apply_temperature(kwargs, TemperatureSpec(mode="free", min=0, max=2))
    assert kwargs == {"temperature": 2.0}

    apply_temperature(kwargs, TemperatureSpec(mode="fixed", fixed_value=0.0))
    assert kwargs == {"temperature": 0.0}

    apply_temperature(kwargs, TemperatureSpec(mode="ignored"))
    assert kwargs == {}


def test_apply_reasoning_control_writes_mode_effort_and_summary():
    kwargs: dict = {}
    cap = CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={"on": {"extra_body": {"thinking": {"type": "enabled"}}}},
            effort_path="reasoning_effort",
            effort_values={"high": "high"},
            summary_path="reasoning.summary",
            summary_values={"auto": "auto"},
        )
    )

    apply_reasoning_control(
        kwargs,
        cap,
        ReasoningControl(mode="on", effort="high", summary="auto"),
    )

    assert kwargs == {
        "extra_body": {"thinking": {"type": "enabled"}},
        "reasoning_effort": "high",
        "reasoning": {"summary": "auto"},
    }


def test_apply_reasoning_control_can_skip_effort_when_off():
    kwargs: dict = {}
    cap = CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={"off": {"reasoning_effort": "minimal"}},
            effort_path="reasoning_effort",
            effort_values={"high": "high"},
            apply_effort_when_off=False,
        )
    )

    apply_reasoning_control(
        kwargs,
        cap,
        ReasoningControl(mode="off", effort="high"),
    )

    assert kwargs == {"reasoning_effort": "minimal"}


def test_lint_warns_for_top_level_thinking_on_openai_chat():
    cap = CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={"off": {"thinking": {"type": "disabled"}}},
        )
    )

    warnings = lint_capability(
        Model(id="glm-5.2", provider_id="volcengine", api="openai-completions"),
        cap,
    )

    assert warnings
    assert warnings[0].code == "openai_chat_top_level_thinking"
