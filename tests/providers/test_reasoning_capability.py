from cubepi.providers.base import Model, ReasoningControl, StreamOptions
from cubepi.providers.capability import (
    CapabilityDescriptor,
    CapabilityWarning,
    PayloadPreview,
    ReasoningCapability,
    apply_reasoning_control,
    lint_capability,
    preview_payload,
)
from cubepi.providers.reasoning_profiles import get_capability_profile


def test_stream_options_default_reasoning_is_off_medium_none():
    opts = StreamOptions()

    assert opts.reasoning == ReasoningControl(
        mode="off",
        effort="medium",
        summary="none",
    )


def test_apply_reasoning_writes_effort_when_off_for_chat_profile():
    cap = get_capability_profile("openai", "chat_completions")
    payload: dict = {}

    apply_reasoning_control(
        payload,
        cap,
        ReasoningControl(mode="off", effort="minimal"),
    )

    assert payload == {"reasoning_effort": "minimal"}


def test_apply_reasoning_writes_nested_summary_and_include_payload():
    cap = get_capability_profile("openai", "responses")
    payload: dict = {}

    apply_reasoning_control(
        payload,
        cap,
        ReasoningControl(mode="auto", effort="high", summary="auto"),
    )

    assert payload == {
        "reasoning": {"effort": "high", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
    }


def test_preview_payload_returns_reasoning_diff():
    cap = CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={
                "off": {"extra_body": {"thinking": {"type": "disabled"}}},
            },
            effort_path="reasoning_effort",
            effort_values={"low": "low"},
        )
    )

    preview = preview_payload(
        Model(id="glm-5.2", provider_id="volcengine", api="openai-completions"),
        cap,
        ReasoningControl(mode="off", effort="low"),
    )

    assert preview == PayloadPreview(
        payload={
            "extra_body": {"thinking": {"type": "disabled"}},
            "reasoning_effort": "low",
        },
        warnings=[],
    )


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
    assert isinstance(warnings[0], CapabilityWarning)
    assert warnings[0].code == "openai_chat_top_level_thinking"
    assert "extra_body.thinking" in warnings[0].message
