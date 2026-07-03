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


def test_apply_reasoning_clamps_mode_to_off_for_non_reasoning_model():
    """A non-reasoning model must never receive an enabled mode payload,
    mirroring the clamping the deleted clamp_thinking_level used to do."""
    cap = get_capability_profile("anthropic", "messages")
    payload: dict = {}

    apply_reasoning_control(
        payload,
        cap,
        ReasoningControl(mode="on", effort="medium"),
        model=Model(id="claude-haiku", provider_id="anthropic", reasoning=False),
    )

    assert payload == {"thinking": {"type": "disabled"}}


def test_apply_reasoning_off_payload_applies_regardless_of_model_reasoning():
    """Hybrid models (e.g. Qwen) must still get their off-mode payload even
    when registered with reasoning=False."""
    cap = CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={
                "off": {"extra_body": {"enable_thinking": False}},
                "on": {"extra_body": {"enable_thinking": True}},
            }
        )
    )
    payload: dict = {}

    apply_reasoning_control(
        payload,
        cap,
        ReasoningControl(mode="off"),
        model=Model(id="qwen", provider_id="test", reasoning=False),
    )

    assert payload == {"extra_body": {"enable_thinking": False}}


def test_apply_reasoning_skips_effort_for_non_reasoning_model_even_with_apply_when_off():
    """model.reasoning=False must suppress effort writes entirely, even when
    the capability sets apply_effort_when_off=True."""
    cap = CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={"on": {"reasoning": {"effort": "low"}}},
            effort_path="reasoning.effort",
            effort_values={"medium": "medium"},
        )
    )
    payload: dict = {}

    apply_reasoning_control(
        payload,
        cap,
        ReasoningControl(mode="on", effort="medium"),
        model=Model(id="gpt-4o", provider_id="test", reasoning=False),
    )

    assert payload == {}


def test_anthropic_profile_has_budgets_for_minimal_and_max_effort():
    cap = get_capability_profile("anthropic", "messages")
    payload: dict = {}

    apply_reasoning_control(
        payload,
        cap,
        ReasoningControl(mode="on", effort="max"),
        model=Model(id="claude-opus", provider_id="anthropic", reasoning=True),
    )

    assert payload["thinking"]["type"] == "enabled"
    assert payload["thinking"]["budget_tokens"] > 0


def test_openai_effort_max_maps_to_xhigh_not_invalid_max_value():
    cap = get_capability_profile("openai", "responses")
    payload: dict = {}

    apply_reasoning_control(
        payload,
        cap,
        ReasoningControl(mode="on", effort="max"),
        model=Model(id="gpt-5", provider_id="openai", reasoning=True),
    )

    assert payload["reasoning"]["effort"] == "xhigh"


def test_get_capability_profile_resolves_openai_completions_alias():
    direct = get_capability_profile("openai", "chat_completions")
    aliased = get_capability_profile("openai", "openai-completions")
    assert aliased == direct


def test_get_capability_profile_unknown_pair_returns_empty_descriptor():
    cap = get_capability_profile("some-vendor", "some-api")
    assert cap == CapabilityDescriptor()
