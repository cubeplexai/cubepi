from __future__ import annotations

from typing import Any

from cubepi.providers.base import ReasoningEffort, ReasoningSummary
from cubepi.providers.capability import CapabilityDescriptor, ReasoningCapability


_OPENAI_EFFORT_VALUES: dict[ReasoningEffort, str] = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    # OpenAI has no "max" effort value; "xhigh" is its highest tier (mirrors
    # the legacy _THINKING_TO_EFFORT mapping's "xhigh" -> "xhigh").
    "max": "xhigh",
}

_OPENAI_SUMMARY_VALUES: dict[ReasoningSummary, Any] = {
    "none": None,
    "auto": "auto",
    "detailed": "detailed",
    "summarized": "summarized",
}


_API_ALIASES: dict[tuple[str, str], tuple[str, str]] = {
    ("openai", "openai-completions"): ("openai", "chat_completions"),
}


def get_capability_profile(
    provider: str, api: str | None = None
) -> CapabilityDescriptor:
    """Return the built-in capability profile for a provider/API pair."""

    provider_key = provider.lower()
    if api is None and "." in provider_key:
        provider_key, api = provider_key.split(".", 1)
    if api is None:
        return CapabilityDescriptor()

    key = _API_ALIASES.get((provider_key, api), (provider_key, api))
    profile = _PROFILES.get(key)
    if profile is not None:
        return profile.model_copy(deep=True)
    return CapabilityDescriptor()


_PROFILES: dict[tuple[str, str], CapabilityDescriptor] = {
    ("openai", "chat_completions"): CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={"off": {"reasoning_effort": "minimal"}},
            effort_path="reasoning_effort",
            effort_values=_OPENAI_EFFORT_VALUES,
            apply_effort_when_off=False,
            unsupported_mode_policy="skip",
        )
    ),
    ("openai", "responses"): CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={"off": {"reasoning": {"effort": "minimal"}}},
            effort_path="reasoning.effort",
            effort_values=_OPENAI_EFFORT_VALUES,
            summary_path="reasoning.summary",
            summary_values=_OPENAI_SUMMARY_VALUES,
            include_payloads={
                "summary:auto": {"include": ["reasoning.encrypted_content"]},
                "summary:detailed": {"include": ["reasoning.encrypted_content"]},
                "summary:summarized": {"include": ["reasoning.encrypted_content"]},
            },
            apply_effort_when_off=False,
            unsupported_mode_policy="skip",
        )
    ),
    ("anthropic", "messages"): CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={
                "off": {"thinking": {"type": "disabled"}},
                "auto": {"thinking": {"type": "enabled"}},
                "on": {"thinking": {"type": "enabled"}},
            },
            effort_path="thinking.budget_tokens",
            effort_values={
                # Anthropic requires budget_tokens >= 1024 for extended
                # thinking; "minimal" uses that floor.
                "minimal": 1024,
                "low": 2048,
                "medium": 8192,
                "high": 16384,
                # No tier above "high" in the legacy budget scale (the old
                # ThinkingLevel="xhigh" clamped down to "high"'s budget).
                "max": 16384,
            },
            apply_effort_when_off=False,
            unsupported_mode_policy="skip",
        )
    ),
}
