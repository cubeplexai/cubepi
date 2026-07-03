from __future__ import annotations

from typing import Any

from cubepi.providers.base import ReasoningEffort, ReasoningSummary
from cubepi.providers.capability import CapabilityDescriptor, ReasoningCapability


_OPENAI_EFFORT_VALUES: dict[ReasoningEffort, str] = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "max",
}

_OPENAI_SUMMARY_VALUES: dict[ReasoningSummary, Any] = {
    "none": None,
    "auto": "auto",
    "detailed": "detailed",
    "summarized": "summarized",
}


def get_capability_profile(provider: str, api: str) -> CapabilityDescriptor:
    """Return the built-in capability profile for a provider/API pair."""

    key = (provider.lower(), api)
    profile = _PROFILES.get(key)
    if profile is not None:
        return profile.model_copy(deep=True)

    if provider.lower() == "openai" and api in {"chat_completions", "openai-completions"}:
        return _PROFILES[("openai", "chat_completions")].model_copy(deep=True)
    if provider.lower() == "openai" and api == "responses":
        return _PROFILES[("openai", "responses")].model_copy(deep=True)
    if provider.lower() == "anthropic":
        return _PROFILES[("anthropic", "messages")].model_copy(deep=True)
    return CapabilityDescriptor()


_PROFILES: dict[tuple[str, str], CapabilityDescriptor] = {
    ("openai", "chat_completions"): CapabilityDescriptor(
        reasoning=ReasoningCapability(
            effort_path="reasoning_effort",
            effort_values=_OPENAI_EFFORT_VALUES,
            apply_effort_when_off=True,
            unsupported_mode_policy="skip",
        )
    ),
    ("openai", "responses"): CapabilityDescriptor(
        reasoning=ReasoningCapability(
            effort_path="reasoning.effort",
            effort_values=_OPENAI_EFFORT_VALUES,
            summary_path="reasoning.summary",
            summary_values=_OPENAI_SUMMARY_VALUES,
            include_payloads={
                "summary:auto": {"include": ["reasoning.encrypted_content"]},
                "summary:detailed": {"include": ["reasoning.encrypted_content"]},
                "summary:summarized": {"include": ["reasoning.encrypted_content"]},
            },
            apply_effort_when_off=True,
            unsupported_mode_policy="skip",
        )
    ),
    ("anthropic", "messages"): CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={"off": {"thinking": {"type": "disabled"}}},
            effort_path="thinking.budget_tokens",
            effort_values={
                "low": 2048,
                "medium": 8192,
                "high": 16384,
            },
            apply_effort_when_off=False,
            unsupported_mode_policy="skip",
        )
    ),
}
