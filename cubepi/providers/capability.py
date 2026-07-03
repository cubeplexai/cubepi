"""Capability descriptor — vendor quirks expressed as data, bound to a Provider.

See docs/dev/specs/2026-05-19-llm-provider-platform-design.md §3.1.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from cubepi.providers.base import (
    Model,
    ReasoningControl,
    ReasoningEffort,
    ReasoningMode,
    ReasoningSummary,
)


class TemperatureSpec(BaseModel):
    mode: Literal["free", "fixed", "ignored"] = "free"
    min: float = 0.0
    max: float = 2.0
    default: float = 1.0
    fixed_value: float | None = None

    @model_validator(mode="after")
    def _validate_fixed(self) -> TemperatureSpec:
        if self.mode == "fixed" and self.fixed_value is None:
            raise ValueError("TemperatureSpec(mode='fixed') requires fixed_value")
        if not (self.min <= self.default <= self.max):
            raise ValueError(
                f"TemperatureSpec.default ({self.default}) must be within [min={self.min}, max={self.max}]"
            )
        return self


UnsupportedModePolicy = Literal["skip", "warn", "error"]


class ReasoningCapability(BaseModel):
    """How one endpoint maps standard reasoning controls onto its wire payload."""

    mode_payloads: dict[ReasoningMode, dict[str, Any]] = Field(default_factory=dict)
    effort_path: str | None = None
    effort_values: dict[ReasoningEffort, Any] = Field(default_factory=dict)
    summary_path: str | None = None
    summary_values: dict[ReasoningSummary, Any] = Field(default_factory=dict)
    include_payloads: dict[str, dict[str, Any]] = Field(default_factory=dict)
    apply_effort_when_off: bool = True
    unsupported_mode_policy: UnsupportedModePolicy = "warn"


class CapabilityWarning(BaseModel):
    code: str
    message: str


class PayloadPreview(BaseModel):
    payload: dict[str, Any]
    warnings: list[CapabilityWarning] = Field(default_factory=list)


class CapabilityDescriptor(BaseModel):
    """Vendor quirks for one endpoint. Empty default = legacy no-op."""

    reasoning: ReasoningCapability | None = None

    temperature: TemperatureSpec = Field(default_factory=TemperatureSpec)

    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"

    supports_tools: bool = True
    supports_images: bool = False
    supports_streaming: bool = True


def merge_capability_payload(kwargs: dict[str, Any], patch: dict[str, Any]) -> None:
    """Deep-merge ``patch`` into ``kwargs`` in place.

    Rules (spec §3.3):
    1. Recurse into nested dicts.
    2. Arrays are atomic — capability replaces caller's array on collision.
    3. On scalar / array key collision, capability (``patch``) wins.
    4. Patch is never mutated; nested dicts are copied on write.
    """

    for key, patch_value in patch.items():
        if (
            key in kwargs
            and isinstance(kwargs[key], dict)
            and isinstance(patch_value, dict)
        ):
            merge_capability_payload(kwargs[key], patch_value)
        elif isinstance(patch_value, dict):
            # Copy nested dict so subsequent kwargs mutation doesn't bleed into patch.
            kwargs[key] = {}
            merge_capability_payload(kwargs[key], patch_value)
        else:
            kwargs[key] = patch_value


def apply_temperature(kwargs: dict[str, Any], spec: TemperatureSpec) -> None:
    """Mutate ``kwargs['temperature']`` in place per ``spec``.

    - mode="ignored": strip the key entirely.
    - mode="fixed": overwrite with ``fixed_value`` (set the key if absent).
    - mode="free":  clamp caller's value to ``[min, max]``; no-op if absent.
    """

    if spec.mode == "ignored":
        kwargs.pop("temperature", None)
        return

    if spec.mode == "fixed":
        if spec.fixed_value is None:
            raise RuntimeError(
                "TemperatureSpec(mode='fixed') reached apply_temperature with no "
                "fixed_value — validator was bypassed"
            )
        kwargs["temperature"] = spec.fixed_value
        return

    if "temperature" in kwargs:
        value = kwargs["temperature"]
        kwargs["temperature"] = max(spec.min, min(spec.max, value))


def apply_reasoning_control(
    kwargs: dict[str, Any],
    capability: CapabilityDescriptor | ReasoningCapability,
    control: ReasoningControl,
    *,
    model: Model | None = None,
) -> list[CapabilityWarning]:
    """Apply provider-independent reasoning controls to a provider payload.

    When ``model.reasoning`` is False, any requested mode other than "off" is
    treated as "off" — the model can't reason, so no effort/summary/include
    fields are written and only the "off" mode payload (if any) applies. This
    is what lets a hybrid model's off-payload (e.g. Qwen's
    ``enable_thinking: false``) keep firing regardless of ``model.reasoning``,
    while never sending enable/effort/summary fields to a model that can't use
    them. Callers should invoke this unconditionally rather than re-deriving
    their own ``model.reasoning`` gate.
    """
    reasoning = (
        capability.reasoning
        if isinstance(capability, CapabilityDescriptor)
        else capability
    )
    if reasoning is None:
        return []

    model_supports_reasoning = model is None or model.reasoning
    effective_mode = control.mode if model_supports_reasoning else "off"

    warnings: list[CapabilityWarning] = []
    mode_payload = reasoning.mode_payloads.get(effective_mode)
    if mode_payload is not None:
        merge_capability_payload(kwargs, mode_payload)
    else:
        _handle_unsupported_mode(reasoning, effective_mode, warnings)

    if (
        model_supports_reasoning
        and reasoning.effort_path is not None
        and (effective_mode != "off" or reasoning.apply_effort_when_off)
    ):
        effort = reasoning.effort_values.get(control.effort)
        if effort is not None:
            _write_dotted_path(kwargs, reasoning.effort_path, effort)

    if (
        model_supports_reasoning
        and reasoning.summary_path is not None
        and effective_mode != "off"
    ):
        summary = reasoning.summary_values.get(control.summary)
        if summary is not None:
            _write_dotted_path(kwargs, reasoning.summary_path, summary)

    if model_supports_reasoning:
        for key in (
            "always",
            f"mode:{effective_mode}",
            f"effort:{control.effort}",
            f"summary:{control.summary}",
        ):
            patch = reasoning.include_payloads.get(key)
            if patch is not None:
                merge_capability_payload(kwargs, patch)

    return warnings


def preview_payload(
    model: Model,
    capability: CapabilityDescriptor,
    control: ReasoningControl,
    base_payload: dict[str, Any] | None = None,
) -> PayloadPreview:
    """Return the payload fragment produced by applying reasoning controls."""

    payload = copy.deepcopy(base_payload) if base_payload is not None else {}
    before = copy.deepcopy(payload)
    warnings = apply_reasoning_control(payload, capability, control)
    warnings.extend(lint_capability(model, capability))
    return PayloadPreview(payload=_payload_delta(before, payload), warnings=warnings)


def lint_capability(
    model: Model,
    capability: CapabilityDescriptor,
) -> list[CapabilityWarning]:
    """Detect capability mappings that are known to be invalid for an API shape."""

    reasoning = capability.reasoning
    if reasoning is None:
        return []

    warnings: list[CapabilityWarning] = []
    if model.api in {"openai-completions", "chat_completions"}:
        for payload in reasoning.mode_payloads.values():
            if "thinking" in payload:
                warnings.append(
                    CapabilityWarning(
                        code="openai_chat_top_level_thinking",
                        message=(
                            "OpenAI-compatible Chat Completions payloads must put "
                            "provider-specific thinking controls under "
                            "extra_body.thinking, not top-level thinking."
                        ),
                    )
                )
                break
    return warnings


def _handle_unsupported_mode(
    reasoning: ReasoningCapability,
    mode: ReasoningMode,
    warnings: list[CapabilityWarning],
) -> None:
    if reasoning.unsupported_mode_policy == "skip":
        return
    message = f"Reasoning mode '{mode}' is not supported by this capability."
    if reasoning.unsupported_mode_policy == "error":
        raise ValueError(message)
    warnings.append(
        CapabilityWarning(code="unsupported_reasoning_mode", message=message)
    )


def _payload_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, value in after.items():
        if key not in before or before[key] != value:
            delta[key] = value
    return delta


def _write_dotted_path(target: dict[str, Any], path: str, value: Any) -> None:
    """Walk a dotted path into ``target``, creating dicts as needed; set the leaf.

    When an intermediate segment holds a non-dict scalar in ``target``, the
    scalar is replaced with a fresh dict before recursion continues. The
    capability layer is the authority on what lives at a provider-controlled
    path, so this "capability wins" rule mirrors :func:`merge_capability_payload`.
    """
    parts = path.split(".")
    cursor: Any = target
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
