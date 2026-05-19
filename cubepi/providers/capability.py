"""Capability descriptor — vendor quirks expressed as data, bound to a Provider.

See docs/dev/specs/2026-05-19-llm-provider-platform-design.md §3.1.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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


class ReasoningLevelSpec(BaseModel):
    """How to express a fine-grain reasoning level on this endpoint."""

    path: str
    kind: Literal["int_budget", "effort", "enum"]
    level_budgets: dict[str, int] | None = None
    level_to_effort: dict[str, str] | None = None
    level_to_enum: dict[str, str] | None = None

    @model_validator(mode="after")
    def _validate_kind_map(self) -> ReasoningLevelSpec:
        if self.kind == "int_budget" and not self.level_budgets:
            raise ValueError("kind='int_budget' requires a non-empty level_budgets map")
        if self.kind == "effort" and not self.level_to_effort:
            raise ValueError("kind='effort' requires a non-empty level_to_effort map")
        if self.kind == "enum" and not self.level_to_enum:
            raise ValueError("kind='enum' requires a non-empty level_to_enum map")
        return self


class CapabilityDescriptor(BaseModel):
    """Vendor quirks for one endpoint. Empty default = legacy no-op."""

    reasoning_off_payload: dict[str, Any] = Field(default_factory=dict)
    reasoning_on_payload: dict[str, Any] = Field(default_factory=dict)
    reasoning_level: ReasoningLevelSpec | None = None

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
        assert spec.fixed_value is not None  # enforced by validator
        kwargs["temperature"] = spec.fixed_value
        return

    if "temperature" in kwargs:
        value = kwargs["temperature"]
        kwargs["temperature"] = max(spec.min, min(spec.max, value))
