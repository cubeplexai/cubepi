"""Catalog types: ProviderPreset, ModelPreset, AuthSpec, WireApi."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cubepi.providers.capability import CapabilityDescriptor

WireApi = Literal["anthropic-messages", "openai-completions", "openai-responses"]


class AuthSpec(BaseModel):
    mode: Literal["api_key", "bearer", "none", "oauth", "iam"]
    header_name: str | None = None
    header_prefix: str | None = "Bearer "


class ModelPreset(BaseModel):
    model_id: str
    display_name: str
    context_window: int
    max_tokens: int
    input_modalities: list[str]
    reasoning: bool = False


class ProviderPreset(BaseModel):
    slug: str
    display_name: str
    short_name: str
    category: Literal["saas", "oss-framework", "custom"]
    description: str
    # @lobehub/icons provider id (lowercase, e.g. "anthropic", "openai",
    # "deepseek"). cubebox frontend renders via
    # <ProviderIcon provider={preset.logo} size=28 type="color" />.
    # None = render generic fallback. Spec §3.6, §7 Q5.
    logo: str | None = None

    api: WireApi
    base_url: str
    auth: AuthSpec

    capability: CapabilityDescriptor
    model_capability_overrides: dict[str, CapabilityDescriptor] = Field(
        default_factory=dict
    )

    default_models: list[ModelPreset] = Field(default_factory=list)
