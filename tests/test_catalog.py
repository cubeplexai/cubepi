from cubepi.providers.catalog.types import (
    AuthSpec, ModelPreset, ProviderPreset, WireApi,
)
from cubepi.providers.capability import CapabilityDescriptor, TemperatureSpec


def test_wire_api_values():
    assert WireApi.__args__ == ("anthropic-messages", "openai-completions", "openai-responses")


def test_minimal_provider_preset_constructs():
    p = ProviderPreset(
        slug="custom-openai",
        display_name="Custom OpenAI",
        short_name="Custom",
        category="custom",
        description="",
        api="openai-completions",
        base_url="https://example.com/v1",
        auth=AuthSpec(mode="api_key"),
        capability=CapabilityDescriptor(),
        default_models=[],
    )
    assert p.slug == "custom-openai"
    assert p.model_capability_overrides == {}
    assert p.logo is None  # custom presets default to no brand mark


def test_model_preset_minimal():
    m = ModelPreset(
        model_id="gpt-4o", display_name="GPT-4o",
        context_window=128000, max_tokens=16384,
        input_modalities=["text", "image"],
    )
    assert m.reasoning is False


def test_auth_spec_api_key_defaults():
    a = AuthSpec(mode="api_key")
    assert a.header_name in (None, "Authorization")
