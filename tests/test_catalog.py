import pytest

from cubepi.providers.catalog import get_provider_preset, list_provider_presets
from cubepi.providers.catalog.types import (
    AuthSpec,
    ModelPreset,
    ProviderPreset,
    WireApi,
)
from cubepi.providers.capability import CapabilityDescriptor


def test_wire_api_values():
    assert WireApi.__args__ == (
        "anthropic-messages",
        "openai-completions",
        "openai-responses",
    )


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
        model_id="gpt-4o",
        display_name="GPT-4o",
        context_window=128000,
        max_tokens=16384,
        input_modalities=["text", "image"],
    )
    assert m.reasoning is False


def test_auth_spec_api_key_defaults():
    a = AuthSpec(mode="api_key")
    assert a.header_name in (None, "Authorization")


def test_list_provider_presets_returns_all_entries():
    presets = list_provider_presets()
    slugs = [p.slug for p in presets]
    for required in (
        "anthropic",
        "openai",
        "qwen-dashscope",
        "doubao-volcengine",
        "openrouter",
        "custom-openai",
        "custom-anthropic",
    ):
        assert required in slugs


def test_every_preset_parses_into_typed_model():
    presets = list_provider_presets()
    assert len(presets) == 20
    valid_apis = WireApi.__args__
    for p in presets:
        assert p.api in valid_apis, p.slug
        assert p.slug == p.slug.lower()
        assert p.capability.temperature.min <= p.capability.temperature.max


def test_get_provider_preset_by_slug():
    qwen = get_provider_preset("qwen-dashscope")
    assert qwen.api == "openai-completions"
    assert qwen.capability.reasoning_off_payload == {
        "extra_body": {"enable_thinking": False}
    }


def test_get_provider_preset_unknown_raises():
    with pytest.raises(KeyError):
        get_provider_preset("nonexistent")


def test_openrouter_has_model_capability_overrides():
    p = get_provider_preset("openrouter")
    assert "deepseek/deepseek-r1" in p.model_capability_overrides
    over = p.model_capability_overrides["deepseek/deepseek-r1"]
    assert over.reasoning_level is not None
    assert over.reasoning_level.kind == "effort"
