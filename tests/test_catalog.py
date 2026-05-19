import pytest

import cubepi.providers.catalog as catalog_module
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
    assert len(presets) == 22
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


def test_load_raises_when_yaml_is_not_a_list(tmp_path, monkeypatch):
    """Defensive: if providers.yaml is a dict at top level instead of list, raise."""
    bad = tmp_path / "providers.yaml"
    bad.write_text("not-a-list: just-a-string\n", encoding="utf-8")
    monkeypatch.setattr(catalog_module, "_DATA_FILE", bad)
    catalog_module._load.cache_clear()

    with pytest.raises(ValueError, match="top-level list"):
        catalog_module._load()

    catalog_module._load.cache_clear()  # restore for other tests


def test_load_wraps_per_entry_validation_error(tmp_path, monkeypatch):
    """Per-entry pydantic failures are re-raised with the entry index."""
    bad = tmp_path / "providers.yaml"
    # entry #0 has a required field missing (no `api`, no `base_url`, etc.)
    bad.write_text(
        "- slug: broken\n"
        "  display_name: Broken\n"
        "  short_name: B\n"
        "  category: custom\n"
        "  description: missing api/base_url/auth/capability\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(catalog_module, "_DATA_FILE", bad)
    catalog_module._load.cache_clear()

    with pytest.raises(ValueError, match=r"providers\.yaml entry #0"):
        catalog_module._load()

    catalog_module._load.cache_clear()


def test_load_rejects_duplicate_slug(tmp_path, monkeypatch):
    """Two presets sharing the same slug fail the loader."""
    # Two minimal-valid entries with the same slug.
    bad = tmp_path / "providers.yaml"
    bad.write_text(
        "- slug: dup\n"
        "  display_name: D1\n"
        "  short_name: D\n"
        "  category: custom\n"
        "  description: first\n"
        "  api: openai-completions\n"
        '  base_url: ""\n'
        "  auth: { mode: api_key }\n"
        "  capability: {}\n"
        "  default_models: []\n"
        "- slug: dup\n"
        "  display_name: D2\n"
        "  short_name: D\n"
        "  category: custom\n"
        "  description: second\n"
        "  api: openai-completions\n"
        '  base_url: ""\n'
        "  auth: { mode: api_key }\n"
        "  capability: {}\n"
        "  default_models: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(catalog_module, "_DATA_FILE", bad)
    catalog_module._load.cache_clear()

    with pytest.raises(ValueError, match="duplicate slug"):
        catalog_module._load()

    catalog_module._load.cache_clear()
