"""Top-level ``cubepi`` re-export contract tests."""

from __future__ import annotations


def test_capability_types_re_exported():
    import cubepi

    assert hasattr(cubepi, "CapabilityDescriptor")
    assert hasattr(cubepi, "TemperatureSpec")
    assert hasattr(cubepi, "ReasoningLevelSpec")


def test_catalog_re_exported():
    import cubepi

    assert hasattr(cubepi, "list_provider_presets")
    assert hasattr(cubepi, "get_provider_preset")
    presets = cubepi.list_provider_presets()
    assert len(presets) == 20
