"""Top-level ``cubepi`` re-export contract tests."""

from __future__ import annotations


def test_capability_types_re_exported():
    import cubepi

    assert hasattr(cubepi, "CapabilityDescriptor")
    assert hasattr(cubepi, "ReasoningCapability")
    assert hasattr(cubepi, "TemperatureSpec")
    assert not hasattr(cubepi, "ReasoningLevelSpec")
