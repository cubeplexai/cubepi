"""Top-level ``cubepi`` re-export contract tests."""

from __future__ import annotations


def test_capability_types_re_exported():
    import cubeloop

    assert hasattr(cubeloop, "CapabilityDescriptor")
    assert hasattr(cubeloop, "ReasoningCapability")
    assert hasattr(cubeloop, "TemperatureSpec")
    assert not hasattr(cubeloop, "ReasoningLevelSpec")
