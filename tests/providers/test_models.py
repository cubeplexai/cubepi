"""Tests for thinking-level validation, clamping, and model comparison."""

from cubepi.providers.base import Model
from cubepi.providers.models import (
    THINKING_LEVELS,
    clamp_thinking_level,
    get_supported_thinking_levels,
    models_are_equal,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _model(
    *,
    reasoning: bool = False,
    thinking_level_map: dict[str, str | None] | None = None,
    provider: str = "anthropic",
    model_id: str = "claude-sonnet-4-20250514",
) -> Model:
    return Model(
        id=model_id,
        provider_id=provider,
        reasoning=reasoning,
        thinking_level_map=thinking_level_map,
    )


# ---------------------------------------------------------------------------
# get_supported_thinking_levels
# ---------------------------------------------------------------------------


class TestGetSupportedThinkingLevels:
    def test_non_reasoning_model_only_off(self):
        model = _model(reasoning=False)
        assert get_supported_thinking_levels(model) == ["off"]

    def test_reasoning_model_no_map_defaults(self):
        """Without a map, all levels except xhigh are available."""
        model = _model(reasoning=True)
        assert get_supported_thinking_levels(model) == [
            "off",
            "low",
            "medium",
            "high",
        ]

    def test_reasoning_model_with_xhigh_in_map(self):
        model = _model(
            reasoning=True,
            thinking_level_map={"xhigh": "max_tokens_16k"},
        )
        levels = get_supported_thinking_levels(model)
        assert "xhigh" in levels
        # All standard levels should still be present
        for lvl in ("off", "low", "medium", "high"):
            assert lvl in levels

    def test_xhigh_mapped_to_none_is_excluded(self):
        model = _model(
            reasoning=True,
            thinking_level_map={"xhigh": None},
        )
        assert "xhigh" not in get_supported_thinking_levels(model)

    def test_level_mapped_to_none_is_excluded(self):
        model = _model(
            reasoning=True,
            thinking_level_map={"low": None},
        )
        levels = get_supported_thinking_levels(model)
        assert "low" not in levels
        # Others remain
        assert "off" in levels
        assert "medium" in levels
        assert "high" in levels

    def test_empty_map_same_as_no_map(self):
        model = _model(reasoning=True, thinking_level_map={})
        levels = get_supported_thinking_levels(model)
        assert levels == ["off", "low", "medium", "high"]

    def test_all_levels_disabled_except_off(self):
        model = _model(
            reasoning=True,
            thinking_level_map={
                "low": None,
                "medium": None,
                "high": None,
            },
        )
        assert get_supported_thinking_levels(model) == ["off"]


# ---------------------------------------------------------------------------
# clamp_thinking_level
# ---------------------------------------------------------------------------


class TestClampThinkingLevel:
    def test_supported_level_passes_through(self):
        model = _model(reasoning=True)
        assert clamp_thinking_level(model, "medium") == "medium"

    def test_off_always_passes_for_non_reasoning(self):
        model = _model(reasoning=False)
        assert clamp_thinking_level(model, "off") == "off"

    def test_non_reasoning_clamps_to_off(self):
        model = _model(reasoning=False)
        assert clamp_thinking_level(model, "high") == "off"

    def test_clamp_xhigh_without_support_goes_down(self):
        """xhigh not in map -> clamp down to high."""
        model = _model(reasoning=True)
        assert clamp_thinking_level(model, "xhigh") == "high"

    def test_clamp_down_to_nearest(self):
        """When a level is disabled, search downward first (prefer cheaper)."""
        model = _model(
            reasoning=True,
            thinking_level_map={"medium": None},
        )
        # "medium" disabled -> next down is "low"
        assert clamp_thinking_level(model, "medium") == "low"

    def test_clamp_down_when_nothing_above(self):
        """When all higher levels are disabled, clamp down."""
        model = _model(
            reasoning=True,
            thinking_level_map={"high": None},
        )
        # "high" disabled, xhigh not in map -> search down -> "medium"
        assert clamp_thinking_level(model, "high") == "medium"

    def test_clamp_unknown_level_returns_first_available(self):
        model = _model(reasoning=True)
        # An invalid level should fall back to the first available
        result = clamp_thinking_level(model, "ultra")  # type: ignore[arg-type]
        assert result == "off"

    def test_clamp_with_only_off_available(self):
        model = _model(
            reasoning=True,
            thinking_level_map={
                "low": None,
                "medium": None,
                "high": None,
            },
        )
        assert clamp_thinking_level(model, "medium") == "off"

    def test_clamp_preserves_exact_match_with_map(self):
        model = _model(
            reasoning=True,
            thinking_level_map={"medium": "budget_4096", "xhigh": "budget_max"},
        )
        assert clamp_thinking_level(model, "medium") == "medium"
        assert clamp_thinking_level(model, "xhigh") == "xhigh"


# ---------------------------------------------------------------------------
# models_are_equal
# ---------------------------------------------------------------------------


class TestModelsAreEqual:
    def test_equal_models(self):
        a = _model(model_id="claude-sonnet-4-20250514", provider="anthropic")
        b = _model(model_id="claude-sonnet-4-20250514", provider="anthropic")
        assert models_are_equal(a, b) is True

    def test_different_id(self):
        a = _model(model_id="claude-sonnet-4-20250514", provider="anthropic")
        b = _model(model_id="claude-opus-4-20250514", provider="anthropic")
        assert models_are_equal(a, b) is False

    def test_different_provider(self):
        a = _model(model_id="claude-sonnet-4-20250514", provider="anthropic")
        b = _model(model_id="claude-sonnet-4-20250514", provider="bedrock")
        assert models_are_equal(a, b) is False

    def test_none_a(self):
        b = _model()
        assert models_are_equal(None, b) is False

    def test_none_b(self):
        a = _model()
        assert models_are_equal(a, None) is False

    def test_both_none(self):
        assert models_are_equal(None, None) is True


# ---------------------------------------------------------------------------
# THINKING_LEVELS ordering
# ---------------------------------------------------------------------------


class TestThinkingLevelsOrdering:
    def test_correct_order(self):
        assert THINKING_LEVELS == [
            "off",
            "low",
            "medium",
            "high",
            "xhigh",
        ]

    def test_all_levels_present(self):
        expected = {"off", "low", "medium", "high", "xhigh"}
        assert set(THINKING_LEVELS) == expected
