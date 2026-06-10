from cubepi.providers.base import ThinkingBudgets, adjust_max_tokens_for_thinking


class TestThinkingBudgets:
    def test_defaults(self):
        b = ThinkingBudgets()
        assert b.low == 2048
        assert b.medium == 8192
        assert b.high == 16384

    def test_custom_values(self):
        b = ThinkingBudgets(low=1024, medium=4096, high=8192)
        assert b.low == 1024
        assert b.high == 8192


class TestAdjustMaxTokensForThinking:
    def test_off_returns_base_unchanged(self):
        max_tokens, budget = adjust_max_tokens_for_thinking(
            base_max_tokens=8192,
            model_max_tokens=128_000,
            reasoning_level="off",
        )
        assert max_tokens == 8192
        assert budget == 0

    def test_low_adds_2048(self):
        max_tokens, budget = adjust_max_tokens_for_thinking(
            base_max_tokens=8192,
            model_max_tokens=128_000,
            reasoning_level="low",
        )
        assert budget == 2048
        assert max_tokens == 8192 + 2048

    def test_medium_adds_8192(self):
        max_tokens, budget = adjust_max_tokens_for_thinking(
            base_max_tokens=8192,
            model_max_tokens=128_000,
            reasoning_level="medium",
        )
        assert budget == 8192
        assert max_tokens == 8192 + 8192

    def test_high_adds_16384(self):
        max_tokens, budget = adjust_max_tokens_for_thinking(
            base_max_tokens=8192,
            model_max_tokens=128_000,
            reasoning_level="high",
        )
        assert budget == 16384
        assert max_tokens == 8192 + 16384

    def test_xhigh_clamps_to_high(self):
        max_tokens, budget = adjust_max_tokens_for_thinking(
            base_max_tokens=8192,
            model_max_tokens=128_000,
            reasoning_level="xhigh",
        )
        # xhigh should be clamped to high (16384)
        assert budget == 16384
        assert max_tokens == 8192 + 16384

    def test_model_cap_limits_max_tokens(self):
        # Model only allows 10000 total, but base + budget = 8192 + 8192 = 16384
        max_tokens, budget = adjust_max_tokens_for_thinking(
            base_max_tokens=8192,
            model_max_tokens=10000,
            reasoning_level="medium",
        )
        assert max_tokens == 10000
        assert budget == 8192  # budget still fits (10000 > 8192)

    def test_budget_reduced_when_model_too_small(self):
        # Model allows only 2000 total, budget would be 8192 (medium)
        # Since max_tokens(2000) <= budget(8192), budget = max(0, 2000 - 1024) = 976
        max_tokens, budget = adjust_max_tokens_for_thinking(
            base_max_tokens=8192,
            model_max_tokens=2000,
            reasoning_level="medium",
        )
        assert max_tokens == 2000
        assert budget == 976

    def test_budget_zero_when_model_extremely_small(self):
        # Model allows only 500 total, budget would be 8192
        # max_tokens = min(8192 + 8192, 500) = 500
        # 500 <= 8192 so budget = max(0, 500 - 1024) = 0
        max_tokens, budget = adjust_max_tokens_for_thinking(
            base_max_tokens=8192,
            model_max_tokens=500,
            reasoning_level="medium",
        )
        assert max_tokens == 500
        assert budget == 0

    def test_custom_budgets_override_defaults(self):
        custom = ThinkingBudgets(low=512, medium=2048, high=4096)
        max_tokens, budget = adjust_max_tokens_for_thinking(
            base_max_tokens=8192,
            model_max_tokens=128_000,
            reasoning_level="medium",
            custom_budgets=custom,
        )
        assert budget == 2048
        assert max_tokens == 8192 + 2048

    def test_custom_budgets_xhigh_uses_custom_high(self):
        custom = ThinkingBudgets(high=4096)
        max_tokens, budget = adjust_max_tokens_for_thinking(
            base_max_tokens=8192,
            model_max_tokens=128_000,
            reasoning_level="xhigh",
            custom_budgets=custom,
        )
        assert budget == 4096
        assert max_tokens == 8192 + 4096

    def test_exact_model_cap_equals_budget(self):
        # Edge case: model cap exactly equals the budget
        # max_tokens = min(8192 + 2048, 2048) = 2048
        # 2048 - 2048 = 0 < min_output_tokens(1024), budget = max(0, 2048 - 1024) = 1024
        max_tokens, budget = adjust_max_tokens_for_thinking(
            base_max_tokens=8192,
            model_max_tokens=2048,
            reasoning_level="low",
        )
        assert max_tokens == 2048
        assert budget == 1024

    def test_model_cap_just_above_budget(self):
        # base=8192, model=2049, low budget=2048
        # max_tokens = min(8192 + 2048, 2049) = 2049
        # 2049 - 2048 = 1 < min_output_tokens(1024), budget = max(0, 2049 - 1024) = 1025
        max_tokens, budget = adjust_max_tokens_for_thinking(
            base_max_tokens=8192,
            model_max_tokens=2049,
            reasoning_level="low",
        )
        assert max_tokens == 2049
        assert budget == 1025

    def test_output_tokens_minimum_guaranteed(self):
        # Bug case: base=8192, model=9000, medium budget=8192
        # max_tokens = min(16384, 9000) = 9000
        # Without fix: 9000 > 8192 so budget stays 8192, only 808 for output
        # With fix: 9000 - 8192 = 808 < 1024, so budget = 9000 - 1024 = 7976
        max_tokens, budget = adjust_max_tokens_for_thinking(
            base_max_tokens=8192,
            model_max_tokens=9000,
            reasoning_level="medium",
        )
        assert max_tokens == 9000
        assert budget == 7976
        assert max_tokens - budget >= 1024
