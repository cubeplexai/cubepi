from __future__ import annotations

from cubepi.middleware.compaction.tokens import (
    approx_tokens,
    real_context_estimate,
)
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def test_empty_returns_zero() -> None:
    assert approx_tokens([]) == 0


def test_text_chars_use_conservative_cjk_safe_estimate() -> None:
    messages = [UserMessage(content=[TextContent(text="x" * 200)])]

    assert approx_tokens(messages) == 100


def test_usage_metadata_does_not_scale_estimate() -> None:
    # approx_tokens is now a pure char estimate — usage no longer calibrates it.
    # (1000 chars / 2 = 500, regardless of the reported usage.)
    messages = [
        UserMessage(content=[TextContent(text="x" * 900)]),
        AssistantMessage(
            content=[TextContent(text="y" * 100)],
            usage=Usage(input_tokens=400, output_tokens=10),
        ),
    ]

    assert approx_tokens(messages) == 500


def test_tool_result_text_is_counted() -> None:
    messages = [
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="search",
            content=[TextContent(text="x" * 200)],
        )
    ]

    assert approx_tokens(messages) == 100


def test_thinking_content_is_counted() -> None:
    # Thinking blocks are serialised back into the next request, so they count
    # toward the prompt — omitting them undercounts thinking-heavy runs.
    messages = [AssistantMessage(content=[ThinkingContent(thinking="x" * 200)])]

    assert approx_tokens(messages) == 100


def test_tool_call_arguments_are_counted() -> None:
    messages = [
        AssistantMessage(
            content=[
                ToolCall(
                    id="c1",
                    name="search",
                    arguments={"query": "x" * 20},
                )
            ],
        )
    ]

    assert approx_tokens(messages) > 0


# --- real_context_estimate: trigger anchored to real provider usage ---


def test_real_estimate_sums_all_usage_fields() -> None:
    # Under prompt caching most of the prompt is cache_read; input_tokens alone
    # is a large undercount. The estimate must sum input + cache_read + cache_write.
    messages = [
        UserMessage(content=[TextContent(text="x" * 1000)]),
        AssistantMessage(
            content=[],
            usage=Usage(
                input_tokens=1_000,
                cache_read_tokens=50_000,
                cache_write_tokens=2_000,
            ),
        ),
    ]
    # base = 1000 + 50000 + 2000 = 53000; the trailing assistant has empty
    # content so the char delta from it is 0.
    assert real_context_estimate(messages) == 53_000


def test_real_estimate_adds_char_delta_from_last_usage_onward() -> None:
    messages = [
        AssistantMessage(content=[], usage=Usage(input_tokens=10_000)),
        UserMessage(content=[TextContent(text="x" * 400)]),  # 200 tokens
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="t",
            content=[TextContent(text="x" * 200)],  # 100 tokens
        ),
    ]
    # base=10000 (only input present) + approx_tokens(messages[0:]) = 300.
    assert real_context_estimate(messages) == 10_300


def test_real_estimate_uses_most_recent_usage() -> None:
    messages = [
        AssistantMessage(content=[], usage=Usage(input_tokens=5_000)),
        UserMessage(content=[TextContent(text="x" * 100)]),
        AssistantMessage(content=[], usage=Usage(input_tokens=20_000)),
    ]
    # The most recent usage-bearing assistant (index 2) wins; nothing follows it.
    assert real_context_estimate(messages) == 20_000


def test_real_estimate_cold_start_falls_back_to_approx() -> None:
    messages = [UserMessage(content=[TextContent(text="x" * 200)])]  # no usage
    assert real_context_estimate(messages) == approx_tokens(messages) == 100


def test_real_estimate_skips_zero_usage_error_assistant() -> None:
    # error/abort/partial assistants carry usage=Usage() (all zeros). Anchoring
    # on one would drop the earlier history and stop compaction after a failure.
    messages = [
        UserMessage(content=[TextContent(text="x" * 1000)]),
        AssistantMessage(content=[], usage=Usage(input_tokens=80_000)),
        ToolResultMessage(
            tool_call_id="c",
            tool_name="t",
            content=[TextContent(text="x" * 200)],  # 100 tokens
        ),
        AssistantMessage(content=[TextContent(text="boom")], usage=Usage()),  # zero-sum
    ]
    # Anchors on the real 80_000 at index 1, not the zero error at index 3:
    # 80000 + approx_tokens(messages[1:]) = 80000 + (0 + 100 + 2) = 80102.
    assert real_context_estimate(messages) == 80_102


def test_real_estimate_only_zero_usage_falls_back_to_approx() -> None:
    messages = [
        UserMessage(content=[TextContent(text="x" * 200)]),  # 100 tokens
        AssistantMessage(content=[], usage=Usage()),  # zero-sum, not an anchor
    ]
    assert real_context_estimate(messages) == approx_tokens(messages) == 100


def test_real_estimate_falls_back_when_no_usage_reported() -> None:
    # A provider that never emits usage leaves every assistant with usage=None.
    # The estimate must degrade to the plain char estimate — not crash, not
    # anchor on a missing measurement, not undercount.
    messages = [
        UserMessage(content=[TextContent(text="x" * 400)]),  # 200 tokens
        AssistantMessage(content=[TextContent(text="x" * 200)]),  # 100, usage=None
        UserMessage(content=[TextContent(text="x" * 200)]),  # 100 tokens
        AssistantMessage(content=[TextContent(text="x" * 100)]),  # 50, usage=None
    ]
    assert real_context_estimate(messages) == approx_tokens(messages) == 450
