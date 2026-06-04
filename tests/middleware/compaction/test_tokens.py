from __future__ import annotations

from cubepi.middleware.compaction.tokens import approx_tokens
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def test_empty_returns_zero() -> None:
    assert approx_tokens([]) == 0


def test_text_chars_use_conservative_cjk_safe_estimate() -> None:
    messages = [UserMessage(content=[TextContent(text="x" * 200)])]

    assert approx_tokens(messages) == 100


def test_usage_metadata_scales_estimate_up() -> None:
    messages = [
        UserMessage(content=[TextContent(text="x" * 900)]),
        AssistantMessage(
            content=[TextContent(text="y" * 100)],
            usage=Usage(input_tokens=400, output_tokens=10),
        ),
    ]

    assert approx_tokens(messages) == 625


def test_tool_result_text_is_counted() -> None:
    messages = [
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="search",
            content=[TextContent(text="x" * 200)],
        )
    ]

    assert approx_tokens(messages) == 100
