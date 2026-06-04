from __future__ import annotations

import json

from cubepi.providers.base import (
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

_CHARS_PER_TOKEN = 2.0
_SCALE_MIN_TOKENS = 100


def approx_tokens(messages: list[Message]) -> int:
    """Conservative token estimate for the exact message view sent to the LLM."""
    if not messages:
        return 0

    total_chars = 0
    scale_factor: float | None = None

    for message in messages:
        if isinstance(message, UserMessage):
            for user_block in message.content:
                if isinstance(user_block, TextContent):
                    total_chars += len(user_block.text)
        elif isinstance(message, AssistantMessage):
            for assistant_block in message.content:
                if isinstance(assistant_block, TextContent):
                    total_chars += len(assistant_block.text)
                elif isinstance(assistant_block, ToolCall):
                    total_chars += len(json.dumps(assistant_block.arguments or {}))
            usage = message.usage
            if (
                usage
                and usage.input_tokens >= _SCALE_MIN_TOKENS
                and scale_factor is None
            ):
                chars_estimate = usage.input_tokens * _CHARS_PER_TOKEN
                if chars_estimate > 0:
                    raw_factor = total_chars / chars_estimate
                    scale_factor = max(1.0, min(raw_factor, 1.25))
        elif isinstance(message, ToolResultMessage):
            for result_block in message.content:
                if isinstance(result_block, TextContent):
                    total_chars += len(result_block.text)

    estimate = total_chars / _CHARS_PER_TOKEN
    if scale_factor is not None:
        return int(estimate * scale_factor)
    return int(estimate)
