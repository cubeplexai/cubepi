from __future__ import annotations

import json

from cubepi.providers.base import (
    AssistantMessage,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

_CHARS_PER_TOKEN = 2.0


def approx_tokens(messages: list[Message]) -> int:
    """Conservative char-based token estimate for a message view.

    A *relative*-sizing helper: used for tail budgeting, pruning decisions, and
    the summary budget, where a consistent estimator bias cancels out. For the
    compaction *trigger* (an absolute count compared against a threshold) use
    :func:`real_context_estimate` instead — it is anchored to real provider
    usage rather than a character heuristic.
    """
    if not messages:
        return 0

    total_chars = 0
    for message in messages:
        if isinstance(message, UserMessage):
            for user_block in message.content:
                if isinstance(user_block, TextContent):
                    total_chars += len(user_block.text)
        elif isinstance(message, AssistantMessage):
            for assistant_block in message.content:
                if isinstance(assistant_block, TextContent):
                    total_chars += len(assistant_block.text)
                elif isinstance(assistant_block, ThinkingContent):
                    # Thinking blocks are serialised back into the next request
                    # (extended thinking + tool use), so they count toward the
                    # next prompt — omitting them undercounts thinking-heavy runs.
                    total_chars += len(assistant_block.thinking)
                elif isinstance(assistant_block, ToolCall):
                    total_chars += len(json.dumps(assistant_block.arguments or {}))
        elif isinstance(message, ToolResultMessage):
            for result_block in message.content:
                if isinstance(result_block, TextContent):
                    total_chars += len(result_block.text)

    return int(total_chars / _CHARS_PER_TOKEN)


def real_context_estimate(messages: list[Message]) -> int:
    """Best estimate of the true context fill, for the compaction trigger.

    Anchored to real provider usage: walks backward to the most recent
    ``AssistantMessage`` whose usage is a real measurement and takes that turn's
    actual prompt size — ``input_tokens + cache_read_tokens + cache_write_tokens``.
    cubepi normalises ``input_tokens`` to the *uncached* portion on every
    provider, so under prompt caching most of the prompt lives in the cache
    fields; all three must be summed or the count is a large undercount. The
    char estimate of everything from that assistant onward (its own output,
    which becomes input next turn, plus any messages appended since) is added on
    top.

    A zero-sum usage is *not* a measurement: error / abort / partial assistant
    messages carry ``usage=Usage()`` (see agent.py, faux.py, the *_responses
    providers). Anchoring on one would drop the entire earlier history from the
    estimate and silently stop compaction right after a failure, so the walk
    skips them and keeps looking for a real anchor.

    Falls back to :func:`approx_tokens` when no measured usage is present — the
    cold start before the first model response, or a tail of only zero-usage
    error messages.
    """
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if isinstance(message, AssistantMessage) and message.usage is not None:
            usage = message.usage
            base = (
                usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
            )
            if base > 0:
                return base + approx_tokens(messages[i:])
    return approx_tokens(messages)
