from __future__ import annotations

from cubepi.middleware.compaction.boundary import safe_boundary
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def _user(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)])


def _assistant(
    text: str = "", tool_calls: list[ToolCall] | None = None
) -> AssistantMessage:
    content = []
    if text:
        content.append(TextContent(text=text))
    if tool_calls:
        content.extend(tool_calls)
    return AssistantMessage(content=content)


def _tool_result(call_id: str) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=call_id,
        tool_name="tool",
        content=[TextContent(text="ok")],
    )


def test_returns_boundary_at_user_message_start() -> None:
    messages: list[Message] = [
        _user("q1"),
        _assistant("a1"),
        _user("q2"),
        _assistant("a2"),
        _user("q3"),
        _assistant("a3"),
    ]

    assert safe_boundary(messages, keep_recent=2, min_compact=1) == 4


def test_rejects_suffix_with_orphan_tool_result() -> None:
    call = ToolCall(id="c1", name="search", arguments={})
    messages: list[Message] = [
        _user("q1"),
        _assistant(tool_calls=[call]),
        _tool_result("c1"),
        _user("q2"),
        _tool_result("orphan"),
        _assistant("done"),
    ]

    assert safe_boundary(messages, keep_recent=2, min_compact=1) is None


def test_enforces_min_compact() -> None:
    messages: list[Message] = [
        _user("q1"),
        _assistant("a1"),
        _user("q2"),
        _assistant("a2"),
    ]

    assert safe_boundary(messages, keep_recent=2, min_compact=3) is None
    assert safe_boundary(messages, keep_recent=2, min_compact=1) == 2
