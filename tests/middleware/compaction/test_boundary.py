from __future__ import annotations

from cubepi.middleware.compaction.boundary import (
    safe_boundary,
    tail_start_by_tokens,
)
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


def _big_user(chars: int) -> UserMessage:
    return UserMessage(content=[TextContent(text="x" * chars)])


# --- safe_boundary with explicit tail_start ---


def test_returns_boundary_at_user_message_start() -> None:
    messages: list[Message] = [
        _user("q1"),
        _assistant("a1"),
        _user("q2"),
        _assistant("a2"),
        _user("q3"),
        _assistant("a3"),
    ]
    # tail_start=4 → msgs[4:] are the protected tail, walk back from 4 → 4 is
    # already a UserMessage, suffix [4..] self-contained, return 4.
    assert safe_boundary(messages, tail_start=4, min_compact=1) == 4


def test_no_safe_boundary_when_tail_start_zero() -> None:
    messages: list[Message] = [_user("q1"), _assistant("a1")]
    assert safe_boundary(messages, tail_start=0) is None


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
    # tail_start=4 brings the orphan ToolResult into the tail — no safe split.
    assert safe_boundary(messages, tail_start=4, min_compact=1) is None


def test_accepts_suffix_with_matching_tool_call_and_result() -> None:
    call = ToolCall(id="c1", name="search", arguments={})
    messages: list[Message] = [
        _user("q1"),
        _assistant("a1"),
        _user("q2"),
        _assistant(tool_calls=[call]),
        _tool_result("c1"),
        _assistant("done"),
    ]
    # tail_start=3 lands inside a tool-call/result pair (3 is the assistant
    # with the call). Walk back: 3 not UserMessage → 2 is UserMessage,
    # suffix [2..] is self-contained → return 2.
    assert safe_boundary(messages, tail_start=3, min_compact=1) == 2


def test_enforces_min_compact() -> None:
    messages: list[Message] = [
        _user("q1"),
        _assistant("a1"),
        _user("q2"),
        _assistant("a2"),
    ]
    assert safe_boundary(messages, tail_start=2, min_compact=3) is None
    assert safe_boundary(messages, tail_start=2, min_compact=1) == 2


# --- tail_start_by_tokens ---


def test_tail_start_protects_what_fits_in_budget() -> None:
    # 4 messages × ~1000 tokens each. Budget 1500 fits only the last:
    # i=3 → accum 0+1000=1000 (not > 1500), accum=1000.
    # i=2 → 1000+1000=2000 > 1500 AND accum>0 → return 3.
    msgs = [_big_user(2000) for _ in range(4)]
    assert tail_start_by_tokens(msgs, 1500) == 3


def test_tail_start_all_fit_returns_zero() -> None:
    msgs = [_big_user(10) for _ in range(3)]
    assert tail_start_by_tokens(msgs, 100_000) == 0


def test_tail_start_oversized_last_message_still_in_tail() -> None:
    # The final message alone exceeds budget — must still end up in the tail.
    msgs = [_big_user(100), _big_user(100), _big_user(100_000)]
    assert tail_start_by_tokens(msgs, 1000) == 2


def test_tail_start_empty_input() -> None:
    assert tail_start_by_tokens([], 1000) == 0


def test_tail_start_returns_index_in_bounds() -> None:
    msgs = [_big_user(2000) for _ in range(5)]
    result = tail_start_by_tokens(msgs, 500)
    assert 0 <= result <= len(msgs) - 1


# --- safe_boundary + tail_start interaction (the documented composition) ---


def test_safe_boundary_using_tail_start_by_tokens() -> None:
    messages: list[Message] = [
        _user("q1"),
        _assistant("a1"),
        _user("q2"),
        _assistant("a2"),
        _user("q3"),
        _assistant("a3"),
    ]
    tail_start = tail_start_by_tokens(messages, budget=1)  # tiny budget → tail=5
    assert tail_start == 5
    # safe_boundary then walks back to find the latest UserMessage <= 5.
    assert safe_boundary(messages, tail_start=tail_start, min_compact=1) == 4


def test_safe_boundary_rejects_out_of_range_tail_start() -> None:
    messages: list[Message] = [_user("q")]
    # Negative or too-large tail_start → None (no work to do).
    assert safe_boundary(messages, tail_start=-1) is None
    assert safe_boundary(messages, tail_start=99) is None
