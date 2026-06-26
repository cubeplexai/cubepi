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


def test_accepts_assistant_turn_start_as_boundary() -> None:
    call = ToolCall(id="c1", name="search", arguments={})
    messages: list[Message] = [
        _user("q1"),
        _assistant("a1"),
        _user("q2"),
        _assistant(tool_calls=[call]),
        _tool_result("c1"),
        _assistant("done"),
    ]
    # tail_start=3 is the assistant that opens a tool turn. Its suffix
    # [assistant(c1), tool_result(c1), assistant(done)] is self-contained, so
    # the boundary lands there (closest to the tail) rather than walking all the
    # way back to the user message at index 2.
    assert safe_boundary(messages, tail_start=3, min_compact=1) == 3


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
    # safe_boundary walks back to the latest self-contained turn boundary <= 5;
    # the last assistant (index 5) has a self-contained suffix, so it wins.
    assert safe_boundary(messages, tail_start=tail_start, min_compact=1) == 5


def test_safe_boundary_rejects_out_of_range_tail_start() -> None:
    messages: list[Message] = [_user("q")]
    # Negative or too-large tail_start → None (no work to do).
    assert safe_boundary(messages, tail_start=-1) is None
    assert safe_boundary(messages, tail_start=99) is None


def test_safe_boundary_tail_start_equals_len_clamps_in_bounds() -> None:
    """When tail_start == len(messages), the search starts at the last valid
    index — exercises the clamp at boundary.py's ``candidate -= 1``."""
    messages: list[Message] = [
        _user("q1"),
        _assistant("a1"),
        _user("q2"),
        _assistant("a2"),
    ]
    # tail_start == 4 == len(messages). Clamp to 3; index 3 (assistant a2) has a
    # self-contained suffix, so the boundary lands there.
    assert safe_boundary(messages, tail_start=4, min_compact=1) == 3


# --- run-scoped compaction: cut at turn boundaries inside a single run ---


def _run_with_tool_chain() -> list[Message]:
    """A long run with NO intervening UserMessage — just a tool-call chain."""
    return [
        _user("open"),
        _assistant(tool_calls=[ToolCall(id="c1", name="t", arguments={})]),
        _tool_result("c1"),
        _assistant(tool_calls=[ToolCall(id="c2", name="t", arguments={})]),
        _tool_result("c2"),
        _assistant(tool_calls=[ToolCall(id="c3", name="t", arguments={})]),
        _tool_result("c3"),
    ]


def test_run_scoped_cut_lands_on_inner_turn_not_run_start() -> None:
    messages = _run_with_tool_chain()
    # Protect only the last turn ([a(c3), tool_result c3]); the boundary must
    # advance to the assistant that opens that turn (index 5), NOT stay pinned to
    # the run's opening user message at index 0.
    boundary = safe_boundary(messages, tail_start=5, min_compact=1)
    assert boundary == 5
    assert _is_self_contained(messages[boundary:])


def test_run_scoped_never_splits_a_multi_tool_turn() -> None:
    messages: list[Message] = [
        _user("open"),
        _assistant(
            tool_calls=[
                ToolCall(id="A", name="t", arguments={}),
                ToolCall(id="B", name="t", arguments={}),
            ]
        ),
        _tool_result("A"),
        _tool_result("B"),
        _assistant(tool_calls=[ToolCall(id="C", name="t", arguments={})]),
        _tool_result("C"),
    ]
    # Searching back from the tail must skip the tool_result for C (orphan) and
    # land on the assistant opening the C turn (index 4) — never between the A/B
    # results of the multi-tool turn.
    boundary = safe_boundary(messages, tail_start=5, min_compact=1)
    assert boundary == 4
    assert _is_self_contained(messages[boundary:])


def test_every_returned_boundary_is_self_contained() -> None:
    """Core invariant: whatever index safe_boundary returns, the kept suffix
    never orphans a tool_use/tool_result pair — across every tail_start."""
    messages = _run_with_tool_chain()
    for tail_start in range(1, len(messages) + 1):
        boundary = safe_boundary(messages, tail_start=tail_start, min_compact=1)
        if boundary is None:
            continue
        assert not isinstance(messages[boundary], ToolResultMessage)
        assert _is_self_contained(messages[boundary:])


def _is_self_contained(suffix: list[Message]) -> bool:
    available: set[str] = set()
    for message in suffix:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolCall) and block.id:
                    available.add(block.id)
        elif isinstance(message, ToolResultMessage):
            if message.tool_call_id and message.tool_call_id not in available:
                return False
    return True
