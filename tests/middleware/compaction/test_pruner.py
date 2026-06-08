from __future__ import annotations

from cubepi.middleware.compaction.pruner import prune_tool_results
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def _user(text: str = "hi") -> UserMessage:
    return UserMessage(content=[TextContent(text=text)])


def _assistant_with_call(tool_name: str, call_id: str) -> AssistantMessage:
    return AssistantMessage(
        content=[ToolCall(id=call_id, name=tool_name, arguments={})]
    )


def _result(call_id: str, text: str, tool_name: str = "tool") -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=call_id,
        tool_name=tool_name,
        content=[TextContent(text=text)],
    )


def test_large_result_outside_tail_replaced_with_one_liner() -> None:
    big = "x" * 5000
    msgs = [
        _user(),
        _assistant_with_call("bash", "c1"),
        _result("c1", big, "bash"),
        _user(),
        _assistant_with_call("bash", "c2"),
        _result("c2", "ok2", "bash"),
    ]
    pruned = prune_tool_results(msgs, tail_start=4)
    assert "bash" in pruned[2].content[0].text
    assert "chars" in pruned[2].content[0].text
    assert pruned[5].content[0].text == "ok2"


def test_large_result_replaced_with_one_liner() -> None:
    big = "x" * 5000
    msgs = [
        _user(),
        _assistant_with_call("read_file", "c1"),
        _result("c1", big, "read_file"),
        _user(),
    ]
    pruned = prune_tool_results(msgs, tail_start=3)
    result_text = pruned[2].content[0].text
    assert len(result_text) < 200
    assert "read_file" in result_text
    assert "5000" in result_text or "chars" in result_text


def test_tail_messages_kept_intact() -> None:
    big = "x" * 5000
    msgs = [
        _user(),
        _assistant_with_call("bash", "c1"),
        _result("c1", big, "bash"),
    ]
    pruned = prune_tool_results(msgs, tail_start=0)
    assert pruned[2].content[0].text == big


def test_result_already_short_kept_intact() -> None:
    msgs = [
        _user(),
        _assistant_with_call("bash", "c1"),
        _result("c1", "exit 0", "bash"),
        _user(),
    ]
    pruned = prune_tool_results(msgs, tail_start=3)
    assert pruned[2].content[0].text == "exit 0"


def test_non_tool_result_messages_untouched() -> None:
    msgs = [_user("hello"), _user("world")]
    assert prune_tool_results(msgs, tail_start=len(msgs)) == msgs


def test_does_not_mutate_input() -> None:
    big = "x" * 5000
    original = _result("c1", big, "bash")
    msgs = [_user(), _assistant_with_call("bash", "c1"), original, _user()]
    prune_tool_results(msgs, tail_start=3)
    # The pruner returns a NEW list with model_copy()'d messages; originals
    # remain referenced from the caller's list and must be unchanged.
    assert original.content[0].text == big
    assert msgs[2] is original
