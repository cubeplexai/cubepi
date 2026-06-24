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
    pruned, preserved = prune_tool_results(msgs, tail_start=4)
    assert "bash" in pruned[2].content[0].text
    assert "chars" in pruned[2].content[0].text
    assert pruned[5].content[0].text == "ok2"
    assert preserved == {}


def test_large_result_replaced_with_one_liner() -> None:
    big = "x" * 5000
    msgs = [
        _user(),
        _assistant_with_call("read_file", "c1"),
        _result("c1", big, "read_file"),
        _user(),
    ]
    pruned, preserved = prune_tool_results(msgs, tail_start=3)
    result_text = pruned[2].content[0].text
    assert len(result_text) < 200
    assert "read_file" in result_text
    assert "5000" in result_text or "chars" in result_text
    assert preserved == {}


def test_tail_messages_kept_intact() -> None:
    big = "x" * 5000
    msgs = [
        _user(),
        _assistant_with_call("bash", "c1"),
        _result("c1", big, "bash"),
    ]
    pruned, preserved = prune_tool_results(msgs, tail_start=0)
    assert pruned[2].content[0].text == big
    assert preserved == {}


def test_result_already_short_kept_intact() -> None:
    msgs = [
        _user(),
        _assistant_with_call("bash", "c1"),
        _result("c1", "exit 0", "bash"),
        _user(),
    ]
    pruned, preserved = prune_tool_results(msgs, tail_start=3)
    assert pruned[2].content[0].text == "exit 0"
    assert preserved == {}


def test_non_tool_result_messages_untouched() -> None:
    msgs = [_user("hello"), _user("world")]
    pruned, preserved = prune_tool_results(msgs, tail_start=len(msgs))
    assert pruned == msgs
    assert preserved == {}


def test_does_not_mutate_input() -> None:
    big = "x" * 5000
    original = _result("c1", big, "bash")
    msgs = [_user(), _assistant_with_call("bash", "c1"), original, _user()]
    prune_tool_results(msgs, tail_start=3)
    assert original.content[0].text == big
    assert msgs[2] is original


# --- compressor tests ---


def test_compressor_returning_str_preserves_message() -> None:
    big = "x" * 5000
    msgs = [
        _user(),
        _assistant_with_call("chip_metrics", "c1"),
        _result("c1", big, "chip_metrics"),
        _user(),
    ]

    def compressor(msg: ToolResultMessage) -> str | None:
        if msg.tool_name == "chip_metrics":
            return msg.content[0].text
        return None

    pruned, preserved = prune_tool_results(msgs, tail_start=3, compressor=compressor)
    assert preserved == {2: big}
    assert "preserved" in pruned[2].content[0].text


def test_compressor_returning_none_falls_through_to_default() -> None:
    big = "x" * 5000
    msgs = [
        _user(),
        _assistant_with_call("bash", "c1"),
        _result("c1", big, "bash"),
        _user(),
    ]

    def compressor(msg: ToolResultMessage) -> str | None:
        return None

    pruned, preserved = prune_tool_results(msgs, tail_start=3, compressor=compressor)
    assert preserved == {}
    assert "chars" in pruned[2].content[0].text


def test_compressor_mixed_preserve_and_prune() -> None:
    big = "x" * 5000
    msgs = [
        _user(),
        _assistant_with_call("chip_metrics", "c1"),
        _result("c1", big, "chip_metrics"),
        _assistant_with_call("bash", "c2"),
        _result("c2", big, "bash"),
        _user(),
    ]

    def compressor(msg: ToolResultMessage) -> str | None:
        if msg.tool_name == "chip_metrics":
            return "important data"
        return None

    pruned, preserved = prune_tool_results(msgs, tail_start=5, compressor=compressor)
    assert preserved == {2: "important data"}
    assert "preserved" in pruned[2].content[0].text
    assert "chars" in pruned[4].content[0].text


def test_compressor_not_called_for_tail_messages() -> None:
    big = "x" * 5000
    calls: list[str] = []

    def compressor(msg: ToolResultMessage) -> str | None:
        calls.append(msg.tool_name)
        return "kept"

    msgs = [
        _user(),
        _assistant_with_call("chip_metrics", "c1"),
        _result("c1", big, "chip_metrics"),
        _user(),
        _assistant_with_call("chip_metrics", "c2"),
        _result("c2", big, "chip_metrics"),
    ]
    pruned, preserved = prune_tool_results(msgs, tail_start=3, compressor=compressor)
    assert calls == ["chip_metrics"]
    assert 2 in preserved
    assert 5 not in preserved
    assert pruned[5].content[0].text == big


def test_compressor_short_message_still_checked() -> None:
    """Compressor is called even for short messages (< _PRUNE_KEEP_CHARS)."""
    msgs = [
        _user(),
        _assistant_with_call("chip_metrics", "c1"),
        _result("c1", "short", "chip_metrics"),
        _user(),
    ]

    def compressor(msg: ToolResultMessage) -> str | None:
        return "preserved short"

    pruned, preserved = prune_tool_results(msgs, tail_start=3, compressor=compressor)
    assert preserved == {2: "preserved short"}
