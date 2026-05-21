"""AnthropicProvider configurable cache_policy tests (D3)."""

import pytest

from cubepi.providers.anthropic import (
    AnthropicProvider,
    DefaultCacheMarkerPolicy,
)
from cubepi.providers.base import Message, TextContent, UserMessage


def test_default_policy_marks_system() -> None:
    assert DefaultCacheMarkerPolicy().mark_system() is True


def test_default_policy_marks_last_tool() -> None:
    assert DefaultCacheMarkerPolicy().mark_last_tool() is True


def test_default_policy_indices_picks_last() -> None:
    p = DefaultCacheMarkerPolicy()
    msgs: list[Message] = [
        UserMessage(content=[TextContent(text="a")]),
        UserMessage(content=[TextContent(text="b")]),
    ]
    assert p.message_breakpoint_indices(msgs) == [1]


def test_default_policy_indices_empty() -> None:
    assert DefaultCacheMarkerPolicy().message_breakpoint_indices([]) == []


def test_provider_uses_default_policy_when_none_passed() -> None:
    p = AnthropicProvider(api_key="x")
    assert isinstance(p._cache_policy, DefaultCacheMarkerPolicy)


def test_provider_uses_custom_policy() -> None:
    class _NoSystem:
        def mark_system(self) -> bool:
            return False

        def mark_last_tool(self) -> bool:
            return False

        def message_breakpoint_indices(self, messages):
            return []

    p = AnthropicProvider(api_key="x", cache_policy=_NoSystem())
    assert p._cache_policy.mark_system() is False


def test_apply_breakpoint_markers_marks_specified_block() -> None:
    """Marker is applied to the targeted (message, block) position."""
    p = AnthropicProvider(api_key="x")
    api_messages = [
        {"role": "user", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": [{"type": "text", "text": "b"}]},
    ]
    p._apply_breakpoint_markers(
        api_messages, targets=[(0, 0)], cache_control={"type": "ephemeral"}
    )
    assert api_messages[0]["content"][0].get("cache_control") == {"type": "ephemeral"}
    # message 1 untouched
    assert "cache_control" not in api_messages[1]["content"][-1]


def test_apply_breakpoint_markers_marks_interior_block() -> None:
    """A target block that isn't the last one still gets marked precisely."""
    p = AnthropicProvider(api_key="x")
    api_messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t0"},
                {"type": "tool_result", "tool_use_id": "t1"},
            ],
        },
    ]
    p._apply_breakpoint_markers(
        api_messages, targets=[(0, 0)], cache_control={"type": "ephemeral"}
    )
    assert api_messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in api_messages[0]["content"][1]


def test_apply_breakpoint_markers_converts_string_content() -> None:
    """String content is converted to a block list and marked."""
    p = AnthropicProvider(api_key="x")
    api_messages = [
        {"role": "user", "content": "plain string"},
    ]
    p._apply_breakpoint_markers(
        api_messages, targets=[(0, 0)], cache_control={"type": "ephemeral"}
    )
    assert isinstance(api_messages[0]["content"], list)
    assert api_messages[0]["content"][0] == {
        "type": "text",
        "text": "plain string",
        "cache_control": {"type": "ephemeral"},
    }


@pytest.mark.asyncio
async def test_custom_policy_drives_message_marker_placement() -> None:
    """Custom policy that marks index 0 should result in marker on first message, not last."""

    class _FirstOnly:
        def mark_system(self) -> bool:
            return False

        def mark_last_tool(self) -> bool:
            return False

        def message_breakpoint_indices(self, messages):
            return [0] if messages else []

    p = AnthropicProvider(api_key="x", cache_policy=_FirstOnly())
    msgs: list[Message] = [
        UserMessage(content=[TextContent(text="zero")]),
        UserMessage(content=[TextContent(text="one")]),
    ]
    api_msgs, breakpoints = p._build_api_messages(msgs)
    targets = [breakpoints[i] for i in _FirstOnly().message_breakpoint_indices(msgs)]
    p._apply_breakpoint_markers(api_msgs, targets, cache_control={"type": "ephemeral"})

    first_blocks = api_msgs[0]["content"]
    assert isinstance(first_blocks, list)
    assert first_blocks[-1].get("cache_control") == {"type": "ephemeral"}

    second_blocks = api_msgs[1]["content"]
    if isinstance(second_blocks, list) and second_blocks:
        last = second_blocks[-1]
        if isinstance(last, dict):
            assert "cache_control" not in last
