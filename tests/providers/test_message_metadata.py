"""Message.metadata field tests (D5)."""

import pytest
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def test_user_message_default_metadata_is_empty_dict() -> None:
    msg = UserMessage(content=[TextContent(text="hi")])
    assert msg.metadata == {}


def test_assistant_message_default_metadata_is_empty_dict() -> None:
    msg = AssistantMessage(content=[], usage=Usage())
    assert msg.metadata == {}


def test_tool_result_message_default_metadata_is_empty_dict() -> None:
    msg = ToolResultMessage(content=[], tool_call_id="tc-1")
    assert msg.metadata == {}


def test_user_message_accepts_metadata() -> None:
    msg = UserMessage(
        content=[TextContent(text="hi")],
        metadata={"memory_snapshot": {"captured_at": "t1", "ids": ["m1"]}},
    )
    assert msg.metadata["memory_snapshot"]["captured_at"] == "t1"


def test_metadata_independent_between_instances() -> None:
    a = UserMessage(content=[TextContent(text="a")])
    b = UserMessage(content=[TextContent(text="b")])
    a.metadata["x"] = 1
    assert "x" not in b.metadata


def test_metadata_serializes_to_dict_in_model_dump() -> None:
    msg = UserMessage(
        content=[TextContent(text="hi")],
        metadata={"k": "v"},
    )
    dumped = msg.model_dump()
    assert dumped["metadata"] == {"k": "v"}
