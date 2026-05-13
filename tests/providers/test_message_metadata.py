"""Message.metadata field tests (D5)."""

import tempfile
from pathlib import Path

import pytest
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from cubepi.checkpointer import MemoryCheckpointer, SQLiteCheckpointer


def test_user_message_default_metadata_is_empty_dict() -> None:
    msg = UserMessage(content=[TextContent(text="hi")])
    assert msg.metadata == {}


def test_assistant_message_default_metadata_is_empty_dict() -> None:
    msg = AssistantMessage(content=[], usage=Usage())
    assert msg.metadata == {}


def test_tool_result_message_default_metadata_is_empty_dict() -> None:
    msg = ToolResultMessage(content=[], tool_call_id="tc-1", tool_name="test_tool")
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


@pytest.mark.asyncio
async def test_memory_checkpointer_preserves_metadata() -> None:
    cp = MemoryCheckpointer()
    msg = UserMessage(
        content=[TextContent(text="hi")],
        metadata={"k": "v", "nested": {"a": 1}},
    )
    await cp.append("t1", [msg])
    loaded = await cp.load("t1")
    assert loaded is not None
    assert loaded.messages[0].metadata == {"k": "v", "nested": {"a": 1}}


@pytest.mark.asyncio
async def test_sqlite_checkpointer_preserves_metadata() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.db"
        async with SQLiteCheckpointer(str(path)) as cp:
            msg = UserMessage(
                content=[TextContent(text="hi")],
                metadata={"k": "v", "nested": {"a": 1}},
            )
            await cp.append("t1", [msg])
            loaded = await cp.load("t1")
        assert loaded is not None
        assert loaded.messages[0].metadata == {"k": "v", "nested": {"a": 1}}
