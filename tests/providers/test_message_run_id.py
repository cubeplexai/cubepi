from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    UserMessage,
)


def test_user_message_run_id_default_none():
    m = UserMessage(content=[TextContent(text="hi")])
    assert m.run_id is None


def test_assistant_message_run_id_default_none():
    m = AssistantMessage(content=[])
    assert m.run_id is None


def test_tool_result_message_run_id_default_none():
    m = ToolResultMessage(
        tool_call_id="tc1", tool_name="foo", content=[]
    )
    assert m.run_id is None


def test_run_id_round_trip_serialization():
    src = AssistantMessage(content=[], run_id="r-1")
    blob = src.model_dump_json()
    dst = AssistantMessage.model_validate_json(blob)
    assert dst.run_id == "r-1"


def test_run_id_is_keyword_only_in_practice():
    # All existing call sites pass content positionally / by keyword;
    # adding run_id as a defaulted field at the end is non-breaking.
    m = UserMessage(content=[TextContent(text="hi")], run_id="r-2")
    assert m.run_id == "r-2"
