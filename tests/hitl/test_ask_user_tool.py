import asyncio

from cubepi.hitl.ask_user import ask_user_tool
from cubepi.hitl.channel import InMemoryChannel


async def test_ask_user_tool_is_sequential():
    tool = ask_user_tool(InMemoryChannel())
    assert tool.name == "ask_user"
    assert tool.execution_mode == "sequential"


async def test_ask_user_tool_returns_answers_in_details():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, {"color": "red"})

    tool = ask_user_tool(ch)
    asyncio.create_task(host())
    result = await tool.execute(
        "tc-1",
        tool.parameters.model_validate(
            {
                "questions": [{"key": "color", "prompt": "Pick:"}],
            }
        ),
        signal=None,
        on_update=lambda p: None,
    )
    assert result.details["hitl"]["answers"] == {"color": "red"}
    # Content has a human-readable summary too
    assert "color" in result.content[0].text


async def test_ask_user_tool_cancel_becomes_tool_error():
    ch = InMemoryChannel()

    async def canceller():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.cancel(ch.pending.question_id, reason="closed tab")

    tool = ask_user_tool(ch)
    asyncio.create_task(canceller())
    result = await tool.execute(
        "tc-1",
        tool.parameters.model_validate({"questions": [{"key": "x", "prompt": "?"}]}),
        signal=None,
        on_update=lambda p: None,
    )
    assert result.is_error is True
    assert result.details["hitl"]["outcome"] == "cancelled"
    assert result.details["hitl"]["reason"] == "closed tab"


async def test_ask_user_tool_timeout_becomes_tool_error():
    ch = InMemoryChannel(default_timeout=0.05)
    tool = ask_user_tool(ch)
    result = await tool.execute(
        "tc-1",
        tool.parameters.model_validate({"questions": [{"key": "x", "prompt": "?"}]}),
        signal=None,
        on_update=lambda p: None,
    )
    assert result.is_error is True
    assert result.details["hitl"]["outcome"] == "timed_out"


async def test_ask_user_tool_multi_question_form():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, {"color": "red", "size": ["s", "l"]})

    tool = ask_user_tool(ch)
    asyncio.create_task(host())
    result = await tool.execute(
        "tc-1",
        tool.parameters.model_validate(
            {
                "questions": [
                    {"key": "color", "prompt": "Color?"},
                    {"key": "size", "prompt": "Sizes?", "multi_select": True},
                ],
            }
        ),
        signal=None,
        on_update=lambda p: None,
    )
    assert result.details["hitl"]["answers"]["size"] == ["s", "l"]
