import pytest

from cubepi.agent.agent import Agent
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from cubepi.providers.faux import FauxProvider


def _agent(**kw):
    return Agent(model=FauxProvider().model("faux-model"), **kw)


def test_messages_kw_none_keeps_default():
    a = _agent()
    assert a.state.messages == []


def test_messages_kw_seeds_initial_history():
    msgs = [UserMessage(content=[TextContent(text="hi")])]
    a = _agent(messages=msgs)
    assert len(a.state.messages) == 1


def test_messages_kw_conflicts_with_thread_id_checkpointer():
    from cubepi.checkpointer.memory import MemoryCheckpointer

    msgs = [UserMessage(content=[TextContent(text="hi")])]
    with pytest.raises(ValueError):
        _agent(
            messages=msgs,
            thread_id="t",
            checkpointer=MemoryCheckpointer(),
        )


def test_messages_kw_deep_copies_all_three_variants():
    user = UserMessage(content=[TextContent(text="u")], metadata={"k": "v"})
    assistant = AssistantMessage(
        content=[ToolCall(id="c1", name="t", arguments={"k": [1, 2]})],
        metadata={},
    )
    tool = ToolResultMessage(
        tool_call_id="c1", tool_name="t",
        content=[TextContent(text="r")], metadata={"x": 1},
    )
    a = _agent(messages=[user, assistant, tool])
    # Mutate originals.
    user.metadata["k"] = "MUT"
    assistant.content[0].arguments["k"].append(99)
    tool.metadata["x"] = 999
    # Internal copies untouched.
    assert a.state.messages[0].metadata["k"] == "v"
    assert a.state.messages[1].content[0].arguments["k"] == [1, 2]
    assert a.state.messages[2].metadata["x"] == 1
    # Mutate agent's copies; originals untouched.
    a.state.messages[0].metadata["k"] = "AGENT"
    assert user.metadata["k"] == "MUT"
