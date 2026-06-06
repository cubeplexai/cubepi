from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl.ask_user import ask_user_tool
from cubepi.hitl.channel import CheckpointedChannel, InMemoryChannel


def test_ask_user_tool_with_checkpointed_channel_sets_binding():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool = ask_user_tool(ch)
    assert tool.hitl is not None
    assert tool.hitl.checkpointed is True
    assert tool.hitl.run_id == "R1"


def test_ask_user_tool_with_in_memory_channel_sets_binding():
    ch = InMemoryChannel(thread_id="t")
    tool = ask_user_tool(ch)
    assert tool.hitl is not None
    assert tool.hitl.checkpointed is False
    assert tool.hitl.run_id is None
