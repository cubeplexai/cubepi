from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl.channel import CheckpointedChannel, InMemoryChannel
from cubepi.hitl.middleware import ApprovalPolicyMiddleware, ConfirmToolCallMiddleware
from cubepi.hitl.policy import Approve


def _approve_policy(ctx):
    return Approve()


def test_approval_policy_with_checkpointed_channel_sets_binding():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    mw = ApprovalPolicyMiddleware(channel=ch, policy=_approve_policy)
    assert mw.hitl is not None
    assert mw.hitl.checkpointed is True
    assert mw.hitl.run_id == "R1"


def test_approval_policy_with_in_memory_channel_sets_binding():
    ch = InMemoryChannel(thread_id="t")
    mw = ApprovalPolicyMiddleware(channel=ch, policy=_approve_policy)
    assert mw.hitl is not None
    assert mw.hitl.checkpointed is False
    assert mw.hitl.run_id is None


def test_confirm_tool_call_middleware_inherits_binding():
    """ConfirmToolCallMiddleware is a subclass of ApprovalPolicyMiddleware.
    It should inherit the same binding behavior."""
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    mw = ConfirmToolCallMiddleware(channel=ch)
    assert mw.hitl is not None
    assert mw.hitl.checkpointed is True
    assert mw.hitl.run_id == "R1"
