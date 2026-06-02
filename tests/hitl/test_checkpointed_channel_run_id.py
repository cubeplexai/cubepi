"""Tests for CheckpointedChannel.run_id — durable cross-process correlation
of a paused HITL request to the (cubebox/host) run that produced it.

The channel takes ``run_id`` at construction; whenever the channel writes
the pending request through the checkpointer (``save_pending_request``),
it must pass ``run_id=self._run_id`` so the run_id is persisted in the
same atomic statement as the pending JSON. This avoids the race where a
worker crashes between "wrote pending" and "wrote run_id".
"""

from __future__ import annotations

import asyncio

import pytest

from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl import ApproveAnswer
from cubepi.hitl.channel import CheckpointedChannel
from cubepi.hitl.types import Question


async def test_run_id_persisted_on_approve():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1", run_id="r-abc")

    async def host():
        while await cp.load_pending_request("t-1") is None:
            await asyncio.sleep(0)
        # run_id must be readable from the same checkpointer state.
        assert await cp.load_pending_run_id("t-1") == "r-abc"
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    ans = await ch.approve(tool_name="bash", tool_call_id="tc-1", args={})
    assert ans.decision == "approve"
    # On success, pending + run_id are both cleared.
    assert await cp.load_pending_request("t-1") is None
    assert await cp.load_pending_run_id("t-1") is None


async def test_run_id_persisted_on_confirm():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-2", run_id="r-xyz")

    async def host():
        while await cp.load_pending_request("t-2") is None:
            await asyncio.sleep(0)
        assert await cp.load_pending_run_id("t-2") == "r-xyz"
        await ch.answer(ch.pending.question_id, True)

    asyncio.create_task(host())
    result = await ch.confirm("ok?")
    assert result is True


async def test_run_id_persisted_on_ask():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-3", run_id="r-ask")

    async def host():
        while await cp.load_pending_request("t-3") is None:
            await asyncio.sleep(0)
        assert await cp.load_pending_run_id("t-3") == "r-ask"
        await ch.answer(ch.pending.question_id, {"q1": "yes"})

    asyncio.create_task(host())
    result = await ch.ask([Question(key="q1", prompt="why?")])
    assert result == {"q1": "yes"}


async def test_no_run_id_kwarg_leaves_run_id_none():
    """Backwards compat: legacy CheckpointedChannel(no run_id=) → no run_id stored."""
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-4")

    async def host():
        while await cp.load_pending_request("t-4") is None:
            await asyncio.sleep(0)
        assert await cp.load_pending_run_id("t-4") is None
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="deny"))

    asyncio.create_task(host())
    await ch.approve(tool_name="bash", tool_call_id="tc-2", args={})


async def test_run_id_cleared_on_detach_stays_persisted():
    """On detach (HitlDetached) the pending row must remain — and so must run_id."""
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-5", run_id="r-detach")

    async def detacher():
        while ch.pending is None:
            await asyncio.sleep(0)
        from cubepi.hitl.exceptions import HitlDetached

        if ch._future is not None and not ch._future.done():
            ch._future.set_exception(HitlDetached())

    asyncio.create_task(detacher())
    from cubepi.hitl.exceptions import HitlDetached

    with pytest.raises(HitlDetached):
        await ch.confirm("ok?")
    # Both pending and run_id survive a detach (cross-process suspend).
    assert await cp.load_pending_request("t-5") is not None
    assert await cp.load_pending_run_id("t-5") == "r-detach"
