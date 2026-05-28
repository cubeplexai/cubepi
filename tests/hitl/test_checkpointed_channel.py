from __future__ import annotations

import asyncio

import pytest

from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl import ApproveAnswer, HitlDurabilityNotGuaranteed, HitlError
from cubepi.hitl.channel import CheckpointedChannel


async def test_checkpointed_persists_pending_on_ask():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1")

    async def host():
        while True:
            if await cp.load_pending_request("t-1") is not None:
                break
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    ans = await ch.approve(tool_name="bash", tool_call_id="tc-1", args={})
    assert ans.decision == "approve"
    # On success, pending should be cleared from the checkpointer.
    assert await cp.load_pending_request("t-1") is None


async def test_checkpointed_durability_guard_rejects_inside_custom_tool():
    from cubepi.hitl.channel import _in_custom_tool_var

    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1")
    token = _in_custom_tool_var.set(True)
    try:
        with pytest.raises(HitlDurabilityNotGuaranteed):
            await ch.confirm("ok?", timeout=0.05)
    finally:
        _in_custom_tool_var.reset(token)


async def test_checkpointed_durability_optin_allows():
    from cubepi.hitl.channel import _in_custom_tool_var

    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(
        checkpointer=cp,
        thread_id="t-1",
        allow_inside_custom_tool=True,
    )
    token = _in_custom_tool_var.set(True)

    async def host():
        while True:
            if await cp.load_pending_request("t-1") is not None:
                break
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, True)

    try:
        asyncio.create_task(host())
        assert await ch.confirm("ok?") is True
    finally:
        _in_custom_tool_var.reset(token)


async def test_detach_leaves_pending_persisted():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1")

    async def detacher():
        while ch.pending is None:
            await asyncio.sleep(0)
        if ch._future is not None and not ch._future.done():
            from cubepi.hitl.exceptions import HitlDetached

            ch._future.set_exception(HitlDetached())

    asyncio.create_task(detacher())
    from cubepi.hitl.exceptions import HitlDetached

    with pytest.raises(HitlDetached):
        await ch.confirm("ok?")
    # Persisted state must remain on detach (cross-process suspend).
    assert await cp.load_pending_request("t-1") is not None


def test_checkpointed_channel_public_export():
    from cubepi.hitl import CheckpointedChannel as Exported

    assert Exported is CheckpointedChannel


def test_checkpointed_requires_hitl_methods_on_checkpointer():
    """CheckpointedChannel.__init__ validates the checkpointer has
    save_pending_request and load_pending_request methods (codex pass 3)."""

    class _BareCheckpointer:
        pass

    with pytest.raises(HitlError):
        CheckpointedChannel(checkpointer=_BareCheckpointer(), thread_id="t-1")
