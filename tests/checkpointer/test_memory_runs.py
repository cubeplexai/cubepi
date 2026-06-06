import pytest

from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
)
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.base import TextContent, UserMessage


@pytest.mark.asyncio
async def test_claim_then_complete_roundtrip():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "r1")
    await cp.mark_run_complete("t", "r1")
    # Idempotent: second mark is a no-op.
    await cp.mark_run_complete("t", "r1")


@pytest.mark.asyncio
async def test_claim_collision_in_flight_raises_claimed():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "r1")
    with pytest.raises(RunAlreadyClaimedError):
        await cp.claim_run("t", "r1")


@pytest.mark.asyncio
async def test_claim_collision_completed_raises_completed():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "r1")
    await cp.mark_run_complete("t", "r1")
    with pytest.raises(RunAlreadyCompletedError):
        await cp.claim_run("t", "r1")


@pytest.mark.asyncio
async def test_mark_without_claim_raises_not_claimed():
    cp = MemoryCheckpointer()
    with pytest.raises(RunNotClaimedError):
        await cp.mark_run_complete("t", "r1")


@pytest.mark.asyncio
async def test_completion_seq_monotonic_per_thread():
    cp = MemoryCheckpointer()
    for rid in ("A", "B", "C"):
        await cp.claim_run("t", rid)
        await cp.mark_run_complete("t", rid)
    # Internal inspection: completion_seq for A < B < C strictly.
    seqs = [cp._runs["t"][rid].completion_seq for rid in ("A", "B", "C")]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == 3


@pytest.mark.asyncio
async def test_append_on_completed_run_id_rejected():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "r1")
    await cp.mark_run_complete("t", "r1")
    msg = UserMessage(content=[TextContent(text="late")], run_id="r1")
    with pytest.raises(RunAlreadyCompletedError):
        await cp.append("t", [msg])


@pytest.mark.asyncio
async def test_append_in_flight_run_id_ok():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "r1")
    msg = UserMessage(content=[TextContent(text="ok")], run_id="r1")
    await cp.append("t", [msg])
    data = await cp.load("t")
    assert data is not None and len(data.messages) == 1


@pytest.mark.asyncio
async def test_load_pending_returns_tuple_with_run_id():
    from cubepi.hitl.types import ConfirmRequest, HitlRequest

    cp = MemoryCheckpointer()
    req = HitlRequest(
        question_id="q1",
        thread_id="t",
        payload=ConfirmRequest(prompt="hi"),
        created_at=0.0,
    )
    await cp.save_pending_request("t", req, run_id="r-1")
    res = await cp.load_pending("t")
    assert res is not None
    got_req, got_run_id = res
    assert got_req.question_id == "q1"
    assert got_run_id == "r-1"


@pytest.mark.asyncio
async def test_load_pending_returns_none_when_empty():
    cp = MemoryCheckpointer()
    assert await cp.load_pending("t") is None
