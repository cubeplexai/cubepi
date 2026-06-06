import pytest

from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
)
from cubepi.checkpointer.memory import MemoryCheckpointer


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
