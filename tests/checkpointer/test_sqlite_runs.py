import tempfile
from pathlib import Path

import pytest

from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
)
from cubepi.checkpointer.sqlite import SQLiteCheckpointer
from cubepi.providers.base import TextContent, UserMessage


@pytest.mark.asyncio
async def test_claim_then_complete_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            await cp.claim_run("t", "r1")
            await cp.mark_run_complete("t", "r1")
            # Idempotent: second mark is a no-op.
            await cp.mark_run_complete("t", "r1")


@pytest.mark.asyncio
async def test_claim_collision_in_flight_raises_claimed():
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            await cp.claim_run("t", "r1")
            with pytest.raises(RunAlreadyClaimedError):
                await cp.claim_run("t", "r1")


@pytest.mark.asyncio
async def test_claim_collision_completed_raises_completed():
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            await cp.claim_run("t", "r1")
            await cp.mark_run_complete("t", "r1")
            with pytest.raises(RunAlreadyCompletedError):
                await cp.claim_run("t", "r1")


@pytest.mark.asyncio
async def test_mark_without_claim_raises_not_claimed():
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            with pytest.raises(RunNotClaimedError):
                await cp.mark_run_complete("t", "r1")


@pytest.mark.asyncio
async def test_completion_seq_monotonic_per_thread():
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            for rid in ("A", "B", "C"):
                await cp.claim_run("t", rid)
                await cp.mark_run_complete("t", rid)
            cur = await cp._db.execute(
                "SELECT run_id, completion_seq FROM runs WHERE thread_id = ? "
                "ORDER BY completion_seq",
                ("t",),
            )
            rows = await cur.fetchall()
            assert [r[0] for r in rows] == ["A", "B", "C"]
            seqs = [r[1] for r in rows]
            assert seqs == sorted(seqs)
            assert len(set(seqs)) == 3


@pytest.mark.asyncio
async def test_append_on_completed_run_id_rejected():
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            await cp.claim_run("t", "r1")
            await cp.mark_run_complete("t", "r1")
            msg = UserMessage(content=[TextContent(text="late")], run_id="r1")
            with pytest.raises(RunAlreadyCompletedError):
                await cp.append("t", [msg])


@pytest.mark.asyncio
async def test_append_in_flight_run_id_ok():
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            await cp.claim_run("t", "r1")
            msg = UserMessage(content=[TextContent(text="ok")], run_id="r1")
            await cp.append("t", [msg])
            data = await cp.load("t")
            assert data is not None and len(data.messages) == 1


@pytest.mark.asyncio
async def test_load_pending_returns_tuple_with_run_id():
    from cubepi.hitl.types import ConfirmRequest, HitlRequest

    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
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
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            assert await cp.load_pending("t") is None
