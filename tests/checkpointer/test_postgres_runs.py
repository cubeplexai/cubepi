"""PostgresCheckpointer run-lifecycle tests."""

import asyncpg
import pytest

from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
)
from cubepi.checkpointer.postgres import PostgresCheckpointer
from cubepi.providers.base import TextContent, UserMessage


@pytest.mark.asyncio
async def test_claim_run_creates_threads_row_lazily(pg_v4_dsn):
    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        await cp.claim_run("t-lazy", "r1")
    conn = await asyncpg.connect(pg_v4_dsn)
    try:
        row = await conn.fetchrow(
            "SELECT thread_id FROM cubepi_threads WHERE thread_id = $1",
            "t-lazy",
        )
        assert row is not None
        run = await conn.fetchrow(
            "SELECT completed_at FROM cubepi_runs WHERE thread_id = $1 AND run_id = $2",
            "t-lazy",
            "r1",
        )
        assert run is not None
        assert run["completed_at"] is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_claim_collision_in_flight_raises_claimed(pg_v4_dsn):
    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        await cp.claim_run("t", "r1")
        with pytest.raises(RunAlreadyClaimedError):
            await cp.claim_run("t", "r1")


@pytest.mark.asyncio
async def test_append_on_completed_run_id_rejected(pg_v4_dsn):
    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        await cp.claim_run("t", "r1")
        await cp.mark_run_complete("t", "r1")
        msg = UserMessage(content=[TextContent(text="late")], run_id="r1")
        with pytest.raises(RunAlreadyCompletedError):
            await cp.append("t", [msg])


@pytest.mark.asyncio
async def test_claim_then_complete_roundtrip(pg_v4_dsn):
    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        await cp.claim_run("t", "r1")
        await cp.mark_run_complete("t", "r1")
        # Idempotent: second mark is a no-op.
        await cp.mark_run_complete("t", "r1")


@pytest.mark.asyncio
async def test_claim_collision_completed_raises_completed(pg_v4_dsn):
    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        await cp.claim_run("t", "r1")
        await cp.mark_run_complete("t", "r1")
        with pytest.raises(RunAlreadyCompletedError):
            await cp.claim_run("t", "r1")


@pytest.mark.asyncio
async def test_mark_without_claim_raises_not_claimed(pg_v4_dsn):
    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        with pytest.raises(RunNotClaimedError):
            await cp.mark_run_complete("t", "r1")


@pytest.mark.asyncio
async def test_completion_seq_monotonic_per_thread(pg_v4_dsn):
    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        for rid in ("A", "B", "C"):
            await cp.claim_run("t", rid)
            await cp.mark_run_complete("t", rid)
    conn = await asyncpg.connect(pg_v4_dsn)
    try:
        rows = await conn.fetch(
            "SELECT run_id, completion_seq FROM cubepi_runs "
            "WHERE thread_id = $1 ORDER BY completion_seq",
            "t",
        )
        assert [r["run_id"] for r in rows] == ["A", "B", "C"]
        seqs = [r["completion_seq"] for r in rows]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == 3
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_load_pending_returns_tuple_with_run_id(pg_v4_dsn):
    from cubepi.hitl.types import ConfirmRequest, HitlRequest

    async with PostgresCheckpointer(pg_v4_dsn) as cp:
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
async def test_load_pending_returns_none_when_empty(pg_v4_dsn):
    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        assert await cp.load_pending("t") is None


@pytest.mark.asyncio
async def test_append_persists_run_id_into_column(pg_v4_dsn):
    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        await cp.claim_run("t", "r1")
        msg = UserMessage(content=[TextContent(text="hi")], run_id="r1")
        await cp.append("t", [msg])
    conn = await asyncpg.connect(pg_v4_dsn)
    try:
        row = await conn.fetchrow(
            "SELECT run_id FROM cubepi_messages WHERE thread_id = $1",
            "t",
        )
        assert row is not None
        assert row["run_id"] == "r1"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_append_in_flight_run_id_ok(pg_v4_dsn):
    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        await cp.claim_run("t", "r1")
        msg = UserMessage(content=[TextContent(text="ok")], run_id="r1")
        await cp.append("t", [msg])
        data = await cp.load("t")
        assert data is not None
        assert len(data.messages) == 1
