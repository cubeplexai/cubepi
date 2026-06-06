"""PostgresCheckpointer run-lifecycle tests."""

import asyncpg
import pytest

from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
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
            "SELECT completed_at FROM cubepi_runs "
            "WHERE thread_id = $1 AND run_id = $2",
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
        # Mark complete via direct UPDATE — mark_run_complete arrives in Task 16.
        # We exercise the append pre-flight independently here.
        conn = await asyncpg.connect(pg_v4_dsn)
        try:
            await conn.execute(
                "UPDATE cubepi_runs SET completed_at = now(), completion_seq = 1 "
                "WHERE thread_id = $1 AND run_id = $2",
                "t",
                "r1",
            )
        finally:
            await conn.close()
        msg = UserMessage(content=[TextContent(text="late")], run_id="r1")
        with pytest.raises(RunAlreadyCompletedError):
            await cp.append("t", [msg])


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
