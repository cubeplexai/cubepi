"""MySQLCheckpointer run-lifecycle tests — mirrors test_postgres_runs.py."""

import aiomysql
import pytest

from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
)
from cubepi.checkpointer.mysql import MySQLCheckpointer
from cubepi.providers.base import TextContent, UserMessage


async def _connect(dsn: str):
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    return await aiomysql.connect(autocommit=True, **_parse_dsn(dsn))


@pytest.mark.asyncio
async def test_claim_run_creates_threads_row_lazily(mysql_v4_dsn) -> None:
    async with MySQLCheckpointer(mysql_v4_dsn) as cp:
        await cp.claim_run("t-lazy", "r1")
    conn = await _connect(mysql_v4_dsn)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT thread_id FROM cubepi_threads WHERE thread_id = %s",
                ("t-lazy",),
            )
            row = await cur.fetchone()
            assert row is not None
            await cur.execute(
                "SELECT completed_at FROM cubepi_runs "
                "WHERE thread_id = %s AND run_id = %s",
                ("t-lazy", "r1"),
            )
            run = await cur.fetchone()
            assert run is not None
            assert run[0] is None
    finally:
        await conn.ensure_closed()


@pytest.mark.asyncio
async def test_claim_collision_in_flight_raises_claimed(mysql_v4_dsn) -> None:
    async with MySQLCheckpointer(mysql_v4_dsn) as cp:
        await cp.claim_run("t", "r1")
        with pytest.raises(RunAlreadyClaimedError):
            await cp.claim_run("t", "r1")


@pytest.mark.asyncio
async def test_append_on_completed_run_id_rejected(mysql_v4_dsn) -> None:
    async with MySQLCheckpointer(mysql_v4_dsn) as cp:
        await cp.claim_run("t", "r1")
        await cp.mark_run_complete("t", "r1")
        msg = UserMessage(content=[TextContent(text="late")], run_id="r1")
        with pytest.raises(RunAlreadyCompletedError):
            await cp.append("t", [msg])


@pytest.mark.asyncio
async def test_claim_then_complete_roundtrip(mysql_v4_dsn) -> None:
    async with MySQLCheckpointer(mysql_v4_dsn) as cp:
        await cp.claim_run("t", "r1")
        await cp.mark_run_complete("t", "r1")
        # Idempotent: second mark is a no-op.
        await cp.mark_run_complete("t", "r1")


@pytest.mark.asyncio
async def test_claim_collision_completed_raises_completed(mysql_v4_dsn) -> None:
    async with MySQLCheckpointer(mysql_v4_dsn) as cp:
        await cp.claim_run("t", "r1")
        await cp.mark_run_complete("t", "r1")
        with pytest.raises(RunAlreadyCompletedError):
            await cp.claim_run("t", "r1")


@pytest.mark.asyncio
async def test_mark_without_claim_raises_not_claimed(mysql_v4_dsn) -> None:
    async with MySQLCheckpointer(mysql_v4_dsn) as cp:
        with pytest.raises(RunNotClaimedError):
            await cp.mark_run_complete("t", "r1")


@pytest.mark.asyncio
async def test_completion_seq_monotonic_per_thread(mysql_v4_dsn) -> None:
    async with MySQLCheckpointer(mysql_v4_dsn) as cp:
        for rid in ("A", "B", "C"):
            await cp.claim_run("t", rid)
            await cp.mark_run_complete("t", rid)
    conn = await _connect(mysql_v4_dsn)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT run_id, completion_seq FROM cubepi_runs "
                "WHERE thread_id = %s ORDER BY completion_seq",
                ("t",),
            )
            rows = await cur.fetchall()
    finally:
        await conn.ensure_closed()
    assert [r[0] for r in rows] == ["A", "B", "C"]
    seqs = [r[1] for r in rows]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == 3


@pytest.mark.asyncio
async def test_load_pending_returns_tuple_with_run_id(mysql_v4_dsn) -> None:
    from cubepi.hitl.types import ConfirmRequest, HitlRequest

    async with MySQLCheckpointer(mysql_v4_dsn) as cp:
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
async def test_load_pending_returns_none_when_empty(mysql_v4_dsn) -> None:
    async with MySQLCheckpointer(mysql_v4_dsn) as cp:
        assert await cp.load_pending("t") is None


@pytest.mark.asyncio
async def test_append_persists_run_id_into_column(mysql_v4_dsn) -> None:
    async with MySQLCheckpointer(mysql_v4_dsn) as cp:
        await cp.claim_run("t", "r1")
        msg = UserMessage(content=[TextContent(text="hi")], run_id="r1")
        await cp.append("t", [msg])
    conn = await _connect(mysql_v4_dsn)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT run_id FROM cubepi_messages WHERE thread_id = %s",
                ("t",),
            )
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == "r1"
    finally:
        await conn.ensure_closed()


@pytest.mark.asyncio
async def test_append_in_flight_run_id_ok(mysql_v4_dsn) -> None:
    async with MySQLCheckpointer(mysql_v4_dsn) as cp:
        await cp.claim_run("t", "r1")
        msg = UserMessage(content=[TextContent(text="ok")], run_id="r1")
        await cp.append("t", [msg])
        data = await cp.load("t")
        assert data is not None
        assert len(data.messages) == 1
