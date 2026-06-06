"""Task 38: cross-backend fork happy path E2E.

Drives ``Agent.prompt`` twice (R1 then R2) on every backend, then forks
``after_run_id="R1"`` and verifies the destination thread contains R1's
messages only, carries ``parent_thread_id`` and the fork metadata.

The four backends (memory / sqlite / postgres / mysql) share one helper
``_run_happy_path`` so the assertions stay identical. Postgres/MySQL share
the DSN fixtures from ``tests/checkpointer/conftest.py`` (re-exported via
``tests/e2e/conftest.py``'s ``pytest_plugins``) which auto-skip when the
backend is unreachable.
"""

from __future__ import annotations

import pytest

from cubepi.agent.agent import Agent
from cubepi.providers.base import AssistantMessage, TextContent
from cubepi.providers.faux import FauxProvider


def _ok_faux() -> FauxProvider:
    """Two-turn faux provider — one assistant message per prompt."""
    p = FauxProvider()
    p.set_responses(
        [
            AssistantMessage(content=[TextContent(text="r1")], stop_reason="end_turn"),
            AssistantMessage(content=[TextContent(text="r2")], stop_reason="end_turn"),
        ]
    )
    return p


async def _run_happy_path(cp) -> None:
    """Shared body — same assertions across all four backends."""
    p = _ok_faux()
    a = Agent(
        model=p.model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    r1 = await a.prompt("first", run_id="R1")
    assert r1 == "R1"
    r2 = await a.prompt("second", run_id="R2")
    assert r2 == "R2"

    await a.fork("src", "dst", after_run_id="R1", metadata={"label": "branch"})

    loaded = await cp.load("dst")
    assert loaded is not None
    assert loaded.parent_thread_id == "src"
    assert loaded.extra["fork"] == {"label": "branch"}
    # dst contains R1's messages only, not R2's.
    run_ids = {m.run_id for m in loaded.messages if m.run_id}
    assert run_ids == {"R1"}


@pytest.mark.asyncio
async def test_fork_e2e_memory():
    from cubepi.checkpointer.memory import MemoryCheckpointer

    await _run_happy_path(MemoryCheckpointer())


@pytest.mark.asyncio
async def test_fork_e2e_sqlite(tmp_path):
    from cubepi.checkpointer.sqlite import SQLiteCheckpointer

    async with SQLiteCheckpointer(str(tmp_path / "x.db")) as cp:
        await _run_happy_path(cp)


@pytest.mark.asyncio
async def test_fork_e2e_postgres(pg_v4_dsn):
    from cubepi.checkpointer.postgres.checkpointer import PostgresCheckpointer

    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        await _run_happy_path(cp)


@pytest.mark.asyncio
async def test_fork_e2e_mysql(mysql_v4_dsn):
    from cubepi.checkpointer.mysql.checkpointer import MySQLCheckpointer

    async with MySQLCheckpointer(mysql_v4_dsn) as cp:
        await _run_happy_path(cp)
