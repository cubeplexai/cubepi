import tempfile
from pathlib import Path

import pytest

from cubepi.checkpointer.sqlite import SQLiteCheckpointer


@pytest.mark.asyncio
async def test_runs_table_and_columns_exist_after_init():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.db"
        async with SQLiteCheckpointer(str(path)) as cp:
            cur = await cp._db.execute("PRAGMA table_info(runs)")
            cols = {row[1] for row in await cur.fetchall()}
            assert {
                "thread_id",
                "run_id",
                "claimed_at",
                "completed_at",
                "completion_seq",
            } <= cols

            cur = await cp._db.execute("PRAGMA table_info(messages)")
            cols = {row[1] for row in await cur.fetchall()}
            assert "run_id" in cols

            cur = await cp._db.execute("PRAGMA table_info(thread_extra)")
            cols = {row[1] for row in await cur.fetchall()}
            assert "parent_thread_id" in cols


@pytest.mark.asyncio
async def test_busy_timeout_is_set():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.db"
        async with SQLiteCheckpointer(str(path)) as cp:
            cur = await cp._db.execute("PRAGMA busy_timeout")
            (val,) = await cur.fetchone()
            assert val >= 5000
