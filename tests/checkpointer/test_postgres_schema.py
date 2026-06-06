import asyncpg
import pytest

from cubepi.checkpointer.postgres.models import EXPECTED_SCHEMA_VERSION


def test_expected_schema_version_is_4():
    assert EXPECTED_SCHEMA_VERSION == 4


@pytest.mark.asyncio
async def test_cubepi_runs_table_present(pg_v4_dsn):
    conn = await asyncpg.connect(pg_v4_dsn)
    try:
        rows = await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'cubepi_runs'"
        )
        cols = {r["column_name"] for r in rows}
        assert {
            "thread_id",
            "run_id",
            "claimed_at",
            "completed_at",
            "completion_seq",
        } <= cols
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_cubepi_messages_has_run_id_column(pg_v4_dsn):
    conn = await asyncpg.connect(pg_v4_dsn)
    try:
        row = await conn.fetchrow(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'cubepi_messages' AND column_name = 'run_id'"
        )
        assert row is not None
    finally:
        await conn.close()
