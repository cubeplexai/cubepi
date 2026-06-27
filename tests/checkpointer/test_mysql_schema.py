"""MySQLCheckpointer v4 schema tests — mirrors test_postgres_schema.py."""

import aiomysql
import pytest

from cubepi.checkpointer.mysql.models import EXPECTED_SCHEMA_VERSION


def test_expected_schema_version_is_5() -> None:
    assert EXPECTED_SCHEMA_VERSION == 5


@pytest.mark.asyncio
async def test_cubepi_runs_table_present(mysql_v4_dsn) -> None:
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    cfg = _parse_dsn(mysql_v4_dsn)
    conn = await aiomysql.connect(autocommit=True, **cfg)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = 'cubepi_runs'",
                (cfg["db"],),
            )
            rows = await cur.fetchall()
    finally:
        await conn.ensure_closed()
    cols = {r[0].lower() for r in rows}
    assert {
        "thread_id",
        "run_id",
        "claimed_at",
        "completed_at",
        "completion_seq",
    } <= cols


@pytest.mark.asyncio
async def test_cubepi_messages_has_run_id_column(mysql_v4_dsn) -> None:
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    cfg = _parse_dsn(mysql_v4_dsn)
    conn = await aiomysql.connect(autocommit=True, **cfg)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = 'cubepi_messages' "
                "AND column_name = 'run_id'",
                (cfg["db"],),
            )
            row = await cur.fetchone()
    finally:
        await conn.ensure_closed()
    assert row is not None


@pytest.mark.asyncio
async def test_cubepi_hitl_answers_table_present(mysql_v4_dsn) -> None:
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    cfg = _parse_dsn(mysql_v4_dsn)
    conn = await aiomysql.connect(autocommit=True, **cfg)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = %s "
                "AND table_name = 'cubepi_hitl_answers'",
                (cfg["db"],),
            )
            rows = await cur.fetchall()
    finally:
        await conn.ensure_closed()
    cols = {r[0].lower() for r in rows}
    assert {
        "thread_id",
        "run_id",
        "question_id",
        "answer",
        "answered_at",
    } <= cols
