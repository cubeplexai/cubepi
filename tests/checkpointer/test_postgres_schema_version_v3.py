"""Bumping EXPECTED_SCHEMA_VERSION to 3: a v2 database must be refused
with a clear, actionable error so operators know they need to apply the
host-side v2→v3 alembic migration."""

from __future__ import annotations

import asyncpg
import pytest

from cubepi.checkpointer.postgres import PostgresCheckpointer
from cubepi.checkpointer.postgres.exceptions import CubepiSchemaMismatch


@pytest.mark.asyncio
async def test_postgres_v2_database_refused_with_actionable_error(clean_db) -> None:
    conn = await asyncpg.connect(clean_db)
    try:
        await conn.execute(
            "CREATE TABLE cubepi_schema_version (version INTEGER PRIMARY KEY);"
        )
        await conn.execute("INSERT INTO cubepi_schema_version (version) VALUES (2);")
    finally:
        await conn.close()

    with pytest.raises(CubepiSchemaMismatch) as ei:
        async with PostgresCheckpointer(clean_db):
            pass

    msg = str(ei.value)
    assert "expected 5" in msg or "expected=5" in msg
    assert "actual 2" in msg or "actual=2" in msg
    # Hint mentions the host alembic migration path so operators know what to do.
    assert "alembic" in msg.lower()


@pytest.mark.asyncio
async def test_mysql_v2_database_refused_with_actionable_error(clean_mysql_db) -> None:
    """Same policy for MySQL (parallel to the Postgres test above)."""
    import aiomysql

    from cubepi.checkpointer.mysql import MySQLCheckpointer
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn
    from cubepi.checkpointer.mysql.exceptions import (
        CubepiSchemaMismatch as MysqlMismatch,
    )

    conn = await aiomysql.connect(autocommit=True, **_parse_dsn(clean_mysql_db))
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "CREATE TABLE cubepi_schema_version (version INT PRIMARY KEY) ENGINE=InnoDB"
            )
            await cur.execute("INSERT INTO cubepi_schema_version (version) VALUES (2)")
    finally:
        await conn.ensure_closed()

    with pytest.raises(MysqlMismatch) as ei:
        async with MySQLCheckpointer(clean_mysql_db):
            pass

    msg = str(ei.value)
    assert "expected 5" in msg or "expected=5" in msg
    assert "actual 2" in msg or "actual=2" in msg
    assert "alembic" in msg.lower()
