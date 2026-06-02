"""Cross-backend tests for ``save_pending_request(request, run_id=...)``
and ``load_pending_run_id`` — the run_id slot persisted next to a HITL
pending request, used by host applications (e.g. cubebox) to recover the
paused run after worker death.

Postgres + MySQL tests reuse the v3 schema setup helpers (the v2 setup
+ the new run_id column add); Memory and SQLite are pure-Python.
"""

from __future__ import annotations

import tempfile
from typing import Any

import pytest

from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.checkpointer.sqlite import SQLiteCheckpointer
from cubepi.hitl.types import ApproveRequest, HitlRequest


def _req(thread_id: str = "t-1", qid: str = "tc-1") -> HitlRequest:
    return HitlRequest(
        question_id=qid,
        thread_id=thread_id,
        payload=ApproveRequest(tool_name="bash", tool_call_id=qid, args={"cmd": "ls"}),
        created_at=0.0,
        timeout_seconds=30.0,
    )


# ----------------------------- Memory --------------------------------------


async def test_memory_save_with_run_id_roundtrip():
    cp = MemoryCheckpointer()
    await cp.save_pending_request("t-1", _req(), run_id="r-1")
    assert (await cp.load_pending_request("t-1")) == _req()
    assert (await cp.load_pending_run_id("t-1")) == "r-1"


async def test_memory_legacy_save_without_run_id():
    cp = MemoryCheckpointer()
    await cp.save_pending_request("t-1", _req())
    assert (await cp.load_pending_request("t-1")) == _req()
    assert (await cp.load_pending_run_id("t-1")) is None


async def test_memory_clear_clears_both_pending_and_run_id():
    cp = MemoryCheckpointer()
    await cp.save_pending_request("t-1", _req(), run_id="r-1")
    await cp.save_pending_request("t-1", None)
    assert (await cp.load_pending_request("t-1")) is None
    assert (await cp.load_pending_run_id("t-1")) is None


async def test_memory_load_pending_run_id_missing_thread():
    cp = MemoryCheckpointer()
    assert (await cp.load_pending_run_id("never-existed")) is None


# ----------------------------- SQLite --------------------------------------


@pytest.fixture
async def sqlite_cp():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        async with SQLiteCheckpointer(f.name) as cp:
            yield cp


async def test_sqlite_save_with_run_id_roundtrip(sqlite_cp):
    await sqlite_cp.save_pending_request("t-1", _req(), run_id="r-1")
    assert (await sqlite_cp.load_pending_request("t-1")) == _req()
    assert (await sqlite_cp.load_pending_run_id("t-1")) == "r-1"


async def test_sqlite_legacy_save_without_run_id(sqlite_cp):
    await sqlite_cp.save_pending_request("t-1", _req())
    assert (await sqlite_cp.load_pending_run_id("t-1")) is None


async def test_sqlite_clear_clears_both_pending_and_run_id(sqlite_cp):
    await sqlite_cp.save_pending_request("t-1", _req(), run_id="r-1")
    await sqlite_cp.save_pending_request("t-1", None)
    assert (await sqlite_cp.load_pending_request("t-1")) is None
    assert (await sqlite_cp.load_pending_run_id("t-1")) is None


async def test_sqlite_load_pending_run_id_missing_thread(sqlite_cp):
    assert (await sqlite_cp.load_pending_run_id("never-existed")) is None


async def test_sqlite_migration_from_v2_table_adds_run_id_column(tmp_path):
    """Existing SQLite DBs (no run_id column) get the column added on __aenter__.

    Simulates an upgrade: bootstrap a v2-style ``thread_pending_request`` table
    without the run_id column, then open with the new SQLiteCheckpointer and
    confirm the column exists + load_pending_run_id returns None for legacy rows.
    """
    import aiosqlite

    db_path = str(tmp_path / "legacy.db")
    conn = await aiosqlite.connect(db_path)
    try:
        await conn.execute(
            "CREATE TABLE thread_pending_request ("
            "  thread_id TEXT PRIMARY KEY,"
            "  request_json TEXT NOT NULL,"
            "  created_at REAL NOT NULL DEFAULT (julianday('now'))"
            ")"
        )
        # Insert a legacy row to confirm load_pending_run_id returns None for it.
        await conn.execute(
            "INSERT INTO thread_pending_request (thread_id, request_json) VALUES (?, ?)",
            ("t-legacy", _req(qid="legacy-q").model_dump_json()),
        )
        await conn.commit()
    finally:
        await conn.close()

    async with SQLiteCheckpointer(db_path) as cp:
        # Column should now exist on the table.
        cur = await cp._db.execute("PRAGMA table_info(thread_pending_request)")
        cols = {row[1] for row in await cur.fetchall()}
        assert "run_id" in cols
        # Legacy row keeps its pending; run_id is None.
        loaded = await cp.load_pending_request("t-legacy")
        assert loaded is not None
        assert (await cp.load_pending_run_id("t-legacy")) is None


# ----------------------------- Postgres ------------------------------------


async def _setup_pg_schema_v3(dsn: str) -> None:
    """Bootstrap a v3 schema in a fresh Postgres DB."""
    import asyncpg

    from cubepi.checkpointer.postgres.alembic_helpers import (
        add_pending_request_column_op,
        add_run_id_column_op,
        create_message_partitions_op,
        write_schema_version_op,
    )

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("""
            CREATE TABLE cubepi_threads (
                thread_id TEXT PRIMARY KEY,
                parent_thread_id TEXT NULL REFERENCES cubepi_threads(thread_id),
                forked_at_seq BIGINT NULL,
                extra JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        await conn.execute(add_pending_request_column_op())
        await conn.execute(add_run_id_column_op())
        await conn.execute("""
            CREATE TABLE cubepi_messages (
                thread_id TEXT NOT NULL REFERENCES cubepi_threads(thread_id) ON DELETE CASCADE,
                seq BIGINT NOT NULL,
                role TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                payload BYTEA NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (thread_id, seq)
            ) PARTITION BY HASH (thread_id);
        """)
        await conn.execute(create_message_partitions_op())
        await conn.execute("""
            CREATE INDEX ix_cubepi_messages_metadata_gin
            ON cubepi_messages USING GIN (metadata jsonb_path_ops);
        """)
        await conn.execute("""
            CREATE TABLE cubepi_schema_version (version INTEGER PRIMARY KEY);
        """)
        await conn.execute(write_schema_version_op())
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_postgres_save_with_run_id_roundtrip(clean_db) -> None:
    from cubepi.checkpointer.postgres import PostgresCheckpointer

    await _setup_pg_schema_v3(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.save_pending_request("t-1", _req(), run_id="r-1")
        assert (await cp.load_pending_request("t-1")) == _req()
        assert (await cp.load_pending_run_id("t-1")) == "r-1"


@pytest.mark.asyncio
async def test_postgres_legacy_save_without_run_id(clean_db) -> None:
    from cubepi.checkpointer.postgres import PostgresCheckpointer

    await _setup_pg_schema_v3(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.save_pending_request("t-1", _req())
        assert (await cp.load_pending_run_id("t-1")) is None


@pytest.mark.asyncio
async def test_postgres_clear_clears_both_pending_and_run_id(clean_db) -> None:
    from cubepi.checkpointer.postgres import PostgresCheckpointer

    await _setup_pg_schema_v3(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.save_pending_request("t-1", _req(), run_id="r-1")
        await cp.save_pending_request("t-1", None)
        assert (await cp.load_pending_request("t-1")) is None
        assert (await cp.load_pending_run_id("t-1")) is None


@pytest.mark.asyncio
async def test_postgres_load_pending_run_id_missing_thread(clean_db) -> None:
    from cubepi.checkpointer.postgres import PostgresCheckpointer

    await _setup_pg_schema_v3(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        assert (await cp.load_pending_run_id("never-existed")) is None


# ----------------------------- MySQL ---------------------------------------


async def _setup_mysql_schema_v3(dsn: str) -> None:
    import aiomysql

    from cubepi.checkpointer.mysql.alembic_helpers import (
        add_pending_request_column_op,
        add_run_id_column_op,
        messages_partition_clause,
        write_schema_version_op,
    )
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    conn = await aiomysql.connect(autocommit=True, **_parse_dsn(dsn))
    try:
        async with conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE cubepi_threads (
                    thread_id VARCHAR(255) COLLATE utf8mb4_bin PRIMARY KEY,
                    parent_thread_id VARCHAR(255) COLLATE utf8mb4_bin NULL,
                    forked_at_seq BIGINT NULL,
                    extra JSON NOT NULL DEFAULT (JSON_OBJECT()),
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    CONSTRAINT fk_parent FOREIGN KEY (parent_thread_id)
                        REFERENCES cubepi_threads (thread_id)
                ) ENGINE=InnoDB
            """)
            await cur.execute(add_pending_request_column_op())
            await cur.execute(add_run_id_column_op())
            await cur.execute(
                """
                CREATE TABLE cubepi_messages (
                    thread_id VARCHAR(255) COLLATE utf8mb4_bin NOT NULL,
                    seq BIGINT NOT NULL,
                    role VARCHAR(32) NOT NULL,
                    metadata JSON NOT NULL DEFAULT (JSON_OBJECT()),
                    payload LONGBLOB NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, seq)
                ) ENGINE=InnoDB """
                + messages_partition_clause()
            )
            await cur.execute("""
                CREATE TABLE cubepi_schema_version (
                    version INT PRIMARY KEY
                ) ENGINE=InnoDB
            """)
            for stmt in write_schema_version_op().split(";"):
                if stmt.strip():
                    await cur.execute(stmt)
    finally:
        await conn.ensure_closed()


@pytest.mark.asyncio
async def test_mysql_save_with_run_id_roundtrip(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import MySQLCheckpointer

    await _setup_mysql_schema_v3(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        await cp.save_pending_request("t-1", _req(), run_id="r-1")
        assert (await cp.load_pending_request("t-1")) == _req()
        assert (await cp.load_pending_run_id("t-1")) == "r-1"


@pytest.mark.asyncio
async def test_mysql_legacy_save_without_run_id(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import MySQLCheckpointer

    await _setup_mysql_schema_v3(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        await cp.save_pending_request("t-1", _req())
        assert (await cp.load_pending_run_id("t-1")) is None


@pytest.mark.asyncio
async def test_mysql_clear_clears_both_pending_and_run_id(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import MySQLCheckpointer

    await _setup_mysql_schema_v3(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        await cp.save_pending_request("t-1", _req(), run_id="r-1")
        await cp.save_pending_request("t-1", None)
        assert (await cp.load_pending_request("t-1")) is None
        assert (await cp.load_pending_run_id("t-1")) is None


@pytest.mark.asyncio
async def test_mysql_load_pending_run_id_missing_thread(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import MySQLCheckpointer

    await _setup_mysql_schema_v3(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        assert (await cp.load_pending_run_id("never-existed")) is None


# Suppress unused import warning for the typing import in scaffolding.
_unused: Any = None
