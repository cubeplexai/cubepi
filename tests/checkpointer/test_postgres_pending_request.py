import asyncpg
import pytest

from cubepi.checkpointer.postgres import PostgresCheckpointer
from cubepi.checkpointer.postgres.alembic_helpers import (
    add_pending_request_column_op,
    add_run_id_column_op,
    create_message_partitions_op,
    write_schema_version_op,
)
from cubepi.hitl.types import ApproveRequest, HitlRequest


async def _setup_schema_v2(dsn: str) -> None:
    """Bootstrap the current cubepi schema in a fresh DB (now v3; legacy name kept).

    Adds both the v2 pending_request column and the v3 run_id column so the
    HITL pending tests can exercise save_pending_request(... run_id=...)."""
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
        # v2 + v3 column adds
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


def _req(qid="tc-1") -> HitlRequest:
    return HitlRequest(
        question_id=qid,
        thread_id="t-1",
        payload=ApproveRequest(tool_name="bash", tool_call_id=qid, args={"cmd": "ls"}),
        created_at=0.0,
        timeout_seconds=30.0,
    )


@pytest.mark.asyncio
async def test_postgres_save_and_load_pending_request(clean_db) -> None:
    await _setup_schema_v2(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.save_pending_request("t-1", _req())
        loaded = await cp.load_pending_request("t-1")
    assert loaded == _req()


@pytest.mark.asyncio
async def test_postgres_clear_pending_request(clean_db) -> None:
    await _setup_schema_v2(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.save_pending_request("t-1", _req())
        await cp.save_pending_request("t-1", None)
        loaded = await cp.load_pending_request("t-1")
    assert loaded is None


@pytest.mark.asyncio
async def test_postgres_pending_request_creates_thread_row_lazily(clean_db) -> None:
    """save_pending_request must INSERT … ON CONFLICT DO NOTHING for the thread row,
    so calling it on an unknown thread doesn't FK-violate."""
    await _setup_schema_v2(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.save_pending_request("brand-new-thread", _req(qid="tc-x"))
        loaded = await cp.load_pending_request("brand-new-thread")
    assert loaded is not None
    assert loaded.question_id == "tc-x"


@pytest.mark.asyncio
async def test_postgres_v2_schema_version_enforced(clean_db) -> None:
    """If the host's alembic only ran v1 (no pending_request column, schema_version=1),
    PostgresCheckpointer().__aenter__ raises CubepiSchemaMismatch."""
    from cubepi.checkpointer.postgres.exceptions import CubepiSchemaMismatch

    conn = await asyncpg.connect(clean_db)
    try:
        await conn.execute(
            "CREATE TABLE cubepi_schema_version (version INTEGER PRIMARY KEY);"
        )
        await conn.execute("INSERT INTO cubepi_schema_version (version) VALUES (1);")
    finally:
        await conn.close()
    with pytest.raises(CubepiSchemaMismatch):
        async with PostgresCheckpointer(clean_db):
            pass
