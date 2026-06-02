import aiomysql
import pytest

from cubepi.checkpointer.mysql import MySQLCheckpointer
from cubepi.checkpointer.mysql.alembic_helpers import (
    add_pending_request_column_op,
    add_run_id_column_op,
    messages_partition_clause,
    write_schema_version_op,
)
from cubepi.checkpointer.mysql.checkpointer import _parse_dsn
from cubepi.hitl.types import ApproveRequest, HitlRequest


async def _setup_schema_v2(dsn: str) -> None:
    """Bootstrap a v2 schema in a fresh DB (mirrors test_mysql.py::_setup_schema)."""
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
            # v2 + v3 column adds
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


def _req(qid="tc-1") -> HitlRequest:
    return HitlRequest(
        question_id=qid,
        thread_id="t-1",
        payload=ApproveRequest(tool_name="bash", tool_call_id=qid, args={"cmd": "ls"}),
        created_at=0.0,
        timeout_seconds=30.0,
    )


@pytest.mark.asyncio
async def test_mysql_save_and_load_pending_request(clean_mysql_db) -> None:
    await _setup_schema_v2(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        await cp.save_pending_request("t-1", _req())
        loaded = await cp.load_pending_request("t-1")
    assert loaded == _req()


@pytest.mark.asyncio
async def test_mysql_clear_pending_request(clean_mysql_db) -> None:
    await _setup_schema_v2(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        await cp.save_pending_request("t-1", _req())
        await cp.save_pending_request("t-1", None)
        loaded = await cp.load_pending_request("t-1")
    assert loaded is None


@pytest.mark.asyncio
async def test_mysql_pending_request_creates_thread_row_lazily(clean_mysql_db) -> None:
    """save_pending_request must upsert the thread row so calling it on an unknown
    thread doesn't violate the FK constraint."""
    await _setup_schema_v2(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        await cp.save_pending_request("brand-new-thread", _req(qid="tc-x"))
        loaded = await cp.load_pending_request("brand-new-thread")
    assert loaded is not None
    assert loaded.question_id == "tc-x"


@pytest.mark.asyncio
async def test_mysql_v2_schema_version_enforced(clean_mysql_db) -> None:
    """If the host's alembic only ran v1 (schema_version=1), MySQLCheckpointer
    __aenter__ raises CubepiSchemaMismatch."""
    from cubepi.checkpointer.mysql.exceptions import CubepiSchemaMismatch

    conn = await aiomysql.connect(autocommit=True, **_parse_dsn(clean_mysql_db))
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "CREATE TABLE cubepi_schema_version (version INT PRIMARY KEY) ENGINE=InnoDB"
            )
            await cur.execute("INSERT INTO cubepi_schema_version (version) VALUES (1)")
    finally:
        await conn.ensure_closed()

    with pytest.raises(CubepiSchemaMismatch):
        async with MySQLCheckpointer(clean_mysql_db):
            pass
