"""PostgresCheckpointer tests (D1) — smoke imports.

Full E2E tests are in D1.3 once PostgresCheckpointer is implemented.
"""

import asyncpg
import pytest


def test_models_import() -> None:
    from cubepi.checkpointer.postgres.models import (
        EXPECTED_SCHEMA_VERSION,
        CubepiHitlAnswer,
        PARTITION_COUNT,
        CubepiMessage,
        CubepiSchemaVersion,
        CubepiThread,
        cubepi_metadata,
    )

    assert EXPECTED_SCHEMA_VERSION == 5
    assert PARTITION_COUNT == 64
    assert CubepiThread.__tablename__ == "cubepi_threads"
    assert CubepiMessage.__tablename__ == "cubepi_messages"
    assert CubepiHitlAnswer.__tablename__ == "cubepi_hitl_answers"
    assert CubepiSchemaVersion.__tablename__ == "cubepi_schema_version"
    assert "cubepi_threads" in cubepi_metadata.tables
    assert "cubepi_messages" in cubepi_metadata.tables
    assert "cubepi_hitl_answers" in cubepi_metadata.tables
    assert "cubepi_schema_version" in cubepi_metadata.tables


def test_cubepi_message_has_partition_by() -> None:
    """The CubepiMessage table declares HASH partitioning."""
    from cubepi.checkpointer.postgres.models import cubepi_metadata

    msgs = cubepi_metadata.tables["cubepi_messages"]
    # SQLAlchemy stores PG partition clause in info or dialect-specific args
    # Verify via dialect kwargs
    assert msgs.kwargs.get("postgresql_partition_by") == "HASH (thread_id)"


def test_cubepi_message_has_gin_index() -> None:
    """The GIN index on metadata is registered."""
    from cubepi.checkpointer.postgres.models import cubepi_metadata

    msgs = cubepi_metadata.tables["cubepi_messages"]
    idx_names = [i.name for i in msgs.indexes]
    assert "ix_cubepi_messages_metadata_gin" in idx_names


def test_create_message_partitions_op_yields_64_statements() -> None:
    from cubepi.checkpointer.postgres.alembic_helpers import (
        create_message_partitions_op,
    )

    sql = create_message_partitions_op()
    assert sql.count("CREATE TABLE cubepi_messages_p") == 64
    assert "modulus 64, remainder 0" in sql
    assert "modulus 64, remainder 63" in sql


def test_create_message_partitions_op_partitions_are_zero_padded() -> None:
    from cubepi.checkpointer.postgres.alembic_helpers import (
        create_message_partitions_op,
    )

    sql = create_message_partitions_op()
    # Partition names use 2-digit padding so they sort lexicographically
    assert "cubepi_messages_p00 " in sql
    assert "cubepi_messages_p63 " in sql


def test_write_schema_version_op_includes_expected_version() -> None:
    from cubepi.checkpointer.postgres.alembic_helpers import write_schema_version_op
    from cubepi.checkpointer.postgres.models import EXPECTED_SCHEMA_VERSION

    sql = write_schema_version_op()
    assert "INSERT INTO cubepi_schema_version" in sql
    assert f"VALUES ({EXPECTED_SCHEMA_VERSION})" in sql
    assert "ON CONFLICT" in sql


def test_write_schema_version_op_clears_stale_rows() -> None:
    """A prior version's row must be removed so _verify_schema sees the new one."""
    from cubepi.checkpointer.postgres.alembic_helpers import write_schema_version_op
    from cubepi.checkpointer.postgres.models import EXPECTED_SCHEMA_VERSION

    sql = write_schema_version_op()
    # Must DELETE rows whose version is not the expected one before INSERT.
    assert "DELETE FROM cubepi_schema_version" in sql
    assert f"WHERE version <> {EXPECTED_SCHEMA_VERSION}" in sql
    # And the DELETE must come before the INSERT in the statement order.
    assert sql.index("DELETE") < sql.index("INSERT")


def test_schema_uninitialized_is_schema_error() -> None:
    from cubepi.checkpointer.postgres.exceptions import (
        CubepiSchemaError,
        CubepiSchemaUninitialized,
    )

    err = CubepiSchemaUninitialized("tables missing")
    assert isinstance(err, CubepiSchemaError)


def test_schema_mismatch_carries_expected_actual() -> None:
    from cubepi.checkpointer.postgres.exceptions import CubepiSchemaMismatch

    err = CubepiSchemaMismatch(expected=2, actual=1, hint="run alembic")
    assert err.expected == 2
    assert err.actual == 1
    assert "expected=2" in str(err)
    assert "actual=1" in str(err)
    assert "run alembic" in str(err)


def test_schema_mismatch_without_hint() -> None:
    from cubepi.checkpointer.postgres.exceptions import CubepiSchemaMismatch

    err = CubepiSchemaMismatch(expected=2, actual=1)
    # No hint suffix
    assert "expected=2" in str(err)
    assert "actual=1" in str(err)


def test_role_of_known_message_types() -> None:
    """_role_of maps each concrete Message subclass to its role string."""
    from cubepi.checkpointer.postgres.checkpointer import _role_of
    from cubepi.providers.base import (
        AssistantMessage,
        TextContent,
        ToolResultMessage,
        Usage,
        UserMessage,
    )

    assert _role_of(UserMessage(content=[TextContent(text="x")])) == "user"
    assert (
        _role_of(AssistantMessage(content=[TextContent(text="x")], usage=Usage()))
        == "assistant"
    )
    tr = ToolResultMessage(
        tool_call_id="tc-1",
        tool_name="t",
        content=[TextContent(text="ok")],
    )
    assert _role_of(tr) == "tool"


def test_role_of_rejects_unknown_message_type() -> None:
    """_role_of raises for anything that isn't User/Assistant/ToolResult."""
    from cubepi.checkpointer.postgres.checkpointer import _role_of

    class FakeMessage:
        pass

    with pytest.raises(TypeError, match="unknown Message type"):
        _role_of(FakeMessage())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_append_empty_messages_is_noop() -> None:
    """append([]) returns early without touching the pool — no DB needed."""
    from cubepi.checkpointer.postgres import PostgresCheckpointer

    cp = PostgresCheckpointer("postgresql://unreachable-host/none")
    # Pool intentionally never created — early return must precede the assert.
    assert cp._pool is None
    await cp.append("thread-x", [])


# ---------------------------------------------------------------------------
# D1.3 E2E tests — require a real Postgres instance
# ---------------------------------------------------------------------------


async def _setup_schema(dsn: str) -> None:
    """Build the cubepi schema (matching what host alembic would generate)."""
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("""
            CREATE TABLE cubepi_threads (
                thread_id TEXT PRIMARY KEY,
                parent_thread_id TEXT REFERENCES cubepi_threads(thread_id),
                forked_at_seq BIGINT,
                extra JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
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
        from cubepi.checkpointer.postgres.alembic_helpers import (
            add_pending_request_column_op,
            add_run_id_column_op,
            create_message_partitions_op,
            upgrade_v3_to_v4_op,
            upgrade_v4_to_v5_op,
            write_schema_version_op,
        )

        # Bring cubepi_threads up to the v3 shape (pending_request + run_id).
        await conn.execute(add_pending_request_column_op())
        await conn.execute(add_run_id_column_op())
        await conn.execute(create_message_partitions_op())
        await conn.execute("""
            CREATE INDEX ix_cubepi_messages_metadata_gin
            ON cubepi_messages USING GIN (metadata jsonb_path_ops);
        """)
        await conn.execute("""
            CREATE TABLE cubepi_schema_version (
                version INTEGER PRIMARY KEY
            );
        """)
        # Apply v3→v4 (run_id on messages + cubepi_runs partitioned table).
        await conn.execute(upgrade_v3_to_v4_op())
        await conn.execute(upgrade_v4_to_v5_op())
        await conn.execute(write_schema_version_op())
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_postgres_checkpointer_round_trip(clean_db) -> None:
    """Append + load round-trips messages with metadata."""
    from cubepi.checkpointer.postgres import PostgresCheckpointer
    from cubepi.providers.base import (
        AssistantMessage,
        TextContent,
        Usage,
        UserMessage,
    )

    await _setup_schema(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        msg1 = UserMessage(
            content=[TextContent(text="hello")],
            metadata={"memory_snapshot": {"id": "m1"}},
        )
        msg2 = AssistantMessage(
            content=[TextContent(text="hi back")],
            usage=Usage(),
            metadata={"cost_cents": 5},
        )
        await cp.append("t-1", [msg1, msg2])
        data = await cp.load("t-1")

    assert data is not None
    assert len(data.messages) == 2
    assert isinstance(data.messages[0], UserMessage)
    assert isinstance(data.messages[1], AssistantMessage)
    assert data.messages[0].metadata == {"memory_snapshot": {"id": "m1"}}
    assert data.messages[1].metadata == {"cost_cents": 5}
    # Round-trip content
    assert data.messages[0].content[0].text == "hello"
    assert data.messages[1].content[0].text == "hi back"


@pytest.mark.asyncio
async def test_postgres_checkpointer_save_extra_merges(clean_db) -> None:
    from cubepi.checkpointer.postgres import PostgresCheckpointer
    from cubepi.providers.base import TextContent, UserMessage

    await _setup_schema(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.append("t-2", [UserMessage(content=[TextContent(text="x")])])
        await cp.save_extra("t-2", {"a": 1})
        await cp.save_extra("t-2", {"b": 2})
        data = await cp.load("t-2")

    assert data is not None
    assert data.extra == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_postgres_checkpointer_seq_monotonic(clean_db) -> None:
    """Multiple append batches produce strictly monotonic seqs."""
    from cubepi.checkpointer.postgres import PostgresCheckpointer
    from cubepi.providers.base import TextContent, UserMessage

    await _setup_schema(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        msgs1 = [UserMessage(content=[TextContent(text=str(i))]) for i in range(5)]
        await cp.append("t-3", msgs1)
        msgs2 = [UserMessage(content=[TextContent(text=str(i))]) for i in range(5, 10)]
        await cp.append("t-3", msgs2)
        data = await cp.load("t-3")

    assert data is not None
    assert len(data.messages) == 10
    texts = [m.content[0].text for m in data.messages]
    assert texts == [str(i) for i in range(10)]


@pytest.mark.asyncio
async def test_uninitialized_schema_raises(clean_db) -> None:
    """Empty DB (no cubepi tables) → CubepiSchemaUninitialized."""
    from cubepi.checkpointer.postgres import (
        CubepiSchemaUninitialized,
        PostgresCheckpointer,
    )

    with pytest.raises(CubepiSchemaUninitialized):
        async with PostgresCheckpointer(clean_db):
            pass


@pytest.mark.asyncio
async def test_version_mismatch_raises(clean_db) -> None:
    """Schema present but version != EXPECTED → CubepiSchemaMismatch."""
    from cubepi.checkpointer.postgres import (
        CubepiSchemaMismatch,
        PostgresCheckpointer,
    )

    await _setup_schema(clean_db)
    conn = await asyncpg.connect(clean_db)
    try:
        await conn.execute("UPDATE cubepi_schema_version SET version = 999")
    finally:
        await conn.close()

    with pytest.raises(CubepiSchemaMismatch) as exc_info:
        async with PostgresCheckpointer(clean_db):
            pass
    from cubepi.checkpointer.postgres.models import EXPECTED_SCHEMA_VERSION

    assert exc_info.value.expected == EXPECTED_SCHEMA_VERSION
    assert exc_info.value.actual == 999


@pytest.mark.asyncio
async def test_empty_thread_load_returns_none(clean_db) -> None:
    """Loading an unknown thread returns None."""
    from cubepi.checkpointer.postgres import PostgresCheckpointer

    await _setup_schema(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        data = await cp.load("nonexistent-thread")
    assert data is None
