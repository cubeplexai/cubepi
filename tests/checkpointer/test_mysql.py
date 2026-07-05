"""MySQLCheckpointer tests — mirrors test_postgres.py."""

import aiomysql
import pytest

from cubepi.checkpointer.mysql.alembic_helpers import (
    messages_partition_clause,
    write_schema_version_op,
)

# ---------------------------------------------------------------------------
# Unit tests (no DB)
# ---------------------------------------------------------------------------


def test_exceptions_are_postgres_aliases() -> None:
    from cubepi.checkpointer.mysql.exceptions import (
        CubepiSchemaError,
        CubepiSchemaMismatch,
        CubepiSchemaUninitialized,
    )
    from cubepi.checkpointer.postgres import exceptions as pg_exc

    assert CubepiSchemaError is pg_exc.CubepiSchemaError
    assert CubepiSchemaMismatch is pg_exc.CubepiSchemaMismatch
    assert CubepiSchemaUninitialized is pg_exc.CubepiSchemaUninitialized


def test_models_import() -> None:
    from cubepi.checkpointer.mysql.models import (
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


def test_threads_parent_self_fk_present() -> None:
    """parent_thread_id keeps a self-FK (threads table is not partitioned)."""
    from cubepi.checkpointer.mysql.models import cubepi_metadata

    threads = cubepi_metadata.tables["cubepi_threads"]
    fk_targets = {
        fk.column.table.name for col in threads.columns for fk in col.foreign_keys
    }
    assert "cubepi_threads" in fk_targets


def test_messages_has_no_foreign_keys() -> None:
    """messages table is partitioned, so no FK is allowed."""
    from cubepi.checkpointer.mysql.models import cubepi_metadata

    msgs = cubepi_metadata.tables["cubepi_messages"]
    assert msgs.foreign_keys == set()


def test_messages_has_no_metadata_index() -> None:
    """Only the v4 (thread_id, run_id) composite index is allowed.

    The MySQL backend never indexes the JSON ``metadata`` column (Postgres
    has a GIN index there; MySQL has no equivalent for the host-app cost).
    """
    from cubepi.checkpointer.mysql.models import cubepi_metadata

    msgs = cubepi_metadata.tables["cubepi_messages"]
    idx_names = {idx.name for idx in msgs.indexes}
    assert idx_names == {"ix_cubepi_messages_thread_run"}


def test_messages_partition_clause() -> None:
    clause = messages_partition_clause()
    assert clause == "PARTITION BY KEY (thread_id) PARTITIONS 64"


def test_write_schema_version_op_clears_stale_then_inserts() -> None:
    from cubepi.checkpointer.mysql.models import EXPECTED_SCHEMA_VERSION

    sql = write_schema_version_op()
    assert "DELETE FROM cubepi_schema_version" in sql
    assert f"WHERE version <> {EXPECTED_SCHEMA_VERSION}" in sql
    assert "INSERT IGNORE INTO cubepi_schema_version" in sql
    assert f"VALUES ({EXPECTED_SCHEMA_VERSION})" in sql
    assert sql.index("DELETE") < sql.index("INSERT")


def test_role_of_known_message_types() -> None:
    from cubepi.checkpointer.mysql.checkpointer import _role_of
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
        tool_call_id="tc-1", tool_name="t", content=[TextContent(text="ok")]
    )
    assert _role_of(tr) == "tool"


def test_role_of_rejects_unknown_message_type() -> None:
    from cubepi.checkpointer.mysql.checkpointer import _role_of

    class FakeMessage:
        pass

    with pytest.raises(TypeError, match="unknown Message type"):
        _role_of(FakeMessage())  # type: ignore[arg-type]


def test_parse_dsn_full() -> None:
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    cfg = _parse_dsn("mysql://user:pw@db.example.com:3307/mydb")
    assert cfg == {
        "host": "db.example.com",
        "port": 3307,
        "user": "user",
        "password": "pw",
        "db": "mydb",
    }


def test_parse_dsn_defaults_port_3306() -> None:
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    cfg = _parse_dsn("mysql://root@localhost/x")
    assert cfg["port"] == 3306
    assert cfg["password"] == ""


def test_decode_json_handles_str_and_dict() -> None:
    from cubepi.checkpointer.mysql.checkpointer import _decode_json

    assert _decode_json('{"a": 1}') == {"a": 1}
    assert _decode_json({"a": 1}) == {"a": 1}
    assert _decode_json(None) == {}


@pytest.mark.asyncio
async def test_append_empty_messages_is_noop() -> None:
    from cubepi.checkpointer.mysql.checkpointer import MySQLCheckpointer

    cp = MySQLCheckpointer("mysql://root@unreachable-host/none")
    assert cp._pool is None
    await cp.append("thread-x", [])
    assert cp._pool is None


def test_top_level_lazy_import() -> None:
    import cubepi.checkpointer as cp_pkg

    assert cp_pkg.MySQLCheckpointer is not None
    assert "MySQLCheckpointer" in cp_pkg.__all__


def test_mysql_import_does_not_require_asyncpg() -> None:
    """The cubepi[mysql] extra must not pull in asyncpg (the Postgres driver).

    Run in a fresh interpreter with asyncpg import forced to fail, to catch any
    transitive import into the Postgres package.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import builtins
        real = builtins.__import__
        def fake(name, *a, **k):
            if name == "asyncpg" or name.startswith("asyncpg."):
                raise ModuleNotFoundError("No module named asyncpg")
            return real(name, *a, **k)
        builtins.__import__ = fake
        from cubepi.checkpointer import MySQLCheckpointer
        assert MySQLCheckpointer is not None
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# E2E tests — require a real MySQL instance (8.0.13+)
# ---------------------------------------------------------------------------


async def _setup_schema(dsn: str) -> None:
    """Build the cubepi schema (matching what host alembic would generate)."""
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
            # Bring cubepi_threads up to the v3 shape via the public helpers.
            from cubepi.checkpointer.mysql.alembic_helpers import (
                add_pending_request_column_op,
                add_run_id_column_op,
                upgrade_v3_to_v4_op,
                upgrade_v4_to_v5_op,
            )

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
            # v3 → v4: run_id on cubepi_messages + cubepi_runs partitioned table.
            for stmt in upgrade_v3_to_v4_op().split(";"):
                if stmt.strip():
                    await cur.execute(stmt)
            for stmt in upgrade_v4_to_v5_op().split(";"):
                if stmt.strip():
                    await cur.execute(stmt)
            for stmt in write_schema_version_op().split(";"):
                if stmt.strip():
                    await cur.execute(stmt)
    finally:
        await conn.ensure_closed()


@pytest.mark.asyncio
async def test_mysql_checkpointer_round_trip(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import MySQLCheckpointer
    from cubepi.providers.base import (
        AssistantMessage,
        TextContent,
        Usage,
        UserMessage,
    )

    await _setup_schema(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
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
    assert data.messages[0].content[0].text == "hello"
    assert data.messages[1].content[0].text == "hi back"


@pytest.mark.asyncio
async def test_mysql_checkpointer_save_extra_merges(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import MySQLCheckpointer
    from cubepi.providers.base import TextContent, UserMessage

    await _setup_schema(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        await cp.append("t-2", [UserMessage(content=[TextContent(text="x")])])
        await cp.save_extra("t-2", {"a": 1})
        await cp.save_extra("t-2", {"b": 2})
        data = await cp.load("t-2")

    assert data is not None
    assert data.extra == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_mysql_checkpointer_save_extra_shallow_merge_semantics(
    clean_mysql_db,
) -> None:
    """Shallow top-level merge (dict.update), NOT JSON_MERGE_PATCH.

    JSON_MERGE_PATCH would (a) delete keys whose value is null and
    (b) deep-merge nested objects. dict.update overwrites top-level keys,
    keeps null values, and replaces nested objects wholesale.
    """
    from cubepi.checkpointer.mysql import MySQLCheckpointer

    await _setup_schema(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        await cp.save_extra("t-sem", {"a": 1, "nested": {"x": 1}, "keep": "v"})
        await cp.save_extra("t-sem", {"a": 2, "nested": {"y": 2}, "z": None})
        data = await cp.load("t-sem")

    assert data is not None
    assert data.extra == {
        "a": 2,  # overwritten
        "nested": {"y": 2},  # replaced wholesale, not deep-merged
        "keep": "v",  # untouched
        "z": None,  # null preserved, not deleted
    }


@pytest.mark.asyncio
async def test_mysql_checkpointer_seq_monotonic(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import MySQLCheckpointer
    from cubepi.providers.base import TextContent, UserMessage

    await _setup_schema(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
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
async def test_uninitialized_schema_raises(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import (
        CubepiSchemaUninitialized,
        MySQLCheckpointer,
    )

    with pytest.raises(CubepiSchemaUninitialized):
        async with MySQLCheckpointer(clean_mysql_db):
            pass


@pytest.mark.asyncio
async def test_version_mismatch_raises(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import CubepiSchemaMismatch, MySQLCheckpointer
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    await _setup_schema(clean_mysql_db)
    conn = await aiomysql.connect(autocommit=True, **_parse_dsn(clean_mysql_db))
    try:
        async with conn.cursor() as cur:
            await cur.execute("UPDATE cubepi_schema_version SET version = 999")
    finally:
        await conn.ensure_closed()

    with pytest.raises(CubepiSchemaMismatch) as exc_info:
        async with MySQLCheckpointer(clean_mysql_db):
            pass
    from cubepi.checkpointer.mysql.models import EXPECTED_SCHEMA_VERSION

    assert exc_info.value.expected == EXPECTED_SCHEMA_VERSION
    assert exc_info.value.actual == 999


@pytest.mark.asyncio
async def test_empty_thread_load_returns_none(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import MySQLCheckpointer

    await _setup_schema(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        data = await cp.load("nonexistent-thread")
    assert data is None


@pytest.mark.asyncio
async def test_mysql_checkpointer_rich_content_round_trip(clean_mysql_db) -> None:
    """Round-trip thinking + tool-call + tool-result content types."""
    from cubepi.checkpointer.mysql import MySQLCheckpointer
    from cubepi.providers.base import (
        AssistantMessage,
        TextContent,
        ThinkingContent,
        ToolCall,
        ToolResultMessage,
        Usage,
    )

    await _setup_schema(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        assistant = AssistantMessage(
            content=[
                ThinkingContent(thinking="let me think"),
                TextContent(text="calling a tool"),
                ToolCall(id="call-1", name="search", arguments={"q": "cubepi"}),
            ],
            usage=Usage(),
        )
        tool_result = ToolResultMessage(
            tool_call_id="call-1",
            tool_name="search",
            content=[TextContent(text="result text")],
        )
        await cp.append("t-rich", [assistant, tool_result])
        data = await cp.load("t-rich")

    assert data is not None
    assert len(data.messages) == 2
    am, tr = data.messages
    assert isinstance(am, AssistantMessage)
    assert isinstance(am.content[0], ThinkingContent)
    assert am.content[0].thinking == "let me think"
    assert isinstance(am.content[2], ToolCall)
    assert am.content[2].id == "call-1"
    assert am.content[2].arguments == {"q": "cubepi"}
    assert isinstance(tr, ToolResultMessage)
    assert tr.tool_call_id == "call-1"
    assert tr.content[0].text == "result text"


@pytest.mark.asyncio
async def test_mysql_checkpointer_concurrent_append_seq_unique(
    clean_mysql_db,
) -> None:
    """Concurrent appends to one thread serialize via FOR UPDATE.

    Two appends race on the same thread; the row lock must give every
    message a unique, contiguous seq with no gaps or collisions.
    """
    import asyncio

    from cubepi.checkpointer.mysql import MySQLCheckpointer
    from cubepi.providers.base import TextContent, UserMessage

    await _setup_schema(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db, max_pool_size=4) as cp:
        batch_a = [UserMessage(content=[TextContent(text=f"a{i}")]) for i in range(10)]
        batch_b = [UserMessage(content=[TextContent(text=f"b{i}")]) for i in range(10)]
        await asyncio.gather(
            cp.append("t-conc", batch_a),
            cp.append("t-conc", batch_b),
        )
        data = await cp.load("t-conc")

    assert data is not None
    assert len(data.messages) == 20
    texts = {m.content[0].text for m in data.messages}
    assert texts == {f"a{i}" for i in range(10)} | {f"b{i}" for i in range(10)}


@pytest.mark.asyncio
async def test_empty_version_table_raises_uninitialized(clean_mysql_db) -> None:
    """Schema present but cubepi_schema_version has no row → Uninitialized."""
    from cubepi.checkpointer.mysql import (
        CubepiSchemaUninitialized,
        MySQLCheckpointer,
    )
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    await _setup_schema(clean_mysql_db)
    conn = await aiomysql.connect(autocommit=True, **_parse_dsn(clean_mysql_db))
    try:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM cubepi_schema_version")
    finally:
        await conn.ensure_closed()

    with pytest.raises(CubepiSchemaUninitialized):
        async with MySQLCheckpointer(clean_mysql_db):
            pass


@pytest.mark.asyncio
async def test_load_unknown_role_raises(clean_mysql_db) -> None:
    """A message row with an unrecognized role string → typed corruption
    error on load, naming the row."""
    import msgpack

    from cubepi.checkpointer.exceptions import CheckpointCorruptionError

    from cubepi.checkpointer.mysql import MySQLCheckpointer
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    await _setup_schema(clean_mysql_db)
    payload = msgpack.packb({"content": []}, use_bin_type=True)
    conn = await aiomysql.connect(autocommit=True, **_parse_dsn(clean_mysql_db))
    try:
        async with conn.cursor() as cur:
            await cur.execute("INSERT INTO cubepi_threads (thread_id) VALUES ('t-bad')")
            await cur.execute(
                "INSERT INTO cubepi_messages "
                "(thread_id, seq, role, metadata, payload) "
                "VALUES ('t-bad', 1, 'bogus', '{}', %s)",
                (payload,),
            )
    finally:
        await conn.ensure_closed()

    async with MySQLCheckpointer(clean_mysql_db) as cp:
        with pytest.raises(CheckpointCorruptionError, match="unknown role in DB") as ei:
            await cp.load("t-bad")
    assert isinstance(ei.value.__cause__, ValueError)
    assert ei.value.row_ref == "cubepi_messages.seq=1"


@pytest.mark.asyncio
async def test_missing_version_column_raises_uninitialized(clean_mysql_db) -> None:
    """A malformed cubepi_schema_version table (no `version` column) → 1054 path."""
    from cubepi.checkpointer.mysql import (
        CubepiSchemaUninitialized,
        MySQLCheckpointer,
    )
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    conn = await aiomysql.connect(autocommit=True, **_parse_dsn(clean_mysql_db))
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "CREATE TABLE cubepi_schema_version (wrong_col INT PRIMARY KEY) "
                "ENGINE=InnoDB"
            )
    finally:
        await conn.ensure_closed()

    with pytest.raises(CubepiSchemaUninitialized):
        async with MySQLCheckpointer(clean_mysql_db):
            pass


@pytest.mark.asyncio
async def test_mysql_load_corrupt_row_raises_typed(clean_mysql_db) -> None:
    """One bad payload row surfaces as CheckpointCorruptionError naming the
    row — not a raw msgpack error that hides which row is bad."""
    from cubepi.checkpointer.exceptions import CheckpointCorruptionError
    from cubepi.checkpointer.mysql import MySQLCheckpointer
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn
    from cubepi.providers.base import TextContent, UserMessage

    await _setup_schema(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        await cp.append(
            "t-corrupt",
            [
                UserMessage(content=[TextContent(text="ok")]),
                UserMessage(content=[TextContent(text="will corrupt")]),
            ],
        )
        conn = await aiomysql.connect(autocommit=True, **_parse_dsn(clean_mysql_db))
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE cubepi_messages SET payload = %s "
                    "WHERE thread_id = 't-corrupt' AND seq = ("
                    "  SELECT max_seq FROM (SELECT max(seq) AS max_seq "
                    "  FROM cubepi_messages WHERE thread_id = 't-corrupt') AS sub)",
                    (b"\xc1 not msgpack",),
                )
        finally:
            await conn.ensure_closed()

        with pytest.raises(CheckpointCorruptionError) as excinfo:
            await cp.load("t-corrupt")

    err = excinfo.value
    assert err.thread_id == "t-corrupt"
    assert err.backend == "mysql"
    assert err.row_ref.startswith("cubepi_messages.seq=")
    assert err.__cause__ is not None
