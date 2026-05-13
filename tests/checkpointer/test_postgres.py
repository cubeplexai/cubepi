"""PostgresCheckpointer tests (D1) — smoke imports.

Full E2E tests are in D1.3 once PostgresCheckpointer is implemented.
"""


def test_models_import() -> None:
    from cubepi.checkpointer.postgres.models import (
        EXPECTED_SCHEMA_VERSION,
        PARTITION_COUNT,
        CubepiMessage,
        CubepiSchemaVersion,
        CubepiThread,
        cubepi_metadata,
    )
    assert EXPECTED_SCHEMA_VERSION == 1
    assert PARTITION_COUNT == 64
    # All three tables registered on cubepi_metadata
    assert "cubepi_threads" in cubepi_metadata.tables
    assert "cubepi_messages" in cubepi_metadata.tables
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
    from cubepi.checkpointer.postgres.alembic_helpers import create_message_partitions_op
    sql = create_message_partitions_op()
    assert sql.count("CREATE TABLE cubepi_messages_p") == 64
    assert "modulus 64, remainder 0" in sql
    assert "modulus 64, remainder 63" in sql


def test_create_message_partitions_op_partitions_are_zero_padded() -> None:
    from cubepi.checkpointer.postgres.alembic_helpers import create_message_partitions_op
    sql = create_message_partitions_op()
    # Partition names use 2-digit padding so they sort lexicographically
    assert "cubepi_messages_p00 " in sql
    assert "cubepi_messages_p63 " in sql


def test_write_schema_version_op_includes_expected_version() -> None:
    from cubepi.checkpointer.postgres.alembic_helpers import write_schema_version_op
    sql = write_schema_version_op()
    assert "INSERT INTO cubepi_schema_version" in sql
    assert "VALUES (1)" in sql
    assert "ON CONFLICT" in sql


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
