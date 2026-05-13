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
