"""SQL helpers for host application alembic migrations."""

from cubepi.checkpointer.postgres.models import (
    EXPECTED_SCHEMA_VERSION,
    PARTITION_COUNT,
)


def create_message_partitions_op() -> str:
    """Return SQL DDL creating all 64 child partitions of cubepi_messages.

    Call inside an alembic upgrade() via op.execute(), AFTER the parent
    cubepi_messages table has been created.
    """
    return "\n".join(
        f"CREATE TABLE cubepi_messages_p{i:02d} "
        f"PARTITION OF cubepi_messages "
        f"FOR VALUES WITH (modulus {PARTITION_COUNT}, remainder {i});"
        for i in range(PARTITION_COUNT)
    )


def add_pending_request_column_op() -> str:
    """Return SQL adding the v2 `pending_request` column to cubepi_threads.

    Call inside the host's alembic v1→v2 upgrade() via op.execute(). The new
    column is JSONB NULL. Idempotent under repeated execution via IF NOT EXISTS.
    Hosts must also bump `cubepi_schema_version` via write_schema_version_op()
    (already documented; EXPECTED_SCHEMA_VERSION is now 2)."""
    return "ALTER TABLE cubepi_threads ADD COLUMN IF NOT EXISTS pending_request JSONB"


def write_schema_version_op() -> str:
    """Return SQL setting cubepi_schema_version to the current version.

    Call inside alembic upgrade() after CREATE TABLE cubepi_schema_version.

    Atomically clears any stale rows (from prior cubepi versions) and writes
    the current version. ``version`` is the primary key, so a plain INSERT
    would leave older rows in place and ``_verify_schema`` (which does
    ``SELECT version ... LIMIT 1``) could read a stale value and falsely
    report CubepiSchemaMismatch. Idempotent under repeated execution.
    """
    return (
        f"DELETE FROM cubepi_schema_version "
        f"WHERE version <> {EXPECTED_SCHEMA_VERSION}; "
        f"INSERT INTO cubepi_schema_version (version) "
        f"VALUES ({EXPECTED_SCHEMA_VERSION}) ON CONFLICT DO NOTHING;"
    )
