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


def write_schema_version_op() -> str:
    """Return SQL inserting the current schema version.

    Call inside alembic upgrade() after CREATE TABLE cubepi_schema_version.
    Idempotent via ON CONFLICT DO NOTHING.
    """
    return (
        f"INSERT INTO cubepi_schema_version (version) "
        f"VALUES ({EXPECTED_SCHEMA_VERSION}) ON CONFLICT DO NOTHING;"
    )
