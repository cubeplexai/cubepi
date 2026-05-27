"""SQL helpers for host application alembic migrations (MySQL)."""

from cubepi.checkpointer.mysql.models import (
    EXPECTED_SCHEMA_VERSION,
    PARTITION_COUNT,
)


def messages_partition_clause() -> str:
    """Return the KEY-partition clause for the cubepi_messages CREATE TABLE.

    Unlike Postgres (one child partition per modulus), MySQL declares all
    partitions inline. Append this to the messages table DDL, e.g.::

        op.execute("CREATE TABLE cubepi_messages (...) " + messages_partition_clause())

    KEY (not HASH) is required because thread_id is a VARCHAR; MySQL HASH
    partitioning only accepts integer expressions.
    """
    return f"PARTITION BY KEY (thread_id) PARTITIONS {PARTITION_COUNT}"


def write_schema_version_op() -> str:
    """Return SQL setting cubepi_schema_version to the current version.

    Clears any stale rows from prior cubepi versions then inserts the current
    one. Idempotent. Call inside alembic upgrade() after CREATE TABLE
    cubepi_schema_version.
    """
    return (
        f"DELETE FROM cubepi_schema_version "
        f"WHERE version <> {EXPECTED_SCHEMA_VERSION}; "
        f"INSERT IGNORE INTO cubepi_schema_version (version) "
        f"VALUES ({EXPECTED_SCHEMA_VERSION});"
    )
