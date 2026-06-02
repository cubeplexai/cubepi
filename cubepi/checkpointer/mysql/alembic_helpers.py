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


def add_pending_request_column_op() -> str:
    """Return SQL adding the v2 `pending_request` column to cubepi_threads.

    Call inside the host's alembic v1→v2 upgrade() via op.execute(). MySQL does
    not support IF NOT EXISTS for ADD COLUMN; guard with a schema check in the
    alembic migration if idempotence is required. Hosts must also bump
    `cubepi_schema_version` via write_schema_version_op()."""
    return "ALTER TABLE cubepi_threads ADD COLUMN pending_request JSON NULL"


def add_run_id_column_op() -> str:
    """Return SQL adding the v3 `run_id` column to cubepi_threads.

    Call inside the host's alembic v2→v3 upgrade() via op.execute(). MySQL does
    not support IF NOT EXISTS for ADD COLUMN; guard with a schema check in the
    alembic migration if idempotence is required. Hosts must also bump
    `cubepi_schema_version` via write_schema_version_op() (EXPECTED_SCHEMA_VERSION
    is now 3)."""
    # VARCHAR(64) accommodates UUIDs (36) and prefixed-UUID conventions.
    # Hosts needing longer run_ids should subclass MySQLCheckpointer and
    # override the column rather than have cubepi pay the TEXT cost
    # globally.
    return "ALTER TABLE cubepi_threads ADD COLUMN run_id VARCHAR(64) NULL"


def write_schema_version_op() -> str:
    """Return SQL setting cubepi_schema_version to the current version.

    Clears any stale rows from prior cubepi versions then inserts the current
    one. Idempotent. Call inside alembic upgrade() after CREATE TABLE
    cubepi_schema_version.

    Returns two ';'-separated statements (DELETE then INSERT). MySQL/pymysql
    executes a single statement per call, so split before executing::

        for stmt in write_schema_version_op().split(";"):
            if stmt.strip():
                op.execute(stmt)
    """
    return (
        f"DELETE FROM cubepi_schema_version "
        f"WHERE version <> {EXPECTED_SCHEMA_VERSION}; "
        f"INSERT IGNORE INTO cubepi_schema_version (version) "
        f"VALUES ({EXPECTED_SCHEMA_VERSION});"
    )
