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
    Hosts must also bump `cubepi_schema_version` via write_schema_version_op()."""
    return "ALTER TABLE cubepi_threads ADD COLUMN IF NOT EXISTS pending_request JSONB"


def add_run_id_column_op() -> str:
    """Return SQL adding the v3 `run_id` column to cubepi_threads.

    Call inside the host's alembic v2→v3 upgrade() via op.execute(). The new
    column is TEXT NULL — it carries an opaque host-side identifier (e.g. the
    cubebox run_id) persisted atomically with `pending_request`. Idempotent
    under repeated execution via IF NOT EXISTS. Hosts must also bump
    `cubepi_schema_version` via write_schema_version_op() (EXPECTED_SCHEMA_VERSION
    is now 3)."""
    return "ALTER TABLE cubepi_threads ADD COLUMN IF NOT EXISTS run_id TEXT"


def create_runs_partitions_op() -> str:
    """Return SQL DDL creating all child partitions of cubepi_runs.

    Call inside an alembic upgrade() via op.execute(), AFTER the parent
    cubepi_runs partitioned table has been created. Uses the same
    PARTITION_COUNT as cubepi_messages.
    """
    return "\n".join(
        f"CREATE TABLE cubepi_runs_p{i:02d} "
        f"PARTITION OF cubepi_runs "
        f"FOR VALUES WITH (modulus {PARTITION_COUNT}, remainder {i});"
        for i in range(PARTITION_COUNT)
    )


def upgrade_v3_to_v4_op() -> str:
    """Return SQL applying the v3→v4 schema changes.

    Adds the `run_id` column + index to `cubepi_messages` and creates
    the partitioned `cubepi_runs` parent table with its child
    partitions. Hosts must also bump `cubepi_schema_version` via
    write_schema_version_op() (EXPECTED_SCHEMA_VERSION is now 4).

    Call inside the host's alembic v3→v4 upgrade() via op.execute().
    Idempotent under repeated execution via IF NOT EXISTS guards.
    """
    parts = [
        "ALTER TABLE cubepi_messages ADD COLUMN IF NOT EXISTS run_id TEXT;",
        (
            "CREATE INDEX IF NOT EXISTS ix_cubepi_messages_thread_run "
            "ON cubepi_messages (thread_id, run_id);"
        ),
        (
            "CREATE TABLE IF NOT EXISTS cubepi_runs ("
            "  thread_id TEXT NOT NULL REFERENCES cubepi_threads(thread_id) "
            "ON DELETE CASCADE,"
            "  run_id TEXT NOT NULL,"
            "  claimed_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "  completed_at TIMESTAMPTZ,"
            "  completion_seq BIGINT,"
            "  PRIMARY KEY (thread_id, run_id)"
            ") PARTITION BY HASH (thread_id);"
        ),
        (
            "CREATE INDEX IF NOT EXISTS ix_cubepi_runs_thread_seq "
            "ON cubepi_runs (thread_id, completion_seq);"
        ),
        create_runs_partitions_op(),
    ]
    return "\n".join(parts)


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
