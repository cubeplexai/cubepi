"""SQL helpers for host application alembic migrations."""

from cubeloop.checkpointer.postgres.models import PARTITION_COUNT


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

    Call inside the host's alembic v1→v2 upgrade() via op.execute(). Idempotent
    under repeated execution via IF NOT EXISTS.
    """
    return "ALTER TABLE cubepi_threads ADD COLUMN IF NOT EXISTS pending_request JSONB"


def add_run_id_column_op() -> str:
    """Return SQL adding the v3 `run_id` column to cubepi_threads.

    Call inside the host's alembic v2→v3 upgrade() via op.execute().
    """
    return "ALTER TABLE cubepi_threads ADD COLUMN IF NOT EXISTS run_id TEXT"


def create_runs_partitions_op() -> str:
    """Return SQL DDL creating all child partitions of cubepi_runs.

    Call after the partitioned cubepi_runs parent has been created.
    """
    return "\n".join(
        f"CREATE TABLE cubepi_runs_p{i:02d} "
        f"PARTITION OF cubepi_runs "
        f"FOR VALUES WITH (modulus {PARTITION_COUNT}, remainder {i});"
        for i in range(PARTITION_COUNT)
    )


def upgrade_v3_to_v4_op() -> str:
    """Return SQL applying the v3→v4 schema changes.

    Adds the run_id column and index, then creates cubepi_runs and its
    partitions.
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


def upgrade_v4_to_v5_op() -> str:
    """Return SQL applying the v4->v5 schema changes.

    Creates the durable HITL answer ledger table.
    """
    return (
        "CREATE TABLE IF NOT EXISTS cubepi_hitl_answers ("
        "  thread_id TEXT NOT NULL REFERENCES cubepi_threads(thread_id) "
        "ON DELETE CASCADE,"
        "  run_id TEXT NOT NULL,"
        "  question_id TEXT NOT NULL,"
        "  answer JSONB NOT NULL,"
        "  answered_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
        "  PRIMARY KEY (thread_id, run_id, question_id)"
        ");"
    )


def write_schema_version_op() -> str:
    """Return the immutable v5 schema-version write used by host migrations."""
    return (
        "DELETE FROM cubepi_schema_version WHERE version <> 5; "
        "INSERT INTO cubepi_schema_version (version) VALUES (5) "
        "ON CONFLICT DO NOTHING;"
    )
