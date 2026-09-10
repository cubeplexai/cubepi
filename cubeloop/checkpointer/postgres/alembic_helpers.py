"""SQL helpers for host application alembic migrations."""

from cubeloop.checkpointer.postgres.models import (
    EXPECTED_SCHEMA_VERSION,
    PARTITION_COUNT,
)


def create_message_partitions_op() -> str:
    """Return SQL DDL creating all 64 child partitions of cubeloop_messages.

    Call inside an alembic upgrade() via op.execute(), AFTER the parent
    cubeloop_messages table has been created. Emits current (v6) names.
    """
    return "\n".join(
        f"CREATE TABLE cubeloop_messages_p{i:02d} "
        f"PARTITION OF cubeloop_messages "
        f"FOR VALUES WITH (modulus {PARTITION_COUNT}, remainder {i});"
        for i in range(PARTITION_COUNT)
    )


def add_pending_request_column_op() -> str:
    """Return SQL adding the v2 `pending_request` column to cubepi_threads.

    Historical helper: still targets the pre-rename table name. Call inside
    the host's alembic v1→v2 upgrade() via op.execute(). Idempotent under
    repeated execution via IF NOT EXISTS.
    """
    return "ALTER TABLE cubepi_threads ADD COLUMN IF NOT EXISTS pending_request JSONB"


def add_run_id_column_op() -> str:
    """Return SQL adding the v3 `run_id` column to cubepi_threads.

    Historical helper: still targets the pre-rename table name.
    """
    return "ALTER TABLE cubepi_threads ADD COLUMN IF NOT EXISTS run_id TEXT"


def create_runs_partitions_op() -> str:
    """Return SQL DDL creating all child partitions of cubeloop_runs.

    Current-schema factory (v6 names). Historical v3→v4 inlines the old
    cubepi_runs_pXX names and must not call this helper.
    """
    return "\n".join(
        f"CREATE TABLE cubeloop_runs_p{i:02d} "
        f"PARTITION OF cubeloop_runs "
        f"FOR VALUES WITH (modulus {PARTITION_COUNT}, remainder {i});"
        for i in range(PARTITION_COUNT)
    )


def upgrade_v3_to_v4_op() -> str:
    """Return SQL applying the v3→v4 schema changes.

    Historical helper: still emits cubepi_* names, including inlined
    cubepi_runs_pXX partitions (do not call create_runs_partitions_op()).
    """
    run_partitions = "\n".join(
        f"CREATE TABLE cubepi_runs_p{i:02d} "
        f"PARTITION OF cubepi_runs "
        f"FOR VALUES WITH (modulus {PARTITION_COUNT}, remainder {i});"
        for i in range(PARTITION_COUNT)
    )
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
        run_partitions,
    ]
    return "\n".join(parts)


def upgrade_v4_to_v5_op() -> str:
    """Return SQL applying the v4->v5 schema changes.

    Historical helper: still emits cubepi_* names.
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


def upgrade_v5_to_v6_op() -> str:
    """Rename cubepi_* tables/indexes/partitions to cubeloop_*.

    Complete (no-op) iff both cubeloop_threads and cubeloop_schema_version
    exist. If data tables already moved but the version table did not,
    rename the leftover. cubeloop_threads alone is not a no-op sentinel.

    Requires a complete v5 (including cubepi_schema_version). Missing
    version table makes the rename fail as a whole.
    """
    message_parts = "\n".join(
        f"    ALTER TABLE cubepi_messages_p{i:02d} "
        f"RENAME TO cubeloop_messages_p{i:02d};"
        for i in range(PARTITION_COUNT)
    )
    run_parts = "\n".join(
        f"    ALTER TABLE cubepi_runs_p{i:02d} RENAME TO cubeloop_runs_p{i:02d};"
        for i in range(PARTITION_COUNT)
    )
    return f"""
DO $$
BEGIN
  IF to_regclass('cubeloop_threads') IS NOT NULL
     AND to_regclass('cubeloop_schema_version') IS NOT NULL THEN
    RETURN;
  END IF;

  IF to_regclass('cubeloop_threads') IS NOT NULL
     AND to_regclass('cubepi_schema_version') IS NOT NULL THEN
    ALTER TABLE cubepi_schema_version RENAME TO cubeloop_schema_version;
    RETURN;
  END IF;

  ALTER TABLE cubepi_threads RENAME TO cubeloop_threads;
  ALTER TABLE cubepi_messages RENAME TO cubeloop_messages;
{message_parts}
  ALTER TABLE cubepi_runs RENAME TO cubeloop_runs;
{run_parts}
  ALTER TABLE cubepi_hitl_answers RENAME TO cubeloop_hitl_answers;
  ALTER TABLE cubepi_schema_version RENAME TO cubeloop_schema_version;
  ALTER INDEX IF EXISTS ix_cubepi_messages_metadata_gin
    RENAME TO ix_cubeloop_messages_metadata_gin;
  ALTER INDEX IF EXISTS ix_cubepi_messages_thread_run
    RENAME TO ix_cubeloop_messages_thread_run;
  ALTER INDEX IF EXISTS ix_cubepi_runs_thread_seq
    RENAME TO ix_cubeloop_runs_thread_seq;
END $$;
"""


def write_schema_version_op() -> str:
    """Write EXPECTED_SCHEMA_VERSION to cubeloop_schema_version if that
    table exists, else to cubepi_schema_version.

    One ``DO $$`` block so asyncpg / Alembic send it as a single statement.
    """
    n = EXPECTED_SCHEMA_VERSION
    return f"""
DO $$
BEGIN
  IF to_regclass('cubeloop_schema_version') IS NOT NULL THEN
    DELETE FROM cubeloop_schema_version WHERE version <> {n};
    INSERT INTO cubeloop_schema_version (version) VALUES ({n})
      ON CONFLICT DO NOTHING;
  ELSE
    DELETE FROM cubepi_schema_version WHERE version <> {n};
    INSERT INTO cubepi_schema_version (version) VALUES ({n})
      ON CONFLICT DO NOTHING;
  END IF;
END $$;
"""
