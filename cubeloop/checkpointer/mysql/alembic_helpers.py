"""SQL helpers for host application alembic migrations (MySQL)."""

from cubeloop.checkpointer.mysql.models import (
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


def runs_partition_clause() -> str:
    """Return the KEY-partition clause for the cubepi_runs CREATE TABLE.

    Mirrors ``messages_partition_clause()``: KEY-partitioning the runs
    table by thread_id keeps each run's row colocated with its messages
    on the same partition shard. KEY (not HASH) because thread_id is a
    VARCHAR.
    """
    return f"PARTITION BY KEY (thread_id) PARTITIONS {PARTITION_COUNT}"


def add_messages_run_id_column_op() -> str:
    """Return SQL adding the v4 `run_id` column + composite index to
    cubepi_messages.

    Call inside the host's alembic v3→v4 upgrade() via op.execute(). The
    helper returns two ';'-separated statements (ALTER then CREATE INDEX);
    MySQL/pymysql executes a single statement per call, so split before
    executing::

        for stmt in add_messages_run_id_column_op().split(";"):
            if stmt.strip():
                op.execute(stmt)

    MySQL does not support IF NOT EXISTS for ADD COLUMN; guard with a
    schema check in the alembic migration if idempotence is required.
    """
    return (
        "ALTER TABLE cubepi_messages ADD COLUMN run_id VARCHAR(255) NULL; "
        "CREATE INDEX ix_cubepi_messages_thread_run "
        "ON cubepi_messages (thread_id, run_id);"
    )


def create_runs_table_op() -> str:
    """Return SQL creating the partitioned ``cubeloop_runs`` parent table.

    Current-schema factory (v6 names). Historical v3→v4 inlines the old
    cubepi_runs CREATE and must not call this helper. NO FK to
    ``cubeloop_threads`` — MySQL forbids FKs on partitioned tables.
    """
    return (
        "CREATE TABLE cubeloop_runs ("
        "  thread_id VARCHAR(255) COLLATE utf8mb4_bin NOT NULL,"
        "  run_id VARCHAR(255) NOT NULL,"
        "  claimed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  completed_at TIMESTAMP NULL,"
        "  completion_seq BIGINT NULL,"
        "  PRIMARY KEY (thread_id, run_id),"
        "  KEY ix_cubeloop_runs_thread_seq (thread_id, completion_seq)"
        ") ENGINE=InnoDB " + runs_partition_clause()
    )


def create_runs_partitions_op() -> str:
    """Return SQL extending the runs table with KEY partitions.

    Provided for symmetry with the Postgres helper of the same name.
    For MySQL, partitions are declared inline by ``create_runs_table_op()``
    (the KEY clause specifies the partition count). This helper returns
    only the partition clause for callers that want to attach it
    separately, e.g. via ALTER.
    """
    return runs_partition_clause()


def upgrade_v3_to_v4_op() -> str:
    """Return SQL applying the v3→v4 schema changes.

    Adds the ``run_id`` column + composite index to ``cubepi_messages``
    and creates the partitioned ``cubepi_runs`` table. Hosts must also
    bump ``cubepi_schema_version`` via ``write_schema_version_op()``
    (EXPECTED_SCHEMA_VERSION is now 4).

    Returns multiple ';'-separated statements; MySQL/pymysql executes one
    per call, so split before executing::

        for stmt in upgrade_v3_to_v4_op().split(";"):
            if stmt.strip():
                op.execute(stmt)
    """
    # Inline the pre-rename cubepi_runs CREATE. create_runs_table_op() now
    # emits cubeloop_runs names for fresh installs.
    legacy_runs = (
        "CREATE TABLE cubepi_runs ("
        "  thread_id VARCHAR(255) COLLATE utf8mb4_bin NOT NULL,"
        "  run_id VARCHAR(255) NOT NULL,"
        "  claimed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  completed_at TIMESTAMP NULL,"
        "  completion_seq BIGINT NULL,"
        "  PRIMARY KEY (thread_id, run_id),"
        "  KEY ix_cubepi_runs_thread_seq (thread_id, completion_seq)"
        ") ENGINE=InnoDB " + runs_partition_clause()
    )
    return f"{add_messages_run_id_column_op()} {legacy_runs};"


def create_hitl_answers_table_op() -> str:
    """Return SQL creating the durable HITL answer ledger table."""
    return (
        "CREATE TABLE cubepi_hitl_answers ("
        "  thread_id VARCHAR(255) COLLATE utf8mb4_bin NOT NULL,"
        "  run_id VARCHAR(255) NOT NULL,"
        "  question_id VARCHAR(255) NOT NULL,"
        "  answer JSON NOT NULL,"
        "  answered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  PRIMARY KEY (thread_id, run_id, question_id)"
        ") ENGINE=InnoDB"
    )


def upgrade_v4_to_v5_op() -> str:
    """Return SQL applying the v4->v5 schema changes.

    Creates the durable HITL answer ledger table. Hosts must also bump
    ``cubepi_schema_version`` via ``write_schema_version_op()``
    (EXPECTED_SCHEMA_VERSION is now 5).
    """
    return create_hitl_answers_table_op() + ";"


def upgrade_v5_to_v6_op() -> str:
    """Rename cubepi_* tables to cubeloop_* in one atomic RENAME TABLE.

    Includes cubepi_schema_version. KEY partitions ride with the table.

    Returns ';'-joined standalone statements (SET / PREPARE / EXECUTE).
    Hosts must split the same way as ``write_schema_version_op``::

        for stmt in upgrade_v5_to_v6_op().split(";"):
            if stmt.strip():
                op.execute(stmt)

    Complete (no-op) iff both cubeloop_threads and cubeloop_schema_version
    already exist. If data tables moved but the version table did not,
    only the leftover version table is renamed. Requires a complete v5;
    missing cubepi_schema_version fails the whole rename.
    """
    rename_all = (
        "RENAME TABLE "
        "cubepi_threads TO cubeloop_threads, "
        "cubepi_messages TO cubeloop_messages, "
        "cubepi_runs TO cubeloop_runs, "
        "cubepi_hitl_answers TO cubeloop_hitl_answers, "
        "cubepi_schema_version TO cubeloop_schema_version"
    )
    rename_ver = "RENAME TABLE cubepi_schema_version TO cubeloop_schema_version"
    return (
        "SET @cp_new_th = ("
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema = DATABASE() "
        "AND table_name = 'cubeloop_threads'); "
        "SET @cp_new_ver = ("
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema = DATABASE() "
        "AND table_name = 'cubeloop_schema_version'); "
        "SET @cp_old_ver = ("
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema = DATABASE() "
        "AND table_name = 'cubepi_schema_version'); "
        "SET @cp_sql = IF(@cp_new_th > 0 AND @cp_new_ver > 0, "
        "'DO 0', "
        "IF(@cp_new_th > 0 AND @cp_old_ver > 0, "
        f"'{rename_ver}', "
        f"'{rename_all}')); "
        "PREPARE cp_v6 FROM @cp_sql; "
        "EXECUTE cp_v6; "
        "DEALLOCATE PREPARE cp_v6;"
    )


def write_schema_version_op() -> str:
    """Write EXPECTED_SCHEMA_VERSION to cubeloop_schema_version if that
    table exists, else to cubepi_schema_version.

    Returns ';'-joined **standalone** statements. Hosts must split::

        for stmt in write_schema_version_op().split(";"):
            if stmt.strip():
                op.execute(stmt)

    No compound IF/THEN — internal semicolons would break the split contract.
    """
    n = EXPECTED_SCHEMA_VERSION
    return (
        "SET @cp_ver_new = ("
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema = DATABASE() "
        "AND table_name = 'cubeloop_schema_version'); "
        f"SET @cp_del = IF(@cp_ver_new > 0, "
        f"'DELETE FROM cubeloop_schema_version WHERE version <> {n}', "
        f"'DELETE FROM cubepi_schema_version WHERE version <> {n}'); "
        "PREPARE cp_stmt FROM @cp_del; "
        "EXECUTE cp_stmt; "
        "DEALLOCATE PREPARE cp_stmt; "
        f"SET @cp_ins = IF(@cp_ver_new > 0, "
        f"'INSERT IGNORE INTO cubeloop_schema_version (version) VALUES ({n})', "
        f"'INSERT IGNORE INTO cubepi_schema_version (version) VALUES ({n})'); "
        "PREPARE cp_stmt FROM @cp_ins; "
        "EXECUTE cp_stmt; "
        "DEALLOCATE PREPARE cp_stmt;"
    )
