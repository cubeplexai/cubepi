-- Emergency recovery for databases created or migrated by withdrawn 0.14.0.
-- Stop all application instances and take a backup before running this file.
SET @cp_new_tables = (
  SELECT COUNT(*) FROM information_schema.tables
  WHERE table_schema = DATABASE() AND table_name IN (
    'cubeloop_threads', 'cubeloop_messages', 'cubeloop_runs',
    'cubeloop_hitl_answers', 'cubeloop_schema_version'
  )
);
SET @cp_old_tables = (
  SELECT COUNT(*) FROM information_schema.tables
  WHERE table_schema = DATABASE() AND table_name IN (
    'cubepi_threads', 'cubepi_messages', 'cubepi_runs',
    'cubepi_hitl_answers', 'cubepi_schema_version'
  )
);
SET @cp_new_indexes = (
  SELECT COUNT(DISTINCT index_name) FROM information_schema.statistics
  WHERE table_schema = DATABASE() AND index_name IN (
    'ix_cubeloop_messages_metadata_gin',
    'ix_cubeloop_messages_thread_run', 'ix_cubeloop_runs_thread_seq'
  )
);
SET @cp_old_indexes = (
  SELECT COUNT(DISTINCT index_name) FROM information_schema.statistics
  WHERE table_schema = DATABASE() AND index_name IN (
    'ix_cubepi_messages_metadata_gin',
    'ix_cubepi_messages_thread_run', 'ix_cubepi_runs_thread_seq'
  )
);
SET @cp_fresh_indexes = (@cp_new_indexes > 0);
SET @cp_guard = IF(
  @cp_new_tables = 5 AND @cp_old_tables = 0
  AND ((@cp_new_indexes = 0 AND @cp_old_indexes = 2)
       OR (@cp_new_indexes = 2 AND @cp_old_indexes = 0)),
  'DO 0',
  'SELECT * FROM cubeloop_0140_recovery_preflight_failed'
);
PREPARE cp_guard FROM @cp_guard;
EXECUTE cp_guard;
DEALLOCATE PREPARE cp_guard;

RENAME TABLE
  cubeloop_threads TO cubepi_threads,
  cubeloop_messages TO cubepi_messages,
  cubeloop_runs TO cubepi_runs,
  cubeloop_hitl_answers TO cubepi_hitl_answers,
  cubeloop_schema_version TO cubepi_schema_version;

SET @cp_rename_message_index = IF(
  @cp_fresh_indexes,
  'ALTER TABLE cubepi_messages RENAME INDEX ix_cubeloop_messages_thread_run TO ix_cubepi_messages_thread_run',
  'DO 0'
);
PREPARE cp_stmt FROM @cp_rename_message_index;
EXECUTE cp_stmt;
DEALLOCATE PREPARE cp_stmt;
SET @cp_rename_run_index = IF(
  @cp_fresh_indexes,
  'ALTER TABLE cubepi_runs RENAME INDEX ix_cubeloop_runs_thread_seq TO ix_cubepi_runs_thread_seq',
  'DO 0'
);
PREPARE cp_stmt FROM @cp_rename_run_index;
EXECUTE cp_stmt;
DEALLOCATE PREPARE cp_stmt;

DELETE FROM cubepi_schema_version WHERE version <> 5;
INSERT IGNORE INTO cubepi_schema_version (version) VALUES (5);
