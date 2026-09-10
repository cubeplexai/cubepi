-- Emergency recovery for databases migrated by withdrawn cubeloop 0.14.0.
-- Stop all application instances and take a backup before running this file.
BEGIN;

DO $$
DECLARE
  schema_name text := current_schema();
  new_count integer;
  old_count integer;
  new_index_count integer;
  old_index_count integer;
  i integer;
BEGIN
  SELECT count(*) INTO new_count
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE relname IN (
    'cubeloop_threads', 'cubeloop_messages', 'cubeloop_runs',
    'cubeloop_hitl_answers', 'cubeloop_schema_version'
  ) AND n.nspname = schema_name;
  SELECT count(*) INTO old_count
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE relname IN (
    'cubepi_threads', 'cubepi_messages', 'cubepi_runs',
    'cubepi_hitl_answers', 'cubepi_schema_version'
  ) AND n.nspname = schema_name;
  IF new_count <> 5 OR old_count <> 0 THEN
    RAISE EXCEPTION
      'expected a complete 0.14.0 schema (5 cubeloop tables, 0 cubepi tables); found % and %',
      new_count, old_count;
  END IF;
  SELECT count(*) INTO new_index_count
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE relkind IN ('i', 'I') AND relname IN (
    'ix_cubeloop_messages_metadata_gin',
    'ix_cubeloop_messages_thread_run', 'ix_cubeloop_runs_thread_seq'
  ) AND n.nspname = schema_name;
  SELECT count(*) INTO old_index_count
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE relkind IN ('i', 'I') AND relname IN (
    'ix_cubepi_messages_metadata_gin',
    'ix_cubepi_messages_thread_run', 'ix_cubepi_runs_thread_seq'
  ) AND n.nspname = schema_name;
  IF new_index_count <> 3 OR old_index_count <> 0 THEN
    RAISE EXCEPTION
      'index preflight failed: expected 3 cubeloop indexes and 0 cubepi indexes';
  END IF;

  FOR i IN 0..63 LOOP
    IF to_regclass(format('%I.%I', schema_name, 'cubeloop_messages_p' || lpad(i::text, 2, '0'))) IS NULL
       OR to_regclass(format('%I.%I', schema_name, 'cubeloop_runs_p' || lpad(i::text, 2, '0'))) IS NULL
       OR to_regclass(format('%I.%I', schema_name, 'cubepi_messages_p' || lpad(i::text, 2, '0'))) IS NOT NULL
       OR to_regclass(format('%I.%I', schema_name, 'cubepi_runs_p' || lpad(i::text, 2, '0'))) IS NOT NULL THEN
      RAISE EXCEPTION 'partition preflight failed at suffix %', i;
    END IF;
  END LOOP;

  EXECUTE format('ALTER TABLE %I.cubeloop_threads RENAME TO cubepi_threads', schema_name);
  EXECUTE format('ALTER TABLE %I.cubeloop_messages RENAME TO cubepi_messages', schema_name);
  FOR i IN 0..63 LOOP
    EXECUTE format(
      'ALTER TABLE %I.%I RENAME TO %I', schema_name,
      'cubeloop_messages_p' || lpad(i::text, 2, '0'),
      'cubepi_messages_p' || lpad(i::text, 2, '0')
    );
  END LOOP;
  EXECUTE format('ALTER TABLE %I.cubeloop_runs RENAME TO cubepi_runs', schema_name);
  FOR i IN 0..63 LOOP
    EXECUTE format(
      'ALTER TABLE %I.%I RENAME TO %I', schema_name,
      'cubeloop_runs_p' || lpad(i::text, 2, '0'),
      'cubepi_runs_p' || lpad(i::text, 2, '0')
    );
  END LOOP;
  EXECUTE format('ALTER TABLE %I.cubeloop_hitl_answers RENAME TO cubepi_hitl_answers', schema_name);
  EXECUTE format('ALTER TABLE %I.cubeloop_schema_version RENAME TO cubepi_schema_version', schema_name);
  EXECUTE format('ALTER INDEX %I.ix_cubeloop_messages_metadata_gin RENAME TO ix_cubepi_messages_metadata_gin', schema_name);
  EXECUTE format('ALTER INDEX %I.ix_cubeloop_messages_thread_run RENAME TO ix_cubepi_messages_thread_run', schema_name);
  EXECUTE format('ALTER INDEX %I.ix_cubeloop_runs_thread_seq RENAME TO ix_cubepi_runs_thread_seq', schema_name);
  EXECUTE format('DELETE FROM %I.cubepi_schema_version WHERE version <> 5', schema_name);
  EXECUTE format('INSERT INTO %I.cubepi_schema_version (version) VALUES (5) ON CONFLICT DO NOTHING', schema_name);
END $$;

COMMIT;
