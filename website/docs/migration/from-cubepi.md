---
title: From cubepi
description: "CubePi was renamed to CubeLoop in 0.14. How to update installs, imports, checkpointer schema, and traces."
---

# Migrating from cubepi

0.14 renames the project from CubePi / `cubepi` to CubeLoop / `cubeloop`.
GitHub keeps the same repository (renamed in place to `cubeplexai/cubeloop`).
The docs site is https://cubeloop.dev.

## Install and imports

```bash
pip install cubeloop
# extras are unchanged: cubeloop[sqlite], cubeloop[postgres], cubeloop[mysql], …
```

```python
from cubeloop import Agent, tool
from cubeloop.providers.anthropic import AnthropicProvider
```

`pip install -U cubepi` still works: 0.14+ is a wrapper that depends on
`cubeloop==0.14.0`, re-exports the public API, and aliases `cubepi.*` imports.
It warns on first import. Prefer depending on `cubeloop` directly.

CLI: `cubeloop trace` (the wrapper still provides `cubepi trace` and warns).

## Checkpointer schema v5 → v6

Postgres and MySQL table names moved from `cubepi_*` to `cubeloop_*`.
SQLite table names were never prefixed; they stay `messages` / `runs` / ….

Existing databases: add an Alembic revision that runs
`upgrade_v5_to_v6_op()` then `write_schema_version_op()`. Opening 0.14 against
an unmigrated v5 database raises `CubeloopSchemaMismatch` pointing at that
helper — not “tables not found”.

If you have data tables but no `*_schema_version` table, do not apply the
fresh v6 CREATE TABLE. Create `cubepi_schema_version` with version 5 first,
then run the v5→v6 helper.

Ship the package bump and this revision **together**. A process that imports
0.14 and then opens the checkpointer against a still-v5 database will refuse
to start.

### Do not retarget historical helpers blindly

0.14 split the Alembic helpers into two kinds:

| Helper | SQL it emits in 0.14 |
|---|---|
| `create_message_partitions_op()`, `create_runs_partitions_op()` | **Current** names: `cubeloop_messages_pXX PARTITION OF cubeloop_messages`, same for runs. Postgres only. |
| `add_pending_request_column_op()`, `add_run_id_column_op()`, `upgrade_v3_to_v4_op()`, `upgrade_v4_to_v5_op()` | **Historical** names: still `cubepi_*`. Safe to import from `cubeloop.checkpointer.postgres.alembic_helpers`. |
| `write_schema_version_op()` | Writes `EXPECTED_SCHEMA_VERSION` (now **6**) to `cubeloop_schema_version` if that table exists, else to `cubepi_schema_version`. |

A typical host v1 revision created `cubepi_messages` and then called
`create_message_partitions_op()`. After you depend on cubeloop 0.14, that
same call emits `PARTITION OF cubeloop_messages`. A greenfield
`alembic upgrade head` (empty database replaying v1→v6) dies at v1:
the cubeloop parent does not exist yet.

**Fix:** stop calling `create_message_partitions_op()` from that historical
revision. Inline the original SQL against the old parent:

```sql
CREATE TABLE cubepi_messages_p00
  PARTITION OF cubepi_messages
  FOR VALUES WITH (modulus 64, remainder 0);
-- … p01 … p63
```

The same trap applies if a historical revision called
`create_runs_partitions_op()`. CubeLoop's own `upgrade_v3_to_v4_op()`
already inlines `cubepi_runs_pXX` and is safe.

Do the inline **before** (or in the same change as) removing the `cubepi`
package. Do not run Alembic against an empty database in between: v1 still
`import cubepi.checkpointer…` until you retarget the remaining helper
imports, and after the pin that module is gone.

`write_schema_version_op()` writing 6 from a v1 replay is intentional —
it has always written the *current* expected version, and `alembic upgrade
head` runs v1→v6 in one shot. Do not rewrite those calls to insert
historical 1/2/3/4/5 by hand.

### Autogenerate can drop `cubepi_threads`

`cubeloop_metadata` describes the **new** names. After you point Alembic
`target_metadata` at it, a database that is still on v5 still has
`cubepi_*` tables. Autogenerate then proposes `DROP cubepi_*` and
`CREATE cubeloop_*` (empty). `cubepi_threads` is the usual casualty:
many hosts let autogen create it in v1 and never put it on an exclusion
list.

That drop cascades onto messages, runs, and HITL answers.

**Fix, in `env.py`, before generating the v6 revision:** exclude both
generations from autogenerate:

- Tables: `cubepi_threads`, `cubepi_messages`, `cubepi_runs`,
  `cubepi_hitl_answers`, `cubepi_schema_version`, and the matching
  `cubeloop_*` names.
- Postgres partitions: prefixes `cubepi_messages_p`, `cubepi_runs_p`,
  `cubeloop_messages_p`, `cubeloop_runs_p`.

Inspect the generated v6 revision: it must not contain
`DROP TABLE cubepi_threads` or a fresh `CREATE TABLE cubeloop_*`. The
body should be `upgrade_v5_to_v6_op()` plus `write_schema_version_op()`.

### Downgrade is a reverse rename, not DROP

`upgrade_v5_to_v6_op()` renames parents, **each** of the 64 message and
64 run partitions (Postgres), the HITL and version tables, and
`ix_cubepi_*` indexes. PostgreSQL `ALTER TABLE … RENAME TO` on a
partitioned parent does **not** rename children.

A downgrade copied from an older bump (`DROP TABLE cubeloop_runs
CASCADE`) deletes every conversation. Reverse-rename instead, including
every partition, then write version 5 into `cubepi_schema_version`.

## Tracing

New spans use `cubeloop.*` attributes and `cubeloop.turn` span names.
`cubeloop trace` still reads 0.13 JSONL that used `cubepi.*`.
Dashboards you own need their own attribute-name update.

Default JSONL directory is `./cubeloop-traces`. There is no automatic
fallback to `./cubepi-traces`; pass `--dir` (or keep your config path)
to read leftover files.

## Environment variables

`CUBELOOP_TEST_PG_DSN`, `CUBELOOP_TEST_MYSQL_DSN`, `CUBELOOP_PG_DSN`,
`CUBELOOP_MYSQL_DSN` replace the `CUBEPI_*` names. The old names are still
read if the new one is unset, for the rest of 0.x.
