# MySQLCheckpointer Design

**Date:** 2026-05-27
**Status:** Spec
**Topic:** Add a production-grade MySQL `Checkpointer` implementation, mirroring
`PostgresCheckpointer`.

## Motivation

cubepi ships memory / sqlite / postgres checkpointers. Deployments standardized
on MySQL (e.g. cubemanus, whose async stack is `aiomysql`) currently have no
first-class option. This adds `MySQLCheckpointer` with the same production
characteristics as `PostgresCheckpointer`: connection pool, msgpack payloads,
queryable JSON `metadata` column, host-managed schema with startup version
verification.

## Goals / Non-goals

**Goals**
- Implement the `Checkpointer` protocol (`load` / `append` / `save_extra`)
  against MySQL with feature parity to `PostgresCheckpointer`.
- Host-managed schema: ship SQLAlchemy models + alembic helpers; verify
  `cubepi_schema_version` on context entry.
- Monotonic per-thread `seq` allocation under concurrency.
- Ship as an optional `mysql` extra; keep core deps lean.
- User-facing docs in `website/docs/`.

**Non-goals**
- No automatic schema creation by the checkpointer (host owns DDL, like
  Postgres). MySQL JSON metadata indexing is left to the host.
- No thread forking logic beyond the columns the Postgres schema already
  carries (`parent_thread_id`, `forked_at_seq`) — they exist for schema parity
  but are not exercised by this checkpointer.

## Driver

`aiomysql` (matches cubemanus, which depends on
`langgraph-checkpoint-mysql[aiomysql]`). Pool via `aiomysql.create_pool`.
Connection configured via discrete params parsed from a DSN-style URL, or passed
explicitly. Placeholders are `%s` (PyMySQL paramstyle).

## Module layout

Mirror `cubepi/checkpointer/postgres/`:

```
cubepi/checkpointer/mysql/
  __init__.py          # exports MySQLCheckpointer, schema symbols, exceptions
  checkpointer.py      # MySQLCheckpointer
  models.py            # SQLAlchemy table defs on a private cubepi_metadata
  alembic_helpers.py   # host alembic migration helpers
  exceptions.py        # CubepiSchemaError / Uninitialized / Mismatch
```

Exceptions are structurally identical to the Postgres ones. To avoid drift they
are **re-exported from the Postgres exceptions module** rather than redefined:
`from cubepi.checkpointer.postgres.exceptions import (...)`. (They are
backend-agnostic schema errors; this keeps a single source of truth.)

## Schema

Host creates these via alembic; the checkpointer only verifies on entry.

### `cubepi_threads`
| column | type | notes |
|---|---|---|
| `thread_id` | `VARCHAR(255)` | PK |
| `parent_thread_id` | `VARCHAR(255) NULL` | no self-FK (see divergence) |
| `forked_at_seq` | `BIGINT NULL` | |
| `extra` | `JSON NOT NULL` | default `'{}'` |
| `created_at` | `TIMESTAMP NOT NULL` | default `CURRENT_TIMESTAMP` |
| `updated_at` | `TIMESTAMP NOT NULL` | default `CURRENT_TIMESTAMP ON UPDATE` |

### `cubepi_messages`
| column | type | notes |
|---|---|---|
| `thread_id` | `VARCHAR(255) NOT NULL` | part of PK |
| `seq` | `BIGINT NOT NULL` | part of PK |
| `role` | `VARCHAR(32) NOT NULL` | `user`/`assistant`/`tool` |
| `metadata` | `JSON NOT NULL` | default `'{}'`, no index |
| `payload` | `LONGBLOB NOT NULL` | msgpack |
| `created_at` | `TIMESTAMP NOT NULL` | default `CURRENT_TIMESTAMP` |

PK `(thread_id, seq)`. Table is `PARTITION BY HASH (thread_id) PARTITIONS 64`.
No foreign key (see divergence). MySQL requires every unique key to contain the
partition column — satisfied since `thread_id` leads the PK.

### `cubepi_schema_version`
| column | type | notes |
|---|---|---|
| `version` | `INT` | PK |

`EXPECTED_SCHEMA_VERSION = 1`, `PARTITION_COUNT = 64` (constants in `models.py`,
parity with Postgres).

## Divergences from PostgresCheckpointer

Each is a deliberate MySQL adaptation; called out per project convention.

| Aspect | Postgres | MySQL | Why |
|---|---|---|---|
| Driver | asyncpg | aiomysql | Matches cubemanus stack |
| Placeholder | `$1` | `%s` | PyMySQL paramstyle |
| `seq` lock | `pg_advisory_xact_lock(hashtext(tid))` | `SELECT … FROM cubepi_threads WHERE thread_id=%s FOR UPDATE` | MySQL has no advisory xact lock; lock the (lazily-created) thread row via InnoDB row lock, auto-released at txn end |
| Upsert | `INSERT … ON CONFLICT DO …` | `INSERT … ON DUPLICATE KEY UPDATE …` | MySQL syntax |
| `extra` merge | `extra \|\| EXCLUDED.extra` | `extra = JSON_MERGE_PATCH(extra, %s)` | JSONB op vs MySQL JSON fn; `JSON_MERGE_PATCH` is shallow-replace per top-level key, matching dict `.update()` semantics |
| Lazy thread insert | `INSERT … ON CONFLICT DO NOTHING` | `INSERT IGNORE INTO cubepi_threads …` | MySQL idiom |
| JSON type | `JSONB` | `JSON` | MySQL has no JSONB |
| metadata index | GIN (`jsonb_path_ops`) | none | MySQL can't index JSON directly; host adds generated-column index if needed |
| Partition + FK | both present | partition only, FK dropped | MySQL partitioned tables disallow foreign keys |
| `now()` | `now()` | `CURRENT_TIMESTAMP` | MySQL spelling |

The self-FK on `cubepi_threads.parent_thread_id` and the
`cubepi_messages → cubepi_threads` FK are both dropped on MySQL. Referential
integrity for messages is guaranteed by the lazy thread-row insert in `append`.

## Behavior

### `__aenter__` / `__aexit__`
Create pool (`min_pool_size`/`max_pool_size`), run `_verify_schema`, return self.
Exit closes the pool. `autocommit=False`; explicit transactions in `append`.

### `_verify_schema`
`SELECT version FROM cubepi_schema_version LIMIT 1`.
- Table missing (`aiomysql`/PyMySQL error code 1146 `ER_NO_SUCH_TABLE`) →
  `CubepiSchemaUninitialized`.
- No row → `CubepiSchemaUninitialized`.
- `version != EXPECTED_SCHEMA_VERSION` → `CubepiSchemaMismatch(expected, actual)`.

### `load(thread_id)`
Fetch messages `WHERE thread_id=%s ORDER BY seq` and the thread's `extra`.
Return `None` if no messages **and** no thread row. Decode each `payload` with
msgpack; the `metadata` column is the source of truth for `Message.metadata`
(aiomysql returns JSON columns as `str` → `json.loads`). Reconstruct via
`_ROLE_TO_CLS` (shared mapping identical to Postgres).

### `append(thread_id, messages)`
Early-return on empty list (no pool touch — testable without a DB). Otherwise, in
one transaction:
1. `INSERT IGNORE INTO cubepi_threads (thread_id) VALUES (%s)` (lazy create).
2. `SELECT thread_id FROM cubepi_threads WHERE thread_id=%s FOR UPDATE` (row
   lock serializes concurrent appends to this thread).
3. `SELECT COALESCE(MAX(seq),0) FROM cubepi_messages WHERE thread_id=%s`.
4. `executemany` insert of `(thread_id, seq, role, metadata_json, payload)` with
   `seq = last + i + 1`.
Commit. msgpack uses `model_dump(mode="json")`; metadata column gets
`json.dumps(m.metadata)`.

### `save_extra(thread_id, extra)`
```sql
INSERT INTO cubepi_threads (thread_id, extra)
VALUES (%s, %s)
ON DUPLICATE KEY UPDATE
  extra = JSON_MERGE_PATCH(extra, %s),
  updated_at = CURRENT_TIMESTAMP
```
Passing the JSON twice (insert value + merge arg). Commit.

## alembic_helpers.py

- `create_messages_table_op() -> str` (or document the partition clause) — unlike
  Postgres, MySQL `PARTITION BY HASH … PARTITIONS 64` is a single inline clause on
  the `CREATE TABLE`, so no per-partition DDL loop is needed. Provide a helper
  that returns the partition clause / full DDL the host appends.
- `write_schema_version_op() -> str` — same intent as Postgres: clear stale rows
  then insert current version, idempotent:
  `DELETE FROM cubepi_schema_version WHERE version <> 1; INSERT IGNORE INTO cubepi_schema_version (version) VALUES (1);`

Exact helper surface finalized in the plan; the principle is parity with the
Postgres helpers adjusted for MySQL's single-statement partitioning.

## Packaging

`pyproject.toml`:
```toml
mysql = [
    "sqlalchemy>=2.0",
    "aiomysql>=0.2",
    "msgpack>=1.0",
]
```
Add `aiomysql>=0.2` to the dev dependency-group. Add `MySQLCheckpointer` lazy
import to `cubepi/checkpointer/__init__.py` `__getattr__` and `__all__`.

## Testing

Mirror `tests/checkpointer/test_postgres.py` + its conftest exactly:

**Unit (no DB):** models import & registration on `cubepi_metadata`; partition
clause present (`PARTITIONS 64` / `mysql_partition_by`-equivalent assertion); no
metadata index; alembic helper SQL (schema-version op clears stale + inserts 1);
exceptions; `_role_of` mapping + rejects unknown; `append([])` is a no-op without
a pool.

**E2E (real MySQL, skipped if unavailable):** add
`tests/checkpointer/conftest.py` fixtures `mysql_dsn`
(`CUBEPI_TEST_MYSQL_DSN`, default `mysql://root:root@localhost:3306/mysql`),
`_mysql_available` probe (connect with short timeout, skip on failure),
`clean_mysql_db` (create fresh `cubepi_test_<hex>` database per test via aiomysql,
drop after). A `_setup_schema(dsn)` in the test file issues the CREATE TABLE +
partition + schema-version DDL (mirroring what host alembic generates). Tests:
round-trip with metadata, `save_extra` merges, `seq` monotonic across batches,
uninitialized schema raises, version mismatch raises, empty-thread load → None.

The existing postgres conftest fixtures are not disturbed; MySQL fixtures are
added alongside.

## Docs

Add a MySQL section to the checkpointer docs under `website/docs/` (same page /
guide that documents the Postgres checkpointer): install extra, host alembic
setup, usage example, and the divergence notes (no metadata index, host-managed
schema).

## Open questions

None blocking. Helper function names in `alembic_helpers.py` will be settled in
the implementation plan.
