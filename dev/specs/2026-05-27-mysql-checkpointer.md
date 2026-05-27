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

**Minimum MySQL version: 8.0.13+** — required for expression defaults on JSON
columns (`DEFAULT (JSON_OBJECT())`). Engine: InnoDB (row locks, transactions).

`thread_id` columns use collation **`utf8mb4_bin`** so thread IDs are
case-sensitive and byte-exact, matching Postgres `TEXT`. (MySQL's default
`utf8mb4_0900_ai_ci` is case- and accent-insensitive, which would collide
distinct IDs like `UserA` / `usera`.) The 255-char limit is documented; cubepi
thread IDs are short (UUIDs / slugs) so this is not a practical constraint.

### `cubepi_threads`
| column | type | notes |
|---|---|---|
| `thread_id` | `VARCHAR(255) utf8mb4_bin` | PK |
| `parent_thread_id` | `VARCHAR(255) utf8mb4_bin NULL` | self-FK → `cubepi_threads.thread_id` (kept; this table is not partitioned) |
| `forked_at_seq` | `BIGINT NULL` | |
| `extra` | `JSON NOT NULL` | `DEFAULT (JSON_OBJECT())` |
| `created_at` | `TIMESTAMP NOT NULL` | default `CURRENT_TIMESTAMP` |
| `updated_at` | `TIMESTAMP NOT NULL` | default `CURRENT_TIMESTAMP ON UPDATE` |

### `cubepi_messages`
| column | type | notes |
|---|---|---|
| `thread_id` | `VARCHAR(255) utf8mb4_bin NOT NULL` | part of PK |
| `seq` | `BIGINT NOT NULL` | part of PK |
| `role` | `VARCHAR(32) NOT NULL` | `user`/`assistant`/`tool` |
| `metadata` | `JSON NOT NULL` | `DEFAULT (JSON_OBJECT())`, no index |
| `payload` | `LONGBLOB NOT NULL` | msgpack |
| `created_at` | `TIMESTAMP NOT NULL` | default `CURRENT_TIMESTAMP` |

PK `(thread_id, seq)`. Table is `PARTITION BY KEY (thread_id) PARTITIONS 64`.
**`KEY`, not `HASH`** — MySQL `HASH` partitioning requires an integer expression,
whereas `KEY` applies MySQL's internal hash to any column type including
`VARCHAR`. No foreign key on this table (see divergence). MySQL requires every
unique key to contain the partition column — satisfied since `thread_id` leads
the PK.

### `cubepi_schema_version`
| column | type | notes |
|---|---|---|
| `version` | `INT` | PK |

`EXPECTED_SCHEMA_VERSION = 1`, `PARTITION_COUNT = 64` (constants in `models.py`,
parity with Postgres).

**Note on partitioning in the model:** SQLAlchemy has no `mysql_partition_by`
dialect kwarg analogous to Postgres's `postgresql_partition_by`. The `KEY`
partition clause therefore cannot live on the declarative model; it is emitted
only by the `messages_partition_clause()` alembic helper, which the host appends
to the `CREATE TABLE` DDL. The model defines columns/PK/self-FK; the partition
clause is a helper concern. Unit tests assert on the helper output, not model
kwargs.

## Divergences from PostgresCheckpointer

Each is a deliberate MySQL adaptation; called out per project convention.

| Aspect | Postgres | MySQL | Why |
|---|---|---|---|
| Driver | asyncpg | aiomysql | Matches cubemanus stack |
| Placeholder | `$1` | `%s` | PyMySQL paramstyle |
| `seq` lock | `pg_advisory_xact_lock(hashtext(tid))` | `SELECT … FROM cubepi_threads WHERE thread_id=%s FOR UPDATE` | MySQL has no advisory xact lock; lock the (lazily-created) thread row via InnoDB row lock, auto-released at txn end |
| Upsert | `INSERT … ON CONFLICT DO …` | `INSERT … ON DUPLICATE KEY UPDATE …` | MySQL syntax |
| `extra` merge | `extra \|\| EXCLUDED.extra` (shallow, top-level) | read-modify-write under `FOR UPDATE` (Python `dict.update`) | `JSON_MERGE_PATCH` is **not** equivalent — it deletes keys whose value is JSON `null` and deep-merges nested objects. PG `\|\|` and sqlite `.update()` are shallow top-level replace; RMW reproduces that exactly |
| Lazy thread insert | `INSERT … ON CONFLICT DO NOTHING` | `INSERT IGNORE INTO cubepi_threads …` | MySQL idiom |
| JSON type | `JSONB` | `JSON` | MySQL has no JSONB |
| JSON readback | native dict | `str` → `json.loads` (both `metadata` and `extra`) | aiomysql returns JSON columns as text |
| JSON default | `'{}'::jsonb` | `DEFAULT (JSON_OBJECT())` | MySQL needs an expression default (8.0.13+) |
| metadata index | GIN (`jsonb_path_ops`) | none | MySQL can't index JSON directly; host adds generated-column index if needed |
| Partition | `HASH (thread_id)` | `KEY (thread_id)` | HASH needs an integer expr; KEY hashes any type |
| messages FK | `→ cubepi_threads` | dropped | MySQL partitioned tables disallow FKs |
| threads self-FK | `parent_thread_id →` | **kept** | this table is not partitioned, so MySQL allows it; preserves parity |
| `thread_id` collation | `TEXT` (case-sensitive) | `VARCHAR(255) utf8mb4_bin` | match PG case-sensitivity/byte-exactness |
| `now()` | `now()` | `CURRENT_TIMESTAMP` | MySQL spelling |

Only the `cubepi_messages → cubepi_threads` FK is dropped (because that table is
partitioned). The `cubepi_threads.parent_thread_id` self-FK is **kept**, matching
Postgres. Referential integrity for messages is guaranteed by the lazy
thread-row insert in `append`.

## Behavior

### `__aenter__` / `__aexit__`
Create pool (`min_pool_size`/`max_pool_size`), run `_verify_schema`, return self.
Exit closes the pool. Pool is created with **`autocommit=True`**; write methods
(`append`, `save_extra`) wrap their work in an explicit transaction
(`conn.begin()` / `commit` / rollback on error). This avoids the InnoDB pitfall
where, under `autocommit=False`, a read-only `load` would open a transaction that
never commits and return the connection to the pool holding a stale REPEATABLE
READ snapshot.

### `_verify_schema`
`SELECT version FROM cubepi_schema_version LIMIT 1`.
- Table missing (PyMySQL error code `1146 ER_NO_SUCH_TABLE`) →
  `CubepiSchemaUninitialized`.
- `version` column missing (`1054 ER_BAD_FIELD_ERROR`, a malformed/stale table) →
  `CubepiSchemaUninitialized` (MySQL-specific defensive handling; Postgres has no
  analogous catch).
- No row → `CubepiSchemaUninitialized`.
- `version != EXPECTED_SCHEMA_VERSION` → `CubepiSchemaMismatch(expected, actual)`.

Error codes are read from `pymysql.err.*` / `e.args[0]` rather than matched on
message text.

### `load(thread_id)`
Fetch messages `WHERE thread_id=%s ORDER BY seq` and the thread's `extra`.
Return `None` if no messages **and** no thread row. Decode each `payload` with
msgpack; the `metadata` column is the source of truth for `Message.metadata`.
**Both** JSON columns (`metadata` and `extra`) come back from aiomysql as `str`
and are decoded with `json.loads` (guard against an already-parsed value for
driver-version robustness). Reconstruct via `_ROLE_TO_CLS` (shared mapping
identical to Postgres).

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
Shallow top-level merge matching Postgres `||` / sqlite `dict.update`. Because
`JSON_MERGE_PATCH` has different semantics (null-key deletion, deep merge), do a
read-modify-write inside one transaction:
1. `INSERT IGNORE INTO cubepi_threads (thread_id) VALUES (%s)` (lazy create).
2. `SELECT extra FROM cubepi_threads WHERE thread_id=%s FOR UPDATE` (row lock).
3. In Python: `merged = {**current, **extra}`.
4. `UPDATE cubepi_threads SET extra=%s, updated_at=CURRENT_TIMESTAMP WHERE thread_id=%s`
   with `json.dumps(merged)`.
Commit. The `FOR UPDATE` serializes concurrent `save_extra` on the same thread.

## alembic_helpers.py

- `messages_partition_clause() -> str` — unlike Postgres (which needs a 64-row
  per-partition DDL loop), MySQL `PARTITION BY KEY (thread_id) PARTITIONS 64` is a
  single inline clause on the `CREATE TABLE`. The helper returns that clause for
  the host to append to its messages-table DDL.
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
clause present (`PARTITION BY KEY (thread_id) PARTITIONS 64`); no metadata index;
self-FK on `parent_thread_id` present while no FK on `cubepi_messages`; alembic
helper SQL (schema-version op clears stale + inserts 1); exceptions; `_role_of`
mapping + rejects unknown; `append([])` is a no-op without a pool.

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
