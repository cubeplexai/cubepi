---
title: Postgres Checkpointing
description: "Use PostgresCheckpointer for production-grade agent state persistence with asyncpg and advisory locks."
---

# Postgres Checkpointing

`PostgresCheckpointer` is the production-grade persistence backend.
It uses `asyncpg` with a connection pool, `msgpack` for payloads, and
a per-thread Postgres advisory lock so multiple processes can write
the same `thread_id` without trampling each other.

Install the extra:

```bash
pip install "cubepi[postgres]"
```

This pulls in `asyncpg`, `sqlalchemy`, and `msgpack`.

## Basic usage

```python
import asyncio
from cubepi import Agent, Model
from cubepi.checkpointer import PostgresCheckpointer
from cubepi.providers.anthropic import AnthropicProvider


async def main():
    provider = AnthropicProvider(provider_id="anthropic", api_key="…")
    async with PostgresCheckpointer("postgresql://user:pass@host/dbname") as cp:
        agent = Agent(
            model=provider.model("claude-sonnet-4-5-20250929"),
            checkpointer=cp,
            thread_id="user-42",
        )
        await agent.prompt("hello")


asyncio.run(main())
```

The DSN is whatever `asyncpg.create_pool(...)` accepts. Pool sizing:

```python
async with PostgresCheckpointer(
    "postgresql://…",
    min_pool_size=2,
    max_pool_size=20,
) as cp:
    …
```

## Schema

The checkpointer expects three tables: `cubepi_threads`,
`cubepi_messages`, and `cubepi_schema_version`. Unlike SQLite, CubePi
**does not create these for you** — it verifies on `__aenter__` that
they exist with the expected `schema_version`.

If they're missing, you get `CubepiSchemaUninitialized`. If the
version doesn't match this CubePi release, you get
`CubepiSchemaMismatch`.

The reason: a production database belongs to the host application's
migration system (Alembic, Atlas, …), not to a third-party library
that might fight your existing migrations.

### Bootstrapping via Alembic

CubePi exposes the SQLAlchemy `MetaData` so your migrations can adopt
the schema:

```python
# alembic/env.py
from cubepi.checkpointer.postgres import cubepi_metadata, EXPECTED_SCHEMA_VERSION

target_metadata = [my_app_metadata, cubepi_metadata]
```

Then autogenerate a revision:

```bash
alembic revision --autogenerate -m "add cubepi checkpointer"
```

Autogenerate emits the `CREATE TABLE`s from `cubepi_metadata`, but
SQLAlchemy `MetaData` **cannot model two things CubePi needs**, so add
them to the generated migration by hand: the 64 hash partitions of
`cubepi_messages` (via `create_message_partitions_op()`) and the
`cubepi_schema_version` row (via `write_schema_version_op()`). Use the
helpers:

```python
# In a migration's upgrade():
from cubepi.checkpointer.postgres.alembic_helpers import (
    create_message_partitions_op,
    write_schema_version_op,
)

def upgrade():
    op.create_table(...)                            # auto-generated from cubepi_metadata
    op.execute(create_message_partitions_op())      # creates the 64 hash partitions
    op.execute(write_schema_version_op())           # records EXPECTED_SCHEMA_VERSION
```

Both helpers return a SQL string — you pass them to `op.execute(...)`.
`write_schema_version_op()` is idempotent: it deletes any rows from a
prior CubePi version and inserts the current one.

When CubePi later upgrades and bumps `EXPECTED_SCHEMA_VERSION`, you
generate a new revision and call `op.execute(write_schema_version_op())`
again.

## Data model

```
cubepi_threads
    thread_id (PK)
    parent_thread_id   -- for forks
    forked_at_seq      -- seq number at fork point
    extra              -- JSONB
    created_at / updated_at

cubepi_messages
    thread_id, seq     -- composite PK; partitioned by HASH(thread_id) into 64
    role               -- "user" | "assistant" | "tool"
    metadata           -- JSONB (indexed via GIN)
    payload            -- bytea (msgpack)
    created_at

cubepi_schema_version
    version (PK)
```

Important properties:

- **`(thread_id, seq)` is the message identity.** `seq` is monotonic
  per thread, allocated under a `pg_advisory_xact_lock(hashtext(thread_id))`.
  Two concurrent writers on the same thread serialize cleanly.
- **`payload` is msgpack-encoded `model.model_dump(mode="json")`.**
  CubePi reconstructs the Pydantic model on read.
- **`metadata` is JSONB, queryable.** The full message also has
  `metadata` inside the payload, but the column is the canonical view
  for SQL queries.
- **Tables are partitioned by `HASH(thread_id)` into 64 partitions.**
  Even distribution across partitions, no per-thread bottleneck.

## Concurrency

The advisory lock makes append-on-the-same-thread safe across
processes:

```python
# Process A and Process B both append to thread "user-42" at the same time.
# They serialize through pg_advisory_xact_lock and each gets a consecutive seq.
```

Reads (`load`) take no lock — they're snapshot-consistent within the
transaction.

The pool default of `min=1, max=10` is fine for most apps; bump
`max_pool_size` if you have many concurrent agents.

## `save_extra` semantics

`save_extra` does a JSONB merge, not a replace:

```sql
extra = cubepi_threads.extra || EXCLUDED.extra
```

So writing `{"foo": 1}` then `{"bar": 2}` leaves `{"foo": 1, "bar":
2}`. Middleware can safely write partial dicts without losing prior
keys.

## Forks

The `parent_thread_id` + `forked_at_seq` columns exist for future fork
support. CubePi v0.3 doesn't expose a fork API yet — they're written
to so the schema is forward-compatible.

## Common pitfalls

- **`CubepiSchemaUninitialized`** — Your DB is empty or your
  migrations didn't run. Apply the host alembic upgrade first.
- **`CubepiSchemaMismatch`** — You upgraded CubePi but didn't generate
  a new migration. Generate one, apply it, and CubePi will start.
  
  :::info Schema v2 (HITL)

  CubePi ≥ the HITL feature bumps `EXPECTED_SCHEMA_VERSION` from 1 to 2
  and adds a `pending_request JSONB NULL` column to `cubepi_threads`.
  Your host alembic upgrade must call
  `add_pending_request_column_op()` (from
  `cubepi.checkpointer.postgres.alembic_helpers`) before bumping the
  schema_version row. See the [HITL guide](../hitl/overview) for the full
  cross-process flow.
  :::
- **Connection pool exhaustion under load** — Default `max_pool_size=10`.
  Bump it if your app's concurrent agent count is higher than that.
- **`asyncpg.exceptions.UndefinedTableError` outside `__aenter__`** —
  Means you're using the checkpointer outside of `async with`. The
  pool isn't connected yet. Wrap in the context manager.
- **Mixing host SQLAlchemy `MetaData`** — CubePi ships its own
  `MetaData` instance precisely so it can coexist with your app's
  models without colliding. Don't merge them into your global metadata
  — pass both to Alembic separately.

## See also

- [SQLite Checkpointing](./sqlite) — single-process alternative.
- [Custom Backends](./custom) — Protocol details.
- [Recipes → Postgres + FastAPI Service](../../recipes/postgres-fastapi)
  — a deployable HTTP-fronted agent.
- [Package README](https://github.com/cubeplexai/cubepi/blob/main/cubepi/checkpointer/postgres/README.md)
  — the full host-integration runbook next to the code.
