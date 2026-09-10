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

`pip install -U cubepi` still works: 0.14.1+ is a wrapper that depends on a
compatible CubeLoop 0.14 release, re-exports the public API, and aliases
`cubepi.*` imports.
It warns on first import. Prefer depending on `cubeloop` directly.

CLI: `cubeloop trace` (the wrapper still provides `cubepi trace` and warns).

## Checkpointer schema stays at v5

Postgres and MySQL keep their `cubepi_*` physical table and index names. They
are persistent protocol identifiers, not product branding. Schema version 5 is
unchanged, so an existing 0.13.6 database needs no migration for 0.14.1.
SQLite table names were never prefixed; they stay `messages` / `runs` / ….

Historical host Alembic revisions may import helpers from
`cubeloop.checkpointer.*.alembic_helpers`; those helpers retain their original
v1–v5 SQL. Do not add a rename revision.

### Emergency recovery from withdrawn 0.14.0

0.14.0 was yanked because it briefly renamed these database objects to
`cubeloop_*`. Only use this recovery if 0.14.0 created the database or you ran its withdrawn
migration. Stop every application instance, take a verified backup, then run
the script for your database with an administrative SQL client:

- [Postgres v6→v5 recovery SQL](/recovery/0.14.0/postgres-v6-to-v5.sql)
- [MySQL v6→v5 recovery SQL](/recovery/0.14.0/mysql-v6-to-v5.sql)

The scripts refuse mixed/colliding schemas before renaming anything. Afterward,
verify table and row counts, then deploy 0.14.1. Do not run them on a normal v5
database.

## Tracing

New spans use `cubeloop.*` attributes and `cubeloop.turn` span names.
`cubeloop trace` still reads 0.13 JSONL that used `cubepi.*`.
Dashboards you own need their own attribute-name update.

Default JSONL directory is `./cubeloop-traces`.

## Environment variables

`CUBELOOP_TEST_PG_DSN`, `CUBELOOP_TEST_MYSQL_DSN`, `CUBELOOP_PG_DSN`,
`CUBELOOP_MYSQL_DSN` replace the `CUBEPI_*` names. The old names are still
read if the new one is unset, for the rest of 0.x.
