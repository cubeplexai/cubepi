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

## Tracing

New spans use `cubeloop.*` attributes and `cubeloop.turn` span names.
`cubeloop trace` still reads 0.13 JSONL that used `cubepi.*`.
Dashboards you own need their own attribute-name update.

Default JSONL directory is `./cubeloop-traces`.

## Environment variables

`CUBELOOP_TEST_PG_DSN`, `CUBELOOP_TEST_MYSQL_DSN`, `CUBELOOP_PG_DSN`,
`CUBELOOP_MYSQL_DSN` replace the `CUBEPI_*` names. The old names are still
read if the new one is unset, for the rest of 0.x.
