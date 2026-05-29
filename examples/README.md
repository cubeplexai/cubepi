# Examples

Runnable scripts demonstrating CubePi features. They use `FauxProvider`, so
they need no API key — only the relevant service where noted.

Run any example with `uv`:

```bash
uv run python examples/<name>.py
```

## Checkpointing

| Example | What it shows | Requires |
|---|---|---|
| [`checkpointing_postgres.py`](checkpointing_postgres.py) | Persist an agent conversation in Postgres and resume it after a simulated restart | A reachable Postgres |
| [`checkpointing_mysql.py`](checkpointing_mysql.py) | Same, on MySQL 8.0.13+ | A reachable MySQL |

Both create a **throwaway database** and drop it on exit, so they are safe to
re-run against a dev server. Point them at your server with an env var:

```bash
CUBEPI_PG_DSN=postgresql://user:pass@host:5432/dbname \
    uv run python examples/checkpointing_postgres.py

CUBEPI_MYSQL_DSN=mysql://user:pass@host:3306/dbname \
    uv run python examples/checkpointing_mysql.py
```

Each script bootstraps the CubePi schema inline so it runs standalone, but in
production the schema is owned by your host application's Alembic migration.
The inline DDL mirrors exactly what that migration produces and uses the same
`alembic_helpers`. See the host-integration runbooks for the migration recipe
and version-upgrade flow:

- [`cubepi/checkpointer/postgres/README.md`](../cubepi/checkpointer/postgres/README.md)
- [`cubepi/checkpointer/mysql/README.md`](../cubepi/checkpointer/mysql/README.md)
- User guides: [Postgres](../website/docs/guides/checkpointing/postgres.md) ·
  [MySQL](../website/docs/guides/checkpointing/mysql.md)
