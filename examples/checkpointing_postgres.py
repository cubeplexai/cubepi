"""Persist an agent conversation in Postgres and resume it after a restart.

This is a runnable, end-to-end example of `PostgresCheckpointer`. It uses
`FauxProvider`, so it needs no API key — only a reachable Postgres.

    CUBEPI_PG_DSN=postgresql://user:pass@host:5432/dbname \
        uv run python examples/checkpointing_postgres.py

Defaults to `postgresql://postgres:postgres@localhost:5432/postgres`.

What it shows:

1. Bootstrapping the cubepi v2 schema. In production this is your host
   application's Alembic migration — see
   `cubepi/checkpointer/postgres/README.md`. The DDL here mirrors exactly
   what that migration produces, using the same `alembic_helpers`.
2. Running an `Agent` whose turns are persisted by the checkpointer.
3. A simulated process restart: a brand-new checkpointer loads the thread's
   full history back.

The example creates a throwaway database and drops it on exit, so it is safe
to run repeatedly against a dev server.
"""

import asyncio
import os
import secrets

import asyncpg

from cubepi.agent.agent import Agent
from cubepi.checkpointer.postgres import PostgresCheckpointer
from cubepi.checkpointer.postgres.alembic_helpers import (
    add_pending_request_column_op,
    add_run_id_column_op,
    create_message_partitions_op,
    upgrade_v3_to_v4_op,
    write_schema_version_op,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message

ADMIN_DSN = os.environ.get(
    "CUBEPI_PG_DSN",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)
THREAD_ID = "user-42"


async def bootstrap_schema(dsn: str) -> None:
    """Create the cubepi v2 schema.

    In a real deployment this is your Alembic migration. The columns come
    straight from `cubepi_metadata`; only the partitioning and the
    schema-version row are added by hand (autogenerate can't model them).
    See cubepi/checkpointer/postgres/README.md for the migration recipe.
    """
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("""
            CREATE TABLE cubepi_threads (
                thread_id TEXT PRIMARY KEY,
                parent_thread_id TEXT REFERENCES cubepi_threads(thread_id),
                forked_at_seq BIGINT,
                extra JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        await conn.execute(add_pending_request_column_op())  # v1 -> v2 column
        await conn.execute(add_run_id_column_op())  # v2 -> v3 column
        await conn.execute("""
            CREATE TABLE cubepi_messages (
                thread_id TEXT NOT NULL
                    REFERENCES cubepi_threads(thread_id) ON DELETE CASCADE,
                seq BIGINT NOT NULL,
                role TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                payload BYTEA NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (thread_id, seq)
            ) PARTITION BY HASH (thread_id);
        """)
        await conn.execute(create_message_partitions_op())  # 64 child partitions
        await conn.execute("""
            CREATE INDEX ix_cubepi_messages_metadata_gin
            ON cubepi_messages USING GIN (metadata jsonb_path_ops);
        """)
        await conn.execute(
            "CREATE TABLE cubepi_schema_version (version INTEGER PRIMARY KEY);"
        )
        # v3 -> v4: run_id on cubepi_messages + cubepi_runs partitioned table.
        await conn.execute(upgrade_v3_to_v4_op())
        await conn.execute(write_schema_version_op())  # record EXPECTED_SCHEMA_VERSION
    finally:
        await conn.close()


def build_agent(checkpointer: PostgresCheckpointer) -> Agent:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("Hi! I'll remember this.")])
    return Agent(
        model=provider.model("faux"),
        checkpointer=checkpointer,
        thread_id=THREAD_ID,
    )


def transcript(messages) -> list[str]:
    return [
        f"{type(m).__name__}: {getattr(c, 'text', '')}"
        for m in messages
        for c in m.content
        if getattr(c, "text", "")
    ]


async def main() -> None:
    # Throwaway DB so the example is safe to re-run.
    db_name = f"cubepi_example_{secrets.token_hex(5)}"
    admin = await asyncpg.connect(ADMIN_DSN)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    dsn = f"{ADMIN_DSN.rsplit('/', 1)[0]}/{db_name}"
    try:
        await bootstrap_schema(dsn)
        print(f"Created schema in throwaway database {db_name}\n")

        # First "process": run a turn. Entering the context manager verifies
        # the schema; the agent persists each turn through the checkpointer.
        async with PostgresCheckpointer(dsn) as cp:
            agent = build_agent(cp)
            await agent.prompt("Remember that my favourite colour is teal.")
            print("First run transcript:")
            for line in transcript(agent.state.messages):
                print(f"  {line}")

        # Second "process": a fresh checkpointer loads the whole thread back.
        async with PostgresCheckpointer(dsn) as cp:
            data = await cp.load(THREAD_ID)
            assert data is not None
            print(f"\nAfter restart, loaded {len(data.messages)} messages:")
            for line in transcript(data.messages):
                print(f"  {line}")
    finally:
        admin = await asyncpg.connect(ADMIN_DSN)
        try:
            await admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = $1",
                db_name,
            )
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
        finally:
            await admin.close()
        print(f"\nDropped throwaway database {db_name}")


if __name__ == "__main__":
    asyncio.run(main())
