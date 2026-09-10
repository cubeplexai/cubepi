"""Persist an agent conversation in MySQL and resume it after a restart.

This is a runnable, end-to-end example of `MySQLCheckpointer`. It uses
`FauxProvider`, so it needs no API key — only a reachable MySQL 8.0.13+.

    CUBELOOP_MYSQL_DSN=mysql://user:pass@host:3306/dbname \
        uv run python examples/checkpointing_mysql.py

Defaults to `mysql://root:root@localhost:3306/mysql`.

What it shows:

1. Bootstrapping the cubepi v2 schema. In production this is your host
   application's Alembic migration — see
   `cubepi/checkpointer/mysql/README.md`. The DDL here mirrors exactly what
   that migration produces, using the same `alembic_helpers`.
2. Running an `Agent` whose turns are persisted by the checkpointer.
3. A simulated process restart: a brand-new checkpointer loads the thread's
   full history back.

The example creates a throwaway database and drops it on exit, so it is safe
to run repeatedly against a dev server.
"""

import asyncio
import os
import secrets

import aiomysql

from cubeloop.agent.agent import Agent
from cubeloop.checkpointer.mysql import MySQLCheckpointer
from cubeloop.checkpointer.mysql.alembic_helpers import (
    create_runs_table_op,
    messages_partition_clause,
    write_schema_version_op,
)
from cubeloop.checkpointer.mysql.checkpointer import _parse_dsn
from cubeloop.providers.faux import FauxProvider, faux_assistant_message

ADMIN_DSN = os.environ.get(
    "CUBELOOP_MYSQL_DSN",
    os.environ.get(
        "CUBEPI_MYSQL_DSN",
        "mysql://root:root@localhost:3306/mysql",
    ),
)
THREAD_ID = "user-42"


async def bootstrap_schema(dsn: str) -> None:
    """Create the CubeLoop checkpointer schema v5.

    In a real deployment this DDL belongs in the host's Alembic history.
    """
    conn = await aiomysql.connect(autocommit=True, **_parse_dsn(dsn))
    try:
        async with conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE cubepi_threads (
                    thread_id VARCHAR(255) COLLATE utf8mb4_bin PRIMARY KEY,
                    parent_thread_id VARCHAR(255) COLLATE utf8mb4_bin NULL,
                    forked_at_seq BIGINT NULL,
                    extra JSON NOT NULL DEFAULT (JSON_OBJECT()),
                    pending_request JSON NULL,
                    run_id VARCHAR(64) NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    CONSTRAINT fk_parent FOREIGN KEY (parent_thread_id)
                        REFERENCES cubepi_threads (thread_id)
                ) ENGINE=InnoDB
            """)
            await cur.execute(
                """
                CREATE TABLE cubepi_messages (
                    thread_id VARCHAR(255) COLLATE utf8mb4_bin NOT NULL,
                    seq BIGINT NOT NULL,
                    role VARCHAR(32) NOT NULL,
                    metadata JSON NOT NULL DEFAULT (JSON_OBJECT()),
                    payload LONGBLOB NOT NULL,
                    run_id VARCHAR(255) NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, seq),
                    KEY ix_cubepi_messages_thread_run (thread_id, run_id)
                ) ENGINE=InnoDB """
                + messages_partition_clause()
            )
            await cur.execute(
                "CREATE TABLE cubepi_schema_version (version INT PRIMARY KEY) "
                "ENGINE=InnoDB"
            )
            await cur.execute(create_runs_table_op())
            await cur.execute("""
                CREATE TABLE cubepi_hitl_answers (
                    thread_id VARCHAR(255) COLLATE utf8mb4_bin NOT NULL,
                    run_id VARCHAR(255) NOT NULL,
                    question_id VARCHAR(255) NOT NULL,
                    answer JSON NOT NULL,
                    answered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, run_id, question_id)
                ) ENGINE=InnoDB
            """)
            for stmt in write_schema_version_op().split(";"):
                if stmt.strip():
                    await cur.execute(stmt)
    finally:
        await conn.ensure_closed()


def build_agent(checkpointer: MySQLCheckpointer) -> Agent:
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
    db_name = f"cubeloop_example_{secrets.token_hex(5)}"
    admin_cfg = _parse_dsn(ADMIN_DSN)
    admin = await aiomysql.connect(autocommit=True, **admin_cfg)
    try:
        async with admin.cursor() as cur:
            await cur.execute(f"CREATE DATABASE `{db_name}`")
    finally:
        await admin.ensure_closed()

    dsn = f"{ADMIN_DSN.rsplit('/', 1)[0]}/{db_name}"
    try:
        await bootstrap_schema(dsn)
        print(f"Created schema in throwaway database {db_name}\n")

        # First "process": run a turn. Entering the context manager verifies
        # the schema; the agent persists each turn through the checkpointer.
        async with MySQLCheckpointer(dsn) as cp:
            agent = build_agent(cp)
            await agent.prompt("Remember that my favourite colour is teal.")
            print("First run transcript:")
            for line in transcript(agent.state.messages):
                print(f"  {line}")

        # Second "process": a fresh checkpointer loads the whole thread back.
        async with MySQLCheckpointer(dsn) as cp:
            data = await cp.load(THREAD_ID)
            assert data is not None
            print(f"\nAfter restart, loaded {len(data.messages)} messages:")
            for line in transcript(data.messages):
                print(f"  {line}")
    finally:
        admin = await aiomysql.connect(autocommit=True, **admin_cfg)
        try:
            async with admin.cursor() as cur:
                await cur.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
        finally:
            await admin.ensure_closed()
        print(f"\nDropped throwaway database {db_name}")


if __name__ == "__main__":
    asyncio.run(main())
