"""PostgresCheckpointer — Checkpointer protocol against PostgreSQL.

Append-only message log + per-thread KV (extra). Uses asyncpg pool +
msgpack payload encoding. Schema version verified on context entry.
"""
from __future__ import annotations

import json
from typing import Any

import asyncpg
import msgpack

from cubepi.checkpointer.base import CheckpointData
from cubepi.checkpointer.postgres.exceptions import (
    CubepiSchemaMismatch,
    CubepiSchemaUninitialized,
)
from cubepi.checkpointer.postgres.models import EXPECTED_SCHEMA_VERSION
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)


def _role_of(msg: Message) -> str:
    if isinstance(msg, UserMessage):
        return "user"
    if isinstance(msg, AssistantMessage):
        return "assistant"
    if isinstance(msg, ToolResultMessage):
        return "tool"
    raise TypeError(f"unknown Message type: {type(msg).__name__}")


_ROLE_TO_CLS: dict[str, type[Message]] = {
    "user": UserMessage,
    "assistant": AssistantMessage,
    "tool": ToolResultMessage,
}


class PostgresCheckpointer:
    """Checkpointer backed by PostgreSQL.

    Usage:
        cp = PostgresCheckpointer(dsn="postgresql://...")
        async with cp:
            await cp.append(thread_id, [msg1, msg2])
            data = await cp.load(thread_id)
            await cp.save_extra(thread_id, {"k": "v"})

    Raises CubepiSchemaUninitialized / CubepiSchemaMismatch at __aenter__
    if the DB schema isn't compatible with this cubepi version.
    """

    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._min = min_pool_size
        self._max = max_pool_size
        self._pool: asyncpg.Pool | None = None

    async def __aenter__(self) -> "PostgresCheckpointer":
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min,
            max_size=self._max,
        )
        await self._verify_schema()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _verify_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    "SELECT version FROM cubepi_schema_version LIMIT 1"
                )
            except asyncpg.UndefinedTableError as e:
                raise CubepiSchemaUninitialized(
                    "cubepi tables not found. Run host application's alembic upgrade."
                ) from e
            if row is None:
                raise CubepiSchemaUninitialized(
                    "cubepi_schema_version table is empty. Host alembic migration "
                    "must INSERT the current version (use write_schema_version_op())."
                )
            if row["version"] != EXPECTED_SCHEMA_VERSION:
                raise CubepiSchemaMismatch(
                    expected=EXPECTED_SCHEMA_VERSION,
                    actual=row["version"],
                    hint="cubepi was upgraded but host alembic is behind. "
                    "Generate a new revision and apply.",
                )

    async def load(self, thread_id: str) -> CheckpointData | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            msg_rows = await conn.fetch(
                "SELECT seq, role, metadata, payload FROM cubepi_messages "
                "WHERE thread_id = $1 ORDER BY seq",
                thread_id,
            )
            extra_row = await conn.fetchrow(
                "SELECT extra FROM cubepi_threads WHERE thread_id = $1",
                thread_id,
            )

        if not msg_rows and extra_row is None:
            return None

        messages: list[Message] = []
        for r in msg_rows:
            cls = _ROLE_TO_CLS.get(r["role"])
            if cls is None:
                raise ValueError(f"unknown role in DB: {r['role']!r}")
            data = msgpack.unpackb(bytes(r["payload"]), raw=False)
            # The DB metadata column is the source of truth for Message.metadata.
            # (payload also contains it, but column is the canonical view for querying.)
            raw_meta = r["metadata"]
            data["metadata"] = (
                json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})
            )
            messages.append(cls.model_validate(data))

        if extra_row is not None:
            raw_extra = extra_row["extra"]
            extra = (
                json.loads(raw_extra)
                if isinstance(raw_extra, str)
                else (raw_extra or {})
            )
        else:
            extra = {}

        return CheckpointData(messages=messages, extra=extra)

    async def append(self, thread_id: str, messages: list[Message]) -> None:
        if not messages:
            return
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Per-thread advisory lock for monotonic seq allocation
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))", thread_id,
                )
                # Lazy thread row creation
                await conn.execute(
                    "INSERT INTO cubepi_threads (thread_id) "
                    "VALUES ($1) ON CONFLICT DO NOTHING",
                    thread_id,
                )
                last_seq = await conn.fetchval(
                    "SELECT COALESCE(MAX(seq), 0) FROM cubepi_messages "
                    "WHERE thread_id = $1",
                    thread_id,
                ) or 0

                rows = []
                for i, m in enumerate(messages):
                    seq = last_seq + i + 1
                    payload = msgpack.packb(m.model_dump(mode="json"), use_bin_type=True)
                    rows.append((
                        thread_id,
                        seq,
                        _role_of(m),
                        json.dumps(m.metadata),
                        payload,
                    ))
                await conn.executemany(
                    "INSERT INTO cubepi_messages "
                    "(thread_id, seq, role, metadata, payload) "
                    "VALUES ($1, $2, $3, $4, $5)",
                    rows,
                )

    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO cubepi_threads (thread_id, extra, updated_at) "
                "VALUES ($1, $2::jsonb, now()) "
                "ON CONFLICT (thread_id) DO UPDATE "
                "SET extra = cubepi_threads.extra || EXCLUDED.extra, "
                "    updated_at = now()",
                thread_id,
                json.dumps(extra),
            )
