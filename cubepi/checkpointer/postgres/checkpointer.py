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
from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
)
from cubepi.checkpointer.postgres.exceptions import (
    CubepiSchemaMismatch,
    CubepiSchemaUninitialized,
)
from cubepi.checkpointer.postgres.models import EXPECTED_SCHEMA_VERSION
from cubepi.hitl.types import HitlRequest
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)
from cubepi.types import JsonObject


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
                    hint=(
                        "cubepi was upgraded but host alembic is behind. "
                        "Generate a new alembic revision that calls "
                        "add_run_id_column_op() + write_schema_version_op() "
                        "(see cubepi.checkpointer.postgres.alembic_helpers) "
                        "and run `alembic upgrade head` against this database."
                    ),
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
        run_ids = {
            rid for m in messages if (rid := getattr(m, "run_id", None)) is not None
        }
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Per-thread advisory lock for monotonic seq allocation
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    thread_id,
                )
                # Lazy thread row creation
                await conn.execute(
                    "INSERT INTO cubepi_threads (thread_id) "
                    "VALUES ($1) ON CONFLICT DO NOTHING",
                    thread_id,
                )
                # Pre-flight: reject append on any completed run_id.
                if run_ids:
                    done_rows = await conn.fetch(
                        "SELECT run_id FROM cubepi_runs "
                        "WHERE thread_id = $1 AND run_id = ANY($2::text[]) "
                        "AND completed_at IS NOT NULL",
                        thread_id,
                        list(run_ids),
                    )
                    if done_rows:
                        bad = ", ".join(r["run_id"] for r in done_rows)
                        raise RunAlreadyCompletedError(
                            f"append on completed run thread={thread_id} runs={bad}"
                        )
                last_seq = (
                    await conn.fetchval(
                        "SELECT COALESCE(MAX(seq), 0) FROM cubepi_messages "
                        "WHERE thread_id = $1",
                        thread_id,
                    )
                    or 0
                )

                rows = []
                for i, m in enumerate(messages):
                    seq = last_seq + i + 1
                    payload = msgpack.packb(
                        m.model_dump(mode="json"), use_bin_type=True
                    )
                    rows.append(
                        (
                            thread_id,
                            seq,
                            _role_of(m),
                            json.dumps(m.metadata),
                            payload,
                            getattr(m, "run_id", None),
                        )
                    )
                await conn.executemany(
                    "INSERT INTO cubepi_messages "
                    "(thread_id, seq, role, metadata, payload, run_id) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    rows,
                )

    async def claim_run(self, thread_id: str, run_id: str) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    thread_id,
                )
                # Lazy thread row creation (claim may precede any append).
                await conn.execute(
                    "INSERT INTO cubepi_threads (thread_id) "
                    "VALUES ($1) ON CONFLICT (thread_id) DO NOTHING",
                    thread_id,
                )
                # Pre-check under the advisory lock so a duplicate doesn't
                # raise UniqueViolation (which would abort the txn before
                # we could distinguish in-flight vs completed).
                row = await conn.fetchrow(
                    "SELECT completed_at FROM cubepi_runs "
                    "WHERE thread_id = $1 AND run_id = $2",
                    thread_id,
                    run_id,
                )
                if row is not None:
                    if row["completed_at"] is not None:
                        raise RunAlreadyCompletedError(
                            f"thread={thread_id} run={run_id} already completed"
                        )
                    raise RunAlreadyClaimedError(
                        f"thread={thread_id} run={run_id} in flight"
                    )
                await conn.execute(
                    "INSERT INTO cubepi_runs (thread_id, run_id) "
                    "VALUES ($1, $2)",
                    thread_id,
                    run_id,
                )

    async def mark_run_complete(self, thread_id: str, run_id: str) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    thread_id,
                )
                row = await conn.fetchrow(
                    "SELECT completed_at FROM cubepi_runs "
                    "WHERE thread_id = $1 AND run_id = $2",
                    thread_id,
                    run_id,
                )
                if row is None:
                    raise RunNotClaimedError(
                        f"thread={thread_id} run={run_id} has no claim row"
                    )
                if row["completed_at"] is not None:
                    return  # idempotent success
                next_seq = await conn.fetchval(
                    "SELECT COALESCE(MAX(completion_seq), 0) + 1 "
                    "FROM cubepi_runs WHERE thread_id = $1 "
                    "AND completion_seq IS NOT NULL",
                    thread_id,
                )
                await conn.execute(
                    "UPDATE cubepi_runs SET completed_at = now(), "
                    "completion_seq = $3 "
                    "WHERE thread_id = $1 AND run_id = $2",
                    thread_id,
                    run_id,
                    next_seq,
                )

    async def load_pending(
        self, thread_id: str
    ) -> tuple[HitlRequest, str | None] | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT pending_request, run_id FROM cubepi_threads "
                "WHERE thread_id = $1",
                thread_id,
            )
        if row is None or row["pending_request"] is None:
            return None
        raw = row["pending_request"]
        if isinstance(raw, str):  # pragma: no cover — codec-dependent
            req = HitlRequest.model_validate_json(raw)
        else:
            req = HitlRequest.model_validate(raw)
        return req, row["run_id"]

    async def save_extra(self, thread_id: str, extra: JsonObject) -> None:
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

    async def save_pending_request(
        self,
        thread_id: str,
        request: HitlRequest | None,
        *,
        run_id: str | None = None,
    ) -> None:
        """Persist a pending HITL request and its owning run_id.

        When ``request is None``, both ``pending_request`` and ``run_id``
        are cleared; the ``run_id`` kwarg is ignored in that case (the
        pending row's run_id is always cleared alongside the pending).

        pending and run_id are set in ONE UPDATE; the surrounding
        transaction also covers the lazy thread INSERT.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Ensure thread row exists (lazy creation matches save_extra path).
                await conn.execute(
                    "INSERT INTO cubepi_threads (thread_id) "
                    "VALUES ($1) ON CONFLICT DO NOTHING",
                    thread_id,
                )
                if request is None:
                    await conn.execute(
                        "UPDATE cubepi_threads "
                        "SET pending_request = NULL, run_id = NULL, "
                        "updated_at = now() WHERE thread_id = $1",
                        thread_id,
                    )
                else:
                    payload = request.model_dump_json()
                    await conn.execute(
                        "UPDATE cubepi_threads "
                        "SET pending_request = $2::jsonb, run_id = $3, "
                        "updated_at = now() WHERE thread_id = $1",
                        thread_id,
                        payload,
                        run_id,
                    )

    async def load_pending_request(self, thread_id: str) -> HitlRequest | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT pending_request FROM cubepi_threads WHERE thread_id = $1",
                thread_id,
            )
        if row is None or row["pending_request"] is None:
            return None
        raw = row["pending_request"]
        # asyncpg returns JSONB as already-parsed dict OR str depending on codec config.
        if isinstance(raw, str):  # pragma: no cover — codec-dependent
            return HitlRequest.model_validate_json(raw)
        return HitlRequest.model_validate(raw)

    async def load_pending_run_id(self, thread_id: str) -> str | None:
        """Return the run_id of the currently pending HITL request.

        Filters on ``pending_request IS NOT NULL`` so the result reflects
        a real pending, not a leftover run_id from a cleared row. Returns
        None when: the thread is unknown, has no pending request, or was
        written by a pre-v3 host (legacy rows have run_id NULL).
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT run_id FROM cubepi_threads "
                "WHERE thread_id = $1 AND pending_request IS NOT NULL",
                thread_id,
            )
        return row["run_id"] if row is not None else None
