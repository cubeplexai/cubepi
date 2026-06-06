"""MySQLCheckpointer — Checkpointer protocol against MySQL.

Append-only message log + per-thread KV (extra). aiomysql pool + msgpack
payloads. Schema version verified on context entry. Mirrors
PostgresCheckpointer; see dev/specs/2026-05-27-mysql-checkpointer.md for the
list of deliberate MySQL divergences.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote, urlparse

import aiomysql
import msgpack
import pymysql

from cubepi.checkpointer.base import CheckpointData
from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
)
from cubepi.checkpointer.mysql.exceptions import (
    CubepiSchemaMismatch,
    CubepiSchemaUninitialized,
)
from cubepi.checkpointer.mysql.models import EXPECTED_SCHEMA_VERSION
from cubepi.hitl.types import HitlRequest
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)
from cubepi.types import JsonObject

_ER_NO_SUCH_TABLE = 1146
_ER_BAD_FIELD_ERROR = 1054


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


def _parse_dsn(dsn: str) -> dict[str, Any]:
    """Parse a mysql:// URL into aiomysql.create_pool kwargs."""
    u = urlparse(dsn)
    db = u.path.lstrip("/")
    return {
        "host": u.hostname or "localhost",
        "port": u.port or 3306,
        "user": unquote(u.username) if u.username else "",
        "password": unquote(u.password) if u.password else "",
        "db": db,
    }


def _decode_json(value: Any) -> dict[str, Any]:
    """aiomysql returns JSON columns as str; tolerate already-parsed dicts."""
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    return value


class MySQLCheckpointer:
    """Checkpointer backed by MySQL (8.0.13+, InnoDB).

    Usage:
        cp = MySQLCheckpointer("mysql://user:pw@host:3306/db")
        async with cp:
            await cp.append(thread_id, [msg1, msg2])
            data = await cp.load(thread_id)
            await cp.save_extra(thread_id, {"k": "v"})

    Raises CubepiSchemaUninitialized / CubepiSchemaMismatch at __aenter__ if the
    DB schema isn't compatible with this cubepi version.
    """

    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        self._cfg = _parse_dsn(dsn)
        self._min = min_pool_size
        self._max = max_pool_size
        self._pool: aiomysql.Pool | None = None

    async def __aenter__(self) -> "MySQLCheckpointer":
        self._pool = await aiomysql.create_pool(
            minsize=self._min,
            maxsize=self._max,
            autocommit=True,
            **self._cfg,
        )
        # If verification fails, __aexit__ won't run (the context was never
        # entered), so close the pool here to avoid leaking connections.
        try:
            await self._verify_schema()
        except BaseException:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
            raise
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    async def _verify_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                try:
                    await cur.execute(
                        "SELECT version FROM cubepi_schema_version LIMIT 1"
                    )
                    row = await cur.fetchone()
                except pymysql.err.Error as e:
                    # PyMySQL maps 1146 (ER_NO_SUCH_TABLE) to ProgrammingError but
                    # 1054 (ER_BAD_FIELD_ERROR) to OperationalError, so catch the
                    # common base and branch on the errno.
                    code = e.args[0] if e.args else None
                    if code in (_ER_NO_SUCH_TABLE, _ER_BAD_FIELD_ERROR):
                        raise CubepiSchemaUninitialized(
                            "cubepi tables not found or malformed. Run host "
                            "application's alembic upgrade."
                        ) from e
                    raise  # pragma: no cover - non-schema DB errors propagate
        if row is None:
            raise CubepiSchemaUninitialized(
                "cubepi_schema_version table is empty. Host alembic migration "
                "must INSERT the current version (use write_schema_version_op())."
            )
        actual = row[0]
        if actual != EXPECTED_SCHEMA_VERSION:
            raise CubepiSchemaMismatch(
                expected=EXPECTED_SCHEMA_VERSION,
                actual=actual,
                hint=(
                    "cubepi was upgraded but host alembic is behind. "
                    "Generate a new alembic revision that calls "
                    "add_run_id_column_op() + write_schema_version_op() "
                    "(see cubepi.checkpointer.mysql.alembic_helpers) "
                    "and run `alembic upgrade head` against this database."
                ),
            )

    async def load(self, thread_id: str) -> CheckpointData | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT seq, role, metadata, payload FROM cubepi_messages "
                    "WHERE thread_id = %s ORDER BY seq",
                    (thread_id,),
                )
                msg_rows = await cur.fetchall()
                await cur.execute(
                    "SELECT extra FROM cubepi_threads WHERE thread_id = %s",
                    (thread_id,),
                )
                extra_row = await cur.fetchone()

        if not msg_rows and extra_row is None:
            return None

        messages: list[Message] = []
        for _seq, role, metadata, payload in msg_rows:
            cls = _ROLE_TO_CLS.get(role)
            if cls is None:
                raise ValueError(f"unknown role in DB: {role!r}")
            data = msgpack.unpackb(bytes(payload), raw=False)
            data["metadata"] = _decode_json(metadata)
            messages.append(cls.model_validate(data))

        extra = _decode_json(extra_row[0]) if extra_row is not None else {}
        return CheckpointData(messages=messages, extra=extra)

    async def append(self, thread_id: str, messages: list[Message]) -> None:
        if not messages:
            return
        assert self._pool is not None
        run_ids = {
            rid for m in messages if (rid := getattr(m, "run_id", None)) is not None
        }
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    # No-op upsert (idiomatic equivalent of Postgres
                    # ON CONFLICT DO NOTHING); avoids INSERT IGNORE, which would
                    # also swallow unrelated errors and emit a duplicate-key
                    # warning when the row already exists.
                    await cur.execute(
                        "INSERT INTO cubepi_threads (thread_id) VALUES (%s) "
                        "ON DUPLICATE KEY UPDATE thread_id = thread_id",
                        (thread_id,),
                    )
                    await cur.execute(
                        "SELECT thread_id FROM cubepi_threads "
                        "WHERE thread_id = %s FOR UPDATE",
                        (thread_id,),
                    )
                    # Pre-flight: reject append on any completed run_id.
                    if run_ids:
                        placeholders = ", ".join(["%s"] * len(run_ids))
                        await cur.execute(
                            f"SELECT run_id FROM cubepi_runs "
                            f"WHERE thread_id = %s "
                            f"AND run_id IN ({placeholders}) "
                            f"AND completed_at IS NOT NULL",
                            (thread_id, *run_ids),
                        )
                        done_rows = await cur.fetchall()
                        if done_rows:
                            bad = ", ".join(r[0] for r in done_rows)
                            raise RunAlreadyCompletedError(
                                f"append on completed run thread={thread_id} runs={bad}"
                            )
                    await cur.execute(
                        "SELECT COALESCE(MAX(seq), 0) FROM cubepi_messages "
                        "WHERE thread_id = %s",
                        (thread_id,),
                    )
                    (last_seq,) = await cur.fetchone()
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
                    await cur.executemany(
                        "INSERT INTO cubepi_messages "
                        "(thread_id, seq, role, metadata, payload, run_id) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        rows,
                    )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def claim_run(self, thread_id: str, run_id: str) -> None:
        """Atomically claim a run_id on a thread.

        Lazy-creates the cubepi_threads row, takes a per-thread FOR UPDATE
        lock to serialize concurrent claims for the same thread, then
        checks cubepi_runs for an existing row before inserting. The
        pre-check matches the Postgres approach — using INSERT + catching
        IntegrityError would also work, but a pre-SELECT lets us
        distinguish in-flight vs completed cleanly without relying on the
        error-class taxonomy.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    # Lazy thread row creation (claim may precede any append).
                    await cur.execute(
                        "INSERT INTO cubepi_threads (thread_id) VALUES (%s) "
                        "ON DUPLICATE KEY UPDATE thread_id = thread_id",
                        (thread_id,),
                    )
                    # Per-thread fence: serializes claim_run/append/fork on
                    # the same thread (matches Postgres's pg_advisory_xact_lock).
                    await cur.execute(
                        "SELECT thread_id FROM cubepi_threads "
                        "WHERE thread_id = %s FOR UPDATE",
                        (thread_id,),
                    )
                    await cur.execute(
                        "SELECT completed_at FROM cubepi_runs "
                        "WHERE thread_id = %s AND run_id = %s",
                        (thread_id, run_id),
                    )
                    row = await cur.fetchone()
                    if row is not None:
                        if row[0] is not None:
                            raise RunAlreadyCompletedError(
                                f"thread={thread_id} run={run_id} already completed"
                            )
                        raise RunAlreadyClaimedError(
                            f"thread={thread_id} run={run_id} in flight"
                        )
                    await cur.execute(
                        "INSERT INTO cubepi_runs (thread_id, run_id) VALUES (%s, %s)",
                        (thread_id, run_id),
                    )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def mark_run_complete(self, thread_id: str, run_id: str) -> None:
        """Mark (thread_id, run_id) complete with a monotonic completion_seq.

        Idempotent: a second call on an already-completed row is a no-op.
        Raises ``RunNotClaimedError`` if no claim row exists. The
        completion_seq is the per-thread MAX(completion_seq) + 1, allocated
        under the same per-thread FOR UPDATE fence as claim_run/append.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT thread_id FROM cubepi_threads "
                        "WHERE thread_id = %s FOR UPDATE",
                        (thread_id,),
                    )
                    await cur.execute(
                        "SELECT completed_at FROM cubepi_runs "
                        "WHERE thread_id = %s AND run_id = %s",
                        (thread_id, run_id),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise RunNotClaimedError(
                            f"thread={thread_id} run={run_id} has no claim row"
                        )
                    if row[0] is not None:
                        await conn.commit()
                        return  # idempotent success
                    await cur.execute(
                        "SELECT COALESCE(MAX(completion_seq), 0) + 1 "
                        "FROM cubepi_runs WHERE thread_id = %s "
                        "AND completion_seq IS NOT NULL",
                        (thread_id,),
                    )
                    (next_seq,) = await cur.fetchone()
                    await cur.execute(
                        "UPDATE cubepi_runs SET completed_at = CURRENT_TIMESTAMP, "
                        "completion_seq = %s "
                        "WHERE thread_id = %s AND run_id = %s",
                        (next_seq, thread_id, run_id),
                    )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def load_pending(
        self, thread_id: str
    ) -> tuple[HitlRequest, str | None] | None:
        """Return the thread's pending HITL request + the owning run_id.

        Single SELECT covers both columns (vs. ``load_pending_request`` +
        ``load_pending_run_id``, which incur two round trips and can race
        across a clear/set window). Returns None when no pending exists.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT pending_request, run_id FROM cubepi_threads "
                    "WHERE thread_id = %s",
                    (thread_id,),
                )
                row = await cur.fetchone()
        if row is None or row[0] is None:
            return None
        raw = row[0]
        if isinstance(raw, str):
            req = HitlRequest.model_validate_json(raw)
        else:
            req = HitlRequest.model_validate(raw)
        return req, row[1]

    async def save_extra(self, thread_id: str, extra: JsonObject) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    # No-op upsert (idiomatic equivalent of Postgres
                    # ON CONFLICT DO NOTHING); avoids INSERT IGNORE, which would
                    # also swallow unrelated errors and emit a duplicate-key
                    # warning when the row already exists.
                    await cur.execute(
                        "INSERT INTO cubepi_threads (thread_id) VALUES (%s) "
                        "ON DUPLICATE KEY UPDATE thread_id = thread_id",
                        (thread_id,),
                    )
                    await cur.execute(
                        "SELECT extra FROM cubepi_threads "
                        "WHERE thread_id = %s FOR UPDATE",
                        (thread_id,),
                    )
                    row = await cur.fetchone()
                    current = _decode_json(row[0]) if row is not None else {}
                    merged = {**current, **extra}
                    await cur.execute(
                        "UPDATE cubepi_threads "
                        "SET extra = %s, updated_at = CURRENT_TIMESTAMP "
                        "WHERE thread_id = %s",
                        (json.dumps(merged), thread_id),
                    )
                await conn.commit()
            except BaseException:  # pragma: no cover - defensive txn rollback
                await conn.rollback()
                raise

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
        explicit transaction also covers the lazy thread INSERT.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "INSERT INTO cubepi_threads (thread_id) VALUES (%s) "
                        "ON DUPLICATE KEY UPDATE thread_id = thread_id",
                        (thread_id,),
                    )
                    if request is None:
                        await cur.execute(
                            "UPDATE cubepi_threads "
                            "SET pending_request = NULL, run_id = NULL, "
                            "updated_at = CURRENT_TIMESTAMP WHERE thread_id = %s",
                            (thread_id,),
                        )
                    else:
                        payload = request.model_dump_json()
                        await cur.execute(
                            "UPDATE cubepi_threads "
                            "SET pending_request = %s, run_id = %s, "
                            "updated_at = CURRENT_TIMESTAMP WHERE thread_id = %s",
                            (payload, run_id, thread_id),
                        )
                await conn.commit()
            except BaseException:  # pragma: no cover - defensive txn rollback
                await conn.rollback()
                raise

    async def load_pending_request(self, thread_id: str) -> HitlRequest | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT pending_request FROM cubepi_threads WHERE thread_id = %s",
                    (thread_id,),
                )
                row = await cur.fetchone()
        if row is None or row[0] is None:
            return None
        raw = row[0]
        # aiomysql returns JSON columns as str; tolerate already-parsed dicts (same
        # convention as the existing _parse_json helper in this module).
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
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT run_id FROM cubepi_threads "
                    "WHERE thread_id = %s AND pending_request IS NOT NULL",
                    (thread_id,),
                )
                row = await cur.fetchone()
        return row[0] if row is not None else None
