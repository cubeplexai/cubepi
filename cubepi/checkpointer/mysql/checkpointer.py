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
from cubepi.checkpointer.mysql.exceptions import (
    CubepiSchemaMismatch,
    CubepiSchemaUninitialized,
)
from cubepi.checkpointer.mysql.models import EXPECTED_SCHEMA_VERSION
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)

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
                hint="cubepi was upgraded but host alembic is behind. "
                "Generate a new revision and apply.",
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
                            )
                        )
                    await cur.executemany(
                        "INSERT INTO cubepi_messages "
                        "(thread_id, seq, role, metadata, payload) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        rows,
                    )
                await conn.commit()
            except BaseException:  # pragma: no cover - defensive txn rollback
                await conn.rollback()
                raise

    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None:
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
