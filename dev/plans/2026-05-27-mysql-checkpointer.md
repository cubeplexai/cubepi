# MySQLCheckpointer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-grade `MySQLCheckpointer` implementing the cubepi
`Checkpointer` protocol against MySQL, mirroring `PostgresCheckpointer`.

**Architecture:** New package `cubepi/checkpointer/mysql/` (checkpointer / models
/ alembic_helpers / exceptions / __init__) following the Postgres package layout.
Connection pool via aiomysql, msgpack message payloads, JSON `metadata`/`extra`
columns, `KEY(thread_id)` partitioned messages table, host-managed schema
verified on entry. Schema exceptions are re-exported from the Postgres module
(single source of truth). Ships as optional `mysql` extra.

**Tech Stack:** Python 3.11+, aiomysql, SQLAlchemy 2.0 (table defs only), msgpack,
pymysql (error codes, transitive via aiomysql), pytest/pytest-asyncio.

Spec: `dev/specs/2026-05-27-mysql-checkpointer.md`.

---

## File Structure

- Create `cubepi/checkpointer/mysql/__init__.py` — public exports + lazy nothing
  (eager, since importing it already implies the extra is installed).
- Create `cubepi/checkpointer/mysql/exceptions.py` — re-export schema exceptions
  from `cubepi.checkpointer.postgres.exceptions`.
- Create `cubepi/checkpointer/mysql/models.py` — SQLAlchemy table defs on a
  private `cubepi_metadata`; constants `EXPECTED_SCHEMA_VERSION`,
  `PARTITION_COUNT`.
- Create `cubepi/checkpointer/mysql/alembic_helpers.py` —
  `messages_partition_clause()`, `write_schema_version_op()`.
- Create `cubepi/checkpointer/mysql/checkpointer.py` — `MySQLCheckpointer` +
  `_parse_dsn`, `_role_of`, `_ROLE_TO_CLS`, `_decode_json`.
- Modify `cubepi/checkpointer/__init__.py` — add `MySQLCheckpointer` to
  `__getattr__` and `__all__`.
- Modify `pyproject.toml` — add `mysql` optional-dependency extra + dev group dep.
- Create `tests/checkpointer/test_mysql.py` — unit + E2E tests.
- Modify `tests/checkpointer/conftest.py` — add MySQL fixtures alongside PG ones.
- Create/modify docs page under `website/docs/` — MySQL checkpointer section.

---

## Task 1: Dependency + package skeleton + exceptions

**Files:**
- Modify: `pyproject.toml`
- Create: `cubepi/checkpointer/mysql/__init__.py`
- Create: `cubepi/checkpointer/mysql/exceptions.py`
- Test: `tests/checkpointer/test_mysql.py`

- [ ] **Step 0: Add the `mysql` extra and dev dep (must precede any aiomysql import)**

In `pyproject.toml`, under `[project.optional-dependencies]` add after the
`postgres` block:

```toml
mysql = [
    "sqlalchemy>=2.0",
    "aiomysql>=0.2",
    "msgpack>=1.0",
]
```

In `[dependency-groups]` `dev = [ ... ]`, add near the other DB drivers:

```toml
    "aiomysql>=0.2",
```

Then run: `uv sync --all-extras --dev`
Expected: resolves and installs aiomysql (which pulls in PyMySQL).

- [ ] **Step 1: Write the failing test**

```python
# tests/checkpointer/test_mysql.py
"""MySQLCheckpointer tests — mirrors test_postgres.py."""

import pytest


def test_exceptions_are_postgres_aliases() -> None:
    from cubepi.checkpointer.mysql.exceptions import (
        CubepiSchemaError,
        CubepiSchemaMismatch,
        CubepiSchemaUninitialized,
    )
    from cubepi.checkpointer.postgres import exceptions as pg_exc

    assert CubepiSchemaError is pg_exc.CubepiSchemaError
    assert CubepiSchemaMismatch is pg_exc.CubepiSchemaMismatch
    assert CubepiSchemaUninitialized is pg_exc.CubepiSchemaUninitialized
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/checkpointer/test_mysql.py::test_exceptions_are_postgres_aliases -v`
Expected: FAIL — `ModuleNotFoundError: cubepi.checkpointer.mysql`

- [ ] **Step 3: Write minimal implementation**

```python
# cubepi/checkpointer/mysql/exceptions.py
"""Schema exceptions for MySQLCheckpointer.

These are backend-agnostic schema errors; re-exported from the Postgres
module so there is a single source of truth.
"""

from cubepi.checkpointer.postgres.exceptions import (
    CubepiSchemaError,
    CubepiSchemaMismatch,
    CubepiSchemaUninitialized,
)

__all__ = [
    "CubepiSchemaError",
    "CubepiSchemaMismatch",
    "CubepiSchemaUninitialized",
]
```

```python
# cubepi/checkpointer/mysql/__init__.py
"""MySQLCheckpointer package.

Exports are built up incrementally across the plan. Exceptions are available
immediately; MySQLCheckpointer and model symbols are added once their modules
exist (Task 5), to avoid importing not-yet-written submodules.
"""

from cubepi.checkpointer.mysql.exceptions import (
    CubepiSchemaError,
    CubepiSchemaMismatch,
    CubepiSchemaUninitialized,
)

__all__ = [
    "CubepiSchemaError",
    "CubepiSchemaMismatch",
    "CubepiSchemaUninitialized",
]
```

**Critical:** importing the package runs `__init__.py` first, so it must NOT
import `checkpointer`/`models` until those modules exist. The full export list
(adding `MySQLCheckpointer`, `EXPECTED_SCHEMA_VERSION`, `PARTITION_COUNT`,
`cubepi_metadata`) is written in Task 5 Step 2, after all submodules exist.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/checkpointer/test_mysql.py::test_exceptions_are_postgres_aliases -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock cubepi/checkpointer/mysql/__init__.py cubepi/checkpointer/mysql/exceptions.py tests/checkpointer/test_mysql.py
git commit -m "feat(checkpointer): mysql package skeleton, exception aliases, mysql extra"
```

---

## Task 2: SQLAlchemy models

**Files:**
- Create: `cubepi/checkpointer/mysql/models.py`
- Test: `tests/checkpointer/test_mysql.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/checkpointer/test_mysql.py
def test_models_import() -> None:
    from cubepi.checkpointer.mysql.models import (
        EXPECTED_SCHEMA_VERSION,
        PARTITION_COUNT,
        CubepiMessage,
        CubepiSchemaVersion,
        CubepiThread,
        cubepi_metadata,
    )

    assert EXPECTED_SCHEMA_VERSION == 1
    assert PARTITION_COUNT == 64
    assert CubepiThread.__tablename__ == "cubepi_threads"
    assert CubepiMessage.__tablename__ == "cubepi_messages"
    assert CubepiSchemaVersion.__tablename__ == "cubepi_schema_version"
    assert "cubepi_threads" in cubepi_metadata.tables
    assert "cubepi_messages" in cubepi_metadata.tables
    assert "cubepi_schema_version" in cubepi_metadata.tables


def test_threads_parent_self_fk_present() -> None:
    """parent_thread_id keeps a self-FK (threads table is not partitioned)."""
    from cubepi.checkpointer.mysql.models import cubepi_metadata

    threads = cubepi_metadata.tables["cubepi_threads"]
    fk_targets = {
        fk.column.table.name
        for col in threads.columns
        for fk in col.foreign_keys
    }
    # parent_thread_id references cubepi_threads
    assert "cubepi_threads" in fk_targets


def test_messages_has_no_foreign_keys() -> None:
    """messages table is partitioned, so no FK is allowed."""
    from cubepi.checkpointer.mysql.models import cubepi_metadata

    msgs = cubepi_metadata.tables["cubepi_messages"]
    assert msgs.foreign_keys == set()


def test_messages_has_no_metadata_index() -> None:
    from cubepi.checkpointer.mysql.models import cubepi_metadata

    msgs = cubepi_metadata.tables["cubepi_messages"]
    assert msgs.indexes == set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/checkpointer/test_mysql.py -k "models_import or self_fk or foreign_keys or metadata_index" -v`
Expected: FAIL — `ModuleNotFoundError` / attribute errors.

- [ ] **Step 3: Write minimal implementation**

```python
# cubepi/checkpointer/mysql/models.py
"""SQLAlchemy table definitions for cubepi MySQLCheckpointer.

Mirrors the Postgres models with MySQL adaptations: VARCHAR(255) utf8mb4_bin
thread ids, JSON columns, no messages->threads FK (the messages table is
KEY-partitioned and MySQL forbids FKs on partitioned tables), self-FK on
parent_thread_id kept. KEY partitioning is NOT expressible in SQLAlchemy
declarative, so it lives only in alembic_helpers.messages_partition_clause().
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.mysql import JSON, LONGBLOB, VARCHAR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EXPECTED_SCHEMA_VERSION = 1
PARTITION_COUNT = 64

cubepi_metadata = sa.MetaData()

_TID = VARCHAR(255, collation="utf8mb4_bin")


class CubepiBase(DeclarativeBase):
    metadata = cubepi_metadata


class CubepiThread(CubepiBase):
    __tablename__ = "cubepi_threads"
    __table_args__ = {"mysql_engine": "InnoDB"}

    thread_id: Mapped[str] = mapped_column(_TID, primary_key=True)
    parent_thread_id: Mapped[str | None] = mapped_column(
        _TID,
        sa.ForeignKey("cubepi_threads.thread_id"),
        nullable=True,
    )
    forked_at_seq: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)
    extra: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        server_default=sa.text("(JSON_OBJECT())"),
    )
    created_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP,
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP,
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
    )


class CubepiMessage(CubepiBase):
    __tablename__ = "cubepi_messages"
    __table_args__ = {"mysql_engine": "InnoDB"}

    thread_id: Mapped[str] = mapped_column(_TID, primary_key=True)
    seq: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True)
    role: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    # Python attribute renamed to avoid DeclarativeBase's reserved `metadata`
    # ClassVar; DB column stays `metadata`.
    msg_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        server_default=sa.text("(JSON_OBJECT())"),
    )
    payload: Mapped[bytes] = mapped_column(LONGBLOB, nullable=False)
    created_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP,
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )


class CubepiSchemaVersion(CubepiBase):
    __tablename__ = "cubepi_schema_version"
    __table_args__ = {"mysql_engine": "InnoDB"}

    version: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/checkpointer/test_mysql.py -k "models_import or self_fk or foreign_keys or metadata_index" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add cubepi/checkpointer/mysql/models.py tests/checkpointer/test_mysql.py
git commit -m "feat(checkpointer): mysql SQLAlchemy models"
```

---

## Task 3: alembic helpers

**Files:**
- Create: `cubepi/checkpointer/mysql/alembic_helpers.py`
- Test: `tests/checkpointer/test_mysql.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/checkpointer/test_mysql.py
def test_messages_partition_clause() -> None:
    from cubepi.checkpointer.mysql.alembic_helpers import messages_partition_clause

    clause = messages_partition_clause()
    assert clause == "PARTITION BY KEY (thread_id) PARTITIONS 64"


def test_write_schema_version_op_clears_stale_then_inserts() -> None:
    from cubepi.checkpointer.mysql.alembic_helpers import write_schema_version_op

    sql = write_schema_version_op()
    assert "DELETE FROM cubepi_schema_version" in sql
    assert "WHERE version <> 1" in sql
    assert "INSERT IGNORE INTO cubepi_schema_version" in sql
    assert "VALUES (1)" in sql
    assert sql.index("DELETE") < sql.index("INSERT")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/checkpointer/test_mysql.py -k "partition_clause or schema_version_op" -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# cubepi/checkpointer/mysql/alembic_helpers.py
"""SQL helpers for host application alembic migrations (MySQL)."""

from cubepi.checkpointer.mysql.models import (
    EXPECTED_SCHEMA_VERSION,
    PARTITION_COUNT,
)


def messages_partition_clause() -> str:
    """Return the KEY-partition clause for the cubepi_messages CREATE TABLE.

    Unlike Postgres (one child partition per modulus), MySQL declares all
    partitions inline. Append this to the messages table DDL, e.g.::

        op.execute("CREATE TABLE cubepi_messages (...) " + messages_partition_clause())

    KEY (not HASH) is required because thread_id is a VARCHAR; MySQL HASH
    partitioning only accepts integer expressions.
    """
    return f"PARTITION BY KEY (thread_id) PARTITIONS {PARTITION_COUNT}"


def write_schema_version_op() -> str:
    """Return SQL setting cubepi_schema_version to the current version.

    Clears any stale rows from prior cubepi versions then inserts the current
    one. Idempotent. Call inside alembic upgrade() after CREATE TABLE
    cubepi_schema_version.
    """
    return (
        f"DELETE FROM cubepi_schema_version "
        f"WHERE version <> {EXPECTED_SCHEMA_VERSION}; "
        f"INSERT IGNORE INTO cubepi_schema_version (version) "
        f"VALUES ({EXPECTED_SCHEMA_VERSION});"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/checkpointer/test_mysql.py -k "partition_clause or schema_version_op" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cubepi/checkpointer/mysql/alembic_helpers.py tests/checkpointer/test_mysql.py
git commit -m "feat(checkpointer): mysql alembic helpers"
```

---

## Task 4: Checkpointer — helpers, DSN, role mapping, empty-append no-op

**Files:**
- Create: `cubepi/checkpointer/mysql/checkpointer.py`
- Test: `tests/checkpointer/test_mysql.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/checkpointer/test_mysql.py
def test_role_of_known_message_types() -> None:
    from cubepi.checkpointer.mysql.checkpointer import _role_of
    from cubepi.providers.base import (
        AssistantMessage,
        TextContent,
        ToolResultMessage,
        Usage,
        UserMessage,
    )

    assert _role_of(UserMessage(content=[TextContent(text="x")])) == "user"
    assert (
        _role_of(AssistantMessage(content=[TextContent(text="x")], usage=Usage()))
        == "assistant"
    )
    tr = ToolResultMessage(
        tool_call_id="tc-1", tool_name="t", content=[TextContent(text="ok")]
    )
    assert _role_of(tr) == "tool"


def test_role_of_rejects_unknown_message_type() -> None:
    from cubepi.checkpointer.mysql.checkpointer import _role_of

    class FakeMessage:
        pass

    with pytest.raises(TypeError, match="unknown Message type"):
        _role_of(FakeMessage())  # type: ignore[arg-type]


def test_parse_dsn_full() -> None:
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    cfg = _parse_dsn("mysql://user:pw@db.example.com:3307/mydb")
    assert cfg == {
        "host": "db.example.com",
        "port": 3307,
        "user": "user",
        "password": "pw",
        "db": "mydb",
    }


def test_parse_dsn_defaults_port_3306() -> None:
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    cfg = _parse_dsn("mysql://root@localhost/x")
    assert cfg["port"] == 3306
    assert cfg["password"] == ""


def test_decode_json_handles_str_and_dict() -> None:
    from cubepi.checkpointer.mysql.checkpointer import _decode_json

    assert _decode_json('{"a": 1}') == {"a": 1}
    assert _decode_json({"a": 1}) == {"a": 1}
    assert _decode_json(None) == {}


@pytest.mark.asyncio
async def test_append_empty_messages_is_noop() -> None:
    # Import from the submodule, not the package: the package __init__ does not
    # export MySQLCheckpointer until Task 5.
    from cubepi.checkpointer.mysql.checkpointer import MySQLCheckpointer

    cp = MySQLCheckpointer("mysql://root@unreachable-host/none")
    assert cp._pool is None
    await cp.append("thread-x", [])
    assert cp._pool is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/checkpointer/test_mysql.py -k "role_of or parse_dsn or decode_json or append_empty" -v`
Expected: FAIL — module/attr missing.

- [ ] **Step 3: Write minimal implementation**

```python
# cubepi/checkpointer/mysql/checkpointer.py
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

    async def append(self, thread_id: str, messages: list[Message]) -> None:
        if not messages:
            return
        raise NotImplementedError  # completed in Task 6
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/checkpointer/test_mysql.py -k "role_of or parse_dsn or decode_json or append_empty" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cubepi/checkpointer/mysql/checkpointer.py tests/checkpointer/test_mysql.py
git commit -m "feat(checkpointer): mysql checkpointer helpers + dsn parsing"
```

---

## Task 5: Pool lifecycle + schema verification + package wiring

**Files:**
- Modify: `cubepi/checkpointer/mysql/checkpointer.py`
- Modify: `cubepi/checkpointer/mysql/__init__.py`
- Modify: `cubepi/checkpointer/__init__.py`
- Test: `tests/checkpointer/test_mysql.py`

(The `mysql` extra and dev dep were already added in Task 1 Step 0.)

- [ ] **Step 1: Complete the mysql package `__init__.py`**

Now that all submodules exist, replace `cubepi/checkpointer/mysql/__init__.py`
with the full export surface:

```python
# cubepi/checkpointer/mysql/__init__.py
from cubepi.checkpointer.mysql.checkpointer import MySQLCheckpointer
from cubepi.checkpointer.mysql.exceptions import (
    CubepiSchemaError,
    CubepiSchemaMismatch,
    CubepiSchemaUninitialized,
)
from cubepi.checkpointer.mysql.models import (
    EXPECTED_SCHEMA_VERSION,
    PARTITION_COUNT,
    cubepi_metadata,
)

__all__ = [
    "MySQLCheckpointer",
    "CubepiSchemaError",
    "CubepiSchemaMismatch",
    "CubepiSchemaUninitialized",
    "EXPECTED_SCHEMA_VERSION",
    "PARTITION_COUNT",
    "cubepi_metadata",
]
```

- [ ] **Step 2: Wire lazy import in the checkpointer package**

Edit `cubepi/checkpointer/__init__.py` `__getattr__` to add, before the final
`raise AttributeError`:

```python
    if name == "MySQLCheckpointer":
        from cubepi.checkpointer.mysql.checkpointer import MySQLCheckpointer

        return MySQLCheckpointer
```

And add `"MySQLCheckpointer"` to `__all__`.

- [ ] **Step 3: Write the test for the top-level lazy import**

```python
# append to tests/checkpointer/test_mysql.py
def test_top_level_lazy_import() -> None:
    import cubepi.checkpointer as cp_pkg

    assert cp_pkg.MySQLCheckpointer is not None
    assert "MySQLCheckpointer" in cp_pkg.__all__
```

- [ ] **Step 4: Run it (verifies the wiring; the class exists from Task 4)**

Run: `uv run pytest tests/checkpointer/test_mysql.py::test_top_level_lazy_import -v`
Expected: PASS. A failure here means an import error in the wiring — fix it
before continuing. (This step verifies wiring rather than following strict
red→green, because the class already exists from Task 4.)

- [ ] **Step 5: Implement `__aenter__`/`__aexit__`/`_verify_schema`**

Add these methods to `MySQLCheckpointer` (after `__init__`, before `append`):

```python
    async def __aenter__(self) -> "MySQLCheckpointer":
        self._pool = await aiomysql.create_pool(
            minsize=self._min,
            maxsize=self._max,
            autocommit=True,
            **self._cfg,
        )
        await self._verify_schema()
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
                    raise
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
```

- [ ] **Step 6: Run the full unit-test subset to verify nothing broke**

Run: `uv run pytest tests/checkpointer/test_mysql.py -v`
Expected: PASS for all non-E2E tests (E2E tests not written yet).

- [ ] **Step 7: Commit**

```bash
git add cubepi/checkpointer/__init__.py cubepi/checkpointer/mysql/__init__.py cubepi/checkpointer/mysql/checkpointer.py tests/checkpointer/test_mysql.py
git commit -m "feat(checkpointer): mysql pool lifecycle + schema verification + package wiring"
```

---

## Task 6: load / append / save_extra

**Files:**
- Modify: `cubepi/checkpointer/mysql/checkpointer.py`
- Test: `tests/checkpointer/test_mysql.py` (E2E, added in Task 7)

This task has no standalone unit test (the logic needs a DB; E2E tests in Task 7
cover it). Implement, then run the existing unit subset to confirm no import or
syntax regressions.

- [ ] **Step 1: Implement `load`**

Replace the `append` stub region by first adding `load` above it:

```python
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
```

- [ ] **Step 2: Implement `append` (replace the `NotImplementedError` stub)**

```python
    async def append(self, thread_id: str, messages: list[Message]) -> None:
        if not messages:
            return
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "INSERT IGNORE INTO cubepi_threads (thread_id) "
                        "VALUES (%s)",
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
            except BaseException:
                await conn.rollback()
                raise
```

- [ ] **Step 3: Implement `save_extra` (read-modify-write shallow merge)**

```python
    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "INSERT IGNORE INTO cubepi_threads (thread_id) "
                        "VALUES (%s)",
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
            except BaseException:
                await conn.rollback()
                raise
```

- [ ] **Step 4: Run the unit subset to verify no regressions**

Run: `uv run pytest tests/checkpointer/test_mysql.py -v`
Expected: PASS (unit tests; E2E added next task).
Also: `uv run ruff check cubepi/ tests/` and `uv run ruff format --check cubepi/ tests/`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add cubepi/checkpointer/mysql/checkpointer.py
git commit -m "feat(checkpointer): mysql load/append/save_extra"
```

---

## Task 7: E2E tests + conftest fixtures

**Files:**
- Modify: `tests/checkpointer/conftest.py`
- Modify: `tests/checkpointer/test_mysql.py`

E2E tests skip automatically when no MySQL is reachable (mirrors the Postgres
ones). They run when `CUBEPI_TEST_MYSQL_DSN` points at a live server.

- [ ] **Step 1: Add MySQL fixtures to conftest**

Append to `tests/checkpointer/conftest.py`:

```python
import aiomysql  # add near the asyncpg import at top of file


@pytest.fixture(scope="session")
def mysql_dsn() -> str:
    """MySQL DSN for tests. Override via CUBEPI_TEST_MYSQL_DSN env."""
    return os.environ.get(
        "CUBEPI_TEST_MYSQL_DSN",
        "mysql://root:root@localhost:3306/mysql",
    )


def _mysql_cfg(dsn: str) -> dict:
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    return _parse_dsn(dsn)


@pytest_asyncio.fixture(scope="session")
async def _mysql_available(mysql_dsn: str) -> bool:
    """Quick probe — connection refused → skip MySQL tests."""
    try:
        conn = await aiomysql.connect(connect_timeout=2, **_mysql_cfg(mysql_dsn))
        conn.close()
        return True
    except (aiomysql.Error, OSError, ConnectionError):
        return False


@pytest_asyncio.fixture
async def clean_mysql_db(mysql_dsn: str, _mysql_available: bool):
    """Create a fresh database for each test; drop after."""
    if not _mysql_available:
        pytest.skip(
            "MySQL not available for E2E tests; set CUBEPI_TEST_MYSQL_DSN "
            "to a working DSN to enable."
        )

    db_name = f"cubepi_test_{secrets.token_hex(6)}"
    admin_cfg = _mysql_cfg(mysql_dsn)
    admin = await aiomysql.connect(autocommit=True, **admin_cfg)
    try:
        async with admin.cursor() as cur:
            await cur.execute(f"CREATE DATABASE `{db_name}`")
    finally:
        admin.close()

    base = mysql_dsn.rsplit("/", 1)[0]
    test_dsn = f"{base}/{db_name}"
    yield test_dsn

    admin = await aiomysql.connect(autocommit=True, **admin_cfg)
    try:
        async with admin.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
    finally:
        admin.close()
```

(`os`, `secrets`, `pytest`, `pytest_asyncio` are already imported in this file.)

- [ ] **Step 2: Add `_setup_schema` + E2E tests to test_mysql.py**

```python
# append to tests/checkpointer/test_mysql.py
import aiomysql

from cubepi.checkpointer.mysql.alembic_helpers import (
    messages_partition_clause,
    write_schema_version_op,
)


async def _setup_schema(dsn: str) -> None:
    """Build the cubepi schema (matching what host alembic would generate)."""
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    conn = await aiomysql.connect(autocommit=True, **_parse_dsn(dsn))
    try:
        async with conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE cubepi_threads (
                    thread_id VARCHAR(255) COLLATE utf8mb4_bin PRIMARY KEY,
                    parent_thread_id VARCHAR(255) COLLATE utf8mb4_bin NULL,
                    forked_at_seq BIGINT NULL,
                    extra JSON NOT NULL DEFAULT (JSON_OBJECT()),
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
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, seq)
                ) ENGINE=InnoDB """
                + messages_partition_clause()
            )
            await cur.execute("""
                CREATE TABLE cubepi_schema_version (
                    version INT PRIMARY KEY
                ) ENGINE=InnoDB
            """)
            for stmt in write_schema_version_op().split(";"):
                if stmt.strip():
                    await cur.execute(stmt)
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_mysql_checkpointer_round_trip(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import MySQLCheckpointer
    from cubepi.providers.base import (
        AssistantMessage,
        TextContent,
        Usage,
        UserMessage,
    )

    await _setup_schema(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        msg1 = UserMessage(
            content=[TextContent(text="hello")],
            metadata={"memory_snapshot": {"id": "m1"}},
        )
        msg2 = AssistantMessage(
            content=[TextContent(text="hi back")],
            usage=Usage(),
            metadata={"cost_cents": 5},
        )
        await cp.append("t-1", [msg1, msg2])
        data = await cp.load("t-1")

    assert data is not None
    assert len(data.messages) == 2
    assert isinstance(data.messages[0], UserMessage)
    assert isinstance(data.messages[1], AssistantMessage)
    assert data.messages[0].metadata == {"memory_snapshot": {"id": "m1"}}
    assert data.messages[1].metadata == {"cost_cents": 5}
    assert data.messages[0].content[0].text == "hello"
    assert data.messages[1].content[0].text == "hi back"


@pytest.mark.asyncio
async def test_mysql_checkpointer_save_extra_merges(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import MySQLCheckpointer
    from cubepi.providers.base import TextContent, UserMessage

    await _setup_schema(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        await cp.append("t-2", [UserMessage(content=[TextContent(text="x")])])
        await cp.save_extra("t-2", {"a": 1})
        await cp.save_extra("t-2", {"b": 2})
        data = await cp.load("t-2")

    assert data is not None
    assert data.extra == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_mysql_checkpointer_seq_monotonic(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import MySQLCheckpointer
    from cubepi.providers.base import TextContent, UserMessage

    await _setup_schema(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        msgs1 = [UserMessage(content=[TextContent(text=str(i))]) for i in range(5)]
        await cp.append("t-3", msgs1)
        msgs2 = [
            UserMessage(content=[TextContent(text=str(i))]) for i in range(5, 10)
        ]
        await cp.append("t-3", msgs2)
        data = await cp.load("t-3")

    assert data is not None
    assert len(data.messages) == 10
    texts = [m.content[0].text for m in data.messages]
    assert texts == [str(i) for i in range(10)]


@pytest.mark.asyncio
async def test_uninitialized_schema_raises(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import (
        CubepiSchemaUninitialized,
        MySQLCheckpointer,
    )

    with pytest.raises(CubepiSchemaUninitialized):
        async with MySQLCheckpointer(clean_mysql_db):
            pass


@pytest.mark.asyncio
async def test_version_mismatch_raises(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import CubepiSchemaMismatch, MySQLCheckpointer
    from cubepi.checkpointer.mysql.checkpointer import _parse_dsn

    await _setup_schema(clean_mysql_db)
    conn = await aiomysql.connect(autocommit=True, **_parse_dsn(clean_mysql_db))
    try:
        async with conn.cursor() as cur:
            await cur.execute("UPDATE cubepi_schema_version SET version = 999")
    finally:
        conn.close()

    with pytest.raises(CubepiSchemaMismatch) as exc_info:
        async with MySQLCheckpointer(clean_mysql_db):
            pass
    assert exc_info.value.expected == 1
    assert exc_info.value.actual == 999


@pytest.mark.asyncio
async def test_empty_thread_load_returns_none(clean_mysql_db) -> None:
    from cubepi.checkpointer.mysql import MySQLCheckpointer

    await _setup_schema(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        data = await cp.load("nonexistent-thread")
    assert data is None
```

- [ ] **Step 3: Run the full test file**

Run: `uv run pytest tests/checkpointer/test_mysql.py -v`
Expected: unit tests PASS; E2E tests PASS if `CUBEPI_TEST_MYSQL_DSN` is set to a
live 8.0.13+ server, otherwise SKIPPED. Run with a real DB before considering
the feature done:
`CUBEPI_TEST_MYSQL_DSN=mysql://root:root@localhost:3306/mysql uv run pytest tests/checkpointer/test_mysql.py -v`

- [ ] **Step 4: Lint**

Run: `uv run ruff check cubepi/ tests/ && uv run ruff format --check cubepi/ tests/`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add tests/checkpointer/conftest.py tests/checkpointer/test_mysql.py
git commit -m "test(checkpointer): mysql unit + E2E tests"
```

---

## Task 8: Documentation

**Files:**
- Modify: the checkpointer docs page under `website/docs/` (locate the page that
  documents `PostgresCheckpointer`; add a MySQL section there). If Postgres is
  documented in e.g. `website/docs/guides/checkpointing.md` or similar, edit that
  file. Use `grep -rl PostgresCheckpointer website/docs` to find it.

- [ ] **Step 1: Locate the existing checkpointer doc**

Run: `grep -rl "PostgresCheckpointer" website/docs`
Use the returned path as the doc to extend.

- [ ] **Step 2: Add a MySQL section**

Mirror the Postgres section. Content must cover:
- Install: `pip install "cubepi[mysql]"` (driver: aiomysql).
- Minimum MySQL **8.0.13+**, InnoDB engine.
- Host-managed schema: cubepi ships SQLAlchemy models
  (`cubepi.checkpointer.mysql.models.cubepi_metadata`) and alembic helpers
  (`messages_partition_clause()`, `write_schema_version_op()`). The host's
  alembic migration creates the tables; the checkpointer verifies the schema
  version on connect and raises `CubepiSchemaUninitialized` / `CubepiSchemaMismatch`.
- Usage example:

```python
from cubepi.checkpointer import MySQLCheckpointer

cp = MySQLCheckpointer("mysql://user:pw@localhost:3306/mydb")
async with cp:
    await cp.append(thread_id, [msg])
    data = await cp.load(thread_id)
```

- Divergences worth flagging to users: messages table is `KEY(thread_id)`
  partitioned with no FK (integrity from lazy thread insert); `metadata` JSON
  column is **not** indexed (add a generated-column index host-side if you need
  to query by metadata); thread IDs are case-sensitive (`utf8mb4_bin`).

- [ ] **Step 3: Build/verify docs (if a build step exists)**

Run (if present): `grep -q "\"build\"" website/package.json && (cd website && npm run build)` — otherwise skip; verify the markdown renders by eye.

- [ ] **Step 4: Commit**

```bash
git add website/docs
git commit -m "docs: document MySQLCheckpointer"
```

---

## Final verification before PR

- [ ] `uv run pytest tests/ -v` (whole suite; MySQL E2E skipped without DB).
- [ ] With a live MySQL 8.0.13+: `CUBEPI_TEST_MYSQL_DSN=... uv run pytest tests/checkpointer/test_mysql.py -v` — all E2E pass.
- [ ] `uv run ruff check cubepi/ tests/` clean.
- [ ] `uv run ruff format --check cubepi/ tests/` clean.
- [ ] Spec divergences all reflected in code.
- [ ] Docs page updated.
