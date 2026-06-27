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

EXPECTED_SCHEMA_VERSION = 5
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
    pending_request: Mapped[dict[str, Any] | None] = mapped_column(
        sa.JSON,
        nullable=True,
    )
    # v3: host-side run identifier persisted alongside pending_request. See
    # the parallel docstring on cubepi/checkpointer/postgres/models.py.
    # VARCHAR(64) accommodates UUIDs (36), prefixed UUIDs (e.g. "run_<uuid>"),
    # and typical opaque ids. Hosts that need longer identifiers should
    # subclass MySQLCheckpointer and override the column type — cubepi
    # doesn't pay the TEXT-column cost for everyone to accommodate a
    # minority case.
    run_id: Mapped[str | None] = mapped_column(
        VARCHAR(64),
        nullable=True,
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
    __table_args__ = (
        sa.Index("ix_cubepi_messages_thread_run", "thread_id", "run_id"),
        {"mysql_engine": "InnoDB"},
    )

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
    # v4: opaque host-side run identifier stamped on each message. Lets
    # fork/snapshot include only messages from completed runs. VARCHAR(255)
    # to match the cubepi_messages indexing convention (thread_id is also
    # VARCHAR(255) — keeps the composite index cardinality consistent).
    run_id: Mapped[str | None] = mapped_column(
        VARCHAR(255),
        nullable=True,
    )
    created_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP,
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )


class CubepiRun(CubepiBase):
    """v4 per-run lifecycle row.

    Primary key (thread_id, run_id). KEY-partitioned by thread_id to match
    cubepi_messages. NO FK to cubepi_threads — MySQL forbids FKs on
    partitioned tables. The partition clause cannot be expressed in
    SQLAlchemy declarative and lives in
    ``alembic_helpers.runs_partition_clause()``.
    """

    __tablename__ = "cubepi_runs"
    __table_args__ = (
        sa.Index("ix_cubepi_runs_thread_seq", "thread_id", "completion_seq"),
        {"mysql_engine": "InnoDB"},
    )

    thread_id: Mapped[str] = mapped_column(_TID, primary_key=True)
    run_id: Mapped[str] = mapped_column(VARCHAR(255), primary_key=True)
    claimed_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP,
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )
    completed_at: Mapped[_dt.datetime | None] = mapped_column(
        sa.TIMESTAMP,
        nullable=True,
    )
    completion_seq: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)


class CubepiHitlAnswer(CubepiBase):
    __tablename__ = "cubepi_hitl_answers"
    __table_args__ = {"mysql_engine": "InnoDB"}

    thread_id: Mapped[str] = mapped_column(_TID, primary_key=True)
    run_id: Mapped[str] = mapped_column(VARCHAR(255), primary_key=True)
    question_id: Mapped[str] = mapped_column(VARCHAR(255), primary_key=True)
    answer: Mapped[Any] = mapped_column(JSON, nullable=False)
    answered_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP,
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )


class CubepiSchemaVersion(CubepiBase):
    __tablename__ = "cubepi_schema_version"
    __table_args__ = {"mysql_engine": "InnoDB"}

    version: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
