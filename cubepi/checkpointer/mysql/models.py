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

EXPECTED_SCHEMA_VERSION = 2
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
