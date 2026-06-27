"""SQLAlchemy table definitions for cubepi PostgresCheckpointer.

cubepi uses a private MetaData instance (not SQLModel's global one)
so downstreams can compose it into alembic regardless of which ORM
they use. SQLAlchemy 2.0 declarative is the chosen style for cubepi
internal definitions.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EXPECTED_SCHEMA_VERSION = 5
PARTITION_COUNT = 64

cubepi_metadata = sa.MetaData()


class CubepiBase(DeclarativeBase):
    metadata = cubepi_metadata


class CubepiThread(CubepiBase):
    __tablename__ = "cubepi_threads"

    thread_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    parent_thread_id: Mapped[str | None] = mapped_column(
        sa.Text,
        sa.ForeignKey("cubepi_threads.thread_id"),
        nullable=True,
    )
    forked_at_seq: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)
    extra: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    pending_request: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        server_default=sa.text("NULL"),
    )
    # v3: host-side run identifier (e.g. cubebox run_id) persisted alongside
    # pending_request so a worker that recovers after crash can map the paused
    # HITL back to the run that produced it. Written atomically with
    # pending_request via save_pending_request(..., run_id=...).
    run_id: Mapped[str | None] = mapped_column(
        sa.Text,
        nullable=True,
        server_default=sa.text("NULL"),
    )
    created_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )
    updated_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )


class CubepiMessage(CubepiBase):
    __tablename__ = "cubepi_messages"
    __table_args__ = (
        sa.Index(
            "ix_cubepi_messages_metadata_gin",
            "metadata",
            postgresql_using="gin",
            postgresql_ops={"metadata": "jsonb_path_ops"},
        ),
        sa.Index("ix_cubepi_messages_thread_run", "thread_id", "run_id"),
        {"postgresql_partition_by": "HASH (thread_id)"},
    )

    thread_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("cubepi_threads.thread_id", ondelete="CASCADE"),
        primary_key=True,
    )
    seq: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True)
    role: Mapped[str] = mapped_column(sa.Text, nullable=False)
    # Note: Python attribute is `msg_metadata` to avoid collision with
    # SQLAlchemy DeclarativeBase's reserved `metadata` ClassVar.
    # The actual DB column name remains `metadata`.
    msg_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    payload: Mapped[bytes] = mapped_column(sa.LargeBinary, nullable=False)
    # v4: opaque host-side run identifier stamped on each message. Lets
    # fork/snapshot include only messages from completed runs.
    run_id: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )


class CubepiRun(CubepiBase):
    __tablename__ = "cubepi_runs"
    __table_args__ = (
        sa.Index("ix_cubepi_runs_thread_seq", "thread_id", "completion_seq"),
        {"postgresql_partition_by": "HASH (thread_id)"},
    )
    thread_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("cubepi_threads.thread_id", ondelete="CASCADE"),
        primary_key=True,
    )
    run_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    claimed_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )
    completed_at: Mapped[_dt.datetime | None] = mapped_column(
        sa.TIMESTAMP(timezone=True), nullable=True
    )
    completion_seq: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)


class CubepiHitlAnswer(CubepiBase):
    __tablename__ = "cubepi_hitl_answers"

    thread_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("cubepi_threads.thread_id", ondelete="CASCADE"),
        primary_key=True,
    )
    run_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    question_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    answer: Mapped[Any] = mapped_column(JSONB, nullable=False)
    answered_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )


class CubepiSchemaVersion(CubepiBase):
    __tablename__ = "cubepi_schema_version"

    version: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
