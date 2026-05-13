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

EXPECTED_SCHEMA_VERSION = 1
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
    created_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )


class CubepiSchemaVersion(CubepiBase):
    __tablename__ = "cubepi_schema_version"

    version: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
