"""Per-run tagging / metadata context for cubepi.tracing.

Sets a contextvar-scoped ``tags`` / ``metadata`` payload that the
:class:`~cubepi.tracing.recorder.Recorder` reads on ``AgentStartEvent``
and stamps onto the ``invoke_agent`` span. Inspired by LangSmith's
``langsmith.run_helpers.tracing_context`` (same contextvar mechanism).

Namespacing:

- Tags use a single attribute ``cubepi.tags`` (tuple of strings).
- User metadata is namespaced under ``cubepi.metadata.*`` so that
  recorder-owned schema keys (``cubepi.run_id``,
  ``cubepi.turn.index``, …) can never be overridden by
  caller-supplied values.

Usage::

    from cubepi.tracing import tracing_context

    async with tracer.attached(agent):
        with tracing_context(tags=["beta-arm"], metadata={"user_id": "u-42"}):
            await agent.prompt("hello")
        # the next run does NOT carry those tags
        await agent.prompt("goodbye")

Resulting ``invoke_agent`` span attributes:

- ``cubepi.tags`` = ``("beta-arm",)``
- ``cubepi.metadata.user_id`` = ``"u-42"``

Multiple nested ``tracing_context`` blocks merge: inner tags are
appended, inner metadata keys override outer ones (last-write-wins).
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Iterator


_run_metadata: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "cubepi.tracing.run_metadata", default={}
)

_run_tags: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "cubepi.tracing.run_tags", default=()
)


@contextlib.contextmanager
def tracing_context(
    *,
    tags: list[str] | tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Scope tags / metadata onto agent runs started inside this block.

    The recorder reads these contextvars on ``AgentStartEvent`` and
    stamps them on the ``invoke_agent`` span as:

    - ``cubepi.tags`` — tuple of strings (OTel attribute type)
    - one attribute per metadata key, namespaced under
      ``cubepi.metadata.*`` (e.g. ``metadata={"user_id": "u-42"}`` →
      ``cubepi.metadata.user_id = "u-42"``). The dedicated
      sub-namespace keeps recorder-owned schema keys
      (``cubepi.run_id``, ``cubepi.turn.index``, …) safe from
      caller-supplied collisions.

    The contextvar nature means this works for concurrent agents:
    each asyncio task tree gets its own value. Nested blocks merge
    additively (tags concatenate; metadata is union with inner
    keys winning).

    Args:
        tags: Tags to apply to runs started in this scope. Stored as
            a tuple on the span so it round-trips through OTel's
            attribute serializer.
        metadata: Per-run key/value pairs. Values must be types that
            OTel attributes accept (str, bool, int, float, or a
            tuple/list of those); other shapes will be silently
            dropped by the recorder.
    """
    new_meta = {**_run_metadata.get(), **(metadata or {})}
    new_tags = tuple(_run_tags.get()) + tuple(tags or ())
    meta_token = _run_metadata.set(new_meta)
    tag_token = _run_tags.set(new_tags)
    try:
        yield
    finally:
        _run_metadata.reset(meta_token)
        _run_tags.reset(tag_token)


def _current_tags() -> tuple[str, ...]:
    """Internal: return the active tag tuple for the current task.

    Called by :class:`~cubepi.tracing.recorder.Recorder` on
    ``_on_agent_start``; not part of the public API.
    """
    return _run_tags.get()


def _current_metadata() -> dict[str, Any]:
    """Internal: return the active metadata dict for the current task."""
    return _run_metadata.get()
