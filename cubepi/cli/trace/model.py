"""Span wrapper + tree building over OTLP/JSON span dicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from cubepi.tracing import schema

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class Span:
    """Thin typed view over one OTLP/JSON span dict."""

    raw: dict[str, Any]

    @property
    def _ctx(self) -> dict[str, Any]:
        return self.raw.get("context") or {}

    @property
    def span_id(self) -> str | None:
        return self._ctx.get("span_id")

    @property
    def trace_id(self) -> str | None:
        return self._ctx.get("trace_id")

    @property
    def parent_id(self) -> str | None:
        return self.raw.get("parent_id")

    @property
    def name(self) -> str:
        return self.raw.get("name", "")

    @property
    def attributes(self) -> dict[str, Any]:
        return self.raw.get("attributes") or {}

    @property
    def start(self) -> datetime | None:
        return _parse_ts(self.raw.get("start_time"))

    @property
    def end(self) -> datetime | None:
        return _parse_ts(self.raw.get("end_time"))

    @property
    def duration_ms(self) -> float | None:
        s, e = self.start, self.end
        if s is None or e is None:
            return None
        return (e - s).total_seconds() * 1000.0

    @property
    def status_code(self) -> str:
        return (self.raw.get("status") or {}).get("status_code", "UNSET")

    @property
    def is_error(self) -> bool:
        return self.status_code == "ERROR"

    @property
    def is_aborted(self) -> bool:
        # The recorder sets a boolean cubepi.aborted attribute (and also an
        # error.type="cubepi.aborted" string). The boolean is the canonical
        # signal; OTel status stays UNSET for aborts, so don't gate on status.
        return self.attributes.get(schema.CUBEPI_ABORTED) is True

    @property
    def run_id(self) -> str | None:
        return self.attributes.get(schema.CUBEPI_RUN_ID)

    @property
    def operation(self) -> str | None:
        """gen_ai.operation.name — the reliable span classifier.

        Span *names* carry suffixes ("chat <model>", "execute_tool <tool>"),
        so classify on this attribute, not on ``name``. The cubepi.turn span
        has no operation name.
        """
        return self.attributes.get(schema.GEN_AI_OPERATION_NAME)

    @property
    def is_chat(self) -> bool:
        return self.operation == schema.OP_CHAT

    @property
    def is_tool(self) -> bool:
        return self.operation == schema.OP_EXECUTE_TOOL

    @property
    def is_invoke_agent(self) -> bool:
        return self.operation == schema.OP_INVOKE_AGENT

    @property
    def sort_start(self) -> datetime:
        """start_time, or epoch when missing — for stable ordering."""
        return self.start or _EPOCH

    @property
    def sort_end(self) -> datetime:
        """end_time (falling back to start, then epoch) — completion order."""
        return self.end or self.start or _EPOCH


@dataclass
class TreeNode:
    span: Span
    children: list["TreeNode"] = field(default_factory=list)
    orphan: bool = False


def build_forest(spans: list[Span]) -> list[TreeNode]:
    """Build a parent→child forest keyed on (trace_id, span_id).

    Spans whose parent_id is not present become roots; if they *had* a
    parent_id, they are flagged ``orphan``.
    """
    nodes: dict[tuple[str | None, str | None], TreeNode] = {
        (sp.trace_id, sp.span_id): TreeNode(span=sp) for sp in spans
    }
    roots: list[TreeNode] = []
    for (trace_id, _), node in nodes.items():
        parent_id = node.span.parent_id
        parent = nodes.get((trace_id, parent_id)) if parent_id else None
        if parent is not None:
            parent.children.append(node)
        else:
            node.orphan = parent_id is not None
            roots.append(node)
    _sort(roots)
    return roots


def _sort(nodes: list[TreeNode]) -> None:
    nodes.sort(key=lambda n: n.span.sort_start)
    for n in nodes:
        _sort(n.children)
