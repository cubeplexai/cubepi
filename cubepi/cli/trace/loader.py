"""Discover trace files, resolve a trace id to file(s), read spans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cubepi.cli.trace.model import Span
from cubepi.tracing import schema

DEFAULT_DIR = Path("./cubepi-traces")
_MIN = datetime.min.replace(tzinfo=timezone.utc)


class RunResolutionError(Exception):
    """Raised when a trace id / path cannot be resolved to any file."""


def resolve_run(arg: str, directory: Path) -> list[Path]:
    """Resolve a CLI argument to one or more JSONL files.

    A ``.jsonl`` path is used directly. Otherwise ``arg`` is a trace id and we
    glob ``<dir>/*/{trace_id}.jsonl`` across date subdirs — a trace that
    crosses UTC midnight is split across date dirs, so ALL matches are
    returned and later merged. Zero matches is an error.
    """
    path = Path(arg)
    if path.suffix == ".jsonl" and path.is_file():
        return [path]
    matches = sorted(directory.glob(f"*/{arg}.jsonl"))
    if matches:
        return matches
    # No exact trace id. `ls` truncates ids, so accept any prefix that names
    # exactly one trace; an ambiguous prefix is reported with its candidates.
    by_trace: dict[str, list[Path]] = {}
    for f in sorted(directory.glob(f"*/{arg}*.jsonl")):
        by_trace.setdefault(f.stem, []).append(f)
    if len(by_trace) == 1:
        (only,) = by_trace.values()
        return sorted(only)
    if len(by_trace) > 1:
        candidates = ", ".join(sorted(by_trace))
        raise RunResolutionError(
            f"trace prefix {arg!r} is ambiguous under {directory}: matches {candidates}"
        )
    raise RunResolutionError(
        f"no trace file for trace {arg!r} under {directory} (try `cubepi trace ls`)"
    )


def load_run(files: list[Path]) -> tuple[list[Span], int]:
    """Read all spans from the given files; return (spans, skipped_count).

    Malformed lines are skipped and tallied, never fatal. Spans are returned
    sorted by start time so a merged cross-midnight run reads in order.
    """
    spans: list[Span] = []
    skipped = 0
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if not isinstance(obj, dict):
                    skipped += 1
                    continue
                spans.append(Span(obj))
    spans.sort(key=lambda s: s.sort_start)
    return spans, skipped


def _run_prompt(spans: list[Span]) -> str | None:
    """The most recent human turn that drove the run, for `ls` identification.

    Reads ``gen_ai.input.messages`` off the root invoke_agent span (only
    present when the Tracer records content). Returns the last user text, so
    successive runs in one thread stay distinguishable. ``None`` when content
    isn't recorded or no user text is present.
    """
    root = next(
        (s for s in spans if s.is_invoke_agent and not s.parent_id), None
    ) or next((s for s in spans if s.is_invoke_agent), None)
    candidates = [root] if root is not None else spans
    for sp in candidates:
        raw = sp.attributes.get(schema.GEN_AI_INPUT_MESSAGES)
        if not raw:
            continue
        try:
            messages = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(messages, list):
            continue
        last: str | None = None
        for m in messages:
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            parts = m.get("parts")
            if isinstance(parts, list):
                text = " ".join(
                    str(p.get("content", ""))
                    for p in parts
                    if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
            else:
                content = m.get("content")
                text = content.strip() if isinstance(content, str) else ""
            if text:
                last = text
        if last is not None:
            return last
    return None


@dataclass
class RunSummary:
    trace_id: str
    files: list[Path]
    start: datetime | None
    span_count: int
    has_error: bool
    duration_ms: float | None
    prompt: str | None


def list_runs(directory: Path, limit: int | None = None) -> list[RunSummary]:
    """Summarize each trace, newest first.

    Files are grouped by trace_id (stem), so a trace split across two date
    dirs (crossed UTC midnight) is summarized as ONE trace, not two. The
    span_count spans the whole trace — parent run plus nested subagent runs.
    """
    by_trace: dict[str, list[Path]] = {}
    for f in directory.glob("*/*.jsonl"):
        by_trace.setdefault(f.stem, []).append(f)
    summaries: list[RunSummary] = []
    for trace_id, files in by_trace.items():
        spans, _ = load_run(sorted(files))
        if not spans:
            continue
        starts = [s.start for s in spans if s.start is not None]
        ends = [s.end for s in spans if s.end is not None]
        start = min(starts) if starts else None
        duration = None
        if start is not None and ends:
            duration = (max(ends) - start).total_seconds() * 1000.0
        summaries.append(
            RunSummary(
                trace_id=trace_id,
                files=sorted(files),
                start=start,
                span_count=len(spans),
                has_error=any(s.is_error for s in spans),
                duration_ms=duration,
                prompt=_run_prompt(spans),
            )
        )
    summaries.sort(key=lambda s: s.start or _MIN, reverse=True)
    return summaries[:limit] if limit else summaries
