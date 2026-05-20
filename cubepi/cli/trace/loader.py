"""Discover trace files, resolve a run id to file(s), read spans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cubepi.cli.trace.model import Span

DEFAULT_DIR = Path("./cubepi-traces")
_MIN = datetime.min.replace(tzinfo=timezone.utc)


class RunResolutionError(Exception):
    """Raised when a run id / path cannot be resolved to any file."""


def resolve_run(arg: str, directory: Path) -> list[Path]:
    """Resolve a CLI argument to one or more JSONL files.

    A ``.jsonl`` path is used directly. Otherwise ``arg`` is a run id and we
    glob ``<dir>/*/{run_id}.jsonl`` across date subdirs — a run that crosses
    UTC midnight is split across date dirs, so ALL matches are returned and
    later merged. Zero matches is an error.
    """
    path = Path(arg)
    if path.suffix == ".jsonl" and path.is_file():
        return [path]
    matches = sorted(directory.glob(f"*/{arg}.jsonl"))
    if not matches:
        raise RunResolutionError(
            f"no trace file for run {arg!r} under {directory} (try `cubepi trace ls`)"
        )
    return matches


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
                    spans.append(Span(json.loads(line)))
                except json.JSONDecodeError:
                    skipped += 1
    spans.sort(key=lambda s: s.sort_start)
    return spans, skipped


@dataclass
class RunSummary:
    run_id: str
    files: list[Path]
    start: datetime | None
    span_count: int
    has_error: bool
    duration_ms: float | None


def list_runs(directory: Path, limit: int | None = None) -> list[RunSummary]:
    """Summarize each run, newest first.

    Files are grouped by run_id (stem), so a run split across two date dirs
    (crossed UTC midnight) is summarized as ONE run, not two.
    """
    by_run: dict[str, list[Path]] = {}
    for f in directory.glob("*/*.jsonl"):
        by_run.setdefault(f.stem, []).append(f)
    summaries: list[RunSummary] = []
    for run_id, files in by_run.items():
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
                run_id=run_id,
                files=sorted(files),
                start=start,
                span_count=len(spans),
                has_error=any(s.is_error for s in spans),
                duration_ms=duration,
            )
        )
    summaries.sort(key=lambda s: s.start or _MIN, reverse=True)
    return summaries[:limit] if limit else summaries
