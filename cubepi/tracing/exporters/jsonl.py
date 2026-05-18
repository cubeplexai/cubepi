"""Append-only JSONL span exporter.

One span per line, OTLP/JSON-shaped via ``span.to_json()``. Files are
sharded ``<directory>/<YYYY-MM-DD>/<run_id>.jsonl`` so each run lands
in its own file and per-day directories don't grow unbounded.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from cubepi.tracing.schema import CUBEPI_RUN_ID


class JsonlSpanExporter(SpanExporter):
    """Write each ReadableSpan as one JSON line.

    Files: ``<directory>/<YYYY-MM-DD>/<run_id>.jsonl``. The date used for
    the subdirectory is the span's start time (UTC). The ``run_id`` comes
    from the ``cubepi.run_id`` attribute set by the cubepi Recorder;
    spans without that attribute fall back to ``"unknown-run"``.

    Permissions: files are created mode ``0o600`` (user-only).
    """

    def __init__(
        self,
        directory: str | os.PathLike[str] = "./cubepi-traces",
    ) -> None:
        self._directory = Path(directory)
        self._shutdown = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._shutdown:
            return SpanExportResult.FAILURE
        if not spans:
            return SpanExportResult.SUCCESS

        # Group by destination file so we open each at most once per
        # export batch.
        grouped: dict[Path, list[str]] = {}
        for span in spans:
            try:
                path = self._path_for(span)
                line = self._encode(span)
            except Exception:
                # Per the SpanExporter contract: never raise from export.
                return SpanExportResult.FAILURE
            grouped.setdefault(path, []).append(line)

        try:
            for path, lines in grouped.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                self._append(path, lines)
        except Exception:
            return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._shutdown = True

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        # Writes are synchronous + ``open(..., "a")`` flushes on close.
        # No background buffer to drain.
        return True

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _path_for(self, span: ReadableSpan) -> Path:
        start_ns = span.start_time or 0
        if start_ns:
            dt = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=timezone.utc)
            date_dir = dt.strftime("%Y-%m-%d")
        else:
            date_dir = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        run_id = (span.attributes or {}).get(CUBEPI_RUN_ID, "unknown-run")
        # Sanitize: run_ids are uuids in practice, but defend against odd
        # characters slipping through extra_attrs.
        safe = _safe_filename(str(run_id))
        return self._directory / date_dir / f"{safe}.jsonl"

    @staticmethod
    def _encode(span: ReadableSpan) -> str:
        # ``span.to_json`` returns indented JSON by default; collapse to
        # one line so each file remains jsonl-grep-friendly.
        raw = span.to_json(indent=None)
        # Some SDK versions emit a trailing newline; strip.
        raw = raw.rstrip("\n")
        # Sanity check: must be one line.
        if "\n" in raw:
            # Re-serialize via stdlib to flatten if SDK ever changes.
            data: Any = json.loads(raw)
            raw = json.dumps(data, separators=(",", ":"))
        return raw

    @staticmethod
    def _append(path: Path, lines: list[str]) -> None:
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        fd = os.open(path, flags, 0o600)
        try:
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line)
                    f.write("\n")
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise


def _safe_filename(name: str) -> str:
    # Allow alnum, dash, underscore, dot. Replace everything else.
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "unknown-run"
