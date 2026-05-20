"""Tail a run file and print spans as a flat chronological event stream.

Spans are written by JsonlSpanExporter when they *end*, and children end
before their parents, so a live tree cannot be drawn correctly. Instead we
print one line per span in the order it completes. The run is considered
finished when the root ``invoke_agent`` span (``parent_id == null``) appears.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cubepi.cli.trace.model import Span


def iter_new_spans(path: Path, state: dict[str, Any]):
    """Yield Spans for newly-completed lines since the last call.

    ``state`` carries ``offset`` (bytes consumed) and ``buffer`` (an
    unterminated trailing line held until its newline arrives). Malformed
    complete lines are skipped and counted in ``state['skipped']``.
    """
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        fh.seek(state["offset"])
        chunk = fh.read()
        state["offset"] = fh.tell()
    data = state["buffer"] + chunk
    lines = data.split("\n")
    state["buffer"] = lines.pop()  # trailing remainder (no newline yet)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            state["skipped"] = state.get("skipped", 0) + 1
            continue
        if not isinstance(obj, dict):
            state["skipped"] = state.get("skipped", 0) + 1
            continue
        yield Span(obj)


def is_run_complete(span: Span) -> bool:
    return span.parent_id is None and span.is_invoke_agent


def format_event(span: Span) -> str:
    end = span.end.isoformat() if span.end else "?"
    dur = f"{span.duration_ms:.1f}ms" if span.duration_ms is not None else "…"
    parts = [end, span.name, dur]
    if span.is_error:
        parts.append("ERROR")
    if span.is_aborted:
        parts.append("aborted")
    return "  ".join(parts)


def follow_run(
    resolve_files: "Callable[[], list[Path]]",
    *,
    interval: float = 0.5,
    timeout: float | None = None,
) -> None:
    """Tail a run, printing each span as it completes.

    ``resolve_files`` is re-invoked every poll so a run crossing UTC midnight
    (a new ``<next-date>/<run_id>.jsonl`` file) is picked up without restarting.
    Each file keeps its own offset/buffer/skipped state. Exits when the root
    invoke_agent span appears, after ``timeout`` idle seconds, or on Ctrl-C.
    """
    states: dict[Path, dict[str, Any]] = {}
    last_activity = time.monotonic()
    try:
        while True:
            # Collect this poll's new spans across all files, then print in
            # end-time order so a cross-midnight run (old + new date file both
            # with backlog) stays chronological.
            batch: list[Span] = []
            for path in resolve_files():
                state = states.setdefault(
                    path, {"offset": 0, "buffer": "", "skipped": 0}
                )
                batch.extend(iter_new_spans(path, state))
            batch.sort(key=lambda s: s.sort_end)
            saw_any = bool(batch)
            done = False
            for span in batch:
                print(format_event(span), flush=True)
                if is_run_complete(span):
                    done = True
            if done:
                break
            now = time.monotonic()
            if saw_any:
                last_activity = now
            elif timeout is not None and (now - last_activity) >= timeout:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    skipped = sum(s.get("skipped", 0) for s in states.values())
    if skipped:
        print(f"({skipped} lines skipped (malformed))")
