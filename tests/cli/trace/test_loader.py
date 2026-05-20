from __future__ import annotations

import json
from pathlib import Path

import pytest

from cubepi.cli.trace.loader import (
    RunResolutionError,
    list_runs,
    load_run,
    resolve_run,
)


def _span(span_id, parent_id, name, start, run_id, **attrs):
    return {
        "name": name,
        "context": {"trace_id": "0xt", "span_id": span_id},
        "parent_id": parent_id,
        "start_time": start,
        "end_time": start,
        "status": {"status_code": "UNSET"},
        "attributes": {"cubepi.run_id": run_id, **attrs},
    }


def _write(path: Path, spans: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for sp in spans:
            f.write(json.dumps(sp) + "\n")


def test_resolve_by_path(tmp_path):
    f = tmp_path / "a.jsonl"
    _write(f, [_span("0x1", None, "invoke_agent", "2026-05-20T00:00:00Z", "r1")])
    assert resolve_run(str(f), tmp_path) == [f]


def test_resolve_by_run_id_merges_cross_midnight(tmp_path):
    f1 = tmp_path / "2026-05-19" / "r1.jsonl"
    f2 = tmp_path / "2026-05-20" / "r1.jsonl"
    _write(f1, [_span("0x1", None, "invoke_agent", "2026-05-19T23:59:59Z", "r1")])
    _write(f2, [_span("0x2", "0x1", "chat", "2026-05-20T00:00:01Z", "r1")])
    resolved = resolve_run("r1", tmp_path)
    assert sorted(resolved) == sorted([f1, f2])


def test_resolve_zero_match_errors(tmp_path):
    with pytest.raises(RunResolutionError):
        resolve_run("missing", tmp_path)


def test_load_run_skips_malformed(tmp_path):
    f = tmp_path / "2026-05-20" / "r1.jsonl"
    f.parent.mkdir(parents=True)
    with f.open("w") as fh:
        fh.write(
            json.dumps(_span("0x1", None, "invoke_agent", "2026-05-20T00:00:00Z", "r1"))
            + "\n"
        )
        fh.write("{ not json\n")
        fh.write(
            json.dumps(_span("0x2", "0x1", "chat", "2026-05-20T00:00:00.1Z", "r1"))
            + "\n"
        )
    spans, skipped = load_run([f])
    assert len(spans) == 2
    assert skipped == 1


def test_list_runs(tmp_path):
    f = tmp_path / "2026-05-20" / "r1.jsonl"
    _write(
        f,
        [
            _span("0x1", None, "invoke_agent", "2026-05-20T00:00:00Z", "r1"),
            _span("0x2", "0x1", "chat", "2026-05-20T00:00:00.1Z", "r1"),
        ],
    )
    runs = list_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].run_id == "r1"
    assert runs[0].span_count == 2


def test_list_runs_merges_cross_midnight(tmp_path):
    f1 = tmp_path / "2026-05-19" / "r1.jsonl"
    f2 = tmp_path / "2026-05-20" / "r1.jsonl"
    _write(f1, [_span("0x1", None, "invoke_agent", "2026-05-19T23:59:59Z", "r1")])
    _write(f2, [_span("0x2", "0x1", "chat", "2026-05-20T00:00:01Z", "r1")])
    runs = list_runs(tmp_path)
    assert len(runs) == 1  # one run, not two
    assert runs[0].span_count == 2
