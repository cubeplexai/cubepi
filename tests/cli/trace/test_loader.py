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


def test_resolve_by_trace_id_merges_cross_midnight(tmp_path):
    f1 = tmp_path / "2026-05-19" / "r1.jsonl"
    f2 = tmp_path / "2026-05-20" / "r1.jsonl"
    _write(f1, [_span("0x1", None, "invoke_agent", "2026-05-19T23:59:59Z", "r1")])
    _write(f2, [_span("0x2", "0x1", "chat", "2026-05-20T00:00:01Z", "r1")])
    resolved = resolve_run("r1", tmp_path)
    assert sorted(resolved) == sorted([f1, f2])


def test_resolve_zero_match_errors(tmp_path):
    with pytest.raises(RunResolutionError):
        resolve_run("missing", tmp_path)


def test_resolve_by_unique_prefix(tmp_path):
    # `ls` truncates trace ids, so a copied prefix must resolve to its one trace.
    f = tmp_path / "2026-05-20" / "66f1806f-4c90-4d50.jsonl"
    _write(f, [_span("0x1", None, "invoke_agent", "2026-05-20T00:00:00Z", "r1")])
    assert resolve_run("66f1806f", tmp_path) == [f]


def test_resolve_prefix_merges_cross_midnight(tmp_path):
    f1 = tmp_path / "2026-05-19" / "abc123def.jsonl"
    f2 = tmp_path / "2026-05-20" / "abc123def.jsonl"
    _write(f1, [_span("0x1", None, "invoke_agent", "2026-05-19T23:59:59Z", "r1")])
    _write(f2, [_span("0x2", "0x1", "chat", "2026-05-20T00:00:01Z", "r1")])
    assert sorted(resolve_run("abc123", tmp_path)) == sorted([f1, f2])


def test_resolve_ambiguous_prefix_errors(tmp_path):
    _write(
        tmp_path / "2026-05-20" / "abc111.jsonl", [_span("0x1", None, "x", None, "r1")]
    )
    _write(
        tmp_path / "2026-05-20" / "abc222.jsonl", [_span("0x2", None, "y", None, "r2")]
    )
    with pytest.raises(RunResolutionError, match="ambiguous"):
        resolve_run("abc", tmp_path)


def test_resolve_exact_match_preferred_over_prefix(tmp_path):
    # An exact trace id must not be shadowed by a longer-named sibling trace.
    exact = tmp_path / "2026-05-20" / "run1.jsonl"
    _write(exact, [_span("0x1", None, "x", None, "r1")])
    _write(
        tmp_path / "2026-05-20" / "run1extra.jsonl",
        [_span("0x2", None, "y", None, "r2")],
    )
    assert resolve_run("run1", tmp_path) == [exact]


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


def test_load_run_skips_non_dict_json(tmp_path):
    f = tmp_path / "2026-05-20" / "r1.jsonl"
    f.parent.mkdir(parents=True)
    with f.open("w") as fh:
        fh.write("null\n")  # valid JSON, but not a span object
        fh.write("[1, 2]\n")  # valid JSON array, not a span object
        fh.write(
            json.dumps(_span("0x1", None, "invoke_agent", "2026-05-20T00:00:00Z", "r1"))
            + "\n"
        )
    spans, skipped = load_run([f])
    assert len(spans) == 1
    assert skipped == 2


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
    assert runs[0].trace_id == "r1"
    assert runs[0].span_count == 2


def test_list_runs_extracts_prompt(tmp_path):
    msgs = json.dumps(
        [
            {"role": "user", "parts": [{"type": "text", "content": "first question"}]},
            {"role": "assistant", "parts": [{"type": "text", "content": "answer"}]},
            {
                "role": "user",
                "parts": [{"type": "text", "content": "北京明天天气如何\n\n"}],
            },
        ]
    )
    f = tmp_path / "2026-05-20" / "r1.jsonl"
    _write(
        f,
        [
            _span(
                "0x1",
                None,
                "invoke_agent",
                "2026-05-20T00:00:00Z",
                "r1",
                **{
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.input.messages": msgs,
                },
            ),
        ],
    )
    runs = list_runs(tmp_path)
    # Most recent human turn, so multiple runs in a thread stay distinguishable.
    assert runs[0].prompt == "北京明天天气如何"


def test_list_runs_prompt_from_string_content(tmp_path):
    # Older message shape: content is a bare string, not parts.
    msgs = json.dumps([{"role": "user", "content": "  plain string prompt  "}])
    f = tmp_path / "2026-05-20" / "r1.jsonl"
    _write(
        f,
        [
            _span(
                "0x1",
                None,
                "invoke_agent",
                "2026-05-20T00:00:00Z",
                "r1",
                **{
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.input.messages": msgs,
                },
            )
        ],
    )
    assert list_runs(tmp_path)[0].prompt == "plain string prompt"


def test_list_runs_prompt_handles_bad_messages(tmp_path):
    # Malformed JSON and a non-list payload both yield no prompt, never raise.
    f1 = tmp_path / "2026-05-20" / "r1.jsonl"
    _write(
        f1,
        [
            _span(
                "0x1",
                None,
                "invoke_agent",
                "2026-05-20T00:00:00Z",
                "r1",
                **{
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.input.messages": "{ not json",
                },
            )
        ],
    )
    f2 = tmp_path / "2026-05-20" / "r2.jsonl"
    _write(
        f2,
        [
            _span(
                "0x2",
                None,
                "invoke_agent",
                "2026-05-20T00:00:01Z",
                "r2",
                **{
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.input.messages": '"a string"',
                },
            )
        ],
    )
    prompts = {r.trace_id: r.prompt for r in list_runs(tmp_path)}
    assert prompts == {"r1": None, "r2": None}


def test_list_runs_prompt_none_without_content(tmp_path):
    f = tmp_path / "2026-05-20" / "r1.jsonl"
    _write(
        f,
        [
            _span(
                "0x1",
                None,
                "invoke_agent",
                "2026-05-20T00:00:00Z",
                "r1",
                **{"gen_ai.operation.name": "invoke_agent"},
            )
        ],
    )
    assert list_runs(tmp_path)[0].prompt is None


def test_list_runs_merges_cross_midnight(tmp_path):
    f1 = tmp_path / "2026-05-19" / "r1.jsonl"
    f2 = tmp_path / "2026-05-20" / "r1.jsonl"
    _write(f1, [_span("0x1", None, "invoke_agent", "2026-05-19T23:59:59Z", "r1")])
    _write(f2, [_span("0x2", "0x1", "chat", "2026-05-20T00:00:01Z", "r1")])
    runs = list_runs(tmp_path)
    assert len(runs) == 1  # one run, not two
    assert runs[0].span_count == 2


def test_run_prompt_prefers_root_over_subagent_invoke_agent():
    # A trace file now holds the parent run PLUS nested subagent runs, each
    # with its own invoke_agent span. The prompt shown in `ls` must come from
    # the ROOT (parent-less) invoke_agent, not a subagent's. The subagent span
    # is placed FIRST in the list so this distinguishes the parent-less filter
    # from plain iteration order (which would otherwise pick the subagent).
    from cubepi.cli.trace.loader import _run_prompt
    from cubepi.cli.trace.model import Span

    sub = Span(
        _span(
            "0x9",
            "0x5",
            "invoke_agent",
            "2026-05-20T00:00:02Z",
            "r-sub",
            **{
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.input.messages": json.dumps(
                    [{"role": "user", "content": "subagent prompt"}]
                ),
            },
        )
    )
    root = Span(
        _span(
            "0x1",
            None,
            "invoke_agent",
            "2026-05-20T00:00:00Z",
            "r-root",
            **{
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.input.messages": json.dumps(
                    [{"role": "user", "content": "root prompt"}]
                ),
            },
        )
    )
    assert _run_prompt([sub, root]) == "root prompt"


def test_list_runs_filters_by_meta(tmp_path):
    f1 = tmp_path / "2026-05-20" / "tA.jsonl"
    f2 = tmp_path / "2026-05-20" / "tB.jsonl"
    _write(f1, [_span("0x1", None, "invoke_agent", "2026-05-20T00:00:00Z", "rA",
                      **{"gen_ai.operation.name": "invoke_agent",
                         "cubepi.metadata.conversation_id": "conv_A",
                         "cubepi.metadata.user_id": "u1"})])
    _write(f2, [_span("0x2", None, "invoke_agent", "2026-05-20T00:00:01Z", "rB",
                      **{"gen_ai.operation.name": "invoke_agent",
                         "cubepi.metadata.conversation_id": "conv_B"})])
    # metadata surfaced on the summary
    by_id = {r.trace_id: r for r in list_runs(tmp_path)}
    assert by_id["tA"].metadata == {"conversation_id": "conv_A", "user_id": "u1"}
    assert by_id["tB"].metadata == {"conversation_id": "conv_B"}
    # single-key filter
    assert [r.trace_id for r in list_runs(tmp_path, meta={"conversation_id": "conv_A"})] == ["tA"]
    # AND: every key must match
    assert [r.trace_id for r in
            list_runs(tmp_path, meta={"conversation_id": "conv_A", "user_id": "u1"})] == ["tA"]
    assert list_runs(tmp_path, meta={"conversation_id": "conv_A", "user_id": "nope"}) == []
    # unknown key matches nothing
    assert list_runs(tmp_path, meta={"missing": "x"}) == []


def test_filter_spans_by_meta():
    from cubepi.cli.trace.loader import filter_spans_by_meta
    from cubepi.cli.trace.model import Span

    def sp(trace_id, span_id, parent, name, **attrs):
        return Span({
            "name": name,
            "context": {"trace_id": trace_id, "span_id": span_id},
            "parent_id": parent,
            "start_time": "2026-05-20T00:00:00Z",
            "end_time": "2026-05-20T00:00:00Z",
            "status": {"status_code": "UNSET"},
            "attributes": attrs,
        })

    a_root = sp("0xA", "0x1", None, "invoke_agent",
                **{"gen_ai.operation.name": "invoke_agent", "cubepi.metadata.user_id": "u1"})
    a_chat = sp("0xA", "0x2", "0x1", "chat", **{"gen_ai.operation.name": "chat"})
    b_root = sp("0xB", "0x3", None, "invoke_agent",
                **{"gen_ai.operation.name": "invoke_agent", "cubepi.metadata.user_id": "u2"})
    spans = [a_root, a_chat, b_root]
    kept = filter_spans_by_meta(spans, {"user_id": "u1"})
    assert {s.span_id for s in kept} == {"0x1", "0x2"}
    # empty filter is a no-op (same list)
    assert filter_spans_by_meta(spans, {}) is spans
