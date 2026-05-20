"""rich-based rendering. rich is imported lazily so the package import does
not require the trace-cli extra until something is actually rendered."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from cubepi.cli.trace.loader import RunSummary
from cubepi.cli.trace.model import Span, TreeNode
from cubepi.cli.trace.stats import StatRow
from cubepi.tracing import schema

if TYPE_CHECKING:  # pragma: no cover
    from rich.console import Console


class RichMissingError(Exception):
    """Raised when rich is needed but not installed."""


def _require_rich() -> None:
    try:
        import rich  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RichMissingError(
            "this command needs rich. Install it via: pip install cubepi[trace-cli]"
        ) from exc


def _console(record: bool = False) -> "Console":
    _require_rich()
    from rich.console import Console

    return Console(record=record)


def _dur(span: Span) -> str:
    d = span.duration_ms
    return f"{d:.1f}ms" if d is not None else "…"


def _node_label(node: TreeNode) -> str:
    sp = node.span
    parts = [f"[bold]{sp.name}[/bold]", _dur(sp)]
    if sp.is_chat:
        model = sp.attributes.get(schema.GEN_AI_REQUEST_MODEL)
        if model:
            parts.append(str(model))
        in_t = sp.attributes.get(schema.GEN_AI_USAGE_INPUT_TOKENS)
        out_t = sp.attributes.get(schema.GEN_AI_USAGE_OUTPUT_TOKENS)
        if in_t is not None or out_t is not None:
            parts.append(f"tok {in_t or 0}/{out_t or 0}")
    elif sp.is_tool:
        tool = sp.attributes.get(schema.GEN_AI_TOOL_NAME)
        if tool:
            parts.append(str(tool))
    if sp.is_error:
        parts.append("[red]ERROR[/red]")
    if sp.is_aborted:
        parts.append("[yellow]aborted[/yellow]")
    if node.orphan:
        parts.append("[dim](orphan)[/dim]")
    return "  ".join(parts)


def _build_rich_tree(roots: list[TreeNode], verbose: bool, content: bool):
    from rich.tree import Tree

    forest = Tree("trace")

    def add(parent_tree, node: TreeNode) -> None:
        branch = parent_tree.add(_node_label(node))
        if verbose:
            for k, v in node.span.attributes.items():
                branch.add(f"[dim]{k}[/dim] = {v!r}")
        if content:
            _add_content(branch, node.span)
        for child in node.children:
            add(branch, child)

    for root in roots:
        add(forest, root)
    return forest


_CONTENT_KEYS = (
    schema.GEN_AI_INPUT_MESSAGES,
    schema.GEN_AI_OUTPUT_MESSAGES,
    schema.GEN_AI_SYSTEM_INSTRUCTIONS,
    schema.GEN_AI_TOOL_CALL_ARGUMENTS,
    schema.GEN_AI_TOOL_CALL_RESULT,
)


def _add_content(branch, span: Span) -> None:
    for key in _CONTENT_KEYS:
        raw = span.attributes.get(key)
        if raw is None:
            continue
        try:
            decoded = json.loads(raw) if isinstance(raw, str) else raw
            pretty = json.dumps(decoded, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pretty = str(raw)
        branch.add(f"[cyan]{key}[/cyan]\n{pretty}")


def render_tree(
    roots: list[TreeNode], *, verbose: bool = False, content: bool = False
) -> None:
    console = _console()
    console.print(_build_rich_tree(roots, verbose, content))


def render_tree_to_text(
    roots: list[TreeNode], *, verbose: bool = False, content: bool = False
) -> str:
    console = _console(record=True)
    console.print(_build_rich_tree(roots, verbose, content))
    return console.export_text()


def render_runs(runs: list[RunSummary]) -> None:
    from rich.table import Table

    console = _console()
    table = Table(title="cubepi runs")
    for col in ("started", "run_id", "spans", "status", "duration"):
        table.add_column(col)
    for r in runs:
        started = r.start.isoformat() if r.start else "?"
        dur = f"{r.duration_ms:.0f}ms" if r.duration_ms is not None else "?"
        status = "[red]error[/red]" if r.has_error else "ok"
        table.add_row(started, r.run_id, str(r.span_count), status, dur)
    console.print(table)


def render_stats(rows: list[StatRow], by: str) -> None:
    from rich.table import Table

    console = _console()
    table = Table(title=f"stats by {by}")
    table.add_column(by)
    table.add_column("calls")
    table.add_column("p50")
    table.add_column("p95")
    table.add_column("err%")
    if by == "model":
        table.add_column("in_tok")
        table.add_column("out_tok")
        table.add_column("cache_tok")
    else:
        table.add_column("aborted")
    for row in rows:
        p50 = row.percentile(50)
        p95 = row.percentile(95)
        cells = [
            row.key,
            str(row.count),
            f"{p50:.0f}ms" if p50 is not None else "?",
            f"{p95:.0f}ms" if p95 is not None else "?",
            f"{row.error_rate * 100:.0f}%",
        ]
        if by == "model":
            cells += [
                str(row.input_tokens),
                str(row.output_tokens),
                str(row.cache_tokens),
            ]
        else:
            cells.append(str(row.aborted))
        table.add_row(*cells)
    console.print(table)
