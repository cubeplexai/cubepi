"""Aggregate spans by model or tool."""

from __future__ import annotations

from dataclasses import dataclass

from cubepi.cli.trace.model import Span
from cubepi.tracing import schema


@dataclass
class StatRow:
    key: str
    count: int
    durations_ms: list[float]
    errors: int
    aborted: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_tokens: int | None = None

    @property
    def error_rate(self) -> float:
        return self.errors / self.count if self.count else 0.0

    def percentile(self, p: float) -> float | None:
        vals = sorted(d for d in self.durations_ms if d is not None)
        if not vals:
            return None
        if len(vals) == 1:
            return vals[0]
        # Linear interpolation between closest ranks (so p50 of [100, 300] is 200).
        rank = (p / 100.0) * (len(vals) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(vals) - 1)
        frac = rank - lo
        return vals[lo] + (vals[hi] - vals[lo]) * frac


def aggregate(spans: list[Span], by: str) -> list[StatRow]:
    """Group spans by model (``chat`` spans) or tool (``execute_tool`` spans).

    ``by='model'`` includes token totals; ``by='tool'`` leaves token fields
    None — tokens live on chat spans, not tool spans.
    """
    want_tokens = by == "model"
    key_attr = schema.GEN_AI_REQUEST_MODEL if by == "model" else schema.GEN_AI_TOOL_NAME
    rows: dict[str, StatRow] = {}
    for sp in spans:
        # Classify on gen_ai.operation.name, not span name (names carry
        # "<model>"/"<tool>" suffixes).
        if by == "model" and not sp.is_chat:
            continue
        if by == "tool" and not sp.is_tool:
            continue
        key = str(sp.attributes.get(key_attr, "<unknown>"))
        row = rows.get(key)
        if row is None:
            row = StatRow(
                key=key,
                count=0,
                durations_ms=[],
                errors=0,
                aborted=0,
                input_tokens=0 if want_tokens else None,
                output_tokens=0 if want_tokens else None,
                cache_tokens=0 if want_tokens else None,
            )
            rows[key] = row
        row.count += 1
        if sp.duration_ms is not None:
            row.durations_ms.append(sp.duration_ms)
        if sp.is_error:
            row.errors += 1
        if sp.is_aborted:
            row.aborted += 1
        if want_tokens:
            a = sp.attributes
            # `or 0` guards against the attr being present but JSON null.
            row.input_tokens += int(a.get(schema.GEN_AI_USAGE_INPUT_TOKENS) or 0)
            row.output_tokens += int(a.get(schema.GEN_AI_USAGE_OUTPUT_TOKENS) or 0)
            # Cache total = read + creation (both are cache-related input tokens).
            row.cache_tokens += int(
                a.get(schema.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS) or 0
            ) + int(a.get(schema.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS) or 0)
    return sorted(rows.values(), key=lambda r: r.count, reverse=True)
