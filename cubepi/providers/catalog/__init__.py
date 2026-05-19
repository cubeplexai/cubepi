"""Provider preset catalog. See docs/dev/specs/2026-05-19-llm-provider-platform-design.md §3.6."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

from cubepi.providers.catalog.types import ProviderPreset

_DATA_FILE = Path(__file__).parent / "data" / "providers.yaml"


@cache
def _load() -> dict[str, ProviderPreset]:
    with _DATA_FILE.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise ValueError(f"providers.yaml must be a top-level list, got {type(raw)}")
    presets: dict[str, ProviderPreset] = {}
    for idx, entry in enumerate(raw):
        try:
            preset = ProviderPreset.model_validate(entry)
        except Exception as exc:
            raise ValueError(f"providers.yaml entry #{idx}: {exc}") from exc
        if preset.slug in presets:
            raise ValueError(f"providers.yaml: duplicate slug {preset.slug!r}")
        presets[preset.slug] = preset
    return presets


def list_provider_presets() -> list[ProviderPreset]:
    """All registered provider presets, in catalog order."""
    return list(_load().values())


def get_provider_preset(slug: str) -> ProviderPreset:
    """Look up by slug. Raises KeyError if not found."""
    presets = _load()
    if slug not in presets:
        raise KeyError(slug)
    return presets[slug]


__all__ = ["list_provider_presets", "get_provider_preset"]
