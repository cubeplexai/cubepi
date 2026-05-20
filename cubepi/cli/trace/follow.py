from __future__ import annotations

from pathlib import Path


def follow_run(path: Path, *, interval: float = 0.5,
               timeout: float | None = None) -> None:
    raise NotImplementedError
