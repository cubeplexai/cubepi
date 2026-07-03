from __future__ import annotations

from cubepi.providers.base import Model


def models_are_equal(a: Model | None, b: Model | None) -> bool:
    """Return true when both model specs refer to the same provider/model id."""

    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.id == b.id and a.provider_id == b.provider_id
