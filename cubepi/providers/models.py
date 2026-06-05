"""Thinking-level validation and model comparison utilities.

Mirrors pi-agent-core's ``getSupportedThinkingLevels``, ``clampThinkingLevel``,
and ``modelsAreEqual`` functions.
"""

from __future__ import annotations

from cubepi.providers.base import Model, ThinkingLevel

# Ordered list of all thinking levels from lowest to highest.
THINKING_LEVELS: list[ThinkingLevel] = [
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]


def get_supported_thinking_levels(model: Model) -> list[ThinkingLevel]:
    """Return the thinking levels supported by *model*.

    * Non-reasoning models only support ``["off"]``.
    * For reasoning models, levels are filtered through the model's
      ``thinking_level_map``.  A level mapped to ``None`` is unsupported.
      ``"xhigh"`` is only included when it has an explicit (non-None) mapping.
      All other levels are included by default when the map omits them.
    """
    if not model.reasoning:
        return ["off"]

    tlm = model.thinking_level_map

    def _is_supported(level: ThinkingLevel) -> bool:
        if tlm is not None:
            mapped = tlm.get(level)
            if mapped is None and level in tlm:
                # Explicitly mapped to None -> unsupported
                return False
            # "xhigh" requires an explicit mapping to be available
            if level == "xhigh":
                return level in tlm and tlm[level] is not None
        else:
            # No map at all: xhigh is excluded by default
            if level == "xhigh":
                return False
        return True

    return [lvl for lvl in THINKING_LEVELS if _is_supported(lvl)]


def clamp_thinking_level(model: Model, level: ThinkingLevel) -> ThinkingLevel:
    """Clamp *level* to the nearest supported level for *model*.

    If *level* is already supported, return it unchanged.  Otherwise search
    upward first (higher intensity), then downward, through the ordered level
    list to find the closest available level.
    """
    available = get_supported_thinking_levels(model)

    if level in available:
        return level

    # Unknown level -> fall back to first available
    if level not in THINKING_LEVELS:
        return available[0] if available else "off"

    requested_idx = THINKING_LEVELS.index(level)

    # Search downward first (prefer cheaper/lower intensity)
    for i in range(requested_idx - 1, -1, -1):
        candidate = THINKING_LEVELS[i]
        if candidate in available:
            return candidate

    # Then upward
    for i in range(requested_idx + 1, len(THINKING_LEVELS)):
        candidate = THINKING_LEVELS[i]
        if candidate in available:
            return candidate

    return available[0] if available else "off"


def models_are_equal(a: Model | None, b: Model | None) -> bool:
    """Return ``True`` if *a* and *b* refer to the same model.

    Comparison is by ``id`` and ``provider_id``.  Returns ``False`` when either
    argument is ``None``.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.id == b.id and a.provider_id == b.provider_id
