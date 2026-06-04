"""Typed provider errors.

cubepi providers catch raw SDK exceptions and re-raise as one of these
subclasses, so downstream callers can ``except cubepi.errors.X`` instead
of pattern-matching on SDK-specific strings or status codes.

Adding a new subclass: add it at the bottom of the file (never reorder)
and re-export from ``cubepi/__init__.py`` and ``__all__``.
"""

from __future__ import annotations

import math
import re
from typing import NoReturn

from cubepi.providers.base import Message, Model


class ProviderError(Exception):
    """Base class for typed cubepi provider errors.

    Always carries provider / model context. ``raw_exception`` is the
    original SDK exception (kept on ``__cause__`` via ``raise … from``).
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.status_code = status_code
        super().__init__(message or self.__class__.__name__)


class ContextLengthExceeded(ProviderError):
    """The request exceeded the model's context window.

    ``tokens_in`` is an estimate (chars/4) of the prompt size at the time
    of the failure; ``context_window`` is the model's advertised window.
    Both may be None if the provider couldn't measure them.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        tokens_in: int | None = None,
        context_window: int | None = None,
    ) -> None:
        self.tokens_in = tokens_in
        self.context_window = context_window
        super().__init__(
            message, provider=provider, model=model, status_code=status_code
        )


class RateLimited(ProviderError):
    """Provider rate-limit / quota error.

    ``retry_after`` is the recommended retry delay in seconds, parsed from
    the SDK response when available.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            message, provider=provider, model=model, status_code=status_code
        )


class ProviderAuthFailed(ProviderError):
    """API-key invalid, account suspended, or 401/403 with no quota wording."""


class ProviderUnavailable(ProviderError):
    """5xx, timeout, or connection failure."""


class ProviderBadRequest(ProviderError):
    """Other 4xx provider error (model_not_found, schema rejection, etc.)."""


# ---------------------------------------------------------------------------
# Heuristics: turn a raw SDK exception into one of the typed errors above.
# ---------------------------------------------------------------------------

_CONTEXT_LENGTH_PATTERNS = (
    re.compile(r"maximum context length", re.IGNORECASE),
    re.compile(r"context.{0,10}length.{0,20}exceed", re.IGNORECASE),
    re.compile(r"too many tokens", re.IGNORECASE),
    re.compile(r"prompt is too long", re.IGNORECASE),
    re.compile(r"reduce.{0,10}messages", re.IGNORECASE),
)

_RATE_LIMIT_PATTERNS = (
    re.compile(r"rate ?limit", re.IGNORECASE),
    re.compile(r"quota (?:exceed|exhaust|limit|reach)", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
)


def _status_of(exc: BaseException) -> int | None:
    """Best-effort status code extraction from an SDK exception."""

    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    resp = getattr(exc, "response", None)
    if resp is not None:
        rc = getattr(resp, "status_code", None)
        if isinstance(rc, int):
            return rc
    return None


def _estimate_input_tokens(messages: list[Message] | None) -> int | None:
    """Rough chars/4 estimate over message text, used only for diagnostics.

    Returns None when ``messages`` is empty or missing.
    """

    if not messages:
        return None
    total = 0
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    total += len(text)
                else:
                    text = getattr(block, "content", None)
                    if isinstance(text, str):
                        total += len(text)
        elif isinstance(content, str):
            total += len(content)
    if total == 0:
        return None
    return max(1, math.ceil(total / 4))


def _retry_after_from(exc: BaseException) -> float | None:
    """Pull retry-after seconds from an SDK exception's response headers."""

    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    val = headers.get("retry-after") or headers.get("Retry-After")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def classify_and_raise(
    exc: BaseException,
    *,
    model: Model,
    messages: list[Message] | None = None,
) -> NoReturn:
    """Inspect a raw SDK exception and raise the typed cubepi error.

    Heuristics (first match wins):
      1. Explicit context-length wording → ContextLengthExceeded.
      2. status==400 + estimated tokens_in within 5% of model.context_window
         → ContextLengthExceeded (covers Volcano ARK's opaque
         ``InvalidParameter`` 400).
      3. status==429 OR quota/rate-limit wording → RateLimited.
      4. status==401 / 403 → ProviderAuthFailed.
      5. TimeoutError / ConnectionError / 5xx → ProviderUnavailable.
      6. Any other 4xx → ProviderBadRequest.
      7. Else: re-raise the original (caller should let it propagate).

    ``raise classify_and_raise(...) from exc`` is the idiomatic call site.
    """

    # Already typed — re-raise unchanged so callers that catch ProviderError
    # don't get double-wrapped.
    if isinstance(exc, ProviderError):
        raise exc

    msg = str(exc) or getattr(exc, "message", "")
    status = _status_of(exc)
    provider = model.provider
    model_id = model.id

    tokens_in = _estimate_input_tokens(messages)
    context_window = model.context_window if model.context_window else None

    for pat in _CONTEXT_LENGTH_PATTERNS:
        if pat.search(msg):
            raise ContextLengthExceeded(
                msg,
                provider=provider,
                model=model_id,
                status_code=status,
                tokens_in=tokens_in,
                context_window=context_window,
            ) from exc

    if (
        status == 400
        and tokens_in is not None
        and context_window is not None
        and tokens_in >= int(context_window * 0.95)
    ):
        raise ContextLengthExceeded(
            msg,
            provider=provider,
            model=model_id,
            status_code=status,
            tokens_in=tokens_in,
            context_window=context_window,
        ) from exc

    if status == 429 or any(pat.search(msg) for pat in _RATE_LIMIT_PATTERNS):
        raise RateLimited(
            msg,
            provider=provider,
            model=model_id,
            status_code=status,
            retry_after=_retry_after_from(exc),
        ) from exc

    if status in (401, 403):
        raise ProviderAuthFailed(
            msg, provider=provider, model=model_id, status_code=status
        ) from exc

    if isinstance(exc, (TimeoutError, ConnectionError)):
        raise ProviderUnavailable(
            msg, provider=provider, model=model_id, status_code=status
        ) from exc

    if status is not None and 500 <= status < 600:
        raise ProviderUnavailable(
            msg, provider=provider, model=model_id, status_code=status
        ) from exc

    if status is not None and 400 <= status < 500:
        raise ProviderBadRequest(
            msg, provider=provider, model=model_id, status_code=status
        ) from exc

    # Unknown → let original propagate. Callers might still need to handle it.
    raise exc


__all__ = [
    "ProviderError",
    "ContextLengthExceeded",
    "RateLimited",
    "ProviderAuthFailed",
    "ProviderUnavailable",
    "ProviderBadRequest",
    "classify_and_raise",
]
