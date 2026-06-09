from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from cubepi.errors import (
    ContextLengthExceeded,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
)
from cubepi.providers.base import (
    AssistantMessage,
    BoundModel,
    Message,
    MessageStream,
    Model,
    Provider,
    StreamEvent,
    StreamOptions,
    ThinkingBudgets,
    ThinkingLevel,
    ToolDefinition,
    Usage,
)

try:
    from loguru import logger as _log
except ImportError:  # pragma: no cover
    import logging as _logging

    _log = _logging.getLogger("cubepi.providers.fallback")


DEFAULT_TRIGGER_ERRORS: frozenset[type[ProviderError]] = frozenset(
    {RateLimited, ProviderUnavailable, ContextLengthExceeded}
)


@dataclass(frozen=True)
class FallbackBoundModel:
    """Ordered chain of BoundModels — tries each in turn on retriable errors.

    chain[0] is the primary model. On a trigger_errors exception or a first-event
    error from stream(), the next model in the chain is tried transparently.
    For generate(), an error AssistantMessage (stop_reason="error") also triggers
    failover. Mid-stream errors (after the first non-error event) are forwarded
    as-is.

    provider and spec proxy chain[0] so tracing/billing code that reads
    agent._model.provider / agent._model.spec continues to work unchanged.

    Tracer / Meter coverage: Recorder.attach() and Meter.attach() detect
    FallbackBoundModel and subscribe to every unique BaseProvider in the
    chain (via chain_providers() below), so post-failover chat spans and
    provider metrics land in the trace / metric stream like primary-leg
    calls do.
    """

    chain: tuple[BoundModel, ...]
    trigger_errors: frozenset[type[ProviderError]] = DEFAULT_TRIGGER_ERRORS
    on_failover: (
        Callable[
            [BoundModel, BoundModel | None, BaseException | str], Awaitable[None] | None
        ]
        | None
    ) = None

    def __post_init__(self) -> None:
        # An empty chain is a configuration mistake: provider/spec proxy,
        # stream(), and generate() would all IndexError or silently exhaust
        # without ever attempting a call. Fail fast at construction time.
        if not self.chain:
            raise ValueError(
                "FallbackBoundModel.chain must contain at least one BoundModel"
            )

    @property
    def provider(self) -> Provider:
        return self.chain[0].provider

    @property
    def spec(self) -> Model:
        return self.chain[0].spec

    async def _notify(
        self,
        failed: BoundModel,
        next_model: BoundModel | None,
        error: BaseException | str,
        attempt: int,
    ) -> None:
        failed_label = f"{failed.spec.provider_id}/{failed.spec.id}"
        next_label = (
            f"{next_model.spec.provider_id}/{next_model.spec.id}"
            if next_model
            else "none (exhausted)"
        )
        _log.warning(
            "cubepi.providers.fallback: failover triggered  "
            "failed=%s  →  next=%s  reason=%s  attempt=%s/%s",
            failed_label,
            next_label,
            error,
            attempt,
            len(self.chain),
        )
        if self.on_failover is not None:
            try:
                result = self.on_failover(failed, next_model, error)
                if inspect.isawaitable(result):
                    await result
            except Exception as cb_exc:  # noqa: BLE001
                _log.warning(
                    "cubepi.providers.fallback: on_failover callback raised; swallowed: %s",
                    cb_exc,
                )

    async def stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        last_error: BaseException | str = "no providers in chain"
        trigger = tuple(self.trigger_errors)

        for attempt, bound in enumerate(self.chain, start=1):
            next_bound = self.chain[attempt] if attempt < len(self.chain) else None

            try:
                inner = await bound.stream(
                    messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    options=options,
                )
            except trigger as exc:
                last_error = exc
                await self._notify(bound, next_bound, exc, attempt)
                continue
            except Exception:
                raise

            iterator = inner.__aiter__()
            try:
                first = await iterator.__anext__()
            except StopAsyncIteration:
                last_error = "stream ended before producing any events"
                await self._notify(bound, next_bound, last_error, attempt)
                continue

            if first.type == "error":
                last_error = first.error_message or "stream error"
                await self._notify(bound, next_bound, last_error, attempt)
                continue

            outer = MessageStream()

            async def _forward(
                first_ev: StreamEvent = first,
                src: Any = iterator,
                src_stream: MessageStream = inner,
                out: MessageStream = outer,
            ) -> None:
                try:
                    out.push(first_ev)
                    async for ev in src:
                        out.push(ev)
                    out.set_result(await src_stream.result())
                except BaseException as exc:  # noqa: BLE001
                    err_msg = AssistantMessage(
                        content=[],
                        stop_reason="error",
                        error_message=str(exc),
                        usage=Usage(),
                        timestamp=time.time(),
                    )
                    out.push(StreamEvent(type="error", error_message=str(exc)))
                    out.set_result(err_msg)
                    if not isinstance(exc, Exception):
                        raise

            outer.attach_task(asyncio.create_task(_forward()))
            return outer

        raise ProviderUnavailable(
            f"all providers exhausted; last error: {last_error!r}"
        )

    async def generate(
        self,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking: ThinkingLevel | None = None,
        thinking_budgets: ThinkingBudgets | None = None,
    ) -> AssistantMessage:
        last_error: BaseException | str = "no providers in chain"
        trigger = tuple(self.trigger_errors)

        for attempt, bound in enumerate(self.chain, start=1):
            next_bound = self.chain[attempt] if attempt < len(self.chain) else None

            try:
                result = await bound.generate(
                    messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    options=options,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    thinking=thinking,
                    thinking_budgets=thinking_budgets,
                )
            except trigger as exc:
                last_error = exc
                await self._notify(bound, next_bound, exc, attempt)
                continue
            except Exception:
                raise

            if result.stop_reason == "error":
                last_error = result.error_message or "generate error"
                await self._notify(bound, next_bound, last_error, attempt)
                continue

            return result

        raise ProviderUnavailable(
            f"all providers exhausted; last error: {last_error!r}"
        )


def chain_providers(model: object) -> list[Any]:
    """Return the unique BaseProvider instances backing a bound model.

    For :class:`FallbackBoundModel`, iterates ``chain`` and dedupes providers
    by identity. For a plain :class:`BoundModel` returns ``[model.provider]``.
    For anything else (or ``None``) returns ``[]``.

    Used by :meth:`cubepi.tracing.recorder.Recorder.attach` and
    :meth:`cubepi.tracing.meter.Meter.attach` to subscribe to every leg of a
    fallback chain so post-failover provider events land in the trace /
    metric stream.
    """
    from cubepi.providers.base import BaseProvider

    if model is None:
        return []
    if isinstance(model, FallbackBoundModel):
        seen: set[int] = set()
        out: list[Any] = []
        for bm in model.chain:
            p = bm.provider
            if isinstance(p, BaseProvider) and id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out
    provider = getattr(model, "provider", None)
    if isinstance(provider, BaseProvider):
        return [provider]
    return []
