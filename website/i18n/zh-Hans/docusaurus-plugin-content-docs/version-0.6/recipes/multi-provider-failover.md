---
title: 多 Provider 故障转移
description: "在 CubePi 中配置多 provider 故障转移以提升弹性 —— provider 间自动降级。"
---

# Recipe：多 Provider 故障转移

当 Anthropic 触发限速或发生故障时，自动切换到 OpenAI 而不让 agent 崩溃。
我们将把两个 provider 封装在一个单独的 `Provider` 适配器后面，
由该适配器自行实现重试/故障转移逻辑。

**预计耗时：** 10 分钟。
**依赖：** `cubepi`、`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`。

## 封装 provider

```python title="failover.py"
import asyncio
import logging
import time
from typing import Sequence

from cubepi.providers.base import (
    AssistantMessage,
    Message,
    MessageStream,
    Model,
    Provider,
    StreamEvent,
    StreamOptions,
    ToolDefinition,
    Usage,
)
from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.openai import OpenAIProvider

log = logging.getLogger(__name__)


class FailoverProvider:
    """按顺序尝试各 provider；在构造时或首个事件出错时进行故障转移。

    内置 provider 会吞掉 API/网络错误，并将其作为
    `StreamEvent(type="error")` 暴露在返回的流上 —— 而不是从
    `provider.stream()` 抛出异常。因此我们先查看每个内部流的第一个事件，
    只有当看到非错误事件时才确认切换到该 provider。

    限制：*首个事件之后*发生的错误（如流传输过半时的限速、服务端断连）
    会原样转发给 agent。对故障 provider 完整重放半条流式响应
    需要缓冲整轮内容 —— 超出本 recipe 范围。
    """

    def __init__(self, primary_pair: tuple[Provider, Model], *fallbacks: tuple[Provider, Model]) -> None:
        self._chain: list[tuple[Provider, Model]] = [primary_pair, *fallbacks]

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        last_error: str | None = None

        for provider, mapped_model in self._chain:
            # 构造时失败（罕见 —— 大多数错误留在生产者任务内部）。
            try:
                inner = await provider.stream(
                    mapped_model,
                    messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    options=options,
                )
            except Exception as e:
                log.warning("provider %s failed at construction: %s", mapped_model.provider, e)
                last_error = repr(e)
                continue

            # 查看第一个事件以判断流是否健康。
            iterator = inner.__aiter__()
            try:
                first = await iterator.__anext__()
            except StopAsyncIteration:
                last_error = "stream ended before producing any events"
                continue

            if first.type == "error":
                log.warning("provider %s errored on first event: %s",
                            mapped_model.provider, first.error_message)
                last_error = first.error_message or "stream error"
                continue

            # 健康 —— 提交到此 provider。通过全新的外层 MessageStream
            # 转发 `first` 及后续事件，让调用方看到从头开始的完整流。
            outer = MessageStream()

            async def _forward(first_event=first, src=iterator, src_stream=inner):
                try:
                    outer.push(first_event)
                    async for ev in src:
                        outer.push(ev)
                    final = await src_stream.result()
                    outer.set_result(final)
                except Exception as exc:
                    fallback_msg = AssistantMessage(
                        content=[],
                        stop_reason="error",
                        error_message=str(exc),
                        usage=Usage(),
                        timestamp=time.time(),
                    )
                    outer.push(StreamEvent(type="error", error_message=str(exc)))
                    outer.set_result(fallback_msg)

            outer.attach_task(asyncio.create_task(_forward()))
            return outer

        raise RuntimeError(f"all providers exhausted; last error: {last_error!r}")
```

## 使用方式

```python title="main.py"
import asyncio
import os

from cubepi import Agent, Model
from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.openai import OpenAIProvider
from failover import FailoverProvider


async def main():
    failover = FailoverProvider(
        (
            AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"]),
            Model(id="claude-sonnet-4-5-20250929", provider="anthropic"),
        ),
        (
            OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"]),
            Model(id="gpt-5", provider="openai"),
        ),
    )

    # 此处传入的 model 会被 FailoverProvider 内部覆盖；可传任意占位符。
    # 我们使用主 provider 的 model，以便用量统计标签与正常路径保持一致。
    agent = Agent(
        provider=failover,
        model=Model(id="claude-sonnet-4-5-20250929", provider="anthropic"),
        system_prompt="You answer concisely.",
    )
    agent.subscribe(lambda e, s=None: None)
    await agent.prompt("Capital of Mongolia?")
    last = agent.state.messages[-1]
    print(last.content[0].text)


asyncio.run(main())
```

## 更智能的故障转移策略

上面的示例在**任何**错误事件时都会降级。这对于 `RateLimitError`、
`APIConnectionError` 或 5xx 是合理的 —— 但对于 `BadRequestError`
则未必正确（代码有误；下一个 provider 也会以同样方式失败）。

首个事件的 `error_message` 来自底层 SDK 异常的 `str(exc)`。
可以按子字符串过滤，或者 —— 更好的做法 —— 封装每个 provider 的
`_produce` 方法，为错误打上类别标签：

```python
NON_RETRYABLE_HINTS = ("bad request", "invalid_request_error", "401", "403")

if first.type == "error":
    msg = (first.error_message or "").lower()
    if any(h in msg for h in NON_RETRYABLE_HINTS):
        raise RuntimeError(f"non-retryable error from {mapped_model.provider}: {msg}")
    last_error = first.error_message
    continue
```

更健壮的方法是 fork 内置 provider 并从 `_produce` 中重新抛出
特定的 SDK 异常，使其作为真正的 Python 异常到达 `provider.stream()` ——
但这需要对 CubePi 本身做较大改动。

## 添加熔断器

不要持续重试一个明显已宕机的 provider。一个简单的计数器：

```python
import time

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 60) -> None:
        self._failures = 0
        self._opened_at: float | None = None
        self._threshold = failure_threshold
        self._recovery = recovery_seconds

    def can_attempt(self) -> bool:
        if self._opened_at and time.monotonic() - self._opened_at < self._recovery:
            return False
        if self._opened_at:
            self._opened_at = None   # half-open
        return True

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()
            self._failures = 0

    def record_success(self) -> None:
        self._failures = 0
```

在 `FailoverProvider` 中为每个 provider 持有一个 `CircuitBreaker`，
若 `can_attempt()` 返回 False 则跳过该 provider。

## 按工具故障转移不适用

本 recipe 处理的是 **provider** 故障。工具故障是另一回事 ——
参见 [Middleware → 重试](../guides/middleware/examples#retries-with-backoff)
了解该模式。

## 常见陷阱

- **不同 provider 的工具 schema 不同** —— 两个内置 provider 都接受相同的
  `ToolDefinition`，但 extra-body 定制（如 OpenAI 的
  `parallel_tool_calls=False`）不会带到 Anthropic。请将跨 provider 的
  行为放在 [`transform_context`](../guides/middleware/hooks#transform_context)
  中，而非 `extra_body`。
- **成本不同** —— 从 Anthropic 故障转移到 OpenAI 会改变每 token 的费用。
  跟踪是哪个 provider 响应（通过 `on_response` 或
  `AssistantMessage.provider_id`）并据此计费。
- **流一致性** —— 封装器通过全新的 `MessageStream` 转发事件，因此
  消费者无论哪个 provider 响应都能看到相同的 `StreamEvent` 结构。
  原始的 `start` 事件由内部 provider 原样传递。
- **流传输中途的错误无法恢复** —— 一旦看到健康的首个事件，封装器即
  提交到该 provider。如果在长响应传输途中出错，agent 会看到该错误。
  完整的中途重放需要缓冲 —— 超出本 recipe 范围。

## 另请参见

- [Providers / Anthropic](../guides/providers/anthropic) 和
  [OpenAI](../guides/providers/openai) —— provider 专属细节。
- [编写自定义 Provider](../guides/providers/custom) —— 本封装器使用的同一 Protocol。
