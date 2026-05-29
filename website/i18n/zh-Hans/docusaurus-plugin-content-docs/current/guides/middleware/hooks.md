---
title: 7 个 Hook
description: "CubePi 的 7 个中间件 hook 参考——transform_context、before_tool_call、after_tool_call 等。"
---

# 7 个 Hook

`Middleware` 是一个最多包含七个可选异步方法的类。每个 hook 在 agent
循环中的精确位置触发。只实现你需要的——CubePi 只会连接你重写了的方法。

```python
from cubepi import Middleware

class MyMiddleware(Middleware):
    async def transform_context(self, messages, *, signal=None):
        ...
```

将实例传入 `Agent(middleware=[MyMiddleware(), …])`。

## `transform_context`

```python
async def transform_context(self, messages: list[Message], *, signal=None) -> list[Message]:
    ...
```

**每次调用模型之前**触发，作用于完整的消息列表。用于：

- 截断或总结以适配上下文窗口。
- 注入系统提醒（更好用 `transform_system_prompt`）。
- 动态添加或删除特定消息。

返回（可能是新的）列表。**不要修改输入**——返回一个新列表以免其他持有原引用的代码感到意外。

组合方式：链式——每个中间件看到上一个的输出。

## `convert_to_llm`

```python
async def convert_to_llm(self, messages: list[Message]) -> list[Message]:
    ...
```

**在序列化给 provider 之前**触发。这是调整 LLM 看到的最终内容的最后机会。用于：

- 将工具结果精简为纯文本。
- 为非多模态 provider 将图片内容替换为文字描述。
- 压缩较长的工具输出。

组合方式：**最后一个实现胜出**（非链式）。当多个中间件可能冲突而你希望只有一个所有者时使用。

## `transform_system_prompt`

```python
async def transform_system_prompt(self, system_prompt: str, *, signal=None) -> str:
    ...
```

**每次调用模型之前**触发，作用于 system prompt 字符串。用于：

- 注入运行时信息（当前时间、用户角色）。
- 组合模块化的 system prompt 片段。
- A/B 测试 prompt 变体。

组合方式：链式。

## `before_tool_call`

```python
async def before_tool_call(self, ctx: BeforeToolCallContext, *, signal=None) -> BeforeToolCallResult | None:
    ...
```

**每个工具调用时**触发，在参数校验之后、`tool.execute` 之前。
context 提供：

- `ctx.assistant_message` —— 发起调用的消息。
- `ctx.tool_call` —— `ToolCall` 块。
- `ctx.args` —— *校验后的* Pydantic 实例。
- `ctx.context` —— 完整的 `AgentContext`。

返回 `BeforeToolCallResult(block=True, reason="…")` 以短路——
CubePi 会将该原因作为工具结果返回，并标记 `is_error=True`。
返回 `None`（或无返回）则继续执行。

用于：权限控制、速率限制、dry-run 模式、沙箱、
人机协同确认（参见 [HITL 指南](../hitl/overview)）。

组合方式：**第一个 `block=True` 短路**整个链。

## `after_tool_call`

```python
async def after_tool_call(self, ctx: AfterToolCallContext, *, signal=None) -> AfterToolCallResult | None:
    ...
```

**每个工具调用时**触发，在 `tool.execute` 返回（或抛出异常）之后。
context 额外提供：

- `ctx.result` —— execute 返回的 `AgentToolResult`。
- `ctx.is_error` —— 工具是否出错。

返回 `AfterToolCallResult(content=…, details=…, is_error=…, terminate=…)`
以覆盖结果的单个字段（任何 `None` 字段保持原值）。返回 `None` 则原样通过。

用于：编辑、重试、日志记录、结果转换。

组合方式：后面的覆盖前面的（返回值中的每个非 `None` 字段覆盖前一个值）。

## `should_stop_after_turn`

```python
async def should_stop_after_turn(self, ctx: ShouldStopAfterTurnContext) -> bool:
    ...
```

**在每个轮次边界**触发（任何工具批次之后）。返回 `True` 以结束本次运行，
不进行下一次模型调用。

用于：最大轮次限制、预算上限、应用定义的停止条件。

组合方式：**任意一个返回 `True` 即停止**（逻辑 OR 跨链）。

## `after_model_response`

```python
async def after_model_response(
    self,
    response: AssistantMessage,
    ctx: AgentContext,
    *,
    signal=None,
) -> TurnAction | None:
    ...
```

**在 assistant 消息落定后立即**触发，在 `message_end` 发出**之前**、
在任何工具调用分发**之前**。该 hook 返回一个 `TurnAction`：

```python
from cubepi.middleware.base import TurnAction
from cubepi.providers.base import UserMessage, TextContent

TurnAction(
    response=modified_message,            # 替换消息；None 则保留原消息
    inject_messages=[UserMessage(...)],   # 在下一轮之前追加的额外消息
    decision="natural",                   # "natural" | "stop" | "loop_to_model"
)
```

三个控制流旋钮：

- `decision="natural"`（默认）—— 正常进入工具执行 / 下一轮。
- `decision="stop"` —— 在发出 `turn_end` 和 `agent_end` 后结束运行。
  不执行工具，不再调用模型。
- `decision="loop_to_model"` —— 跳过工具执行，立即重新调用模型（配合
  `inject_messages` 使用以先添加上下文）。

用于：响应审核、带重新提示的结构化输出验证、条件路由。

组合方式：链式——每个中间件看到前一个中间件的 `response`；
`inject_messages` 跨链拼接；最后一个中间件的 `decision` 胜出。

## 中间件的构成

一个中间件不需要实现每一个 hook。只覆盖你需要的即可；基类中未实现
的 hook 会抛出 `NotImplementedError`，但 `compose_middleware` 会自动跳过它们。

```python
from cubepi import Middleware

class MaxTurnsMiddleware(Middleware):
    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
        self.turns = 0

    async def should_stop_after_turn(self, ctx) -> bool:
        self.turns += 1
        return self.turns >= self.max_turns


agent = Agent(provider=…, model=…, middleware=[MaxTurnsMiddleware(5)])
```

## 另请参阅

- [组合规则](./composition) —— 多个中间件定义同一 hook 时的精确语义。
- [示例](./examples) —— 速率限制、日志、重试、滑动窗口上下文的实用中间件。
