---
title: 第一个 Agent
description: "从零开始创建并运行你的第一个 CubePi agent。"
---

# 构建你的第一个 Agent

这是 [快速开始](../../getting-started/quick-start) 的"加长版"。我们
端到端构建一个单工具 Agent,然后逐层加上你后续大概率会想要的东西：
流式 UI、错误处理、取消按钮。

## 第 1 步 —— 配 provider 和 model

provider 是到 LLM API 的连接；`Model` 描述你想调哪个模型 + 一些能力
开关。

```python
import os
from cubepi import Model
from cubepi.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
model = Model(
    id="claude-sonnet-4-5-20250929",
    provider="anthropic",
    max_tokens=4096,         # 回复 token 上限
    context_window=200_000,  # 模型硬上限；通常默认就够
    temperature=0.7,
)
```

`Model.provider` 是一个字符串标签(比如 `"anthropic"`、`"openai"`)。
框架内部用来 clamp 思考等级和打 tag —— 在你的代码库里保持稳定即可,
不必和 provider 内部命名一致。

## 第 2 步 —— 声明一个工具

每个工具 = Pydantic 参数模型 + 一个异步 `execute` 函数：

```python
from pydantic import BaseModel
from cubepi import AgentTool, AgentToolResult, TextContent


class GetWeatherParams(BaseModel):
    city: str


async def get_weather(tool_call_id, params: GetWeatherParams, *, signal=None, on_update=None):
    # 真实工作放在这里 —— 调 HTTP API、查 DB 等。
    return AgentToolResult(
        content=[TextContent(text=f"{params.city} 现在 72°F,晴")]
    )


weather_tool = AgentTool(
    name="get_weather",
    description="获取一个城市的当前天气,返回简短文字。",
    parameters=GetWeatherParams,
    execute=get_weather,
)
```

注意几点：

- Pydantic 模型自动转成 JSON Schema,作为工具定义发给模型。
- `execute` 签名是固定的:`(tool_call_id, params, *, signal, on_update)`。
  最后两个 keyword-only 参数一定会被传入 —— 哪怕你不用,也得在签名里。
- `signal` 是用户取消时会被 set 的 `asyncio.Event`。在长时间运行的
  代码里检查它,及时退出。
- `on_update(partial)` 让你流式回报进度(见
  [工具使用](./tool-use))。

## 第 3 步 —— 组装 Agent

```python
from cubepi import Agent

agent = Agent(
    provider=provider,
    model=model,
    system_prompt="你是一个简洁的天气助手。",
    tools=[weather_tool],
)
```

不需要工具时传 `tools=[]`（或干脆省掉)。

## 第 4 步 —— 订阅事件

`agent.subscribe(listener)` 是观察运行过程的入口。Listener 会收到
每一个 `AgentEvent`:

```python
def on_event(event, signal=None):
    if event.type == "message_update" and event.stream_event.type == "text_delta":
        print(event.stream_event.delta, end="", flush=True)
    elif event.type == "tool_execution_start":
        print(f"\n→ 调用 {event.tool_name}({event.args})")
    elif event.type == "tool_execution_end":
        print(f"  ✓ 完成")

agent.subscribe(on_event)
```

可以注册多个 listener,它们都会收到所有事件。**在 `prompt()` 之前
订阅** —— 事件在 run 一开始就会发出。

## 第 5 步 —— prompt 然后跑

```python
import asyncio

async def main():
    await agent.prompt("东京天气怎么样？")

asyncio.run(main())
```

`agent.prompt()` 不返回任何值。结果存在 `agent.state.messages`
（完整历史）和 `agent.state.streaming_message`（当前正在产生的消息,
在两轮之间为 `None`）上。

## 加上错误处理

当 `provider.stream()` 抛异常时,Agent 循环仍然会产出一条
`AssistantMessage`,其 `stop_reason="error"`、`error_message` 已填好。
事件序列是:`message_start` → `message_end` → `turn_end` → `agent_end`。

两种处理方式：

1. 在 subscriber 里捕获 `event.type == "agent_end"`,看最后一条消息
   的 `stop_reason`：

    ```python
    def on_event(event, signal=None):
        if event.type == "agent_end":
            last = event.messages[-1]
            if getattr(last, "stop_reason", "") == "error":
                print(f"\nerror: {last.error_message}")
    ```

2. 或者在 `await agent.prompt(...)` 返回之后,检查
   `agent.state.error_message`。

## 加一个"取消按钮"

`agent.abort()` 会 set run 级别的 signal。Provider 流短路成
`"aborted"`,正在运行的工具看到 `signal.is_set() == True`,
循环干净地发出 `agent_end`。

```python
async def main():
    task = asyncio.create_task(agent.prompt("搜索…"))
    await asyncio.sleep(0.5)
    agent.abort()
    await task              # 一定会 return —— 不会抛异常
    await agent.wait_for_idle()
```

## 常见坑

- **`RuntimeError: Agent is already processing a prompt.`** ——
  上一次 `prompt()` 没 await 完你又调了一次。用 `await
  agent.wait_for_idle()`,或用 `steer()` / `follow_up()` 排队。
- **没有 `text_delta` 事件** —— 是不是在 `prompt()` 之后才订阅的？
  Listener 只看得到注册之后的事件。
- **Tool not found** —— 模型调了一个 `name` 不在 `tools=[...]` 里的
  工具。CubePi 把这种情况包装成一个 `is_error=True` 的工具结果,
  不会崩 —— 在 `tool_execution_end` 事件的 `result` 里能看到。
- **Pydantic ValidationError 被吞掉了** —— 如果模型产出格式错误的
  JSON,CubePi 会把 validation error 也包成工具的 error result 喂
  回去,模型通常下一轮自动纠正。

## 下一步

- [工具使用与并行执行](./tool-use) —— 同时多个工具、sequential 模式、
  `terminate`、增量进度。
- [流式事件](./streaming) —— 完整的事件分类。
- [多轮会话](./multi-turn) —— 跨轮保留状态、`steer`、`follow_up`、
  `resume`。
