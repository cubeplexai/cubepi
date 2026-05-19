---
title: 核心概念
---

# 核心概念

CubePi 的全部能力可以归纳为六个概念。这一页读一遍,后面的文档基本就
变成查表了。

## Agent

`Agent` 是有状态的门面：你用 provider、model、可选的工具、可选的
middleware/checkpointer 构造它。然后通过三个方法驱动它：

- `await agent.prompt(message)` —— 用一条 user 消息开启新一轮。
- `await agent.resume()` —— 从最后一条已持久化的消息继续（配合
  checkpointer 使用）。
- `agent.steer(message)` / `agent.follow_up(message)` —— 在正在跑的
  过程中插入消息,或为当前运行结束后排队下一条。

Agent 持有一个 `AgentState`（system prompt、tools、model、消息历史、
未结束的 tool call、流式标志）和一个 subscribers 列表：

```python
unsubscribe = agent.subscribe(my_listener)
# ...
unsubscribe()
```

Subscriber 会收到循环发出的每一个 `AgentEvent`。可以是同步或异步函数。

## Tool

`AgentTool` = name + description + Pydantic 参数模型 + 异步 `execute`：

```python
from pydantic import BaseModel
from cubepi import AgentTool, AgentToolResult, TextContent

class SearchParams(BaseModel):
    query: str
    limit: int = 10

async def execute(tool_call_id, params: SearchParams, *, signal=None, on_update=None):
    # 干活；如果可以被打断,记得检查 signal
    return AgentToolResult(content=[TextContent(text=f"…")])

search = AgentTool(
    name="search",
    description="搜索语料库",
    parameters=SearchParams,
    execute=execute,
)
```

Pydantic schema 会自动转成 JSON Schema 喂给模型。参数解析、错误包装、
并行执行都由框架处理。`execution_mode`、`on_update`（增量进度）、
`terminate`（在工具里结束本轮）见 [工具使用](../guides/agents/tool-use)。

## Provider

任何匹配下面 Protocol 的对象就是 Provider：

```python
class Provider(Protocol):
    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream: ...
```

它返回一个 `MessageStream` —— 一个统一的异步迭代器,产出 `StreamEvent`,
并通过 `await stream.result()` 暴露最终的 `AssistantMessage`。内置 Provider：

- `AnthropicProvider` —— Claude(Messages API,支持思考、缓存、工具使用)。
- `OpenAIProvider` —— GPT 家族(Chat Completions API)。
- `OpenAIResponsesProvider` —— GPT 家族(Responses API,服务端状态)。
- `FauxProvider` —— 确定性测试替身(不发任何网络请求)。

实现一个方法就能写自己的。见 [Providers / 自定义](../guides/providers/custom)。

## Stream 和事件

流和事件分两层：

- **Provider 流** —— `MessageStream` 产出的是 *provider* 事件：
  `start`、`text_start`、`text_delta`、`text_end`、`thinking_*`、
  `toolcall_*`、`done`、`error`。原始 token 流。
- **Agent 事件** —— `agent.subscribe(...)` 收到的内容。十一种类型
  覆盖整个循环：`agent_start`、`agent_end`、`turn_start`、`turn_end`、
  `message_start`、`message_update`、`message_end`、
  `tool_execution_start`、`tool_execution_update`、
  `tool_execution_end`。`message_update` 把 provider 事件嵌套在
  `event.stream_event` 里。

做 UI 订阅 Agent 事件；做底层 token 路由就钻 `event.stream_event`。
见 [流式事件](../guides/agents/streaming)。

## Middleware

`Middleware` 是有最多七个类型化 hook 的类：

| Hook | 何时触发 | 组合规则 |
|---|---|---|
| `transform_context` | 每次调模型之前,处理消息列表 | 链式 —— 每个收到上一个的输出 |
| `convert_to_llm` | provider 序列化之前 | 最后一个实现生效 |
| `transform_system_prompt` | 每次调模型之前,处理 system prompt | 链式 |
| `before_tool_call` | 每个工具调用之前(在参数校验后) | 第一个 `block=True` 短路 |
| `after_tool_call` | 每个工具调用之后(在 `execute` 之后) | 后写覆盖先写 |
| `after_model_response` | assistant 消息落定之后 | 返回 `TurnAction` 控制流向 |
| `should_stop_after_turn` | 每个轮次结束时 | 任一返回 `True` 即停 |

通过 `Agent(middleware=[...])` 传入。见
[Middleware → 组合规则](../guides/middleware/composition)。

## Checkpointer

任何匹配下面 Protocol 的对象就是 Checkpointer：

```python
class Checkpointer(Protocol):
    async def load(self, thread_id: str) -> CheckpointData | None: ...
    async def append(self, thread_id: str, messages: list[Message]) -> None: ...
    async def save_extra(self, thread_id: str, extra: dict) -> None: ...
```

通过 `Agent(checkpointer=cp, thread_id="…")` 绑定到 Agent,循环就会在
每条消息落定时追加一行,并在第一次 `prompt()` 时恢复历史。内置后端：
`MemoryCheckpointer`、`SQLiteCheckpointer`、`PostgresCheckpointer`。
见 [Checkpointing → SQLite](../guides/checkpointing/sqlite)。

## 拼起来

```
用户代码
   │
   ▼
┌──────────────────────────────────────────┐
│ Agent                                     │
│  ├─ AgentState (messages, tools, …)       │
│  ├─ Middleware ── compose_middleware()    │
│  ├─ Checkpointer ── message_end 时追加    │
│  └─ run_agent_loop  ◀──── 真实的循环      │
│       │                                   │
│       ▼                                   │
│  Provider.stream() → MessageStream        │
│       │                                   │
│       └─ events → emit → subscribers      │
└──────────────────────────────────────────┘
```

这张图就是整个框架。文档站的其余部分,都是细节而已。
