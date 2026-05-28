---
title: 人机协同 (HITL)
sidebar_position: 10
---

# 人机协同 (HITL)

cubepi 的 HITL channel 让 agent 能够**暂停并等待人类输入**后再继续。
它用一个统一原语覆盖两种典型模式：

1. **沙箱工具确认** —— 危险工具（bash、文件写入、API 变更）在运行前
   需要人类 approve / deny / edit。
2. **运行中结构化提问** —— agent 需要用户做出选择或填写表单才能继续。

Channel 是一个可 `await` 的协程协作者。工具作者写
`await channel.ask(...)`，channel 处理暂停。宿主代码订阅挂起请求并回填
答案。两种后端覆盖全场景：

- `InMemoryChannel` —— CLI、notebook、测试。进程死亡，挂起丢失。
- `CheckpointedChannel` —— web 服务。将挂起请求持久化到 `Checkpointer`，
  让不同进程（或重启后的同一进程）在数小时后拾起并回答。

## 三种动词

### `confirm(prompt, *, details, timeout, signal) → bool`

简单的 yes/no 问题。宿主回答 `True` 或 `False`。

### `approve(tool_name, tool_call_id, args, *, details, timeout, signal) → ApproveAnswer`

沙箱确认动词。返回一个 `ApproveAnswer`，三种决策：

| 决策 | 结果 |
|---|---|
| `"approve"` | 以原始参数运行工具 |
| `"deny"` | 阻塞工具；`tool_result.is_error=True`，`details["hitl"]["decision"]="human_deny"` |
| `"edit"` | 以编辑后的参数运行（会拿工具的 pydantic 参数模型重新校验） |

对 `approve` 请求，信封的 `question_id` 设为 LLM 的 `tool_call_id` —
没有独立的 UUID，所以宿主代码可以直接用它已经在工具流中追踪的同一个
ID 来关联。

### `ask(questions, *, timeout, signal) → dict[str, str | list[str]]`

一个包含一个或多个 `Question` 对象的结构化表单。每个问题可以是：

- **自由文本** (`options=None`)
- **单选** (`options=[...]`, `multi_select=False`)
- **多选** (`options=[...]`, `multi_select=True`)
- **"其他" 可输入**（选项有 `allow_input=True` —— 用户输入自由文本）

```python
from cubepi.hitl.types import Question, Option

answers = await channel.ask([
    Question(key="framework", prompt="选择框架？", options=[
        Option(label="React", value="react"),
        Option(label="Vue", value="vue"),
        Option(label="其他", value="other", allow_input=True),
    ]),
    Question(key="features", prompt="启用功能：", multi_select=True, options=[
        Option(label="认证", value="auth"),
        Option(label="支付", value="payments"),
    ]),
])
# answers == {"framework": "react", "features": ["auth", "payments"]}
```

## 超时

两个 channel 都在构造函数接受 `default_timeout`，每个动词接受一個
per-call 的 `timeout` kwarg（per-call 覆盖默认）。

超时到期从 agent 侧的 `await` 抛出 `HitlTimedOut(BaseException)`。
周围的工具或中间件将其转换为 `tool_result.is_error=True`，
`details["hitl"]["decision"]="timed_out"`，模型看到干净的拒绝结果
并能自然反应。信封的 `HitlRequest.timeout_seconds` 会自动填写，
前端可以渲染倒计时。

## 内置中间件

### `ConfirmToolCallMiddleware`

"在集合中的工具名一律询问人类。"

```python
from cubepi.hitl import ConfirmToolCallMiddleware

agent = Agent(
    ...,
    middleware=[
        ConfirmToolCallMiddleware(
            channel,
            require_confirm={"bash", "write_file"},
            timeout_seconds=180,
        ),
    ],
)
```

`require_confirm` 选项：

| 值 | 行为 |
|---|---|
| `None`（默认） | 确认**所有**工具 |
| `set[str]` | 仅当 `tool_call.name` 在集合中时确认 |
| `Callable[[BeforeToolCallContext], bool]` | 自定义断言 —— 检查参数、上下文等 |

### `ApprovalPolicyMiddleware`

适用于需要**策略引擎**将工具调用分为 auto-allow、hard-deny 和
human-confirm 三类的宿主。

```python
from cubepi.hitl import Approve, ApprovalPolicyMiddleware, AskUser, Deny

def my_policy(ctx):
    if ctx.tool_call.name == "read_file":
        return Approve()                              # 直接放行
    if ctx.tool_call.name.startswith("dangerous_"):
        return Deny(reason="策略阻止")                  # 硬阻止，不询问人类
    return AskUser(timeout_seconds=180)               # 人类确认

agent = Agent(
    ...,
    middleware=[ApprovalPolicyMiddleware(channel, policy=my_policy)],
)
```

策略函数可以是同步或异步（可 `await`）。返回值：

| 返回 | 效果 |
|---|---|
| `Approve()` | 工具运行；channel 从未调用 |
| `Deny(reason)` | 工具阻塞；`hitl_trace["decision"]="policy_deny"` |
| `AskUser(timeout_seconds=..., details=...)` | 调用 channel；人类选择 approve/deny/edit |

## `ask_user` 内置工具

模型在需要用户结构化输入时调用的工具。工厂函数返回一个名为
`"ask_user"` 的 `AgentTool`，`execution_mode="sequential"`。

```python
from cubepi.hitl import ask_user_tool

agent = Agent(
    ...,
    tools=[bash_tool, ask_user_tool(channel)],
)
```

工具描述明确引导模型不要拿 `ask_user` 做自由文本澄清（"对于自由文本
提问，直接用文本结束 turn——用户的下一条消息就是答案"）。

取消和超时以 `tool_result.is_error=True` 体现，
`details["hitl"]["outcome"]="cancelled"` / `"timed_out"` —— 模型看到
干净的���误工具结果并能做出反应。其他 HITL 控制异常（HitlDetached、
HitlAborted）传播到 Agent 层，不暴露给模型。

## 跨进程（web 服务）流程

```python
# ───── 进程 1：HTTP POST /chat ─────

async with PostgresCheckpointer("postgresql://...") as cp:
    channel = CheckpointedChannel(checkpointer=cp, thread_id="conv-42")

    agent = Agent(
        provider=…, model=…,
        tools=[bash_tool],
        middleware=[ConfirmToolCallMiddleware(channel, require_confirm={"bash"})],
        channel=channel, checkpointer=cp, thread_id="conv-42",
    )

    task = asyncio.create_task(agent.prompt("删除临时文件"))

    # 轮询挂起（或订阅 channel 用于 SSE 推送）
    for _ in range(1000):
        pending = channel.pending
        if pending is not None:
            break
        await asyncio.sleep(0.1)

    # 优雅挂起 — 持久化 assistant message + 未决 tool_calls,
    # pending_request 留在 DB, 发射 AgentSuspendedEvent.
    await agent.detach()
    await task


# ───── 进程 2：HTTP POST /respond ─────

async with PostgresCheckpointer("postgresql://...") as cp:
    channel = CheckpointedChannel(checkpointer=cp, thread_id="conv-42")

    agent = Agent(
        provider=…, model=…,
        tools=[bash_tool],
        middleware=[ConfirmToolCallMiddleware(channel, require_confirm={"bash"})],
        channel=channel, checkpointer=cp, thread_id="conv-42",
    )

    await agent.respond(
        question_id=request.json["call_id"],
        answer=ApproveAnswer(decision="approve"),
    )
    # Bash 工具执行，模型收到 tool_result，产生下一个 assistant turn。
```

**用户关闭 tab 未回答时：**

```python
await agent.abort_pending(reason="用户关闭了 tab")
# Phase 1：发信号给进行中的 HITL await（如有），触发 HitlAborted。
# Phase 2：为未决的 tool_calls 追加合成 deny ToolResultMessage，
#   追加终止 AssistantMessage(stop_reason="aborted")，
#   清除持久化 pending, 发射 AgentAbortedEvent。
# 不调用模型。对话关闭。
```

## Agent API

| 属性/方法 | 签名 | 说明 |
|---|---|---|
| `agent.channel` | `HitlChannel \| None` | 绑定的 channel 或 `None` |
| `agent.in_flight_hitl_request` | `HitlRequest \| None` | channel 当前 pending（同步属性） |
| `load_pending_hitl_request()` | `async → HitlRequest \| None` | 从 checkpointer 读取 pending（detach 后也能用） |
| `detach()` | `async → None` | 发射 `AgentSuspendedEvent` 然后触发 `HitlDetached` |
| `respond(*, question_id=, answer=)` | `async → None` | 恢复挂起的运行 |
| `abort_pending(reason=)` | `async → None` | 关闭对话（两阶段：发信号 + 追加合成拒绝） |

## 事件

四个新事件在 agent 事件流上发射：

| 事件 | 何时发射 | 关键字段 |
|---|---|---|
| `HitlRequestEvent` | channel 收到新的 `confirm/approve/ask` | `request: HitlRequest` |
| `HitlAnswerEvent` | `channel.answer()` 或 `channel.cancel()` 触发 | `question_id`, `answer`, `cancelled`, `timed_out` |
| `AgentSuspendedEvent` | `agent.detach()` 被调用时 HITL pending | `pending_request: HitlRequest` |
| `AgentAbortedEvent` | `agent.abort_pending()` 关闭对话 | `reason: str` |

## 追踪 span

安装 `cubepi[tracing]` extra 后，每次 HITL await 都会包装在一个
OpenTelemetry span 中：

| Span 名称 | 属性 |
|---|---|
| `hitl.approve` | `hitl.tool_name`, `hitl.tool_call_id`, `hitl.outcome`, `hitl.from_resume`, `hitl.duration_seconds` |
| `hitl.confirm` | `hitl.question_id`, `hitl.outcome`, `hitl.duration_seconds` |
| `hitl.ask` | `hitl.question_id`, `hitl.outcome`, `hitl.duration_seconds` |

`hitl.outcome` 可以是：`approved`, `denied`, `edited`, `answered`,
`cancelled`, `timed_out`, `aborted`, `detached`。

追踪导入是懒式的 —— 未安装 `opentelemetry` 时，channel 静默回退到
无操作 span。

## 错误参考

| 异常 | 基类 | 含义 |
|---|---|---|
| `HitlCancelled(reason)` | `BaseException` | 宿主调了 `channel.cancel(qid)` |
| `HitlTimedOut(seconds)` | `BaseException` | per-call 或 channel 默认超时到期 |
| `HitlDetached` | `BaseException` | HITL await 期间调了 `agent.detach()` |
| `HitlAborted` | `BaseException` | `agent.abort_pending()` 向 agent 发信号 |
| `HitlConcurrencyError` | `Exception` | channel 已有 pending 时再次调用 `confirm/approve/ask` |
| `HitlStaleAnswer` | `Exception` | `channel.answer(qid)` 的 qid 与当前 pending 不匹配 |
| `HitlNoPendingRequest` | `Exception` | 调了 `agent.respond(…)` 但线程没有 `pending_request` |
| `HitlDurabilityNotGuaranteed` | `Exception` | 自定义工具调了 `CheckpointedChannel.ask()` 但没有 `allow_inside_custom_tool=True` |

`HitlControlException`（四个 `BaseException` 子类的父类）**故意**不被
`cubepi.agent.tools._prepare_tool_call` 和 `_execute_prepared` 中现有
的广泛 `except Exception:` 处理器捕获 —— 这模仿了 `asyncio.CancelledError`
的模式。

## `ask_user` 对比 end of turn

| 目标 | 用法 |
|---|---|
| 对用户的自由文本追问 | 直接用文本结束 turn——用户的下一条消息就是答案 |
| 结构化选择 (N 选一) | `ask_user` 工具带 `options` |
| 多选 ("任意选择") | `ask_user` 工具带 `multi_select=True` |
| "其他" 可自由文本输入 | `ask_user` 工具选项带 `allow_input=True` |
| 确认/编辑工具参数后才运行 | `ConfirmToolCallMiddleware` 或 `ApprovalPolicyMiddleware` |

## 持久化范围

持久的跨进程恢复（进程死亡后仍能继续）在两个定义明确的安全暂停点支持：

1. **`before_tool_call` 确认门** —— 确认中间件在工具的 `execute()` body
   运行*之前*调用 `channel.approve(...)`。此时不存在工具副作用。恢复时
   重新进入循环，执行（可能被编辑过的）工具体或替换为合成 deny
   tool_result。
2. **`ask_user` 工具体** —— 其整个 `execute()` body 就是
   `return await channel.ask(...)`。恢复时不会重放任何内容。

**默认情况下，在 `execute()` 内将 HITL 与其他工作混合的自定义工具不
支持跨进程持久化。** 如果此类工具的进程在执行中途死亡，channel 调用
前运行的所有内容都会丢失。除非使用 `allow_inside_custom_tool=True`
构造 `CheckpointedChannel`，否则将抛出 `HitlDurabilityNotGuaranteed` —
调用者必须承认等幂性契约（工具体在这一点必须是纯 HITL 等待，前面没有
可观察的副作用）。

## 测试辅助

```python
from cubepi.hitl.testing import ScriptedChannel, NoopChannel

# ScriptedChannel：预编程答案，按顺序消费
ch = ScriptedChannel(answers=[
    ApproveAnswer(decision="approve"),
    {"color": "red"},                       # ask 答案
])
assert len(ch.history) == 2  # 所有见过的 HitlRequest

# NoopChannel：自动批准一切。用于测试中的 subagent。
ch = NoopChannel()
assert (await ch.approve("bash", "tc", {})).decision == "approve"
```

## 架构说明

- **每线程单一 pending。** Agent 循环是顺序的——每个 `thread_id` 最多
  有一个 HITL 请求。并发 `confirm/approve/ask` 会抛 `HitlConcurrencyError`。
- **Prompt-cache 前缀不变量。** 暂停与恢复之间，消息列表仅通过在末尾
  追加 tool-result 消息和下一条 assistant turn 来改变。不插入、重排或
  修改先前的消息——否则会使 provider 端的 prompt cache 失效。
- **`question_id == tool_call_id` 适用于 approve 请求。** 无需别名或
  映射——已从工具流中追踪 `call_id` 的宿主可直接传递。
- **恢复不重放。** 恢复时将答案预加载到 channel 并重新进入循环。最后一条
  assistant message 中未决的 tool call 决定了接下来执行什么。没有基于
  节点的重放语义。
