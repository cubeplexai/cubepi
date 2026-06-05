---
title: 上下文压缩
description: "使用 CompactionMiddleware 总结较早轮次，同时保留完整 CubePi 历史。"
---

# 上下文压缩

`CompactionMiddleware` 让长对话保持在模型上下文窗口内，同时不删除 agent
历史。它把较早轮次总结进 `ctx.extra`，再把压缩后的视图发给模型：一条摘要
消息加最近消息。`agent.state.messages` 和 checkpointer 历史仍然完整。

## 基本设置

用便宜模型做摘要，用正常模型运行 agent：

```python
from cubepi import Agent
from cubepi.middleware import CompactionMiddleware

agent = Agent(
    model=provider.model("claude-sonnet-4-5-20250929"),
    checkpointer=checkpointer,
    thread_id="conv_123",
    middleware=[
        CompactionMiddleware(
            summary_model=summary_model,
            max_tokens_before_compact=80_000,
            keep_recent_messages=8,
            max_summary_tokens=1024,
        ),
    ],
)
```

摘要调用使用 `Provider.generate(...)`，并设置 `temperature=0.0`、
`thinking="off"`、`max_output_tokens=max_summary_tokens`。

## 持久化内容

middleware 会向 `AgentContext.extra` 写入两个键：

- `compaction` —— 摘要状态，以及它覆盖的消息引用。
- `compaction_until_msg_index` —— 已总结到的历史边界。

绑定 checkpointer 时，CubePi 会在 `agent_end` 通过 `save_extra` 保存
`ctx.extra`，所以下一个进程可以带着已有摘要继续。如果消息引用与当前历史不再
匹配，middleware 会清除旧状态并重新开始，而不是发送无效摘要。

## 阈值选择

先用保守值：

```python
CompactionMiddleware(
    summary_model=cheap_model,
    max_tokens_before_compact=80_000,
    keep_recent_messages=8,
    max_summary_tokens=1024,
)
```

如果模型上下文很大、希望减少摘要调用，可以提高
`max_tokens_before_compact`。如果最近工具输出或用户修正很重要，可以提高
`keep_recent_messages`。长时间研究或编码会话可提高 `max_summary_tokens`。

## Tracing

挂上 `cubepi.tracing` 时，摘要调用是 trace 树里的一等公民。`summarize()`
在 LLM 调用外包一个 `cubepi.compaction.summarize` 父 span（标签
`cubepi.compaction.message_count`），同时 recorder 自动订阅 summary
provider，所以它的 `chat` span 也落在里面：

```
invoke_agent
└── cubepi.turn
    ├── cubepi.compaction.summarize
    │   └── chat <summary-model>
    └── chat <main-model>
```

没装 OpenTelemetry 时，wrapper span 退化为 no-op context manager，中间件
行为不变。根 `invoke_agent` span 的 `gen_ai.provider.name` /
`cubepi.agent.system_prompt_sha256` / `cubepi.agent.tools` 始终归属
agent 的主 provider/model，不会被先跑的 summarizer 覆盖。

## 失败行为

如果摘要 provider 失败，CubePi 会记录 warning，并继续使用之前的压缩视图或
原始消息。agent 不会仅仅因为压缩刷新失败而失败。

## 什么时候不用

短任务、无状态 agent、或需要模型看到旧工具输出每个 token 的流程，不适合使用
compaction。这些场景里，简单的滑动窗口 `transform_context` hook 更容易推理。

