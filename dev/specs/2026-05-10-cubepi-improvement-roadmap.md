# cubepi 改进路线图

> 基于与 pi-agent-core / pi-ai 的对比审查，按优先级排列。

## P0 — 类型安全与正确性

这些问题直接影响代码正确性、可维护性和开发者体验。

### 1. 消除 `list[Any]` — 引入 `AgentMessage` 联合类型

**现状**: `AgentContext.messages`、`AgentState._messages`、事件类型中的 `message` 字段大量使用 `Any`，导致：
- 无法静态类型检查，IDE 无法补全
- 运行时靠 `hasattr(msg, "role")` 判断类型，脆弱且不安全
- 回调签名全部是 `Callable | None`，无参数类型信息

**Pi 做法**: 定义 `AgentMessage = Message | CustomAgentMessages[keyof CustomAgentMessages]`，所有回调签名完整标注。

**改动范围**:
- 定义 `AgentMessage = UserMessage | AssistantMessage | ToolResultMessage`
- `AgentContext.messages: list[AgentMessage]`
- `AgentState._messages: list[AgentMessage]`
- 所有事件类型的 `message` 字段标注为 `AgentMessage` 或具体类型
- `Agent.__init__` 的回调参数加上完整签名（参考 pi 的 `AgentOptions`）
- 消除 `_default_convert_to_llm` 中的 `hasattr` 检查
- 消除 `_process_event` 中的 `hasattr` 检查

### 2. `StreamEvent` 增加 `content_index`

**现状**: `StreamEvent` 没有 `content_index`，provider 实现依赖"partial.content 的最后一个元素"来判断当前正在流式传输的 content block。当内容块交错时（如 text → tool_call → text），定位会出错。

**Pi 做法**: 所有 content 事件都携带 `contentIndex: number`。

**改动范围**:
- `StreamEvent` 新增 `content_index: int | None = None`
- 三个 provider 实现（Anthropic、OpenAI、OpenAI Responses）在生成事件时填充 `content_index`
- FauxProvider 同步更新
- `MessageUpdateEvent` 传递 `content_index` 信息

### 3. `AssistantMessage` 补充 provider 元数据

**现状**: `AssistantMessage` 不记录是哪个 provider/model 生成的，无法追踪来源。

**Pi 做法**: `AssistantMessage` 包含 `api`、`provider`、`model`、`responseModel`、`responseId`、`diagnostics`。

**改动范围**:
- `AssistantMessage` 新增 `provider_id: str = ""`、`model_id: str = ""`、`response_id: str | None = None`
- 各 provider 在 `_convert_response` / 构造 final message 时填充这些字段
- 对现有代码无破坏性影响（新增字段有默认值）

---

## P1 — 功能补全

这些是 pi-agent 有而 cubepi 缺失的功能，影响实际可用性。

### 4. Checkpointer 集成到 Agent

**现状**: `Agent.__init__` 接受 `checkpointer` 和 `thread_id` 参数，`MemoryCheckpointer` 和 `SQLiteCheckpointer` 实现完整，但 Agent 从未调用它们。这是死代码。

**改动范围**:
- `Agent.prompt()` 启动时，若有 checkpointer + thread_id，调用 `load()` 恢复历史消息
- `_process_event` 中 `message_end` 时调用 `append()`
- `Agent.reset()` 决定是否清除持久化数据（可通过参数控制）
- 补充集成测试

### 5. `ToolResultMessage` 传递 `details`

**现状**: `AgentToolResult` 有 `details` 字段，但 `_make_tool_result_message()` 构造 `ToolResultMessage` 时丢弃了它。

**Pi 做法**: `ToolResultMessage<TDetails>` 有 `details?: TDetails`，全程传递。

**改动范围**:
- `ToolResultMessage` 新增 `details: Any = None`
- `_make_tool_result_message()` 填充 `finalized.result.details`
- `__init__.py` 导出不变（`ToolResultMessage` 已导出）

### 6. 循环开始时轮询 steering 消息

**现状**: `_run_loop` 只在工具执行后检查 steering 消息。如果用户在 agent 初始化期间发送了消息，会被丢失。

**Pi 做法**: `runLoop` 在循环开始前执行 `let pendingMessages = (await config.getSteeringMessages?.()) || []`。

**改动范围**:
- `_run_loop` 入口处增加一次 `get_steering_messages()` 调用
- 将取得的消息注入到第一轮 assistant 响应之前

### 7. OpenAI Provider 支持图片输入

**现状**: `OpenAIProvider._convert_message()` 对 `UserMessage` 只提取 `TextContent`，`ImageContent` 被丢弃。

**改动范围**:
- `_convert_message` 检测 `ImageContent`，转换为 OpenAI 的 `image_url` 格式
- `OpenAIResponsesProvider._build_input` 同理处理图片

---

## P2 — 代码质量与健壮性

### 8. 消除重复的 `_emit` 辅助函数

**现状**: `loop.py` 和 `tools.py` 各定义了一个相同的 `_emit` 函数。

**改动范围**:
- 将 `_emit` 移到共享位置（如 `agent/types.py` 或新的 `agent/_utils.py`）
- 两处导入替换

### 9. 修复 fire-and-forget `asyncio.create_task`

**现状**: 所有 provider 的 `stream()` 方法用 `asyncio.create_task(_produce())` 启动生产者协程，但不保存 task 引用。若 task 在推送 error 事件之前就失败，异常会被静默吞掉。

**改动范围**:
- `MessageStream` 持有 task 引用
- `MessageStream.result()` 等待时同时检查 task 异常
- 或改为在 `stream()` 方法中保存 task 引用并设置 done callback

### 10. 清理 OpenAI Responses 的 system prompt 死代码

**现状**: `openai_responses.py:78-86` 先设置 `kwargs["instructions"]`，又删除它，中间的赋值是死代码。

**改动范围**:
- 删除 `kwargs["instructions"] = system_prompt` 和 `del kwargs["instructions"]`
- 只保留 input 前置逻辑

### 11. 私有函数跨模块导入规范化

**现状**: `anthropic.py` 和 `openai.py` 导入 `_invoke_on_payload` 和 `_invoke_on_response`（下划线前缀表示私有）。

**改动范围**:
- 去掉下划线前缀，改为 `invoke_on_payload` / `invoke_on_response`
- 或将它们移到 `StreamOptions` 的方法中

---

## P3 — API 增强

这些是 pi-agent 有但 cubepi 当前不急需的能力，可在需要时再实现。

### 12. Agent 级别的 EventStream

**现状**: 只有 callback-based 的 `run_agent_loop`。Pi 还提供 `agentLoop()` 返回 `EventStream<AgentEvent, AgentMessage[]>`，可用 `async for` 消费。

**参考**: Pi 的 `EventStream` 类和 `createAgentStream()` 工厂。

### 13. 动态 API Key 解析 (`get_api_key`)

**现状**: Provider 在构造时绑定 API key。对于短期 OAuth token（如 GitHub Copilot），长时间运行的 tool 执行期间 token 可能过期。

**Pi 做法**: `AgentLoopConfig.getApiKey` 在每次 LLM 调用前动态解析 key。

### 14. 工具参数预处理 (`prepare_arguments`)

**现状**: 工具参数直接用 Pydantic 校验，无法做兼容性转换。

**Pi 做法**: `AgentTool.prepareArguments` 可在校验前转换原始参数。

### 15. Queue 管理 API

**现状**: Agent 没有暴露 `clear_steering_queue()`、`clear_follow_up_queue()`、`has_queued_messages()` 等方法。只有 `reset()` 会清除所有状态。

**Pi 做法**: 提供细粒度的队列管理方法。

### 16. 重试与 `max_retry_delay_ms`

**现状**: Provider 没有重试逻辑。

**Pi 做法**: `StreamOptions.maxRetryDelayMs` 控制重试上限，超时则立即失败并向上层报告。

---

## 实施建议

- P0 的 3 项应当在下一个迭代周期内完成，它们是基础设施级改进
- P1 的 4 项可以独立分支并行推进
- P2 的 4 项可以合并为一个代码清理 PR
- P3 的 5 项按实际需求触发，不必提前实现
