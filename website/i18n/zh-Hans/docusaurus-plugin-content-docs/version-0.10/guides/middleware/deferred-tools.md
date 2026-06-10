---
title: 延迟工具组
description: "默认对模型隐藏 MCP 工具 schema，按需展开，减少上下文膨胀。"
---

# 延迟工具组

当一个 agent 接入多个 MCP 服务器时，它们合起来的 tool schema 在每一轮都
能吃掉数千个 token 上下文——即使模型这一回合只需要其中一两组。
`DeferredToolGroup` 用一份紧凑的目录替代完整 schema，让模型按需展开
工具组。

## 工作方式

1. 构造时，agent 的 system prompt 里包含一份简短目录——每个组一行，
   带描述和工具列表。
2. 模型看到一个内置的 `load_tools` 工具，可以调用它来加载一组（或
   组内的某些工具）。
3. 展开时，loader 跑一次，工具被注入到运行中的 tool 集，schema 被
   追加到 system prompt 末尾。

```
# Deferred tool groups

These tool groups are available but not yet loaded. Call `load_tools(group_id)`
to load a group's tools for the rest of this conversation.

- `mcp:github` — GitHub: Issues, PRs, repos, code search (4 tools)
  create_issue, search_repos, create_pr, list_comments
- `mcp:linear` — Linear: Project management and issue tracking (6 tools)
  create_issue, update_issue, list_projects, ...
```

## 基础用法

把 `deferred_tool_groups` 传给 `Agent`。中间件会自动创建——不需要手动
拼装：

```python
from cubepi import Agent
from cubepi.deferred import DeferredToolGroup

github_group = DeferredToolGroup(
    group_id="mcp:github",
    display_name="GitHub",
    description="Issues, PRs, repos, code search",
    tool_names=["create_issue", "search_repos", "create_pr", "list_comments"],
    loader=github_mcp.load_tools,  # async () -> list[AgentTool]
)

linear_group = DeferredToolGroup(
    group_id="mcp:linear",
    display_name="Linear",
    description="Project management and issue tracking",
    tool_names=["create_issue", "update_issue", "list_projects"],
    loader=linear_mcp.load_tools,
)

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    tools=[search_tool, calculator],              # 始终可用的 tool
    deferred_tool_groups=[github_group, linear_group],
)
```

### `DeferredToolGroup` 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `group_id` | `str` | 模型在 `load_tools` 调用里使用的唯一 ID（如 `"mcp:github"`） |
| `display_name` | `str` | 目录里展示的人类可读名称 |
| `description` | `str` | 该组能力的一行摘要 |
| `tool_names` | `list[str]` | 目录里列出的 tool 名 |
| `loader` | `async () -> list[AgentTool]` | 返回该组完整 tool 集的回调 |

## `load_tools` 工具

模型通过 `load_tools` 加载一组的 tool。两种模式：

```
# 展开整组
load_tools(group_id="mcp:github")

# 只展开指定的 tool
load_tools(group_id="mcp:github", tool_names=["create_issue", "search_repos"])
```

工具返回结构化结果：

```json
{
  "group_id": "mcp:github",
  "expanded": true,
  "tool_names": ["create_issue", "search_repos", "create_pr", "list_comments"],
  "remaining": 0
}
```

展开后这些 tool 同一轮就可以被模型调用（通过 `after_tool_call` hook）。

### 选择性展开

模型可以分批次展开一个组——现在要一两个，稍后再要更多：

```
load_tools(group_id="mcp:github", tool_names=["create_issue"])
# → remaining: 3

# 稍后……
load_tools(group_id="mcp:github", tool_names=["search_repos"])
# → remaining: 2
```

已经展开过的 tool 是幂等的——再请求一次是 no-op。

### Loader 缓存

`loader` 回调在**每个组、每次 run** 里只会被调用**一次**。第一次
`load_tools` 调用触发它；后续的选择性展开从缓存结果里筛选。如果 loader
失败，错误会回给模型，组保持未展开。

## Prompt 缓存稳定性

System prompt 的设计目标是 prompt 缓存的前缀稳定：

- **目录**按 `group_id` 字典序排序——输入顺序无关，渲染出来的文本
  字节稳定。
- **展开后的 schema** 按展开顺序（模型调用 `load_tools` 的顺序）追加
  到末尾，从不重排。每次新展开只在尾部追加，保留已有的前缀。

这意味着 LLM API 的 prompt 缓存在轮次间一直有效：system prompt 只增长，
而且只在末尾增长。

## 展开状态

中间件把哪些组已展开记录在 `ctx.extra` 里：

```python
ctx.extra["expanded_groups"] = {
    "mcp:github": None,                    # 完全展开（None = 全部 tool）
    "mcp:linear": ["create_issue"],        # 部分展开
    # mcp:slack 不在 = 未展开
}
```

这份状态会跟着 checkpoint 一起持久化，可用于跨 run 恢复（见下文）。

## 跨 run 恢复

从上一次 run 恢复对话时，你需要把展开状态还原，让模型有相同的工具
可用。`prepare_resumed_state` 负责这件事：

```python
from cubepi.deferred import DeferredToolsMiddleware

# saved_extra 是上一次 run 持久化下来的 ctx.extra
resumed = await DeferredToolsMiddleware.prepare_resumed_state(
    groups=all_groups,
    expanded=saved_extra["expanded_groups"],
)

agent = Agent(
    model=model,
    tools=[*builtin_tools, *resumed.pre_loaded_tools],
    deferred_tool_groups=resumed.remaining_groups,
)
```

`prepare_resumed_state` 返回一个 `ResumedState`：

| 字段 | 说明 |
|---|---|
| `pre_loaded_tools` | 此前已展开组的 tool，已就绪可用 |
| `remaining_groups` | 未展开或部分展开的组 |
| `expanded_schemas` | 用于 system prompt 的 schema 数据（高级用法时传入 `resumed_schemas`） |
| `loader_cache` | 已加载的 tool 缓存（传给 `resumed_loader_cache` 可避免重复调用 loader） |

完全展开的组会被加载并从延迟集合中移除。部分展开的组会加载已选择的
tool，但仍保留为可延迟（模型仍可展开余下部分）。

### 还原 schema 文本

`Agent(deferred_tool_groups=...)` 这种简写覆盖常见用例。如果要做完整的
prompt 缓存连续性——也就是恢复后的 run 的 system prompt 必须和上一次
最终状态字节一致——直接构造中间件并传 `resumed_schemas`：

```python
mw = DeferredToolsMiddleware(
    groups=resumed.remaining_groups,
    extra_ref=lambda: agent_extra,
    resumed_schemas=resumed.expanded_schemas,
    resumed_loader_cache=resumed.loader_cache,
)

agent = Agent(
    model=model,
    tools=[*builtin_tools, *resumed.pre_loaded_tools],
    middleware=[mw],
)
```

## 进阶：直接构造中间件

如果要完全控制目录头、跨 run schema 种子或其他中间件参数，直接构造
`DeferredToolsMiddleware`：

```python
from cubepi.deferred import DeferredToolsMiddleware

mw = DeferredToolsMiddleware(
    groups=[github_group, linear_group],
    extra_ref=lambda: agent_extra,
    catalog_header="# Available integrations\n\nExpand with load_tools().",
    resumed_schemas=None,  # 或者传入上一次 run 的 schema
)

agent = Agent(
    model=model,
    tools=[search_tool],
    middleware=[mw],
)
```

### 构造参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `groups` | `list[DeferredToolGroup]` | 必填 | 要延迟的组 |
| `extra_ref` | `() -> dict` | 必填 | 返回当前的 `ctx.extra` dict |
| `catalog_header` | `str` | *(内置)* | 目录区段的头文本 |
| `resumed_schemas` | `list[tuple[str, list[dict]]] \| None` | `None` | 用于种子的 schema 数据 |
| `resumed_loader_cache` | `dict[str, list[AgentTool]] \| None` | `None` | 上一次 run 的 tool 缓存（恢复时避免重复调 loader） |
| `on_tools_expanded` | `(list[AgentTool]) -> None \| None` | `None` | 新 tool 展开后回调（内部用于跨轮持久化） |

使用 `Agent(deferred_tool_groups=...)` 这种简写时，`extra_ref` 会自动
绑到 `self._extra`。

## 何时使用

**适用：**

- Agent 接入 5+ 个 MCP 服务器，但每次对话通常只用 1–2 组。
- Tool schema 很大（参数多、描述长）。
- 你希望跨轮保持高 prompt 缓存命中率。

**不适用：**

- Agent 只有少量 tool——目录和 `load_tools` 调用的开销不划算。
- 每一轮都要全部 tool——延迟只多一次往返。
- Tool schema 本身很小——上下文节省微乎其微。

## 参见

- [加载 MCP 工具](../mcp/loading)——如何从 MCP 服务器拿到
  `AgentTool` 列表。
- [8 个 Hook](./hooks)——驱动延迟工具的两个中间件 hook
  （`transform_system_prompt`、`after_tool_call`）。
- [组合](./composition)——多个中间件叠加时怎么组合。
