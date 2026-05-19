---
title: 工具调用
---

# 工具使用与并行执行

工具是 Agent 影响世界的方式。CubePi 把每个 `AgentTool` 转成模型可用
的 JSON Schema、用 Pydantic 校验参数、执行你的代码、把结果作为
`ToolResultMessage` 喂回。默认情况下,只要模型在一轮里发出多个工具
调用,它们就会并行执行。

## 工具的结构

```python
from pydantic import BaseModel, Field
from cubepi import AgentTool, AgentToolResult, TextContent


class SearchParams(BaseModel):
    query: str = Field(..., description="自然语言查询")
    limit: int = Field(10, ge=1, le=100)


async def search(tool_call_id, params: SearchParams, *, signal=None, on_update=None):
    results = await my_search_backend(params.query, params.limit)
    return AgentToolResult(
        content=[TextContent(text="\n".join(results))],
        details={"raw_results": results},   # 透传到 ToolResultMessage.details
    )


search_tool = AgentTool(
    name="search",
    description="在内部知识库里搜索。",
    parameters=SearchParams,
    execute=search,
)
```

`description` 是直接展示给模型的 —— 给模型写,不是给人看。Pydantic
的 `Field(description=…)` 会进 JSON Schema,帮模型理解每个参数。

## 默认并行

模型在一条 assistant 消息里发多个工具调用时,CubePi 会用
`asyncio.create_task()` 调度它们并 gather。这通常就是你想要的。

```python
agent = Agent(
    provider=provider,
    model=model,
    tools=[search_tool, fetch_url_tool, summarise_tool],
)
```

事件流会先一次性发出所有 `tool_execution_start`,中间穿插每个工具
报告进度的 `tool_execution_update`,最后按完成顺序发
`tool_execution_end`。

## 强制顺序执行

两种方式：

1. **整个 agent 级别** —— `Agent(tool_execution="sequential")`。所有
   工具批次按模型发出的顺序逐个执行。

2. **单个工具级别** —— 在 `AgentTool` 上设
   `execution_mode="sequential"`。一旦当前批次里 *任意* 一个工具是
   sequential,整个批次都退化为顺序执行。

    ```python
    write_db_tool = AgentTool(
        name="write_db",
        description="持久化一条记录。",
        parameters=WriteDbParams,
        execute=write_db,
        execution_mode="sequential",   # 出于安全考虑放弃并行
    )
    ```

工具会修改共享状态（DB、计数器）且你需要确定顺序时,选 sequential。

## 流式回报工具进度

长耗时工具可以推送增量更新,以 `tool_execution_update` 事件呈现：

```python
async def slow_search(tool_call_id, params, *, signal=None, on_update=None):
    for i, page in enumerate(await fetch_pages(params.query)):
        if signal and signal.is_set():
            break
        if on_update:
            on_update({"progress": i, "total": len(pages), "url": page.url})
        await process(page)
    return AgentToolResult(content=[TextContent(text="done")])
```

事件里的 `partial_result` 就是你传给 `on_update` 的对象。用小 dict 就好,
它不会进模型的 context。

## 取消正在跑的工具

`signal` 就是 `agent.abort()` set 的那个 `asyncio.Event`。在任何循环
里检查它：

```python
async def long_running(tool_call_id, params, *, signal=None, on_update=None):
    for chunk in big_dataset:
        if signal and signal.is_set():
            return AgentToolResult(content=[TextContent(text="cancelled")])
        await process_chunk(chunk)
```

如果工作是一个大 `await`,用 `asyncio.wait_for(..., timeout=…)` 包一下,
或调用底层库自己的取消方法。

## 返回错误

两种姿势：

1. **抛异常。** CubePi 捕获后转成 `is_error=True` 的 `AgentToolResult`,
   异常字符串作为 `TextContent`。
2. **显式返回 `is_error=True`。** 适合你想给结构化错误体的场景：

    ```python
    return AgentToolResult(
        content=[TextContent(text="超出限流,60 秒后再试")],
        is_error=True,
    )
    ```

不管哪种方式,模型都会收到一个明确标记错误的工具结果,通常会自适应
（换参数重试、问用户等）。

## 从工具结束本轮:`terminate`

工具可以声明 *"这次之后,不要再循环到模型了。"* 设
`terminate=True`：

```python
async def submit_final_answer(tool_call_id, params, *, signal=None, on_update=None):
    save_answer(params.answer)
    return AgentToolResult(
        content=[TextContent(text="submitted")],
        terminate=True,
    )
```

CubePi 仅在当前批次中 *每个* 工具结果都是 `terminate=True` 时才终止。
然后循环发 `turn_end`、`agent_end`,退出。

## 常见坑

- **忘了 keyword-only 参数** —— 开发时 `execute(tool_call_id, params)`
  能跑,但框架传 `signal=` 时会崩。签名一定保留
  `*, signal=None, on_update=None`。
- **`details` 塞太大** —— `details` 透传到 agent 事件里,但 **不会**
  给模型看。除非你下游有消费者,否则别堆大 blob。
- **Pydantic 严格度的意外** —— `Field(..., min_length=1)` 让模型通过
  JSON Schema 看到约束 —— 约束有帮助,但模型仍然偶尔发坏 JSON。
  CubePi 把 `ValidationError` 转成工具的 error result,你不用自己包。
- **`tools=[]` 但模型还是想用工具** —— 一般是 system prompt 里提到了
  工具。要么删掉提示,要么真的把工具给它。

## 另请参阅

- [流式事件](./streaming) —— `tool_execution_*` 事件如何嵌入到事件
  分类体系。
- [Middleware → before_tool_call](../middleware/hooks#before_tool_call)
  和 [after_tool_call](../middleware/hooks#after_tool_call) ——
  拦截、策略、重试。
- [Recipes → Weather Agent](../../recipes/weather-agent) —— 一个真发
  HTTP 请求的工具,端到端。
- [MCP 加载](../mcp/loading) —— 一次性把一个 MCP server 的整套工具
  拉下来。
