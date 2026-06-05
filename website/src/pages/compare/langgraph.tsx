import React from 'react';
import ComparePage, { type CompareContent } from '@site/src/components/Compare/ComparePage';
import { useIsZhHans } from '@site/src/hooks/useIsZhHans';

const EN: CompareContent = {
  them: 'LangGraph',
  title: 'CubePi vs LangGraph — a leaner Python agent framework',
  description:
    'CubePi vs LangGraph: a side-by-side comparison of two Python agent frameworks. CubePi models the agent as a plain async while-loop with append-only checkpointing and 3 core dependencies, instead of a state graph of nodes, edges, and channels.',
  keywords:
    'CubePi vs LangGraph, LangGraph alternative, Python agent framework, async agent framework, LangGraph vs CubePi, state graph alternative, langgraph 替代品',
  h1: 'CubePi vs LangGraph',
  intro: [
    'CubePi and LangGraph both build tool-using LLM agents in Python, but they start from opposite mental models. LangGraph asks you to express your agent as a state graph — nodes, edges, and typed channels you wire together. CubePi models the same agent as a plain async while-loop you can read top to bottom.',
    'If you find yourself drawing graphs to express what is fundamentally a linear "call the model → run tools → repeat" loop, CubePi is the leaner alternative. Here is how the two compare.',
  ],
  tableHeading: 'Side-by-side',
  rows: [
    { label: 'Abstraction', them: 'State graph: nodes + edges + channels you wire manually', us: 'A plain async while-loop — run_agent_loop reads top to bottom' },
    { label: 'Control flow', them: 'add_edge / add_conditional_edges reify the loop as a graph', us: 'The tool-call → re-prompt loop IS the runtime; you do not reify it' },
    { label: 'Streaming', them: 'Callback-based with multiple stream_mode flags', us: 'async for event in stream — one pattern, eleven event types' },
    { label: 'Checkpointing', them: 'Full snapshot per step; serializes the entire message list', us: 'Append-only — O(1) DB I/O regardless of conversation length' },
    { label: 'Dependencies', them: 'langchain-core, langgraph-sdk, and transitive deps', us: '3 core deps: pydantic, anthropic, openai' },
    { label: 'Tools', them: 'Tools are graph nodes (ToolNode) with manual wiring', us: 'Declare tools as functions; the framework routes and parallelizes' },
    { label: 'Async', them: 'Split invoke / ainvoke surface', us: 'Async-first — every entry point is async' },
    { label: 'Observability', them: 'LangSmith / Langfuse integration', us: 'Native OpenTelemetry — GenAI semconv, OTLP / JSONL out of the box' },
  ],
  code: {
    h2: 'A tool-using agent',
    themTitle: '# LangGraph',
    them: `from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"72F and sunny in {city}"


llm = ChatAnthropic(model="claude-sonnet-4-5").bind_tools([get_weather])

class State(TypedDict):
    messages: list

def call_model(state):
    return {"messages": [llm.invoke(state["messages"])]}

def should_continue(state):
    return "tools" if state["messages"][-1].tool_calls else END

graph = StateGraph(State)
graph.add_node("llm", call_model)
graph.add_node("tools", ToolNode([get_weather]))
graph.add_edge("__start__", "llm")
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tools", "llm")
app = graph.compile()
`,
    usTitle: '# CubePi',
    us: `from cubepi import Agent, tool
from cubepi.providers.anthropic import AnthropicProvider


@tool
async def get_weather(city: str) -> str:
    "Get current weather for a city."
    return f"72F and sunny in {city}"


agent = Agent(
    model=AnthropicProvider(provider_id="anthropic", api_key="...").model("claude-sonnet-4-5"),
    tools=[get_weather],
)
await agent.prompt("Weather in Tokyo?")
`,
  },
  sections: [
    {
      h2: 'Why the loop instead of a graph',
      body: [
        'A LangGraph agent never branches at runtime the way a general graph suggests — the "graph" is almost always the same shape: call the model, and if it asked for tools, run them and call the model again. CubePi makes that shape the runtime. There is no StateGraph, no END sentinel, no should_continue function, no ToolNode registry, and no State TypedDict to keep in sync.',
        'Flow control that does need to vary — stop early, summarize, gate a tool behind human approval — lives in typed middleware hooks instead of conditional edges, so it is imperative and testable in isolation.',
      ],
    },
    {
      h2: 'Persistence that does not grow with the conversation',
      body: [
        'LangGraph checkpointers snapshot the full state at every step, so write cost grows linearly with conversation length. CubePi checkpointing is append-only: each turn writes O(1) regardless of how long the thread is, and messages stay JSONB-queryable. The same MemorySaver / SqliteSaver / PostgresSaver idea maps onto MemoryCheckpointer / SQLiteCheckpointer / PostgresCheckpointer.',
      ],
    },
    {
      h2: 'When LangGraph is the better fit',
      body: [
        'LangGraph is the better choice if you genuinely need arbitrary multi-agent supervisor graphs, visual graph rendering, or time-travel/fork at arbitrary checkpoints today — CubePi keeps its flow linear by design and emits vendor-neutral OpenTelemetry rather than shipping its own trace UI. If your agent is fundamentally a loop, CubePi removes the graph machinery you were not really using.',
      ],
    },
  ],
  cta: [
    { text: 'Migration guide: from LangGraph →', href: '/docs/migration/from-langgraph' },
    { text: 'Quick Start →', href: '/docs/getting-started/quick-start' },
  ],
};

const ZH: CompareContent = {
  them: 'LangGraph',
  title: 'CubePi vs LangGraph — 更精简的 Python Agent 框架',
  description:
    'CubePi 与 LangGraph 对比:两个 Python Agent 框架的并排比较。CubePi 用普通的 async while 循环建模 agent,配合追加式 checkpointing 和 3 个核心依赖,而非由节点、边、通道组成的状态图。',
  keywords:
    'CubePi vs LangGraph, LangGraph 替代品, Python Agent 框架, 异步 Agent 框架, 状态图替代方案, langgraph alternative',
  h1: 'CubePi vs LangGraph',
  intro: [
    'CubePi 和 LangGraph 都用 Python 构建会调用工具的 LLM agent,但出发点完全相反。LangGraph 要求你把 agent 表达成一张状态图 —— 手动连接节点、边和类型化通道。CubePi 则把同样的 agent 建模为一个可以从上读到下的普通 async while 循环。',
    '如果你发现自己在用画图的方式去表达本质上线性的「调用模型 → 执行工具 → 重复」循环,CubePi 就是更精简的替代方案。下面是两者的对比。',
  ],
  tableHeading: '并排对比',
  rows: [
    { label: '抽象模型', them: '状态图:手动连接节点 + 边 + 通道', us: '普通 async while 循环 —— run_agent_loop 从上读到下' },
    { label: '控制流', them: 'add_edge / add_conditional_edges 把循环具象成图', us: '工具调用 → 再次提示 的循环本身就是运行时,无需具象化' },
    { label: '流式输出', them: '基于回调,多个 stream_mode 标志', us: 'async for event in stream —— 统一模式,11 种事件类型' },
    { label: '检查点', them: '每步全量快照,序列化整个消息列表', us: '追加式 —— 无论对话多长,每轮 O(1) DB I/O' },
    { label: '依赖项', them: 'langchain-core、langgraph-sdk 及传递依赖', us: '3 个核心依赖:pydantic、anthropic、openai' },
    { label: '工具', them: '工具是需要手动连线的图节点(ToolNode)', us: '声明为函数;框架自动路由并并行执行' },
    { label: '异步', them: 'invoke / ainvoke 两套接口', us: '异步优先 —— 每个入口都是 async' },
    { label: '可观测性', them: 'LangSmith / Langfuse 集成', us: '原生 OpenTelemetry —— GenAI semconv,开箱即用 OTLP / JSONL' },
  ],
  code: {
    h2: '一个会调用工具的 agent',
    themTitle: '# LangGraph',
    them: `from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"72F and sunny in {city}"


llm = ChatAnthropic(model="claude-sonnet-4-5").bind_tools([get_weather])

class State(TypedDict):
    messages: list

def call_model(state):
    return {"messages": [llm.invoke(state["messages"])]}

def should_continue(state):
    return "tools" if state["messages"][-1].tool_calls else END

graph = StateGraph(State)
graph.add_node("llm", call_model)
graph.add_node("tools", ToolNode([get_weather]))
graph.add_edge("__start__", "llm")
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tools", "llm")
app = graph.compile()
`,
    usTitle: '# CubePi',
    us: `from cubepi import Agent, tool
from cubepi.providers.anthropic import AnthropicProvider


@tool
async def get_weather(city: str) -> str:
    "Get current weather for a city."
    return f"72F and sunny in {city}"


agent = Agent(
    model=AnthropicProvider(provider_id="anthropic", api_key="...").model("claude-sonnet-4-5"),
    tools=[get_weather],
)
await agent.prompt("Weather in Tokyo?")
`,
  },
  sections: [
    {
      h2: '为什么用循环而不是图',
      body: [
        'LangGraph 的 agent 在运行时其实从不像通用图那样随意分支 —— 那张「图」几乎永远是同一个形状:调用模型,如果它请求了工具就执行,然后再次调用模型。CubePi 直接把这个形状变成运行时。没有 StateGraph、没有 END 哨兵、没有 should_continue 函数、没有 ToolNode 注册表,也没有需要同步维护的 State TypedDict。',
        '确实需要变化的流程控制 —— 提前停止、生成总结、把某个工具放到人工审批之后 —— 都放在类型化的 middleware hook 里,而非条件边,因此是命令式的,也能单独测试。',
      ],
    },
    {
      h2: '不随对话增长的持久化',
      body: [
        'LangGraph 的 checkpointer 在每一步都对完整状态做快照,写入成本随对话长度线性增长。CubePi 的 checkpointing 是追加式的:无论线程多长,每轮写入都是 O(1),且消息保持 JSONB 可查询。MemorySaver / SqliteSaver / PostgresSaver 的思路对应到 MemoryCheckpointer / SQLiteCheckpointer / PostgresCheckpointer。',
      ],
    },
    {
      h2: '什么时候 LangGraph 更合适',
      body: [
        '如果你现在确实需要任意的多 agent supervisor 图、可视化图渲染,或在任意检查点做时间旅行/分叉,那么 LangGraph 更合适 —— CubePi 在设计上保持流程线性,并输出厂商中立的 OpenTelemetry,而不是自带一套 trace UI。但如果你的 agent 本质上就是一个循环,CubePi 帮你去掉了那些你其实没真正用上的图机制。',
      ],
    },
  ],
  cta: [
    { text: '迁移指南:从 LangGraph 迁移 →', href: '/docs/migration/from-langgraph' },
    { text: '快速开始 →', href: '/docs/getting-started/quick-start' },
  ],
};

export default function CompareLangGraph(): React.ReactElement {
  const zh = useIsZhHans();
  return <ComparePage content={zh ? ZH : EN} />;
}
