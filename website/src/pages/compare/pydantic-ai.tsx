import React from 'react';
import ComparePage, { type CompareContent } from '@site/src/components/Compare/ComparePage';
import { useIsZhHans } from '@site/src/hooks/useIsZhHans';

const EN: CompareContent = {
  them: 'PydanticAI',
  title: 'CubePi vs PydanticAI — Python agent frameworks compared',
  description:
    'CubePi vs PydanticAI: both are async-first Python agent frameworks built on Pydantic. PydanticAI centres on dependency injection and structured output. CubePi centres on append-only checkpointing, composable middleware, and vendor-neutral OpenTelemetry observability.',
  keywords:
    'CubePi vs PydanticAI, PydanticAI alternative, pydantic-ai alternative, Python agent framework, async agent framework, PydanticAI vs CubePi, pydantic ai 替代品',
  h1: 'CubePi vs PydanticAI',
  intro: [
    'PydanticAI and CubePi share the same foundation — Pydantic v2, asyncio-native design, and a belief that agents should be plain Python rather than graph-based state machines. Where they diverge is in the abstractions built on top.',
    'PydanticAI centres on type-safe structured output and dependency injection via `RunContext`. CubePi centres on persistent multi-turn conversations, composable middleware hooks, and vendor-neutral OpenTelemetry tracing. Here is how the two compare.',
  ],
  tableHeading: 'Side-by-side',
  rows: [
    { label: 'Core abstraction', them: 'Typed Agent[OutputType] with dependency injection', us: 'Stateful Agent with a while-loop core — composable via middleware' },
    { label: 'Structured output', them: 'First-class — result_type enforces Pydantic model output', us: 'Via tool return types; use a "final answer" tool for structured output' },
    { label: 'Dependency injection', them: 'RunContext[Deps] — deps injected at run time', us: 'Pass context through tool closures or middleware; no DI container' },
    { label: 'Checkpointing', them: 'No built-in persistence layer', us: 'Append-only — O(1) DB I/O; backends: memory, SQLite, Postgres, MySQL' },
    { label: 'Streaming', them: 'async for chunk in agent.run_stream()', us: 'async for event in stream — 11 typed event types including tool lifecycle' },
    { label: 'Multi-provider', them: 'Anthropic, OpenAI, Gemini, Groq, and more', us: 'Anthropic & OpenAI built in; Provider protocol for custom backends' },
    { label: 'Provider fallback', them: 'Not built in', us: 'FallbackBoundModel — auto-failover on rate-limit or outage' },
    { label: 'Observability', them: 'Logfire (Pydantic\'s own platform)', us: 'Native OpenTelemetry — GenAI semconv, OTLP / JSONL; vendor-neutral' },
    { label: 'Middleware', them: 'No middleware system', us: '8 typed hooks with declarative composition rules' },
    { label: 'Testing', them: 'TestModel for deterministic tests', us: 'FauxProvider — realistic streaming deltas, no API keys needed' },
    { label: 'Core deps', them: 'pydantic-ai-slim + provider adapters', us: 'pydantic, anthropic, openai — everything else is an optional extra' },
  ],
  code: {
    h2: 'A tool-using agent',
    themTitle: '# PydanticAI',
    them: `from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel


model = AnthropicModel("claude-sonnet-4-6")
agent = Agent(model, system_prompt="You are a helpful weather assistant.")


@agent.tool_plain
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"72F and sunny in {city}"


result = await agent.run("What's the weather in Tokyo?")
print(result.output)
`,
    usTitle: '# CubePi',
    us: `from cubepi import Agent, tool
from cubepi.providers.anthropic import AnthropicProvider


@tool
async def get_weather(city: str) -> str:
    "Get current weather for a city."
    return f"72F and sunny in {city}"


provider = AnthropicProvider(provider_id="anthropic", api_key="...")
agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant.",
)
await agent.prompt("What's the weather in Tokyo?")
`,
  },
  sections: [
    {
      h2: 'Structured output vs persistent conversation',
      body: [
        'PydanticAI\'s standout feature is `result_type`: you declare the Pydantic model you want back and the framework forces the model to return it. This is excellent for batch extraction pipelines and single-turn tasks where the output shape matters more than the conversation.',
        'CubePi optimises for the other axis — multi-turn conversations that survive restarts. Append-only checkpointing means a thread\'s write cost does not grow with conversation length, which matters when you have thousands of concurrent long-lived sessions. For structured output in CubePi, define a tool whose return value IS the structured result and instruct the model to call it when done.',
      ],
    },
    {
      h2: 'Dependency injection vs middleware',
      body: [
        'PydanticAI\'s `RunContext[Deps]` is a clean pattern for injecting services (databases, HTTP clients) into tool functions. CubePi achieves the same via closures — capture your dependencies when you define the tool function — which requires no new abstraction to learn.',
        'Where CubePi adds structure is in cross-cutting concerns: middleware hooks (`before_tool_call`, `transform_context`, `should_stop_after_turn`, etc.) let you add rate-limiting, safety checks, compaction, or subagent orchestration without touching the core agent loop.',
      ],
    },
    {
      h2: 'Observability: Logfire vs vendor-neutral OTel',
      body: [
        'PydanticAI integrates with Logfire, Pydantic\'s own observability platform. It works well if Logfire fits your stack. CubePi emits standard OpenTelemetry spans with GenAI semantic-convention attributes that land in any OTLP-compatible backend — Jaeger, Grafana Tempo, Honeycomb, Datadog, AWS X-Ray — with no vendor dependency. The `cubepi trace` CLI also lets you inspect traces locally from JSONL files without a backend at all.',
      ],
    },
    {
      h2: 'When PydanticAI is the better fit',
      body: [
        'PydanticAI is the stronger choice for single-turn structured extraction pipelines where you need guaranteed output schemas, or if you are already using Logfire and want tight integration. Choose CubePi when you need production-grade multi-turn persistence, composable middleware, provider failover, or vendor-neutral observability.',
      ],
    },
  ],
  cta: [
    { text: 'Quick Start →', href: '/docs/getting-started/quick-start' },
    { text: 'Core Concepts →', href: '/docs/getting-started/core-concepts' },
  ],
};

const ZH: CompareContent = {
  them: 'PydanticAI',
  title: 'CubePi vs PydanticAI — Python Agent 框架对比',
  description:
    'CubePi 与 PydanticAI 对比：两者都是基于 Pydantic 的异步优先 Python Agent 框架。PydanticAI 聚焦于依赖注入和结构化输出；CubePi 聚焦于追加式 checkpointing、可组合中间件和厂商中立的 OpenTelemetry 可观测性。',
  keywords:
    'CubePi vs PydanticAI, PydanticAI 替代品, pydantic-ai 替代品, Python Agent 框架, 异步 Agent 框架, pydantic ai 替代品',
  h1: 'CubePi vs PydanticAI',
  intro: [
    'PydanticAI 和 CubePi 共享同一基础 —— Pydantic v2、asyncio 原生设计，以及"Agent 应该是普通 Python 而非基于图的状态机"的理念。两者的分歧在于在此之上构建的抽象。',
    'PydanticAI 聚焦于通过 `RunContext` 实现类型安全的结构化输出和依赖注入。CubePi 聚焦于持久化多轮对话、可组合的中间件钩子和厂商中立的 OpenTelemetry 追踪。以下是两者的详细对比。',
  ],
  tableHeading: '并排对比',
  rows: [
    { label: '核心抽象', them: '带依赖注入的类型化 Agent[OutputType]', us: '有状态 Agent，核心为 while 循环 —— 通过中间件可组合' },
    { label: '结构化输出', them: '一等公民 —— result_type 强制 Pydantic 模型输出', us: '通过工具返回类型；用"最终答案"工具实现结构化输出' },
    { label: '依赖注入', them: 'RunContext[Deps] —— 运行时注入依赖', us: '通过工具闭包或中间件传递上下文；无 DI 容器' },
    { label: 'Checkpointing', them: '无内置持久化层', us: '追加式 —— O(1) DB I/O；后端：memory、SQLite、Postgres、MySQL' },
    { label: '流式输出', them: 'async for chunk in agent.run_stream()', us: 'async for event in stream —— 11 种类型化事件，含工具生命周期' },
    { label: '多 Provider', them: 'Anthropic、OpenAI、Gemini、Groq 等', us: '内置 Anthropic 和 OpenAI；Provider 协议支持自定义后端' },
    { label: 'Provider 故障转移', them: '无内置支持', us: 'FallbackBoundModel —— 限速或故障时自动切换' },
    { label: '可观测性', them: 'Logfire（Pydantic 自有平台）', us: '原生 OpenTelemetry —— GenAI 语义约定，OTLP / JSONL；厂商中立' },
    { label: '中间件', them: '无中间件系统', us: '8 个类型化钩子，带声明式组合规则' },
    { label: '测试', them: 'TestModel 用于确定性测试', us: 'FauxProvider —— 真实流式仿真，无需 API Key' },
    { label: '核心依赖', them: 'pydantic-ai-slim + provider 适配器', us: 'pydantic、anthropic、openai —— 其余皆为可选 extra' },
  ],
  sections: [
    {
      h2: '结构化输出 vs 持久化对话',
      body: [
        'PydanticAI 的核心亮点是 `result_type`：你声明想要返回的 Pydantic 模型，框架强制模型输出它。对于输出格式比对话更重要的批量提取流水线和单轮任务，这非常出色。',
        'CubePi 在另一个轴上优化 —— 能在重启后存活的多轮对话。追加式 checkpointing 意味着线程的写入成本不会随对话长度增长，当你有数千个并发长存会话时，这一点至关重要。在 CubePi 中实现结构化输出，可以定义一个返回值即为结构化结果的工具，并指示模型在完成时调用它。',
      ],
    },
    {
      h2: '依赖注入 vs 中间件',
      body: [
        'PydanticAI 的 `RunContext[Deps]` 是将服务（数据库、HTTP 客户端）注入工具函数的简洁模式。CubePi 通过闭包实现同样效果 —— 在定义工具函数时捕获依赖 —— 无需学习新抽象。',
        'CubePi 在横切关注点上增加了结构：中间件钩子（`before_tool_call`、`transform_context`、`should_stop_after_turn` 等）让你在不触碰核心 agent 循环的情况下添加限速、安全检查、上下文压缩或子 Agent 编排。',
      ],
    },
    {
      h2: '可观测性：Logfire vs 厂商中立的 OTel',
      body: [
        'PydanticAI 与 Logfire（Pydantic 自有的可观测性平台）集成。如果 Logfire 适合你的技术栈，效果很好。CubePi 输出带有 GenAI 语义约定属性的标准 OpenTelemetry span，可以落入任何 OTLP 兼容后端 —— Jaeger、Grafana Tempo、Honeycomb、Datadog、AWS X-Ray —— 无厂商依赖。`cubepi trace` CLI 也可以让你在没有后端的情况下从 JSONL 文件本地检查追踪。',
      ],
    },
    {
      h2: 'PydanticAI 更适合的场景',
      body: [
        '如果你需要保证输出 schema 的单轮结构化提取流水线，或者你已经在使用 Logfire 并希望紧密集成，PydanticAI 是更强的选择。当你需要生产级多轮持久化、可组合中间件、Provider 故障转移或厂商中立可观测性时，选择 CubePi。',
      ],
    },
  ],
  cta: [
    { text: '快速开始 →', href: '/docs/getting-started/quick-start' },
    { text: '核心概念 →', href: '/docs/getting-started/core-concepts' },
  ],
};

export default function ComparePydanticAI(): React.ReactElement {
  const zh = useIsZhHans();
  return <ComparePage content={zh ? ZH : EN} />;
}
