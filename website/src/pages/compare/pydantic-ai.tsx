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
    { label: 'Structured output', them: 'Agent-level — Agent[Deps, OutputType] types every run; NativeOutput / PromptedOutput / ToolOutput modes', us: 'Call-level — BoundModel.generate_structured(Pydantic, …) returns a validated instance; tool-output mode under the hood (same as pydantic-ai default)' },
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
      h2: 'Where structured output sits in the API',
      body: [
        'Both frameworks ship first-class structured output backed by Pydantic. CubePi 0.10 added `BoundModel.generate_structured(Pydantic, ...)`, which injects a synthetic tool from the model\'s JSON schema, forces the call via `tool_choice`, and validates the response through `output_type.model_validate()` — the same `ToolOutput` mode PydanticAI uses by default.',
        'Where the two diverge is which abstraction owns the contract. PydanticAI lifts the output type up to the agent itself: `Agent[Deps, Sentiment].run(...)` types the whole run as `Sentiment`, and the framework offers `NativeOutput` (provider `response_format` / JSON-schema endpoints) and `PromptedOutput` as alternative modes alongside `ToolOutput`. CubePi keeps the agent loop as free-form text plus tool calls and exposes structured output as a one-shot `BoundModel` call you reach for when you need it — well-suited for extraction subroutines inside a larger multi-turn agent.',
        'CubePi optimises a different primary axis: multi-turn conversations that survive restarts. Append-only checkpointing keeps a thread\'s write cost flat regardless of conversation length, which matters when you have thousands of concurrent long-lived sessions.',
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
        'PydanticAI is the stronger choice when you want the agent itself typed by its output (`Agent[Deps, OutputType]`), when you need provider-native JSON-schema endpoints (`NativeOutput`) instead of the tool-output mode CubePi uses, or if you are already on Logfire and want tight integration. Choose CubePi when you need production-grade multi-turn persistence, composable middleware, provider failover, or vendor-neutral observability — and you are happy to reach for `BoundModel.generate_structured(...)` as a one-shot when you do need a validated Pydantic instance.',
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
    { label: '结构化输出', them: 'Agent 级 —— Agent[Deps, OutputType] 把整个 run 类型化；支持 NativeOutput / PromptedOutput / ToolOutput 三种模式', us: '调用级 —— BoundModel.generate_structured(Pydantic, …) 返回已校验的实例；底层走 tool-output 模式（与 pydantic-ai 默认相同）' },
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
      h2: '结构化输出在 API 里的位置',
      body: [
        '两者都把结构化输出做成一等公民、并以 Pydantic 为校验后端。CubePi 0.10 加了 `BoundModel.generate_structured(Pydantic, ...)`：它从模型的 JSON schema 注入一个合成 tool、通过 `tool_choice` 强制调用、再用 `output_type.model_validate()` 校验回来——这正是 PydanticAI 默认的 `ToolOutput` 模式。',
        '真正的差异是「契约绑在哪个抽象上」。PydanticAI 把输出类型抬到 Agent 本身：`Agent[Deps, Sentiment].run(...)` 把整个 run 的类型签到 `Sentiment`，且除了 `ToolOutput` 之外还提供 `NativeOutput`（Provider 的 `response_format` / JSON schema 端点）和 `PromptedOutput` 两种备选。CubePi 保持 agent 循环为自由文本 + 工具调用，把结构化输出做成一次性的 `BoundModel` 调用——适合作为多轮 agent 里的「抽取子例程」按需取用。',
        'CubePi 的主轴在另一边：可在重启后存活的多轮对话。追加式 checkpointing 让单线程的写入成本不随对话长度增长，在你有数千个并发长存会话时尤其重要。',
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
        '如果你希望 agent 本身被它的输出类型化（`Agent[Deps, OutputType]`）、需要 Provider 原生 JSON-schema 端点（`NativeOutput`）而不是 CubePi 现在用的 tool-output 模式、或者你已经在 Logfire 上希望紧密集成，那么 PydanticAI 是更强的选择。当你需要生产级多轮持久化、可组合中间件、Provider 故障转移或厂商中立可观测性，并且能接受用 `BoundModel.generate_structured(...)` 作为一次性子例程获取已校验的 Pydantic 实例时，选择 CubePi。',
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
