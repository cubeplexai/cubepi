import React from 'react';
import ComparePage, { type CompareContent } from '@site/src/components/Compare/ComparePage';
import { useIsZhHans } from '@site/src/hooks/useIsZhHans';

const EN: CompareContent = {
  them: 'CrewAI',
  title: 'CubePi vs CrewAI — a leaner Python agent framework',
  description:
    'CubePi vs CrewAI: a side-by-side comparison. CrewAI models agents as role-playing crew members with assigned tasks. CubePi models the agent as a plain async while-loop with typed middleware, append-only checkpointing, and 3 core dependencies.',
  keywords:
    'CubePi vs CrewAI, CrewAI alternative, Python agent framework, async agent framework, CrewAI vs CubePi, multi-agent framework, crewai 替代品',
  h1: 'CubePi vs CrewAI',
  intro: [
    'CrewAI organises your agent system around role-playing metaphors — crews, agents with roles and backstories, and tasks that agents are assigned to. CubePi starts from a different premise: an agent is a plain async while-loop that calls the model, runs tools, and repeats.',
    'If the role-based abstraction is solving a problem you do not have, CubePi removes that layer. Here is how the two compare.',
  ],
  tableHeading: 'Side-by-side',
  rows: [
    { label: 'Abstraction', them: 'Crews of role-playing agents with roles, goals, and backstories', us: 'A plain async while-loop — readable top to bottom in five minutes' },
    { label: 'Multi-agent', them: 'First-class: hierarchical or sequential process types', us: 'Subagent middleware — spawn child agents, await results, compose freely' },
    { label: 'Async', them: 'Synchronous by default; async support added later', us: 'Async-first — every entry point is async' },
    { label: 'Streaming', them: 'Limited streaming support', us: 'async for event in stream — one pattern, eleven event types' },
    { label: 'Checkpointing', them: 'No built-in persistence layer', us: 'Append-only — O(1) DB I/O; backends: memory, SQLite, Postgres, MySQL' },
    { label: 'Dependencies', them: 'langchain-core + many transitive deps', us: '3 core deps: pydantic, anthropic, openai' },
    { label: 'Tools', them: 'Tool classes with schema boilerplate', us: 'Decorate any async function with @tool — schema auto-derived' },
    { label: 'Observability', them: 'Basic logging; no native OTel', us: 'Native OpenTelemetry — GenAI semconv, OTLP / JSONL exporters' },
    { label: 'Testing', them: 'Requires live API calls for most tests', us: 'FauxProvider — deterministic streaming, no API keys needed' },
  ],
  code: {
    h2: 'A tool-using agent',
    themTitle: '# CrewAI',
    them: `from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel

class WeatherInput(BaseModel):
    city: str

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get current weather for a city."
    args_schema: type[BaseModel] = WeatherInput

    def _run(self, city: str) -> str:
        return f"72F and sunny in {city}"

weather_tool = WeatherTool()

agent = Agent(
    role="Weather Assistant",
    goal="Answer weather questions accurately.",
    backstory="You are a helpful weather assistant.",
    tools=[weather_tool],
    llm="claude-sonnet-4-6",
)

task = Task(
    description="What's the weather in {city}?",
    expected_output="A weather report for the city.",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
result = crew.kickoff(inputs={"city": "Tokyo"})
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
      h2: 'Roles, goals, and backstories — or just a system prompt',
      body: [
        'CrewAI\'s "role", "goal", and "backstory" fields are a structured way to write a system prompt. When you have one agent doing one job, the role metaphor adds ceremony without adding capability. In CubePi, `system_prompt` is a string you own completely — no framework vocabulary to learn, no structure to maintain.',
        'For genuine multi-agent scenarios, CubePi provides a `SubagentsMiddleware` that spawns child agents and wires their results back to the parent — without the crew/process/task object model.',
      ],
    },
    {
      h2: 'Persistence that does not require you to wire it yourself',
      body: [
        'CrewAI does not ship a built-in checkpointing layer. If you want conversations to survive restarts you add your own database logic. CubePi\'s append-only checkpointing ships out of the box — `SQLiteCheckpointer`, `PostgresCheckpointer`, and `MySQLCheckpointer` each write only the new messages from each turn, so write cost stays O(1) regardless of how long the thread grows.',
      ],
    },
    {
      h2: 'When CrewAI is the better fit',
      body: [
        'CrewAI is a reasonable choice if your workflow maps naturally onto the crew metaphor — a team of specialist agents each assigned a discrete task in a pipeline. If you are building a single conversational agent, a stateful assistant, or anything that needs production-grade persistence or OTel observability, CubePi is the leaner path.',
      ],
    },
  ],
  cta: [
    { text: 'Quick Start →', href: '/docs/getting-started/quick-start' },
    { text: 'Core Concepts →', href: '/docs/getting-started/core-concepts' },
  ],
};

const ZH: CompareContent = {
  them: 'CrewAI',
  title: 'CubePi vs CrewAI — 更精简的 Python Agent 框架',
  description:
    'CubePi 与 CrewAI 对比：CrewAI 将 Agent 建模为具有角色和背景故事的"角色扮演"团队成员。CubePi 将 Agent 建模为普通的异步 while 循环，具备类型化中间件、追加式 checkpointing 和 3 个核心依赖。',
  keywords:
    'CubePi vs CrewAI, CrewAI 替代品, Python Agent 框架, 异步 Agent 框架, CrewAI vs CubePi, 多 Agent 框架, crewai 替代品',
  h1: 'CubePi vs CrewAI',
  intro: [
    'CrewAI 围绕角色扮演的隐喻组织 Agent 系统 —— 团队（crew）、带有角色和背景故事的 Agent，以及分配给 Agent 的任务。CubePi 从不同的前提出发：Agent 是一个普通的异步 while 循环，调用模型、执行工具、循环往复。',
    '如果角色抽象并不能解决你的问题，CubePi 直接省去这一层。以下是两者的详细对比。',
  ],
  tableHeading: '并排对比',
  rows: [
    { label: '抽象层', them: '具有角色、目标和背景故事的角色扮演 Agent 团队', us: '普通的异步 while 循环 —— 五分钟内可从头读完' },
    { label: '多 Agent', them: '一等公民：层次化或顺序化进程类型', us: 'Subagent 中间件 —— 派生子 Agent，等待结果，自由组合' },
    { label: '异步支持', them: '默认同步；异步支持后来追加', us: '异步优先 —— 每个入口都是 async' },
    { label: '流式输出', them: '流式支持有限', us: 'async for event in stream —— 统一模式，11 种事件类型' },
    { label: 'Checkpointing', them: '无内置持久化层', us: '追加式 —— O(1) DB I/O；后端：memory、SQLite、Postgres、MySQL' },
    { label: '依赖', them: 'langchain-core + 大量传递依赖', us: '3 个核心依赖：pydantic、anthropic、openai' },
    { label: '工具', them: '需要编写 schema 样板代码的工具类', us: '用 @tool 装饰任意 async 函数 —— schema 自动推导' },
    { label: '可观测性', them: '基础日志；无原生 OTel', us: '原生 OpenTelemetry —— GenAI 语义约定，OTLP / JSONL 导出器' },
    { label: '测试', them: '大多数测试需要真实 API 调用', us: 'FauxProvider —— 确定性流式仿真，无需 API Key' },
  ],
  sections: [
    {
      h2: '角色、目标、背景故事 —— 还是直接写 system prompt',
      body: [
        'CrewAI 的"角色"、"目标"和"背景故事"字段本质上是一种结构化写 system prompt 的方式。当你只有一个 Agent 做一件事时，角色隐喻增加的是仪式感而非能力。在 CubePi 中，`system_prompt` 就是你完全掌控的一个字符串 —— 没有框架词汇要学，没有结构要维护。',
        '对于真正的多 Agent 场景，CubePi 提供 `SubagentsMiddleware`，可以派生子 Agent 并将其结果回传给父 Agent —— 无需 crew/process/task 对象模型。',
      ],
    },
    {
      h2: '开箱即用的持久化，无需自己接线',
      body: [
        'CrewAI 没有内置的 checkpointing 层。如果你希望对话在重启后依然存在，需要自己添加数据库逻辑。CubePi 的追加式 checkpointing 开箱即用 —— `SQLiteCheckpointer`、`PostgresCheckpointer` 和 `MySQLCheckpointer` 每轮只写入新消息，无论对话多长，写入成本保持 O(1)。',
      ],
    },
    {
      h2: 'CrewAI 更适合的场景',
      body: [
        '如果你的工作流自然地映射到 crew 的隐喻 —— 一组专家 Agent 各自负责流水线中的一个离散任务 —— CrewAI 是合理的选择。如果你在构建单一的对话 Agent、有状态的助手，或任何需要生产级持久化或 OTel 可观测性的系统，CubePi 是更精简的路径。',
      ],
    },
  ],
  cta: [
    { text: '快速开始 →', href: '/docs/getting-started/quick-start' },
    { text: '核心概念 →', href: '/docs/getting-started/core-concepts' },
  ],
};

export default function CompareCrewAI(): React.ReactElement {
  const zh = useIsZhHans();
  return <ComparePage content={zh ? ZH : EN} />;
}
