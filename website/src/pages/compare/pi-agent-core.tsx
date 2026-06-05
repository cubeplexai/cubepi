import React from 'react';
import ComparePage, { type CompareContent } from '@site/src/components/Compare/ComparePage';
import { useIsZhHans } from '@site/src/hooks/useIsZhHans';

const EN: CompareContent = {
  them: 'pi-agent-core',
  title: 'CubePi vs pi-agent-core — the Python-native sibling',
  description:
    'CubePi vs pi-agent-core: pi-agent-core is a TypeScript agent framework; CubePi is an independent Python reimplementation of the same linear agent-loop architecture, with Pydantic v2 models, asyncio-native primitives, and built-in append-only checkpointing.',
  keywords:
    'CubePi vs pi-agent-core, pi-agent-core alternative, pi-agent-core Python, Python agent framework, async agent framework, TypeScript to Python agent',
  h1: 'CubePi vs pi-agent-core',
  intro: [
    'pi-agent-core is a TypeScript agent framework. CubePi is an independent Python reimplementation of the same architecture — the linear agent loop (agent.ts + agent-loop.ts maps onto CubePi’s agent.py + loop.py) — rebuilt around Pydantic v2, asyncio-native primitives, and built-in checkpointing.',
    'So this is less a "which is better" question and more a "which ecosystem are you in" one. If your stack is TypeScript/Node, pi-agent-core fits natively. If your stack is Python, CubePi gives you the same mental model without dragging a JS runtime into your services.',
  ],
  tableHeading: 'Moving from TypeScript to Python',
  rows: [
    { label: 'Language / runtime', them: 'TypeScript on Node.js', us: 'Python 3.11+' },
    { label: 'Concurrency', them: 'Promises / async-await', us: 'asyncio — async-first, every entry point is async' },
    { label: 'Schemas & validation', them: 'TypeScript types (compile-time)', us: 'Pydantic v2 models — runtime-validated tool params' },
    { label: 'Agent model', them: 'Linear agent loop (agent.ts + agent-loop.ts)', us: 'Same linear loop (agent.py + loop.py)' },
    { label: 'Providers', them: 'Provider abstraction', us: 'Provider protocol — Anthropic & OpenAI built in' },
  ],
  sections: [
    {
      h2: 'Same architecture, Pythonic surface',
      body: [
        'CubePi deliberately keeps pi-agent-core’s core idea: an agent is a loop, not a graph. What changes is the surface. Tools are plain async functions whose parameters are Pydantic models, so arguments coming back from the model are validated at runtime, not just type-hinted. Streaming is a single async iterator. The result reads like idiomatic Python rather than a TypeScript API transliterated into Python.',
      ],
    },
    {
      h2: 'What CubePi adds on the Python side',
      body: [
        'Built-in checkpointing: an append-only persistence layer with memory, SQLite, and Postgres backends, where each turn is O(1) to write regardless of conversation length.',
        'Native OpenTelemetry: a Tracer and Meter emit OTel spans with GenAI semantic-convention attributes out of the box, exportable to any OTLP backend (Jaeger, Tempo, Honeycomb, Langfuse, Datadog, …) — plus a cubepi trace CLI for local JSONL traces.',
        'MCP loaders for HTTP and stdio transports, a streaming-realistic FauxProvider for deterministic tests, and a lean dependency footprint: pydantic, anthropic, and openai are the only core dependencies; everything else is an optional extra.',
      ],
    },
    {
      h2: 'Which should you choose',
      body: [
        'Choose pi-agent-core if your services are TypeScript/Node and you want to stay in that runtime. Choose CubePi if you are building in Python and want the same linear-loop agent model with first-class asyncio, Pydantic-validated tools, append-only persistence, and native OpenTelemetry — without running a JavaScript runtime alongside your Python.',
      ],
    },
  ],
  cta: [
    { text: 'Quick Start →', href: '/docs/getting-started/quick-start' },
    { text: 'Core Concepts →', href: '/docs/getting-started/core-concepts' },
  ],
};

const ZH: CompareContent = {
  them: 'pi-agent-core',
  title: 'CubePi vs pi-agent-core — Python 原生的同源框架',
  description:
    'CubePi 与 pi-agent-core 对比:pi-agent-core 是一个 TypeScript Agent 框架;CubePi 是同一套线性 agent-loop 架构的 Python 独立重实现,采用 Pydantic v2 模型、asyncio 原生原语,以及内置的追加式 checkpointing。',
  keywords:
    'CubePi vs pi-agent-core, pi-agent-core 替代品, pi-agent-core Python, Python Agent 框架, 异步 Agent 框架',
  h1: 'CubePi vs pi-agent-core',
  intro: [
    'pi-agent-core 是一个 TypeScript Agent 框架。CubePi 是同一套架构的 Python 独立重实现 —— 同样的线性 agent 循环(agent.ts + agent-loop.ts 对应 CubePi 的 agent.py + loop.py)—— 围绕 Pydantic v2、asyncio 原生原语和内置 checkpointing 重新构建。',
    '所以这与其说是「谁更好」,不如说是「你在哪个生态」。如果你的技术栈是 TypeScript/Node,pi-agent-core 原生契合。如果你的技术栈是 Python,CubePi 给你同样的心智模型,而无需把 JS 运行时拖进你的服务里。',
  ],
  tableHeading: '从 TypeScript 迁移到 Python',
  rows: [
    { label: '语言 / 运行时', them: 'Node.js 上的 TypeScript', us: 'Python 3.11+' },
    { label: '并发模型', them: 'Promise / async-await', us: 'asyncio —— 异步优先,每个入口都是 async' },
    { label: '模式与校验', them: 'TypeScript 类型(编译期)', us: 'Pydantic v2 模型 —— 工具参数运行时校验' },
    { label: 'Agent 模型', them: '线性 agent 循环(agent.ts + agent-loop.ts)', us: '同样的线性循环(agent.py + loop.py)' },
    { label: 'Provider', them: 'Provider 抽象', us: 'Provider 协议 —— 内置 Anthropic 与 OpenAI' },
  ],
  sections: [
    {
      h2: '同一套架构,Pythonic 的表层',
      body: [
        'CubePi 刻意保留了 pi-agent-core 的核心理念:agent 是一个循环,而不是一张图。改变的是表层。工具是普通的 async 函数,其参数是 Pydantic 模型,因此模型返回的参数会在运行时被校验,而不只是类型提示。流式输出是单个 async 迭代器。最终代码读起来像地道的 Python,而非把 TypeScript API 直译成 Python。',
      ],
    },
    {
      h2: 'CubePi 在 Python 侧带来的增量',
      body: [
        '内置 checkpointing:追加式持久化层,提供 memory、SQLite、Postgres 后端,无论对话多长,每轮写入都是 O(1)。',
        '原生 OpenTelemetry:Tracer 与 Meter 开箱即用地输出带 GenAI 语义约定属性的 OTel span,可导出到任意 OTLP 后端(Jaeger、Tempo、Honeycomb、Langfuse、Datadog……),并附带 cubepi trace CLI 用于本地 JSONL trace。',
        'MCP 加载器(支持 HTTP 与 stdio 传输)、用于确定性测试的流式仿真 FauxProvider,以及精简的依赖足迹:核心依赖只有 pydantic、anthropic、openai,其余皆为可选 extra。',
      ],
    },
    {
      h2: '该怎么选',
      body: [
        '如果你的服务是 TypeScript/Node 并希望留在该运行时,选 pi-agent-core。如果你用 Python 构建,并想要同样的线性循环 agent 模型 —— 一等公民的 asyncio、Pydantic 校验的工具、追加式持久化、原生 OpenTelemetry —— 而不想在 Python 旁边再跑一个 JavaScript 运行时,选 CubePi。',
      ],
    },
  ],
  cta: [
    { text: '快速开始 →', href: '/docs/getting-started/quick-start' },
    { text: '核心概念 →', href: '/docs/getting-started/core-concepts' },
  ],
};

export default function ComparePiAgentCore(): React.ReactElement {
  const zh = useIsZhHans();
  return <ComparePage content={zh ? ZH : EN} />;
}
