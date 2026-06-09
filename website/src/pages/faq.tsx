import React from 'react';
import Head from '@docusaurus/Head';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import { useIsZhHans } from '@site/src/hooks/useIsZhHans';

interface FaqItem {
  q: string;
  a: React.ReactNode;
}

/** Recursively extract plain text from a React node for use in JSON-LD. */
function reactNodeToText(node: React.ReactNode): string {
  if (node === null || node === undefined || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(reactNodeToText).join('');
  // React element
  const el = node as React.ReactElement;
  if (el && typeof el === 'object' && 'props' in el) {
    return reactNodeToText(el.props.children);
  }
  return '';
}

const EN_ITEMS: FaqItem[] = [
  {
    q: 'What is CubePi?',
    a: (
      <>
        CubePi is a Pythonic, async-native agent framework for building LLM-powered agents in Python.
        It models the agent as a plain async while-loop — no state graphs, no nodes/edges to wire — so the
        core algorithm is readable in five minutes. It ships with append-only checkpointing, composable
        middleware, native OpenTelemetry tracing, MCP support, and a <code>FauxProvider</code> for
        deterministic tests.
      </>
    ),
  },
  {
    q: 'How is CubePi different from LangGraph?',
    a: (
      <>
        LangGraph models an agent as a state graph you wire together with nodes, edges, and typed channels.
        CubePi models the same agent as a plain <code>async while</code> loop. There is no <code>StateGraph</code>,
        no <code>add_edge</code>, no <code>ToolNode</code>, and no <code>TypedDict</code> to maintain.
        CubePi also checkpoints append-only (O(1) per turn vs. full snapshot per step) and has only three core
        dependencies. See the full <Link to="/compare/langgraph">LangGraph comparison</Link>.
      </>
    ),
  },
  {
    q: 'How is CubePi different from CrewAI?',
    a: (
      <>
        CrewAI organises agents around role-playing metaphors — crews, roles, goals, and backstories.
        CubePi skips the metaphor: an agent is a function with a system prompt. CubePi also ships
        built-in append-only checkpointing (CrewAI has none), is async-first, and has native OpenTelemetry.
        See the full <Link to="/compare/crewai">CrewAI comparison</Link>.
      </>
    ),
  },
  {
    q: 'How is CubePi different from PydanticAI?',
    a: (
      <>
        Both frameworks are async-first and built on Pydantic. PydanticAI focuses on structured output and
        dependency injection via <code>RunContext</code>; CubePi focuses on persistent multi-turn conversations,
        composable middleware hooks, provider fallback, and vendor-neutral OpenTelemetry (vs. Logfire).
        CubePi also ships built-in checkpointing that PydanticAI lacks.
        See the full <Link to="/compare/pydantic-ai">PydanticAI comparison</Link>.
      </>
    ),
  },
  {
    q: 'Does CubePi support multiple LLM providers?',
    a: (
      <>
        Yes. CubePi ships <code>AnthropicProvider</code> and <code>OpenAIProvider</code> (plus{' '}
        <code>OpenAIResponsesProvider</code> for the Responses API) out of the box. You can add your own
        provider with a single class that implements the <code>Provider</code> protocol. Use{' '}
        <code>FallbackBoundModel</code> to chain providers — on a rate-limit or outage the next model in the
        chain is tried automatically.
      </>
    ),
  },
  {
    q: 'What databases does CubePi support for checkpointing?',
    a: (
      <>
        CubePi ships four checkpointer backends: <code>MemoryCheckpointer</code> (development/testing),{' '}
        <code>SQLiteCheckpointer</code> (lightweight single-node), <code>PostgresCheckpointer</code>{' '}
        (production), and <code>MySQLCheckpointer</code> (production). All use append-only writes — each
        turn writes only the new messages, so write cost stays O(1) regardless of conversation length.
        Postgres and MySQL use Alembic for schema management; your app owns the DDL.
      </>
    ),
  },
  {
    q: 'What is append-only checkpointing and why does it matter?',
    a: (
      <>
        Most agent frameworks checkpoint by snapshotting the entire message list on every step. As
        conversations grow, write cost grows linearly. CubePi writes only the new messages produced in
        each turn — O(1) regardless of how long the thread is. This matters at scale: thousands of
        concurrent long-lived sessions with frequent turns.
      </>
    ),
  },
  {
    q: 'Does CubePi support MCP (Model Context Protocol)?',
    a: (
      <>
        Yes. Install <code>pip install cubepi[mcp]</code> and use <code>StdioMCPLoader</code> or{' '}
        <code>HttpMCPLoader</code> to load tools from any MCP-compatible server. Loaded tools plug into
        the same <code>AgentTool</code> interface as hand-written tools.
      </>
    ),
  },
  {
    q: 'How does CubePi handle observability?',
    a: (
      <>
        CubePi includes a <code>Tracer</code> and <code>Meter</code> that emit OpenTelemetry spans
        and metrics aligned with the GenAI Semantic Conventions. Spans export via OTLP/HTTP to any
        compatible backend (Jaeger, Grafana Tempo, Honeycomb, Datadog, AWS X-Ray, Langfuse, …) or
        to local JSONL files. The <code>cubepi trace</code> CLI lets you inspect JSONL traces in the
        terminal without a backend. Install with <code>pip install cubepi[tracing,trace-cli]</code>.
      </>
    ),
  },
  {
    q: 'How do I test agents without hitting real APIs?',
    a: (
      <>
        Use <code>FauxProvider</code>. It emits realistic streaming deltas (<code>content_block_start</code>,{' '}
        <code>text_delta</code>, etc.) without making any API calls. You script the responses with{' '}
        <code>provider.set_responses([...])</code> using helpers like <code>faux_text()</code> and{' '}
        <code>faux_tool_call()</code>. Tests run fully deterministically with no API keys.
      </>
    ),
  },
  {
    q: 'What Python versions does CubePi support?',
    a: 'CubePi supports Python 3.11, 3.12, 3.13, and 3.14. CI runs on all four versions.',
  },
  {
    q: 'Is CubePi production-ready?',
    a: (
      <>
        CubePi is in beta (v0.9). The core agent loop, checkpointing, middleware, and tracing APIs are
        stable. Breaking changes follow semantic versioning and are documented in the{' '}
        <Link to="/changelog">changelog</Link>. The Postgres and MySQL checkpointers are used in
        production by early adopters.
      </>
    ),
  },
  {
    q: 'How do I install CubePi?',
    a: (
      <>
        <code>pip install cubepi</code> for the core. Add extras for optional features:{' '}
        <code>pip install cubepi[sqlite,postgres,mcp,tracing,trace-cli]</code>.
        With uv: <code>uv add cubepi</code> or <code>uv add cubepi[sqlite,postgres,mcp,tracing]</code>.
      </>
    ),
  },
  {
    q: 'Is CubePi cache-friendly?',
    a: (
      <>
        Yes — CubePi is designed to maximise prompt cache hit rates. LLM providers (Anthropic, OpenAI) let
        you cache the stable prefix of a request so repeated turns don't re-process the same tokens, cutting
        costs by up to 90% and reducing latency. Two CubePi design decisions work together to make this
        reliable:
        <br /><br />
        <strong>Append-only message storage.</strong> CubePi never rewrites or reorders history — it only
        appends new messages. This means the prefix of every request is byte-identical to the previous turn,
        which is exactly what the cache needs to get a hit. Frameworks that rebuild the full message list
        from a snapshot on every step silently break the cache if any serialisation detail changes.
        <br /><br />
        <strong>Automatic cache breakpoints (Anthropic).</strong> <code>AnthropicProvider</code> marks three
        cache breakpoints by default: the system prompt, the last tool definition, and the last message in
        history. The last-message breakpoint moves forward each turn, keeping prior history warm. See the{' '}
        <Link to="/docs/guides/agents/prompt-caching">Prompt Caching guide</Link> for details on how to
        avoid breaking these breakpoints and how to read cache hit metrics from <code>Usage</code>.
      </>
    ),
  },
  {
    q: 'Is CubePi open source?',
    a: (
      <>
        Yes. CubePi is MIT licensed. Source code is on{' '}
        <a href="https://github.com/cubeplexai/cubepi" target="_blank" rel="noopener noreferrer">GitHub</a>.
      </>
    ),
  },
];

const ZH_ITEMS: FaqItem[] = [
  {
    q: 'CubePi 是什么？',
    a: (
      <>
        CubePi 是一个 Pythonic、异步原生的 Agent 框架，用于在 Python 中构建 LLM 驱动的 Agent。
        它将 Agent 建模为普通的异步 while 循环 —— 无状态图，无需连接节点/边 —— 核心算法五分钟内可读懂。
        它内置追加式 checkpointing、可组合中间件、原生 OpenTelemetry 追踪、MCP 支持，以及用于确定性测试的 <code>FauxProvider</code>。
      </>
    ),
  },
  {
    q: 'CubePi 与 LangGraph 有何不同？',
    a: (
      <>
        LangGraph 将 Agent 建模为需要手动连接节点、边和类型化通道的状态图。CubePi 将同样的 Agent 建模为普通的 <code>async while</code> 循环。
        没有 <code>StateGraph</code>，没有 <code>add_edge</code>，没有 <code>ToolNode</code>，也没有需要维护的 <code>TypedDict</code>。
        CubePi 还采用追加式 checkpointing（每轮 O(1) vs. 每步全量快照），且只有三个核心依赖。
        查看完整的 <Link to="/compare/langgraph">LangGraph 对比</Link>。
      </>
    ),
  },
  {
    q: 'CubePi 与 CrewAI 有何不同？',
    a: (
      <>
        CrewAI 围绕角色扮演的隐喻组织 Agent —— 团队、角色、目标和背景故事。CubePi 省去了这层隐喻：Agent 就是一个带有 system prompt 的函数。
        CubePi 还内置了追加式 checkpointing（CrewAI 没有），异步优先，并具有原生 OpenTelemetry。
        查看完整的 <Link to="/compare/crewai">CrewAI 对比</Link>。
      </>
    ),
  },
  {
    q: 'CubePi 与 PydanticAI 有何不同？',
    a: (
      <>
        两者都是异步优先且基于 Pydantic 的框架。PydanticAI 聚焦于结构化输出和通过 <code>RunContext</code> 实现的依赖注入；
        CubePi 聚焦于持久化多轮对话、可组合的中间件钩子、Provider 故障转移，以及厂商中立的 OpenTelemetry（而非 Logfire）。
        CubePi 还内置了 PydanticAI 所缺乏的 checkpointing。
        查看完整的 <Link to="/compare/pydantic-ai">PydanticAI 对比</Link>。
      </>
    ),
  },
  {
    q: 'CubePi 支持多个 LLM Provider 吗？',
    a: (
      <>
        支持。CubePi 内置 <code>AnthropicProvider</code> 和 <code>OpenAIProvider</code>（以及用于 Responses API 的{' '}
        <code>OpenAIResponsesProvider</code>）。你可以用一个实现 <code>Provider</code> 协议的类添加自定义 Provider。
        使用 <code>FallbackBoundModel</code> 链式连接 Provider —— 在限速或故障时自动切换到链中的下一个模型。
      </>
    ),
  },
  {
    q: 'CubePi 的 checkpointing 支持哪些数据库？',
    a: (
      <>
        CubePi 提供四种 checkpointer 后端：<code>MemoryCheckpointer</code>（开发/测试）、
        <code>SQLiteCheckpointer</code>（轻量级单节点）、<code>PostgresCheckpointer</code>（生产环境）
        和 <code>MySQLCheckpointer</code>（生产环境）。所有后端均使用追加式写入 —— 每轮只写入新消息，
        无论对话多长，写入成本保持 O(1)。
      </>
    ),
  },
  {
    q: '什么是追加式 checkpointing，为什么重要？',
    a: (
      <>
        大多数 Agent 框架通过每步快照整个消息列表来进行 checkpoint。随着对话增长，写入成本线性增长。
        CubePi 只写入每轮产生的新消息 —— 无论线程多长，都是 O(1)。这在规模化时至关重要：数千个并发的长存会话，频繁轮次。
      </>
    ),
  },
  {
    q: 'CubePi 支持 MCP（模型上下文协议）吗？',
    a: (
      <>
        支持。安装 <code>pip install cubepi[mcp]</code>，使用 <code>StdioMCPLoader</code> 或{' '}
        <code>HttpMCPLoader</code> 从任何兼容 MCP 的服务器加载工具。加载的工具与手写工具使用相同的 <code>AgentTool</code> 接口。
      </>
    ),
  },
  {
    q: 'CubePi 如何处理可观测性？',
    a: (
      <>
        CubePi 内置 <code>Tracer</code> 和 <code>Meter</code>，输出符合 GenAI 语义约定的 OpenTelemetry span 和指标。
        Span 通过 OTLP/HTTP 导出到任何兼容后端（Jaeger、Grafana Tempo、Honeycomb、Datadog、AWS X-Ray、Langfuse……）
        或本地 JSONL 文件。<code>cubepi trace</code> CLI 可在终端中无需后端直接检查 JSONL trace。
        安装：<code>pip install cubepi[tracing,trace-cli]</code>。
      </>
    ),
  },
  {
    q: '如何在不调用真实 API 的情况下测试 Agent？',
    a: (
      <>
        使用 <code>FauxProvider</code>。它在不发起任何 API 调用的情况下产生真实的流式 delta（<code>content_block_start</code>、
        <code>text_delta</code> 等）。通过 <code>provider.set_responses([...])</code> 配合 <code>faux_text()</code> 和{' '}
        <code>faux_tool_call()</code> 等辅助函数编写响应脚本。测试完全确定性，无需 API Key。
      </>
    ),
  },
  {
    q: 'CubePi 支持哪些 Python 版本？',
    a: 'CubePi 支持 Python 3.11、3.12、3.13 和 3.14。CI 在所有四个版本上运行。',
  },
  {
    q: 'CubePi 是否可用于生产环境？',
    a: (
      <>
        CubePi 处于 Beta 阶段（v0.9）。核心 agent 循环、checkpointing、中间件和追踪 API 均已稳定。
        重大变更遵循语义化版本管理，并记录在 <Link to="/changelog">更新日志</Link> 中。
        Postgres 和 MySQL checkpointer 已被早期用户用于生产环境。
      </>
    ),
  },
  {
    q: '如何安装 CubePi？',
    a: (
      <>
        核心安装：<code>pip install cubepi</code>。通过 extras 添加可选功能：
        <code>pip install cubepi[sqlite,postgres,mcp,tracing,trace-cli]</code>。
        使用 uv：<code>uv add cubepi</code> 或 <code>uv add cubepi[sqlite,postgres,mcp,tracing]</code>。
      </>
    ),
  },
  {
    q: 'CubePi 对缓存友好吗？',
    a: (
      <>
        是的 —— CubePi 的设计目标之一就是最大化 prompt 缓存命中率。LLM 提供商（Anthropic、OpenAI）允许你缓存请求的稳定前缀，
        这样相同的 token 不需要在每轮都重新处理，可降低最高 90% 的费用并减少延迟。CubePi 有两个设计决策共同保障了这一点：
        <br /><br />
        <strong>追加式消息存储。</strong>CubePi 从不重写或重排历史消息 —— 只追加新消息。这意味着每次请求的前缀与上一轮完全相同，
        正是缓存命中所需要的。那些每步都从快照重建完整消息列表的框架，一旦序列化细节有任何变化，就会悄悄破坏缓存。
        <br /><br />
        <strong>自动缓存断点（Anthropic）。</strong><code>AnthropicProvider</code> 默认标记三个缓存断点：system prompt、
        最后一个工具定义，以及历史中的最后一条消息。最后一条消息的断点每轮向前推进，让之前的历史保持热缓存。
        查看 <Link to="/docs/guides/agents/prompt-caching">Prompt Caching 指南</Link> 了解如何避免破坏这些断点，
        以及如何从 <code>Usage</code> 中读取缓存命中指标。
      </>
    ),
  },
  {
    q: 'CubePi 是开源的吗？',
    a: (
      <>
        是的。CubePi 采用 MIT 许可证。源代码托管在{' '}
        <a href="https://github.com/cubeplexai/cubepi" target="_blank" rel="noopener noreferrer">GitHub</a> 上。
      </>
    ),
  },
];

function FaqSection({ items }: { items: FaqItem[] }) {
  return (
    <div style={{ maxWidth: 860, margin: '0 auto', padding: '2rem 1.5rem 4rem' }}>
      {items.map((item, i) => (
        <details
          key={i}
          style={{
            borderBottom: '1px solid var(--ifm-color-emphasis-200)',
            padding: '1rem 0',
          }}
        >
          <summary
            style={{
              fontWeight: 600,
              fontSize: '1.05rem',
              cursor: 'pointer',
              listStyle: 'none',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: '1rem',
            }}
          >
            {item.q}
            <span style={{ fontSize: '1.2rem', flexShrink: 0 }}>＋</span>
          </summary>
          <p style={{ marginTop: '0.75rem', marginBottom: 0, lineHeight: 1.7, color: 'var(--ifm-color-content-secondary)' }}>
            {item.a}
          </p>
        </details>
      ))}
    </div>
  );
}

export default function FAQ(): React.ReactElement {
  const zh = useIsZhHans();

  const title = zh ? 'CubePi 常见问题' : 'Frequently Asked Questions — CubePi';
  const description = zh
    ? 'CubePi 常见问题：与 LangGraph、CrewAI、PydanticAI 的区别，checkpointing、MCP、OpenTelemetry 支持，安装方式，以及生产可用性。'
    : 'Answers to common questions about CubePi — how it compares to LangGraph, CrewAI, and PydanticAI; checkpointing, MCP, OpenTelemetry, installation, and production readiness.';

  const faqJsonLd = {
    '@context': 'https://schema.org',
    '@type': 'FAQPage',
    mainEntity: (zh ? ZH_ITEMS : EN_ITEMS).map((item) => ({
      '@type': 'Question',
      name: item.q,
      acceptedAnswer: {
        '@type': 'Answer',
        text: reactNodeToText(item.a),
      },
    })),
  };

  return (
    <Layout title={title} description={description}>
      <Head>
        <meta
          name="keywords"
          content="CubePi FAQ, CubePi vs LangGraph, CubePi vs CrewAI, CubePi vs PydanticAI, Python agent framework FAQ, async agent FAQ"
        />
        <script type="application/ld+json">{JSON.stringify(faqJsonLd)}</script>
      </Head>
      <main>
        <div
          style={{
            textAlign: 'center',
            padding: '3rem 1.5rem 1.5rem',
            maxWidth: 860,
            margin: '0 auto',
          }}
        >
          <h1>{zh ? '常见问题' : 'Frequently Asked Questions'}</h1>
          <p style={{ fontSize: '1.1rem', color: 'var(--ifm-color-content-secondary)' }}>
            {zh
              ? '关于 CubePi 的常见问题解答'
              : 'Common questions about CubePi, how it compares to alternatives, and how to get started.'}
          </p>
        </div>
        <FaqSection items={zh ? ZH_ITEMS : EN_ITEMS} />
      </main>
    </Layout>
  );
}
