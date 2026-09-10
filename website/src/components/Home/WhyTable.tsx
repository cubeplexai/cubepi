import React from 'react';
import Link from '@docusaurus/Link';
import { useIsZhHans } from '@site/src/hooks/useIsZhHans';
import styles from './WhyTable.module.css';

const ROWS_EN: { label: string; langgraph: string; cubeloop: string }[] = [
  { label: 'Abstraction',     langgraph: 'Graph nodes + edges + channels',                          cubeloop: 'Plain async functions — run_agent_loop is a while loop' },
  { label: 'Streaming',       langgraph: 'Callback-based, multiple handler types',                  cubeloop: 'async for event in stream — one pattern everywhere' },
  { label: 'Checkpointing',   langgraph: 'Full snapshot per step; serializes entire message list',  cubeloop: 'Append-only — O(1) DB I/O regardless of conversation length' },
  { label: 'Dependencies',    langgraph: 'langchain-core, langgraph-sdk, and transitive deps',      cubeloop: '3 core deps: pydantic, anthropic, openai' },
  { label: 'Tool execution',  langgraph: 'Tools are graph nodes with manual wiring',                cubeloop: 'Declare tools as functions; framework routes and parallelizes' },
  { label: 'Multi-provider',  langgraph: 'Via langchain chat model adapters',                       cubeloop: 'Native Provider protocol — Anthropic, OpenAI built in' },
  { label: 'Middleware',      langgraph: 'Graph-level middleware on node entry/exit',               cubeloop: '8 typed hooks with declarative composition rules' },
  { label: 'Observability',   langgraph: 'LangSmith / Langfuse integration',                        cubeloop: 'Native OpenTelemetry — Tracer, Meter, GenAI semconv, OTLP / JSONL out of the box' },
];

const ROWS_ZH: { label: string; langgraph: string; cubeloop: string }[] = [
  { label: '抽象模型',     langgraph: '图节点 + 边 + 通道',                              cubeloop: '普通 async 函数 — run_agent_loop 就是一个 while 循环' },
  { label: '流式输出',     langgraph: '基于回调，多种 handler 类型',                      cubeloop: 'async for event in stream — 统一模式' },
  { label: '检查点',        langgraph: '每步全量快照，序列化整个消息列表',                cubeloop: '追加式 — 无论对话多长，每轮 O(1) DB I/O' },
  { label: '依赖项',       langgraph: 'langchain-core、langgraph-sdk 及传递依赖',        cubeloop: '3 个核心依赖：pydantic、anthropic、openai' },
  { label: '工具执行',     langgraph: '工具是需要手动连线的图节点',                        cubeloop: '声明为函数；框架自动路由并并行执行' },
  { label: '多 Provider',  langgraph: '通过 langchain chat model 适配器',                cubeloop: '原生 Provider 协议 — 内置 Anthropic、OpenAI' },
  { label: 'Middleware',   langgraph: '图级中间件，在节点进出时触发',                     cubeloop: '8 种类型化 hook，声明式组合规则' },
  { label: '可观测性',     langgraph: 'LangSmith / Langfuse 集成',                       cubeloop: '原生 OpenTelemetry — Tracer、Meter、GenAI semconv，开箱即用 OTLP / JSONL' },
];

export default function WhyTable() {
  const zh = useIsZhHans();
  const ROWS = zh ? ROWS_ZH : ROWS_EN;
  return (
    <section className={styles.section}>
      <h2 className={styles.h2}>
        {zh ? '为什么选择 CubeLoop — langgraph 和 pi-agent-core 的替代方案' : 'Why CubeLoop — a langgraph and pi-agent-core alternative'}
      </h2>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th></th>
              <th>langgraph</th>
              <th>CubeLoop</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r.label}>
                <td className={styles.label}>{r.label}</td>
                <td>{r.langgraph}</td>
                <td className={styles.us}>{r.cubeloop}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={styles.more}>
        <Link to="/compare/langgraph">
          {zh ? '完整对比:CubeLoop vs LangGraph →' : 'Full comparison: CubeLoop vs LangGraph →'}
        </Link>
        {'  ·  '}
        <Link to="/compare/pi-agent-core">
          {zh ? 'CubeLoop vs pi-agent-core →' : 'CubeLoop vs pi-agent-core →'}
        </Link>
      </p>
    </section>
  );
}
