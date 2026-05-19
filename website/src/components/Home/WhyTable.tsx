import React from 'react';
import styles from './WhyTable.module.css';

const ROWS: { label: string; langgraph: string; cubepi: string }[] = [
  { label: 'Abstraction',     langgraph: 'Graph nodes + edges + channels',                          cubepi: 'Plain async functions — run_agent_loop is a while loop' },
  { label: 'Streaming',       langgraph: 'Callback-based, multiple handler types',                  cubepi: 'async for event in stream — one pattern everywhere' },
  { label: 'Checkpointing',   langgraph: 'Full snapshot per step; serializes entire message list',  cubepi: 'Append-only — O(1) DB I/O regardless of conversation length' },
  { label: 'Dependencies',    langgraph: 'langchain-core, langgraph-sdk, and transitive deps',      cubepi: '3 core deps: pydantic, anthropic, openai' },
  { label: 'Tool execution',  langgraph: 'Tools are graph nodes with manual wiring',                cubepi: 'Declare tools as functions; framework routes and parallelizes' },
  { label: 'Multi-provider',  langgraph: 'Via langchain chat model adapters',                       cubepi: 'Native Provider protocol — Anthropic, OpenAI built in' },
  { label: 'Middleware',      langgraph: 'Graph-level middleware on node entry/exit',               cubepi: '5 typed hooks with declarative composition rules' },
  { label: 'Observability',   langgraph: 'LangSmith / Langfuse integration',                        cubepi: 'Native OpenTelemetry — Tracer, Meter, GenAI semconv, OTLP / JSONL out of the box' },
];

export default function WhyTable() {
  return (
    <section className={styles.section}>
      <h2 className={styles.h2}>Why CubePi</h2>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th></th>
              <th>langgraph</th>
              <th>CubePi</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r.label}>
                <td className={styles.label}>{r.label}</td>
                <td>{r.langgraph}</td>
                <td className={styles.us}>{r.cubepi}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
