import React from 'react';
import styles from './InstallMatrix.module.css';

const ROWS: { tool: string; cmd: string }[] = [
  { tool: 'pip',    cmd: 'pip install cubepi' },
  { tool: 'uv',     cmd: 'uv add cubepi' },
  { tool: 'poetry', cmd: 'poetry add cubepi' },
  { tool: 'extras', cmd: 'pip install cubepi[sqlite,postgres,mcp,tracing,tracing-otlp]' },
];

export default function InstallMatrix() {
  return (
    <section className={styles.section}>
      <h2 className={styles.h2}>Install</h2>
      <div className={styles.table}>
        {ROWS.map((r) => (
          <div key={r.tool} className={styles.row}>
            <span className={styles.tool}>{r.tool}</span>
            <code className={styles.cmd}>{r.cmd}</code>
            <button
              type="button"
              className={styles.copy}
              onClick={() => navigator.clipboard?.writeText(r.cmd)}
              aria-label={`Copy ${r.tool} command`}
            >Copy</button>
          </div>
        ))}
      </div>
    </section>
  );
}
