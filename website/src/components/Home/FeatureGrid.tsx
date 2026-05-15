import React from 'react';
import Link from '@docusaurus/Link';
import styles from './FeatureGrid.module.css';

type Icon = React.FC<React.SVGProps<SVGSVGElement>>;

const IconBox: Icon = (p) => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" {...p}>
    <rect x="2" y="2" width="12" height="12" rx="2" />
  </svg>
);
const IconStream: Icon = (p) => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" {...p}>
    <path d="M1 4h14M1 8h14M1 12h10" />
  </svg>
);
const IconTool: Icon = (p) => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" {...p}>
    <path d="M11 2l3 3-2 2-3-3zM10 5l-7 7v3h3l7-7" />
  </svg>
);
const IconPlug: Icon = (p) => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" {...p}>
    <path d="M5 1v4M11 1v4M3 5h10v4a4 4 0 01-4 4H7a4 4 0 01-4-4V5zM8 13v2" />
  </svg>
);
const IconDisk: Icon = (p) => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" {...p}>
    <ellipse cx="8" cy="4" rx="6" ry="2" />
    <path d="M2 4v8c0 1.1 2.7 2 6 2s6-.9 6-2V4M2 8c0 1.1 2.7 2 6 2s6-.9 6-2" />
  </svg>
);
const IconMcp: Icon = (p) => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" {...p}>
    <circle cx="8" cy="8" r="6" />
    <path d="M2 8h12M8 2v12" />
  </svg>
);

const CARDS = [
  { Icon: IconBox,    title: 'Agents',        body: 'One async loop, fully typed events.',     href: '/docs/guides/agents/first-agent' },
  { Icon: IconStream, title: 'Streaming',     body: 'async for event in stream.',              href: '/docs/guides/agents/streaming' },
  { Icon: IconTool,   title: 'Tools',         body: 'Plain functions, parallel execution.',    href: '/docs/guides/agents/tool-use' },
  { Icon: IconPlug,   title: 'Providers',     body: 'Anthropic, OpenAI, or write your own.',   href: '/docs/guides/providers/anthropic' },
  { Icon: IconDisk,   title: 'Checkpointing', body: 'Append-only, O(1) per turn.',             href: '/docs/guides/checkpointing/sqlite' },
  { Icon: IconMcp,    title: 'MCP',           body: 'Load remote tools at startup.',           href: '/docs/guides/mcp/loading' },
];

export default function FeatureGrid() {
  return (
    <section className={styles.section}>
      <div className={styles.grid}>
        {CARDS.map((c) => (
          <Link key={c.title} to={c.href} className={styles.card}>
            <c.Icon className={styles.icon} width={16} height={16} />
            <h3 className={styles.title}>{c.title}</h3>
            <p className={styles.body}>{c.body}</p>
            <span className={styles.more}>→ Guides / {c.title}</span>
          </Link>
        ))}
      </div>
    </section>
  );
}
