import React from 'react';
import Head from '@docusaurus/Head';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import CodeBlock from '@theme/CodeBlock';
import styles from './ComparePage.module.css';

export interface CompareRow {
  label: string;
  them: string;
  us: string;
}

export interface CompareSection {
  h2: string;
  /** Prose paragraphs; each string becomes a <p>. */
  body: string[];
}

export interface CompareCode {
  h2: string;
  themTitle: string;
  them: string;
  usTitle: string;
  us: string;
}

export interface CompareContent {
  /** The competitor's display name, e.g. "LangGraph". */
  them: string;
  /** <title> — must NOT lead with "CubePi" (the brand is appended). */
  title: string;
  description: string;
  keywords: string;
  h1: string;
  intro: string[];
  rows: CompareRow[];
  /** Optional side-by-side code sample. */
  code?: CompareCode;
  sections: CompareSection[];
  cta: { text: string; href: string }[];
  tableHeading: string;
}

export default function ComparePage({ content }: { content: CompareContent }): React.ReactElement {
  return (
    <Layout title={content.title} description={content.description}>
      <Head>
        <meta name="keywords" content={content.keywords} />
        <script type="application/ld+json">
          {JSON.stringify({
            '@context': 'https://schema.org',
            '@type': 'TechArticle',
            headline: content.h1,
            description: content.description,
            author: { '@type': 'Organization', name: 'CubePi', url: 'https://cubepi.ai' },
            publisher: { '@type': 'Organization', name: 'CubePi', url: 'https://cubepi.ai' },
          })}
        </script>
      </Head>
      <main className={styles.page}>
        <h1 className={styles.h1}>{content.h1}</h1>
        {content.intro.map((p, i) => (
          <p key={i} className={styles.lead}>{p}</p>
        ))}

        <h2 className={styles.h2}>{content.tableHeading}</h2>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th></th>
                <th>{content.them}</th>
                <th>CubePi</th>
              </tr>
            </thead>
            <tbody>
              {content.rows.map((r) => (
                <tr key={r.label}>
                  <td className={styles.label}>{r.label}</td>
                  <td>{r.them}</td>
                  <td className={styles.us}>{r.us}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {content.code && (
          <section className={styles.codeSection}>
            <h2 className={styles.h2}>{content.code.h2}</h2>
            <div className={styles.codeGrid}>
              <div>
                <div className={styles.codeTitle}>{content.code.themTitle}</div>
                <CodeBlock language="python">{content.code.them}</CodeBlock>
              </div>
              <div>
                <div className={styles.codeTitle}>{content.code.usTitle}</div>
                <CodeBlock language="python">{content.code.us}</CodeBlock>
              </div>
            </div>
          </section>
        )}

        {content.sections.map((s) => (
          <section key={s.h2} className={styles.prose}>
            <h2 className={styles.h2}>{s.h2}</h2>
            {s.body.map((p, i) => (
              <p key={i} className={styles.body}>{p}</p>
            ))}
          </section>
        ))}

        <div className={styles.ctaRow}>
          {content.cta.map((c) => (
            <Link key={c.href} className={styles.cta} to={c.href}>{c.text}</Link>
          ))}
        </div>
      </main>
    </Layout>
  );
}
