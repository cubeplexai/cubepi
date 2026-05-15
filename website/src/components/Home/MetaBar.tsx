import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './MetaBar.module.css';

export default function MetaBar() {
  const { siteConfig } = useDocusaurusContext();
  const sha = (siteConfig.customFields?.GIT_SHA as string | undefined) ?? 'dev';
  return (
    <section className={styles.bar}>
      <span>v0.3.0</span>
      <span className={styles.sep}>·</span>
      <span>py 3.11+</span>
      <span className={styles.sep}>·</span>
      <span>MIT</span>
      <span className={styles.sep}>·</span>
      <span>build {sha}</span>
      <span className={styles.sep}>·</span>
      <span className={styles.ok}>● ci passing</span>
      <span className={styles.sep}>·</span>
      <span>pypi · weekly downloads via shields badge</span>
    </section>
  );
}
