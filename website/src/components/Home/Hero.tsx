import React from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './Hero.module.css';

export default function Hero() {
  const social = useBaseUrl('/img/brand/cubepi-social-preview.png');
  const installCmd = 'pip install cubepi';
  const { siteConfig } = useDocusaurusContext();
  // Sourced from siteConfig.customFields.PACKAGE_VERSION (parsed from
  // pyproject.toml at config-load time) so the eyebrow tracks the
  // released version automatically.
  const version =
    (siteConfig.customFields?.PACKAGE_VERSION as string | undefined) ?? 'dev';

  return (
    <section className={styles.hero}>
      <img
        className={styles.banner}
        src={social}
        alt="CubePi · A Pythonic, async-native agent framework"
        width={1280}
        height={640}
      />
      <div className={styles.eyebrow}>cubepi · v{version} · alpha</div>
      <h1 className={styles.h1}>A Pythonic, async-native agent framework.</h1>
      <p className={styles.lead}>
        Plain async functions instead of graph nodes. 3 deps. Append-only checkpointing.
      </p>
      <div className={styles.actions}>
        <button
          type="button"
          className={`${styles.cta} ${styles.ctaPrimary}`}
          onClick={() => navigator.clipboard?.writeText(installCmd)}
          aria-label={`Copy install command: ${installCmd}`}
        >
          <code>{installCmd}</code>
          <kbd className={styles.kbd}>⌘C</kbd>
        </button>
        <Link className={`${styles.cta} ${styles.ctaGhost}`} to="/docs/getting-started/quick-start">
          Quick Start →
          <kbd className={styles.kbd}>G Q</kbd>
        </Link>
      </div>
    </section>
  );
}
