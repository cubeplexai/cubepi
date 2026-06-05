import React from 'react';
import Layout from '@theme/Layout';
import Hero from '@site/src/components/Home/Hero';
import WhyTable from '@site/src/components/Home/WhyTable';
import HelloAgent from '@site/src/components/Home/HelloAgent';
import FeatureGrid from '@site/src/components/Home/FeatureGrid';
import InstallMatrix from '@site/src/components/Home/InstallMatrix';
import MetaBar from '@site/src/components/Home/MetaBar';
import { useIsZhHans } from '@site/src/hooks/useIsZhHans';

export default function Home(): React.ReactElement {
  const zh = useIsZhHans();
  return (
    <Layout
      title={zh
        ? 'CubePi — Pythonic 异步原生 Agent 框架'
        : 'CubePi — a Pythonic async-native agent framework'}
      description={zh
        ? 'CubePi 是 langgraph 和 pi-agent-core 的 Pythonic 异步原生替代方案。普通 async 函数、追加式持久化、3 个核心依赖。'
        : 'CubePi is a Pythonic async-native agent framework — a leaner alternative to langgraph and pi-agent-core. Plain async functions, append-only checkpointing, 3 core dependencies.'}
    >
      <Hero />
      <WhyTable />
      <HelloAgent />
      <FeatureGrid />
      <InstallMatrix />
      <MetaBar />
    </Layout>
  );
}
