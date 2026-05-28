import React from 'react';
import Layout from '@theme/Layout';
import Hero from '@site/src/components/Home/Hero';
import WhyTable from '@site/src/components/Home/WhyTable';
import HelloAgent from '@site/src/components/Home/HelloAgent';
import FeatureGrid from '@site/src/components/Home/FeatureGrid';
import InstallMatrix from '@site/src/components/Home/InstallMatrix';
import MetaBar from '@site/src/components/Home/MetaBar';

export default function Home(): JSX.Element {
  return (
    <Layout title="CubePi — a Pythonic langgraph and pi-agent-core alternative"
            description="CubePi is a Pythonic, async-native alternative to langgraph and pi-agent-core. Plain async functions, append-only checkpointing, 3 core dependencies.">
      <Hero />
      <WhyTable />
      <HelloAgent />
      <FeatureGrid />
      <InstallMatrix />
      <MetaBar />
    </Layout>
  );
}
