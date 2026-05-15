// Placeholder home page — will be replaced by full implementation in T7/T16
import type {ReactNode} from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description={siteConfig.tagline}>
      <main style={{padding: '4rem 0', textAlign: 'center'}}>
        <Heading as="h1">{siteConfig.title}</Heading>
        <p>{siteConfig.tagline}</p>
      </main>
    </Layout>
  );
}
