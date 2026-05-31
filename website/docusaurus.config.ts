import * as fs from 'fs';
import * as path from 'path';

import type { Config } from '@docusaurus/types';
import type { Options as ClassicOptions } from '@docusaurus/preset-classic';
import { themes as prismThemes } from 'prism-react-renderer';

// Use `||` not `??` here: GitHub Actions expands an unset secret to "" (an
// empty string passes the nullish check, so `??` would leave api_host empty
// and PostHog falls back to the current origin — sending POST /e/ to the
// docs domain and hitting a 405 from Cloudflare Pages).
const POSTHOG_KEY = process.env.POSTHOG_KEY || '';
const POSTHOG_HOST = process.env.POSTHOG_HOST || 'https://us.i.posthog.com';
const GIT_SHA = process.env.GITHUB_SHA?.slice(0, 7) ?? 'dev';

// Single source of truth for the package version shown in the homepage
// MetaBar: read pyproject.toml at config-load time so the site never
// drifts from the actual released version. Plain regex parse — avoids
// a TOML-parser dep just for one field.
const PYPROJECT_TOML = fs.readFileSync(
  path.join(__dirname, '..', 'pyproject.toml'),
  'utf-8',
);
const VERSION_MATCH = PYPROJECT_TOML.match(/^version\s*=\s*"([^"]+)"/m);
const PACKAGE_VERSION = VERSION_MATCH ? VERSION_MATCH[1] : 'dev';

const classicOptions: ClassicOptions = {
  docs: {
    sidebarPath: './sidebars.ts',
    editUrl: 'https://github.com/cubeplexai/cubepi/edit/main/website/',
    lastVersion: '0.6',
    versions: {
      current: { label: 'Next 🚧', path: 'next', banner: 'unreleased' },
      '0.6':   { label: '0.6 (latest)', path: '' },
      '0.5':   { label: '0.5', path: '0.5' },
      '0.4':   { label: '0.4', path: '0.4' },
      '0.3':   { label: '0.3', path: '0.3' },
    },
  },
  blog: false,
  theme: {
    customCss: './src/css/custom.css',
  },
  sitemap: {
    ignorePatterns: [
      '/docs/next/**',
      '/docs/0.3/**',
      '/docs/0.4/**',
      '/docs/0.5/**',
    ],
  },
};

const config: Config = {
  title: 'CubePi',
  tagline: 'A Pythonic, async-native agent framework — an alternative to langgraph and pi-agent-core',
  favicon: 'img/brand/cubepi-logo.svg',

  url: 'https://cubepi.ai',
  baseUrl: '/',
  organizationName: 'cubeplexai',
  projectName: 'cubepi',

  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',
  onBrokenMarkdownLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh-Hans'],
    localeConfigs: {
      en:        { label: 'English' },
      'zh-Hans': { label: '简体中文' },
    },
  },

  headTags: [
    {
      tagName: 'script',
      attributes: { type: 'application/ld+json' },
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'SoftwareApplication',
        name: 'CubePi',
        description: 'A Pythonic, async-native alternative to langgraph and pi-agent-core. Plain async functions, append-only checkpointing, minimal dependencies.',
        url: 'https://cubepi.ai',
        applicationCategory: 'DeveloperApplication',
        operatingSystem: 'Linux, macOS, Windows',
        programmingLanguage: 'Python',
      }),
    },
  ],

  customFields: { POSTHOG_KEY, POSTHOG_HOST, GIT_SHA, PACKAGE_VERSION },

  clientModules: [require.resolve('./src/clientModules/posthog.ts')],

  markdown: {
    // Strip empty ## [Unreleased] sections before MDX compilation so the
    // /changelog page doesn't open with a heading that has no content and
    // makes the first release look unreleased.
    preprocessor: ({fileContent}) =>
      fileContent.replace(/^## \[Unreleased\]\n+(?=##)/m, ''),
  },

  presets: [['classic', classicOptions]],

  plugins: [
    [
      '@docusaurus/plugin-google-gtag',
      {
        trackingID: 'G-NE2PDN0M91',
        anonymizeIP: true,
      },
    ],
  ],

  themeConfig: {
    metadata: [
      { name: 'keywords', content: 'CubePi, langgraph alternative, pi-agent-core alternative, Python agent framework, async agent, LLM agent, AI agent framework, tool-use agent' },
      { name: 'twitter:site', content: '@cubeplexai' },
      { name: 'twitter:creator', content: '@cubeplexai' },
    ],
    image: 'img/brand/cubepi-social-preview.png',
    navbar: {
      title: 'CubePi',
      logo: { alt: 'CubePi logo', src: 'img/brand/cubepi-logo.svg' },
      items: [
        // Version-aware section links (see src/components/VersionAwareDocLink):
        // active state is path-driven so only one lights up per page, and the
        // target stays within the version the reader is currently browsing.
        { type: 'custom-versionAwareDocLink', section: 'docs', label: 'Docs', position: 'left' },
        { type: 'custom-versionAwareDocLink', section: 'api', label: 'API', position: 'left' },
        { type: 'custom-versionAwareDocLink', section: 'recipes', label: 'Recipes', position: 'left' },
        { to: '/changelog', label: 'Changelog', position: 'left' },
        { type: 'docsVersionDropdown', position: 'right' },
        { type: 'localeDropdown', position: 'right' },
        { href: 'https://github.com/cubeplexai/cubepi', label: 'GitHub', position: 'right' },
      ],
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'toml'],
    },
    colorMode: { defaultMode: 'light', respectPrefersColorScheme: true },
  },
};

export default config;
