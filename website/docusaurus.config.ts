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
    lastVersion: '0.5',
    versions: {
      current: { label: 'Next 🚧', path: 'next', banner: 'unreleased' },
      '0.5':   { label: '0.5 (latest)', path: '' },
      '0.4':   { label: '0.4', path: '0.4' },
      '0.3':   { label: '0.3', path: '0.3' },
    },
  },
  blog: false,
  theme: {
    customCss: './src/css/custom.css',
  },
};

const config: Config = {
  title: 'CubePi',
  tagline: 'A Pythonic, async-native agent framework',
  favicon: 'img/brand/cubepi-logo.svg',

  url: 'https://cubepi.pages.dev',
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

  customFields: { POSTHOG_KEY, POSTHOG_HOST, GIT_SHA, PACKAGE_VERSION },

  clientModules: [require.resolve('./src/clientModules/posthog.ts')],

  presets: [['classic', classicOptions]],

  plugins: [
    [
      '@docusaurus/plugin-google-gtag',
      {
        trackingID: 'G-NE2PDN0M91',
        anonymizeIP: false,
      },
    ],
  ],

  themeConfig: {
    image: 'img/brand/cubepi-social-preview.png',
    navbar: {
      title: 'CubePi',
      logo: { alt: 'CubePi logo', src: 'img/brand/cubepi-logo.svg' },
      items: [
        // Path-based active matching so each top-nav item highlights only for
        // its own docs section. `type: 'doc'` can't do this here because every
        // doc lives in one shared sidebar, which lights up all three at once.
        // Each regex tolerates an optional locale prefix (/zh-Hans) and an
        // optional version segment (e.g. /docs/0.4/) before the section.
        {
          to: '/docs/getting-started/installation',
          label: 'Docs',
          position: 'left',
          activeBaseRegex: '^(?:/zh-Hans)?/docs/(?!(?:(?:\\d+\\.\\d+|next)/)?(?:api|recipes)(?:/|$))',
        },
        {
          to: '/docs/api/',
          label: 'API',
          position: 'left',
          activeBaseRegex: '^(?:/zh-Hans)?/docs/(?:(?:\\d+\\.\\d+|next)/)?api(?:/|$)',
        },
        {
          to: '/docs/recipes/weather-agent',
          label: 'Recipes',
          position: 'left',
          activeBaseRegex: '^(?:/zh-Hans)?/docs/(?:(?:\\d+\\.\\d+|next)/)?recipes(?:/|$)',
        },
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
