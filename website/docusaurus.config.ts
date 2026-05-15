import type { Config } from '@docusaurus/types';
import type { Options as ClassicOptions } from '@docusaurus/preset-classic';
import { themes as prismThemes } from 'prism-react-renderer';

const POSTHOG_KEY = process.env.POSTHOG_KEY ?? '';
const POSTHOG_HOST = process.env.POSTHOG_HOST ?? 'https://us.i.posthog.com';
const GIT_SHA = process.env.GITHUB_SHA?.slice(0, 7) ?? 'dev';

const classicOptions: ClassicOptions = {
  docs: {
    sidebarPath: './sidebars.ts',
    editUrl: 'https://github.com/cubeplexai/cubepi/edit/main/website/',
    lastVersion: 'current',
    versions: {
      current: { label: 'Next 🚧', path: 'next', banner: 'unreleased' },
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

  customFields: { POSTHOG_KEY, POSTHOG_HOST, GIT_SHA },

  clientModules: [require.resolve('./src/clientModules/posthog.ts')],

  presets: [['classic', classicOptions]],

  themeConfig: {
    image: 'img/brand/cubepi-social-preview.png',
    navbar: {
      title: 'CubePi',
      logo: { alt: 'CubePi logo', src: 'img/brand/cubepi-logo.svg' },
      items: [
        { type: 'doc', docId: 'getting-started/installation', label: 'Docs', position: 'left' },
        { type: 'doc', docId: 'api/index', label: 'API', position: 'left' },
        { type: 'doc', docId: 'recipes/weather-agent', label: 'Recipes', position: 'left' },
        // { type: 'docsVersionDropdown', position: 'right' }, // enabled in T21
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
