export type Section = 'docs' | 'api' | 'recipes';

export const SECTIONS: Record<Section, {entry: string; activeBaseRegex: string}> = {
  docs: {
    entry: '',
    activeBaseRegex:
      '^(?:/zh-Hans)?/docs/(?!(?:(?:\\d+\\.\\d+|next)/)?(?:api|recipes)(?:/|$))',
  },
  api: {
    entry: 'api/',
    activeBaseRegex: '^(?:/zh-Hans)?/docs/(?:(?:\\d+\\.\\d+|next)/)?api(?:/|$)',
  },
  recipes: {
    entry: 'recipes/weather-agent',
    activeBaseRegex: '^(?:/zh-Hans)?/docs/(?:(?:\\d+\\.\\d+|next)/)?recipes(?:/|$)',
  },
};

export function buildVersionAwareDocTo(section: Section, versionPath: string): string {
  const base = versionPath.replace(/\/$/, '');
  const {entry} = SECTIONS[section];
  return entry ? `${base}/${entry}` : base;
}
