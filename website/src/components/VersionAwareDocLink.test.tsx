import {describe, expect, it} from 'vitest';

import {buildVersionAwareDocTo} from './versionAwareDocLinkConfig';

describe('VersionAwareDocLink', () => {
  it('links the docs section to the docs root', () => {
    expect(buildVersionAwareDocTo('docs', '/docs')).toBe('/docs');
    expect(buildVersionAwareDocTo('docs', '/docs/next')).toBe('/docs/next');
    expect(buildVersionAwareDocTo('docs', '/docs/0.7')).toBe('/docs/0.7');
  });
});
