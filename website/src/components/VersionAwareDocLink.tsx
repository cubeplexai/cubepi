import React, {type ReactNode} from 'react';
import DefaultNavbarItem from '@theme/NavbarItem/DefaultNavbarItem';
import type {Props as DefaultNavbarItemProps} from '@theme/NavbarItem/DefaultNavbarItem';
import {
  useActiveDocContext,
  useLatestVersion,
} from '@docusaurus/plugin-content-docs/client';
import {
  buildVersionAwareDocTo,
  SECTIONS,
  type Section,
} from './versionAwareDocLinkConfig';

// One shared sidebar means a plain `type: 'doc'` item would light up all three
// top-nav links at once. We instead drive the active state from the pathname
// (per-section regex, tolerant of the optional /zh-Hans locale and /docs/0.4/
// version segments) while building a version-aware `to` so clicking stays in
// whatever version the reader is currently browsing.

type Props = Omit<DefaultNavbarItemProps, 'to' | 'activeBaseRegex'> & {
  section: Section;
};

export default function VersionAwareDocLink({
  section,
  ...props
}: Props): ReactNode {
  const {activeVersion} = useActiveDocContext('default');
  const latestVersion = useLatestVersion('default');
  const version = activeVersion ?? latestVersion;
  const {activeBaseRegex} = SECTIONS[section];

  return (
    <DefaultNavbarItem
      {...props}
      to={buildVersionAwareDocTo(section, version.path)}
      activeBaseRegex={activeBaseRegex}
    />
  );
}
