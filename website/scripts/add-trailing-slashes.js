#!/usr/bin/env node
/**
 * Add trailing slashes to URLs in sitemap files.
 * Prevents Google from deindexing due to redirect chains.
 * Runs after docusaurus build.
 */

const fs = require('fs');
const path = require('path');

const sitemapFiles = [
  'build/sitemap.xml',
  'build/zh-Hans/sitemap.xml',
];

const buildDir = path.join(__dirname, '..');

for (const file of sitemapFiles) {
  const filePath = path.join(buildDir, file);
  if (!fs.existsSync(filePath)) {
    console.log(`Skipping ${file} (not found)`);
    continue;
  }

  let content = fs.readFileSync(filePath, 'utf-8');

  // Match URLs in <loc> tags and add trailing slash if not present
  // Don't add slash if URL already ends with .xml or has a hash
  const updated = content.replace(
    /<loc>([^<]+)<\/loc>/g,
    (match, url) => {
      // Skip if already has trailing slash, has a file extension, or is the root
      if (url.endsWith('/') || url.match(/\.\w+$/) || url === 'https://cubepi.ai') {
        return match;
      }
      return `<loc>${url}/<\/loc>`;
    }
  );

  fs.writeFileSync(filePath, updated, 'utf-8');
  console.log(`✓ Added trailing slashes to ${file}`);
}
