# Required GitHub Actions secrets

Before merging this PR, configure these repository secrets at
`Settings → Secrets and variables → Actions`:

| Secret           | Required for                  | Source                                                              |
|------------------|-------------------------------|---------------------------------------------------------------------|
| `CF_API_TOKEN`   | Cloudflare Pages deploy       | Cloudflare dashboard → My Profile → API Tokens → "Cloudflare Pages: Edit" |
| `CF_ACCOUNT_ID`  | Cloudflare Pages deploy       | Cloudflare dashboard → Workers & Pages → account ID                  |
| `POSTHOG_KEY`    | Build-time injection of PostHog client key | PostHog dashboard → Project settings → Project API key |
| `POSTHOG_HOST`   | (optional) PostHog endpoint   | Defaults to `https://us.i.posthog.com` if not set                    |

Also create a Cloudflare Pages project named `cubepi` in "Direct upload" mode before the first deploy.
