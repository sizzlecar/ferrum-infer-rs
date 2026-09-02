# Ferrum website

`cloudflare-worker.js` serves the bilingual Ferrum landing page, favicon,
`robots.txt`, and `sitemap.xml` for `https://ferrum.pandaailabs.com`.

Validate and preview from this directory:

```bash
npx --yes wrangler@4.127.1 deploy --dry-run
npx --yes wrangler@4.127.1 dev --local
```

Production deployment requires the Panda AI Labs Cloudflare account and uses
the custom domain declared in `wrangler.jsonc`:

```bash
npx --yes wrangler@4.127.1 deploy
```

After deployment, verify the English and Chinese canonical pages, security
headers, `robots.txt`, and `sitemap.xml`. Keep public product claims aligned
with the release-supported model matrix and update release-specific links only
after their corresponding release gate passes.
