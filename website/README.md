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

Installers download immutable release assets through `/downloads/vVERSION/NAME`.
The Worker streams these bytes through Cloudflare's fetch cache; it never buffers
large installers in JavaScript. The R2 bucket `ferrum-releases`, exposed at
`ferrum-downloads.pandaailabs.com`, contains verified copies of published assets.
Missing objects fall back to the same public GitHub release at the edge, so new
versions work before an optional R2 mirror is populated. Client-side GitHub
fallback remains available if the CDN cannot be reached. Version lookup stays
dynamic and installers still verify SHA256 before execution.

Only publish existing formal release assets to R2, under their original tag and
filename, after comparing size and SHA256 with GitHub metadata. Set immutable
objects to `Cache-Control: public, max-age=31536000, immutable`; do not overwrite
published version paths. Verify the public download's bytes, `HEAD`/`Range`
responses, cache hits, and the actual README installation command afterward.
