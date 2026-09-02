# Ferrum 0.8.4 release goal

## Scope

Release the already-merged `main` model-adoption work as Ferrum 0.8.4 without
adding another feature family. Release preparation may change version metadata,
user onboarding documentation, release workflows, validators, and packaging.
Any product defect found by the gates must be fixed narrowly and revalidated.

The model-adoption changes since 0.8.3 do not, by themselves, expand Ferrum's
formal release-support matrix. Models outside the existing release-supported
Qwen3.5/Qwen3/Llama rows remain experimental or development evidence until
their release-grade correctness and accelerator gates produce the required
artifacts and PASS lines.

This release is not complete merely because a tag, crate, formula, or GitHub
release exists. Completion requires the final validator line defined below.

## Frozen candidate

- The release candidate must be a clean commit descended from
  `84be21f06dcd8b625de00ca5d62ace1e3046db47`.
- `Cargo.toml`, workspace path dependencies, `Cargo.lock`, README install
  commands, and staged-asset workflows must all identify version `0.8.4`.
- An annotated RC tag must peel to the exact candidate commit.
- CPU, Metal, and CUDA assets are built exactly once from that commit. The same
  bytes are used for prerelease validation and the final release.
- The CUDA build must bind a validated native-operator artifact set produced
  from the current checked-in source bundle. It must not reuse the stale 0.8.3
  operator set merely because the binary links.

## README contract

English and Chinese README Quick Starts must stay behaviorally equivalent and
must use the release-supported Qwen3.5 4B aliases:

- Metal: `qwen3.5:4b-q4_k_m`.
- CUDA: `qwen3.5:4b`.

The first command must include `--disable-thinking`, and the download size must
be stated before the command. The docs must distinguish model download time
from a hung process and must state how to restore the template's default
reasoning behavior. A smaller unsupported model must not become the primary
Quick Start solely to reduce the download.

For both backends, the exact documented flow must cover:

1. binary version and help;
2. `ferrum doctor <MODEL>`;
3. a cold-cache model download through the public model source;
4. `ferrum run` with a non-empty objective response;
5. `ferrum serve` readiness and `/v1/models`;
6. non-stream Chat Completions with HTTP 200 and non-empty content;
7. streaming Chat Completions with positive usage and exactly one `[DONE]`;
8. a log scan rejecting panic, OOM, CUDA/Metal errors, invalid UTF-8, `<unk>`,
   `[PAD]`, and raw internal control-token leakage.

The actual commands, binary SHA256, model revision/files, cache root, start and
finish times, deadline, progress signal, responses, and logs must be saved.

## Required gates before prerelease

- `FERRUM GATE unit PASS: <out_dir>`.
- `FERRUM GATE metal PASS: <out_dir>`.
- `FERRUM GATE cuda-full PASS: <out_dir>` on exactly one RTX 4090.
- `FERRUM GATE cuda-llama-dense PASS: <out_dir>` on exactly one RTX 4090.
- Release-workflow policy and native-operator-set validation PASS.
- Exact staged Metal and CUDA tarballs must pass their prepublication
  `--asset-path` binary gates.

The existing G0 gate rules remain authoritative. A model-adoption PASS or CI
success does not replace these release gates.

## Prerelease download gate

Create the GitHub release for tag `v0.8.4` as a prerelease and upload the exact
validated candidate assets and adjacent checksums/manifests. Do not publish
crates.io or update Homebrew yet.

On a clean Metal environment and a clean CUDA environment:

- download the assets through the public GitHub prerelease URLs, not from the
  Actions artifact store or a local path;
- verify GitHub's asset digest and the adjacent SHA256 file;
- extract into a fresh directory and execute that extracted binary;
- use a fresh model-cache directory and no undocumented behavior-changing
  environment variables;
- execute the complete README contract above.

The prerelease gate must prove that the downloaded bytes equal the staged bytes
and print:

```text
FERRUM 0.8.4 PRERELEASE DOWNLOAD PASS: <out_dir>
```

## Promotion and post-release gates

After the prerelease download gate passes, promote the same GitHub release by
changing only `prerelease: true` to `prerelease: false`. Do not rebuild, replace,
rename, or re-upload assets during promotion.

Then verify:

- the annotated `v0.8.4` tag still peels to the frozen candidate;
- every final asset id, name, size, and digest equals its prerelease identity;
- the release is returned by `releases/latest`;
- every workspace crate is published to crates.io at version `0.8.4`, and the
  published source matches the frozen candidate;
- Homebrew Metal install and CUDA fetch gates use the final public URLs;
- `python3 scripts/release/g0_release_summary.py docs/release/g0/0.8.4`
  prints `G0 RELEASE PASS: docs/release/g0/0.8.4`.

Promotion evidence must print:

```text
FERRUM 0.8.4 PROMOTION PASS: <out_dir>
```

## Outreach

Promotion starts only after all final release gates pass. Posts must disclose
the maintainer relationship, make no unsupported performance comparison, obey
each community's self-promotion and AI-content rules, and link the final 0.8.4
release or repository. Filtered or removed posts are recorded as such and are
not counted as successful distribution.

The official landing page is `https://ferrum.pandaailabs.com`. It must provide
indexable English and Chinese canonical pages, accurate supported-platform and
model-scope language, `robots.txt`, and `sitemap.xml`. It may link the repository
before release, but it must not advertise 0.8.4 as final or link version-specific
0.8.4 assets until the final release gates pass.

## Final completion

The final goal validator must aggregate the source, accelerator, prerelease
download, promotion, release asset, Homebrew, and G0 summary artifacts and end
with exactly:

```text
FERRUM 0.8.4 RELEASE PASS: docs/release/g0/0.8.4
```
