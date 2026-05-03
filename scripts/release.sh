#!/usr/bin/env bash
# release.sh — one-button release: bump version, PR, tag, binaries,
#              cargo publish, brew tap update, smoke test.
#
# Usage:
#   CARGO_REGISTRY_TOKEN=<token> ./scripts/release.sh 0.7.4
#
# Pipeline (per stage):
#   1. preflight       : tree clean, on main, builds, gh CLI logged in
#   2. bump            : edit workspace version in Cargo.{toml,lock} +
#                        ferrum-attention/Cargo.toml; commit on a fresh
#                        release branch; push; open PR
#   3. wait-merge      : poll PR CI; auto-merge once CPU + Metal pass;
#                        sync local main to merged commit
#   4. tag             : push v$VERSION tag → triggers release.yml
#   5. wait-release    : poll release.yml until SUCCESS; verify both
#                        tarballs uploaded
#   6. cargo-publish   : publish all 15 crates in topo order (5s sleep
#                        between for crates.io indexing)
#   7. brew-tap-update : clone homebrew-ferrum, edit Formula/ferrum.rb
#                        with new version + SHA256s, push
#   8. brew-verify     : `brew upgrade ferrum` locally, assert
#                        `ferrum --version == $VERSION`
#
# Re-run safety:
#   - If PR for the same version already exists, the script reuses it.
#   - If the tag is already pushed, the tag step is skipped.
#   - If a crate is already published at $VERSION, cargo publish skips
#     it (cargo errors with "already uploaded", we tolerate that).
#   - The brew tap update is `git push` — diff vs remote handles dups.

set -euo pipefail

# ── parse args ──────────────────────────────────────────────────────
if [[ $# -ne 1 ]]; then
  echo "usage: $0 <version>   (e.g. $0 0.7.4)" >&2
  exit 1
fi
VERSION="$1"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "version must be MAJOR.MINOR.PATCH (got: $VERSION)" >&2
  exit 1
fi
TAG="v${VERSION}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── helpers ─────────────────────────────────────────────────────────
log()  { printf '\n\033[1;36m▶\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m⚠\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }

# Crates in dependency-topological order. Mirrors what we determined
# from `cargo metadata` — see docs/bench/macos-2026-05-02 commit
# history for the derivation. Keep this list in sync if you add a
# crate: search "(workspace_members)" in Cargo.toml.
CRATES=(
  ferrum-types
  ferrum-interfaces
  ferrum-attention
  ferrum-runtime
  ferrum-kernels
  ferrum-kv
  ferrum-quantization
  ferrum-tokenizer
  ferrum-sampler
  ferrum-scheduler
  ferrum-testkit
  ferrum-models
  ferrum-engine
  ferrum-server
  ferrum-cli
)

# ── 1. preflight ────────────────────────────────────────────────────
log "[1/8] preflight"

if ! git diff-index --quiet HEAD --; then
  die "working tree has uncommitted changes — commit or stash first"
fi
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
  die "must be on main (currently on: $CURRENT_BRANCH)"
fi

git fetch origin main --quiet
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)
if [[ "$LOCAL" != "$REMOTE" ]]; then
  die "local main is not up-to-date with origin/main — run: git pull --ff-only"
fi

if ! command -v gh >/dev/null; then die "gh CLI not installed"; fi
if ! gh auth status >/dev/null 2>&1; then die "gh CLI not logged in (run: gh auth login)"; fi

if [[ -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
  die "CARGO_REGISTRY_TOKEN env var is required (export it before running)"
fi

# Catch a typo'd VERSION early — if it's not strictly greater than
# whatever's on crates.io for ferrum-cli, cargo publish will error
# anyway, but we'd rather fail before bumping branches.
CURRENT_PUBLISHED=$(curl -sf "https://crates.io/api/v1/crates/ferrum-cli" 2>/dev/null \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('crate',{}).get('max_version',''))" 2>/dev/null || echo "")
if [[ -n "$CURRENT_PUBLISHED" && "$CURRENT_PUBLISHED" == "$VERSION" ]]; then
  die "ferrum-cli $VERSION is already on crates.io — pick a higher version"
fi

ok "preflight: clean, on main, gh logged in, token set, target=$VERSION (current crates.io: $CURRENT_PUBLISHED)"

# ── 2. bump version + open PR ───────────────────────────────────────
log "[2/8] bump version + open PR"

PREV_VERSION=$(grep -m1 -E '^version\s*=' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
if [[ "$PREV_VERSION" == "$VERSION" ]]; then
  warn "Cargo.toml is already at $VERSION — skipping bump commit"
  BRANCH=""
else
  BRANCH="release/${VERSION}"
  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    warn "branch $BRANCH already exists locally — reusing"
    git checkout "$BRANCH"
  else
    git checkout -b "$BRANCH"
  fi

  # Replace the workspace version in the two relevant files.
  sed -i '' "s/${PREV_VERSION}/${VERSION}/g" Cargo.toml crates/ferrum-attention/Cargo.toml
  cargo update --workspace --quiet

  if ! git diff-index --quiet HEAD --; then
    git add Cargo.toml Cargo.lock crates/ferrum-attention/Cargo.toml
    git commit -m "chore: bump workspace version to ${VERSION}"
    git push -u origin "$BRANCH" 2>&1 | tail -1
  fi

  # Open PR if not already there.
  if ! gh pr view "$BRANCH" --json number >/dev/null 2>&1; then
    gh pr create --title "chore: bump workspace version to ${VERSION}" \
      --body "Automated by scripts/release.sh — workspace version bump for ${VERSION} release." \
      --base main --head "$BRANCH" >/dev/null
  fi
  PR_NUM=$(gh pr view "$BRANCH" --json number --jq .number)
  ok "PR #$PR_NUM opened for $BRANCH"
fi

# ── 3. wait for CI then auto-merge ──────────────────────────────────
log "[3/8] wait for CI + auto-merge"

if [[ -n "$BRANCH" ]]; then
  PR_NUM=$(gh pr view "$BRANCH" --json number --jq .number)
  while :; do
    STATUS=$(gh pr view "$PR_NUM" --json statusCheckRollup --jq \
      '[.statusCheckRollup[] | select(.name=="CPU (Linux)" or .name=="Metal (macOS)") | .conclusion // .status] | join(",")')
    case "$STATUS" in
      "SUCCESS,SUCCESS") break ;;
      *FAILURE*)         die "PR #$PR_NUM CI failed: $STATUS" ;;
      *)                 printf "  CI: %s — sleeping 30s\r" "$STATUS"; sleep 30 ;;
    esac
  done
  echo
  gh pr merge "$PR_NUM" --squash --delete-branch >/dev/null
  ok "PR #$PR_NUM merged"

  git checkout main
  git fetch origin main --quiet
  git pull --ff-only --quiet
fi

# ── 4. tag ──────────────────────────────────────────────────────────
log "[4/8] tag $TAG + trigger release.yml"

if git rev-parse "$TAG" >/dev/null 2>&1; then
  warn "tag $TAG already exists — skipping"
else
  git tag "$TAG"
  git push origin "$TAG"
fi

# ── 5. wait for release.yml ─────────────────────────────────────────
log "[5/8] wait for release.yml binaries"

# Find the run for this tag.
RUN_ID=""
for _ in {1..12}; do
  RUN_ID=$(gh run list --workflow=release.yml --limit 5 \
    --json databaseId,headBranch \
    --jq ".[] | select(.headBranch==\"$TAG\") | .databaseId" | head -1 || true)
  [[ -n "$RUN_ID" ]] && break
  sleep 5
done
[[ -z "$RUN_ID" ]] && die "could not find release.yml run for $TAG"

while :; do
  STATUS=$(gh run view "$RUN_ID" --json status,conclusion --jq '.status + ":" + (.conclusion // "")')
  case "$STATUS" in
    completed:success) break ;;
    completed:*)       die "release.yml failed: $STATUS — see gh run view $RUN_ID" ;;
    *)                 printf "  release.yml: %s — sleeping 30s\r" "$STATUS"; sleep 30 ;;
  esac
done
echo
ok "release.yml completed; tarballs uploaded to v$VERSION release"

# ── 6. cargo publish loop ───────────────────────────────────────────
log "[6/8] cargo publish 15 crates"

for crate in "${CRATES[@]}"; do
  printf "  publish %-22s — " "$crate"
  if cargo publish -p "$crate" --allow-dirty --token "$CARGO_REGISTRY_TOKEN" \
       >/tmp/release-${crate}.log 2>&1; then
    echo "ok"
  else
    if grep -q "already uploaded" /tmp/release-${crate}.log; then
      echo "already published, skip"
    else
      tail -10 /tmp/release-${crate}.log
      die "cargo publish $crate failed; full log: /tmp/release-${crate}.log"
    fi
  fi
  sleep 5  # let crates.io's index settle before the next dependent
done
ok "all crates published at $VERSION"

# ── 7. brew tap update ──────────────────────────────────────────────
log "[7/8] update Homebrew tap"

SHA_MAC=$(curl -sfL "https://github.com/sizzlecar/ferrum-infer-rs/releases/download/${TAG}/ferrum-macos-aarch64.tar.gz.sha256" | cut -d' ' -f1)
SHA_LINUX=$(curl -sfL "https://github.com/sizzlecar/ferrum-infer-rs/releases/download/${TAG}/ferrum-linux-x86_64.tar.gz.sha256" | cut -d' ' -f1)
[[ -z "$SHA_MAC"   ]] && die "could not fetch macOS sha256"
[[ -z "$SHA_LINUX" ]] && die "could not fetch Linux sha256"

TAP_DIR="/tmp/homebrew-ferrum-release-${VERSION}"
rm -rf "$TAP_DIR"
git clone --depth 1 --quiet https://github.com/sizzlecar/homebrew-ferrum.git "$TAP_DIR"

cat > "$TAP_DIR/Formula/ferrum.rb" <<EOF
class Ferrum < Formula
  desc "Production-grade LLM inference in Rust — runs on Apple Silicon and CUDA"
  homepage "https://github.com/sizzlecar/ferrum-infer-rs"
  version "${VERSION}"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/sizzlecar/ferrum-infer-rs/releases/download/${TAG}/ferrum-macos-aarch64.tar.gz"
      sha256 "${SHA_MAC}"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/sizzlecar/ferrum-infer-rs/releases/download/${TAG}/ferrum-linux-x86_64.tar.gz"
      sha256 "${SHA_LINUX}"
    end
  end

  def install
    bin.install "ferrum"
    doc.install "README.md"
  end

  test do
    assert_match "ferrum #{version}", shell_output("#{bin}/ferrum --version")
  end
end
EOF

cd "$TAP_DIR"
if git diff --quiet; then
  warn "tap formula already at $VERSION — skipping push"
else
  git add Formula/ferrum.rb
  git commit -m "ferrum ${VERSION}"
  git push --quiet
  ok "tap pushed: sizzlecar/homebrew-ferrum @ ${VERSION}"
fi
cd "$ROOT"

# ── 8. brew upgrade verify ──────────────────────────────────────────
log "[8/8] brew upgrade + verify"

brew update >/dev/null 2>&1 || true
brew upgrade ferrum >/dev/null 2>&1 || brew install ferrum >/dev/null
INSTALLED=$(ferrum --version | awk '{print $2}')
if [[ "$INSTALLED" == "$VERSION" ]]; then
  ok "brew-installed ferrum is $VERSION"
else
  die "brew installed $INSTALLED but expected $VERSION"
fi

echo
echo "─────────────────────────────────────────────────────"
echo "✅ ferrum $VERSION shipped on all three channels:"
echo "   • GitHub Release: https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/$TAG"
echo "   • crates.io:      https://crates.io/crates/ferrum-cli/$VERSION"
echo "   • Homebrew:       brew tap sizzlecar/ferrum && brew install ferrum"
echo "─────────────────────────────────────────────────────"
