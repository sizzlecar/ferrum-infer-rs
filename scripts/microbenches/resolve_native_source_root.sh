#!/usr/bin/env bash

if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "resolve_native_source_root.sh requires REPO_ROOT" >&2
  return 1 2>/dev/null || exit 1
fi

FERRUM_NATIVE_SOURCE_BUNDLE_MANIFEST="$REPO_ROOT/native-operators/cuda/source-bundles/ferrum-native-cuda-v1.json"
FERRUM_NATIVE_SOURCE_CACHE="${FERRUM_NATIVE_SOURCE_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/ferrum/native-sources}"
FERRUM_NATIVE_SOURCE_BUNDLE_ID="$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["bundle_id"])' \
    "$FERRUM_NATIVE_SOURCE_BUNDLE_MANIFEST"
)"
FERRUM_NATIVE_SOURCE_ROOT="${FERRUM_NATIVE_SOURCE_ROOT:-$FERRUM_NATIVE_SOURCE_CACHE/materialized/$FERRUM_NATIVE_SOURCE_BUNDLE_ID}"

python3 "$REPO_ROOT/scripts/release/native_operator_source_bundle.py" ensure \
  --manifest "$FERRUM_NATIVE_SOURCE_BUNDLE_MANIFEST" \
  --cache "$FERRUM_NATIVE_SOURCE_CACHE/archives" \
  --out "$FERRUM_NATIVE_SOURCE_ROOT"

export FERRUM_NATIVE_SOURCE_ROOT
