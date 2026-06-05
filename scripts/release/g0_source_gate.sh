#!/usr/bin/env bash
set -euo pipefail

LANE="${1:-}"
if [[ -z "$LANE" ]]; then
  echo "usage: scripts/release/g0_source_gate.sh {unit|metal|cuda-smoke|cuda-full|all-source} [OUT_ROOT]" >&2
  exit 2
fi
OUT_ROOT="${2:-docs/release/g0/source-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_ROOT"

pass() { echo "G0 SOURCE ${1} PASS: $OUT_ROOT"; }

run_unit() {
  cargo test --workspace --all-targets | tee "$OUT_ROOT/unit.log"
  python3 -m py_compile \
    scripts/release/inventory_tree.py \
    scripts/release/validate_metal_readme_regression.py \
    scripts/release/release_binary_gate.py \
    scripts/release/g0_release_summary.py \
    scripts/release/selftest_g0_validators.py \
    scripts/release/selftest_g1_g3_g4_release_regression.py | tee "$OUT_ROOT/release-scripts-pycompile.log"
  bash -n scripts/release/g0_source_gate.sh | tee "$OUT_ROOT/g0-source-bashn.log"
  python3 scripts/release/selftest_g0_validators.py | tee "$OUT_ROOT/g0-validator-selftest.log"
  python3 scripts/release/selftest_g1_g3_g4_release_regression.py | tee "$OUT_ROOT/g1-g3-g4-validator-selftest.log"
  echo '{"status":"pass","lane":"unit"}' > "$OUT_ROOT/unit.gate.json"
  pass unit
}

run_metal() {
  cargo build --release -p ferrum-cli --features metal --tests | tee "$OUT_ROOT/metal-build.log"
  local metal_out="$OUT_ROOT/metal-readme"
  python3 scripts/metal_readme_regression.py --out "$metal_out" --ferrum-bin ./target/release/ferrum | tee "$OUT_ROOT/metal-runner.log"
  python3 scripts/release/validate_metal_readme_regression.py "$metal_out" | tee "$OUT_ROOT/metal-validator.log"
  echo '{"status":"pass","lane":"metal","artifact":"metal-readme"}' > "$OUT_ROOT/metal.gate.json"
  pass metal
}

cuda_build() {
  cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source | tee "$OUT_ROOT/cuda-build.log"
}

run_cuda_template() {
  local template="$1"
  local label="$2"
  local change_type
  change_type="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("validation",{}).get("change_type","opt_in_experiment"))' "$template")"
  python3 - "$template" "$OUT_ROOT" <<'PY'
import json, pathlib, sys
src = pathlib.Path(sys.argv[1])
out_root = pathlib.Path(sys.argv[2])
base = json.load(open(src))
cells = base.pop("concurrency_cells")
for i, c in enumerate(cells):
    cfg = dict(base)
    cfg["concurrency"] = c
    cfg["out_root"] = str(out_root / f"c{c}")
    cfg["port_base"] = int(base.get("port_base", 19000)) + i * 10
    cfg["validation"] = dict(base.get("validation", {}))
    cfg["validation"]["required_concurrency_cells"] = [c]
    for j, case in enumerate(cfg.get("cases", [])):
        case["port"] = cfg["port_base"] + j
    path = out_root / f"{src.stem}-c{c}.json"
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    print(path)
PY
  for cfg in "$OUT_ROOT"/${label}-c*.json; do
    python3 scripts/m3_ab_runner.py --config "$cfg"
    art=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["out_root"])' "$cfg")
    python3 scripts/m3_validate_runner_artifact.py "$art"
  done
  python3 scripts/m3_collect_allcell_runner_artifacts.py "$OUT_ROOT" \
    --baseline-case baseline \
    --candidate candidate \
    --change-type "$change_type"
  python3 scripts/m3_validate_runner_artifact.py "$OUT_ROOT"
  echo "{\"status\":\"pass\",\"lane\":\"$label\"}" > "$OUT_ROOT/$label.gate.json"
  pass "$label"
}

case "$LANE" in
  unit) run_unit ;;
  metal) run_metal ;;
  cuda-smoke) cuda_build; run_cuda_template scripts/release/configs/g0_cuda4090_smoke.json g0_cuda4090_smoke ;;
  cuda-full) cuda_build; run_cuda_template scripts/release/configs/g0_cuda4090_full.json g0_cuda4090_full ;;
  all-source)
    run_unit
    if [[ "$(uname -s)" == "Darwin" ]]; then
      run_metal
    else
      echo "G0 SOURCE all-source: skipping metal on non-macOS"
    fi
    echo '{"status":"pass","lane":"all-source"}' > "$OUT_ROOT/all-source.gate.json"
    pass all-source
    ;;
  *) echo "unknown lane: $LANE" >&2; exit 2 ;;
esac
