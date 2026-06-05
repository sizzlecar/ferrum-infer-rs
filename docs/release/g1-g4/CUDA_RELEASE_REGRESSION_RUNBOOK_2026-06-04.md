# G1/G3/G4 CUDA and Release Regression Runbook

Date: 2026-06-04

This runbook is the paid-GPU execution plan for the remaining
`G1_G3_G4_改造验收文档_2026-06-04.md` requirements. It is not release evidence
by itself. A claim is valid only after the listed commands produce artifact
directories and exact PASS lines.

## Lane

- Hardware: one RTX 4090 CUDA runner.
- Model for CUDA release performance: `Qwen/Qwen3-30B-A3B-GPTQ-Int4`.
- Small semantic model: `Qwen/Qwen3-0.6B`.
- CUDA features:

```bash
export CUDA_FEATURES=cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
```

## Expected Runtime / Cost

- CUDA small semantic gates plus CUDA smoke: about 2-4 hours on a prepared
  RTX 4090 pod.
- CUDA full c=1/4/16/32 regression: about 3-6 additional hours.
- Target machine price: about $1/hour. Expected total if both smoke and full
  are required in one session: about $5-$10.

## Stop Conditions

Stop the paid GPU run and preserve artifacts immediately when any of these
happen:

- Any correctness gate fails.
- Any `bench-serve --fail-on-error` run returns non-zero.
- Any log scan finds panic, KV cache overflow, `<unk>`, `[PAD]`, invalid UTF-8,
  missing/duplicate `[DONE]`, malformed SSE, or silent feature fallback.
- CUDA build fails or required CUDA/FA2/CUTLASS dependencies are missing.
- The same full sweep fails once; do not rerun full repeatedly before inspecting
  the failing artifact.

After collecting artifacts, stop or destroy the paid GPU instance unless the
user explicitly asks to keep it running.

## Artifact Root

On the CUDA machine, run from the repository root:

```bash
export SHA=$(git rev-parse --short HEAD)
export OUT_ROOT=docs/release/g1-g4/cuda/$(date -u +%Y%m%d-%H%M%S)-${SHA}
mkdir -p "$OUT_ROOT"
```

## Correctness Gates Before Performance Claims

Run source tests before treating any performance data as evidence:

```bash
cargo test --workspace --all-targets
```

Then run the G1/G3/G4 CUDA semantic/product gates:

```bash
python3 scripts/release/g1_vllm_migration_gate.py \
  --out "$OUT_ROOT/g1-vllm-migration" \
  --model Qwen/Qwen3-0.6B \
  --model-name Qwen/Qwen3-0.6B \
  --cargo-features "$CUDA_FEATURES"

python3 scripts/release/g3_cache_product_gate.py \
  --out "$OUT_ROOT/g3-cache-product-small" \
  --model Qwen/Qwen3-0.6B \
  --cargo-features "$CUDA_FEATURES" \
  --bench-timeout 1800

python3 scripts/release/g4_lora_startup_gate.py \
  --out "$OUT_ROOT/g4-lora-inference-small" \
  --model Qwen/Qwen3-0.6B \
  --cargo-features "$CUDA_FEATURES" \
  --num-prompts 8 \
  --warmup-requests 2 \
  --random-input-len 32 \
  --random-output-len 8 \
  --bench-timeout 1800
```

Required PASS lines:

```text
G1 VLLM MIGRATION PASS: <artifact_dir>
G3 CACHE PRODUCT PASS: <artifact_dir>
G4 LORA INFERENCE PASS: <artifact_dir>
```

## CUDA 30B Performance Gates

Use the repository CUDA source gates for the 4090 / Qwen3-30B-A3B GPTQ Int4
path. Start with smoke:

```bash
scripts/release/g0_source_gate.sh cuda-smoke "$OUT_ROOT/g0-cuda-smoke"
```

Required PASS line:

```text
G0 SOURCE g0_cuda4090_smoke PASS: <out_root>
```

Before release-ready claims, run full:

```bash
scripts/release/g0_source_gate.sh cuda-full "$OUT_ROOT/g0-cuda-full"
```

Required PASS line:

```text
G0 SOURCE g0_cuda4090_full PASS: <out_root>
```

The full lane covers c=1/4/16/32 with repeats and default-path performance
regression checks. Do not broaden beyond one RTX 4090 or add more CUDA models
without explicit approval.

## Overall Release Regression Artifact

After the CUDA gates pass, create:

```text
docs/release/g1-g4/release-regression/<timestamp>-<sha>/
```

It must include the Metal artifacts already collected, the CUDA artifacts above,
the `ferrum run` and `ferrum serve` correctness/performance logs, and the
required comparison JSON files listed in the goal document.

The final overall PASS line is:

```text
G1-G3-G4 RELEASE REGRESSION PASS: <artifact_dir>
```

