# W2 Marlin cache-policy product correctness smoke

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_cache_policy_product_correctness_2026-06-16`
- Lane: W2 Marlin evict-first product correctness smoke
- Remote git HEAD: `212b2bf925c998062ef22767a1da41ba47ed5101`
- Remote source state: clean `git status --short --untracked-files=no`
- Vast instance: `40826362`, 1x RTX 4090
- CUDA: driver `565.77`, `nvcc 12.4.131`
- Build command:
  `cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`
- Build rc: `0`
- Binary SHA256:
  `d38caf704f252045c29bdfe02795606937f400ab00edef05647da74179b215d5`
- Vast cleanup: instance `40826362` confirmed `stopped/exited`

## Correctness

`ferrum run`:

- Command shape: `ferrum run gemma3:27b-gptq --backend cuda --max-tokens 8
  --temperature 0 --kv-capacity 2560 --max-num-seqs 2 --output-format jsonl`
- rc: `0`
- Output content: `5`
- finish_reason: `stop`
- n_tokens: `3`

`ferrum serve`:

- Command shape: `ferrum serve --model gemma3:27b-gptq --backend cuda
  --max-num-seqs 16 --max-num-batched-tokens 2048 --kv-capacity 512`
- readiness: `/v1/models` ready
- chat rc: `0`
- Response content: `5`
- finish_reason: `length`
- usage: `prompt_tokens=23`, `completion_tokens=1`, `total_tokens=24`

Log scan:

- `server/error_scan.txt`: `0` lines for panic/error/NaN/`<unk>`/`[PAD]`/
  invalid UTF/mojibake/illegal address/CUDA error patterns used in this artifact.

`correctness_check.json` reports `ok=true`.

## Notes

The first background attempt failed before build because the script did not
include `/root/.cargo/bin` in `PATH`; that environment-only failure is preserved
under `build_initial_env_failure/`. The retry ran in the same Vast instance with
the same clean worktree and passed. This is product-entrypoint correctness
evidence after making Marlin B-weight `L2::evict_first` the CUDA default path.

This is not W2 release-grade evidence. It does not include a release performance
matrix or the final validator PASS line:
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.
