# W2 unified op profile c16 rerun - 2026-06-16

Diagnostic artifact for W2 Gemma3 27B CUDA GPTQ c16 bottleneck localization.
This is not release-grade performance evidence and did not produce
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.

## Scope

- Lane: W2 Gemma3 27B GPTQ c16 minimal diagnostic.
- Source checkpoint:
  `1707b001da835f99484f09dec252a9f3c66823e4`
  (`test(models): sample large unified op profiles`).
- Remote worktree dirty status: empty, saved in
  `remote/meta/git_status_short.txt`.
- Binary SHA256:
  `e2117a1df9613b15a2df470c3f7fa6b50a873b16b6dff61925ce9a9d33d4239f`.
- Hardware: Vast instance `41187356`, 1x NVIDIA GeForce RTX 4090,
  driver `580.95.05`, `nvidia-smi` CUDA `13.0`.
- Build: `cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-paged-attn-v2`.
- Runtime path: product `ferrum serve` plus `ferrum bench-serve`.
- Diagnostic config was saved in `remote/ferrum.toml`, including
  `decode_op_profile=true` and `marlin_profile=true`.

## Commands

The remote runner used the checked-in diagnostic script from
`../w2_unified_op_profile_c16_2026-06-16/run_profile.sh` with:

```bash
OUT=/workspace/artifacts/w2_unified_op_profile_c16_rerun_2026-06-16 \
  bash /workspace/run_profile.sh
```

The benchmark command inside the script was:

```bash
ferrum bench-serve \
  --base-url http://127.0.0.1:18142 \
  --model gemma3:27b-gptq \
  --tokenizer /workspace/hf-cache/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2 \
  --dataset random \
  --random-input-len 64 \
  --random-output-len 16 \
  --concurrency 16 \
  --num-prompts 16 \
  --warmup-requests 4 \
  --n-repeats 1 \
  --fail-on-error \
  --seed 9271 \
  --output json
```

## Correctness and stability

- `ferrum serve` streaming smoke passed: `SMOKE_OK True`.
- `bench-serve` returned rc `0`.
- `completed_per_run=[16]`, `errored_per_run=[0]`.
- `output_token_count_source="usage"`.
- Log scan did not find panic, OOM, illegal address, or CUDA error.
- Vast cleanup confirmed `cur_state=stopped`, `actual_status=exited`.

## Diagnostic performance

Single-repeat diagnostic values:

- request throughput: `9.92982317786992 req/s`
- output throughput: `158.87717084591873 tok/s`
- total throughput: `794.3858542295937 tok/s`
- TTFT p50/p95: `737.506014 ms` / `825.028914 ms`
- TPOT p50/p95: `57.694356 ms` / `86.602136 ms`
- ITL p50/p95: `51.493204 ms` / `93.061995 ms`

These are diagnostic-only because `--require-ci` was not used and
`n_repeats=1`.

## Bottleneck evidence

The target mixed-prefill frame was captured after changing profile sampling:

```text
[unified-op-profile] call#23 m_total=822 num_seqs=16 prefill=11 decode=5 max_kv=77 sampled=16 total=339796us qkv=30569us qkr=2048us attn=18567us o_proj=16654us norm=4492us gate_up=143991us act=5935us down=82787us resid=2542us final_norm=20us final_copy=66us lm_head=3150us readback=22039us generic_matmul=277151us generic_attn=18567us generic_qkr=2048us generic_norm=4512us generic_other=8543us marlin_kernel=255630us marlin_gate_up_kernel=141474us marlin_down_kernel=71062us unwrapped=6936us
```

Interpretation:

- The dominant cost is Gemma3 GPTQ dense MLP Marlin, especially
  `gate_up` and `down`.
- Attention is not the primary target in this frame:
  `attn=18.567 ms` versus `gate_up+down=226.778 ms`.
- The next high-signal step is not another FA2 or graph sweep. It is a
  same-shape vLLM/Ferrum Marlin MLP comparison for `gate_up/down` at
  representative `m_total` values.

## Next action

Compare Ferrum's GPTQ Marlin projection path with local vLLM source at
`/Users/chejinxuan/py_ws/vllm`, then build a minimal native CUDA or Rust CUDA
microbench for representative shapes:

- decode-like: `m=1`, `m=4`
- mixed-prefill: `m=150`, `m=373`, `m=822`
- projections: Gemma3 `qkv`, `o_proj`, `gate_up`, `down`

If Ferrum's single-op Marlin timing is slower than vLLM at the same shape,
fix the wrapper/kernel/packing/gather path. If single-op timing is comparable,
the next lever is scheduler/admission token budgeting to avoid TTFT-heavy
mixed-prefill frames.
