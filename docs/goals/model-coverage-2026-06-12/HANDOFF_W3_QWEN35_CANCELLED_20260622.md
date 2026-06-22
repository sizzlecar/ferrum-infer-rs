# W3 Qwen35 Cancelled Handoff — 2026-06-22

User cancelled the active W3 Qwen3.5 release-grade goal on 2026-06-22 and
asked to clean up machines, organize code, commit, push, and leave a handoff.

## Status

- Goal status: cancelled by user, not complete.
- Final W3 PASS: absent. There is still no
  `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` artifact.
- Performance claim: none. No new same-hardware A/B performance run was
  collected after the latest source changes.
- Correct current performance wording from local copied evidence:
  - The old release-shape L5 artifact reported c32 around `142.839 tok/s`;
    this is not the current post-scheduler diagnostic number.
  - The latest copied local scheduler/cohort diagnostic in `STATUS.md` records
    c32 `651.4 output tok/s`.
  - The accepted local vLLM c32 baseline is `1708.52785 output tok/s`; the
    80% target is `1366.82228 output tok/s`.
  - Therefore the locally evidenced current gap is still about `2.1x` from
    the 80% target. If a remote log has a newer `760+ tok/s` number, it was
    not copied into this local evidence set and must not be claimed without
    the artifact.

## Machine Cleanup

- Vast instance: `41422823`.
- Stop request result: `success=true`.
- Final checked state:
  `cur_state=stopped actual_status=exited intended_status=stopped`.
- No local `cargo`, `bench-serve`, `ferrum serve`, W3 lane, or remote SSH
  process was left running.

## Code Left In This Branch

This final cleanup commit keeps one small product-path config fix:

- `ferrum run` now decides engine-level paged KV from the resolved
  `startup_auto_config.runtime_config`, not the pre-auto-config snapshot.
- Runtime defaults now emit canonical `FERRUM_PAGED_KV` while keeping legacy
  `FERRUM_METAL_PAGED_KV` for compatibility.
- Qwen35 model construction prefers `FERRUM_PAGED_KV` and falls back to
  `FERRUM_METAL_PAGED_KV`.

Why it is kept:

- It prevents a product-path mismatch where auto-config can resolve paged-KV
  behavior for the model, while `ferrum run` still chooses the engine KV
  manager from the older pre-auto-config snapshot.
- This is not a performance-result claim. It only removes a configuration
  inconsistency before any future GPU validation.

## Validation Run

- `cargo fmt --all`
- `cargo fmt --all -- --check`
- `git diff --check`
- `cargo test -p ferrum-models qwen35_paged_kv_prefers_canonical_key_with_legacy_fallback -- --nocapture`
- `cargo test -p ferrum-cli source_resolver -- --nocapture`

## Recent Useful Work Already On Branch

- W3 CUDA lane runner exists and composes build, model prefetch, product
  correctness, L2/L4/L5, fixed-output `bench-serve`, vLLM baseline handling,
  and final manifest generation.
- The lane now rejects invalid historical baselines and requires live vLLM
  baseline preflight to prove importable vLLM and CUDA-visible Torch.
- Qwen35 source hot-path work already on branch includes packed GDN prefill,
  continuation/chunked varlen batch prefill, mixed decode/prefill policy
  handling, and model-side argmax policy plumbing.

## If Work Resumes

- Do not quote the old `142 tok/s` release-shape number as current
  performance.
- First obtain a reachable 1x RTX 4090 and state the paid-GPU lane contract.
- Run the current W3 CUDA lane or a focused same-pod diagnostic that validates:
  product correctness, Qwen35 packed/continuation paths in logs, and c32
  throughput against the vLLM baseline.
- If c32 is still far below target, use profiler evidence around mixed
  prefill+decode and Qwen35 GDN decode/prefill kernels before making further
  code changes.
