# CUDA Build Cache Status - 2026-05-30

Milestone A is not complete. This checkpoint replaces the core PTX build path
with per-kernel content-hash stamps and adds machine-readable CUDA build
summary lines.

## Added

- `crates/ferrum-kernels/build.rs` now builds each core PTX kernel as an
  independently stamped artifact:
  - artifact: `<OUT_DIR>/<kernel>.ptx`
  - stamp: `<OUT_DIR>/<kernel>.ptx.stamp`
  - inputs hash includes the kernel source, `kernels/common.cuh`, CUDA target,
    and nvcc flags.
- A no-content `touch` on a core `.cu` no longer forces nvcc compilation when
  the content hash is unchanged.
- The generated `ptx.rs` is now written directly by the build script with the
  same `include_str!(env!("OUT_DIR")...)` constants used by the runtime.
- Every core PTX artifact and cached CUDA static library emits a summary line:

```text
[cuda-build-summary] artifact=<name> status=<built|cache_hit> reason=<reason> elapsed_ms=<ms> inputs_hash=<fnv1a64:...>
```

- Existing content-hash stamps for Marlin, vLLM-Marlin, vLLM-MoE-Marlin,
  vLLM paged-attn, and FA2 source now also emit the same summary format on
  both cache-hit and build paths.
- 2026-06-01 restoration note: the `fa2-source` static library now compiles
  only `crates/ferrum-kernels/kernels/fa2_source/ferrum_fa2_paged_varlen.cu`.
  It no longer reads `FERRUM_FA2_SRC_DIR`, `FERRUM_CUTLASS_INCLUDE_DIR`, or
  `CUTLASS_INCLUDE_DIR`, and no longer depends on an external FlashAttention
  checkout or CUTLASS header tree for the product build path.
- `scripts/check_fa2_source_native.py` guards that source boundary so future
  product `fa2-source` changes cannot silently reintroduce external
  FlashAttention/CUTLASS build dependencies.
- `scripts/validate_cuda_build_summary.py` validates the machine-readable
  summary log format, required artifacts, and required
  `status=cache_hit`/`status=built` expectations. CI runs its self-test so
  future changes to the summary line format break a lightweight local gate
  before GPU timing artifacts are produced.
- `scripts/m3_cuda_build_boundary_probe.py` runs the Milestone A release
  touch/content-change probe for N iterations, validates the same structured
  CUDA summary rows for every run, computes nearest-rank p50/p95 timing, and
  writes `build_boundary_manifest.json`.
- `scripts/validate_cuda_build_boundary_manifest.py` validates the timing
  manifest itself: schema version, mutation type, run count, per-run exit
  status, embedded CUDA summary validation, p50/p95 values, and whether
  `limits_pass` matches the configured timing limits. This gives the eventual
  GPU timing artifact a machine-readable acceptance gate instead of relying on
  the console line alone.
- CI now runs the lightweight self-tests for both CUDA build validation tools:
  `scripts/validate_cuda_build_summary.py --self-test`,
  `scripts/validate_cuda_build_boundary_manifest.py --self-test`, and
  `scripts/m3_cuda_build_boundary_probe.py --self-test`. This keeps the
  parser, timing manifest validator, and release-boundary probe
  manifest/summary validation logic checked before spending GPU time on the
  full timing artifact.
- `ferrum-kernels/build.rs` no longer registers core CUDA `.cu` inputs before
  the `cuda` feature check. CUDA builds still watch the same core PTX inputs in
  `compile_core_ptx`, but non-CUDA builds are no longer dirtied by unrelated
  core CUDA kernel touches.

## Validation

Local:

```bash
cargo fmt --all -- --check
cargo check -q -p ferrum-kernels
python3 -m py_compile scripts/validate_cuda_build_summary.py
python3 scripts/validate_cuda_build_summary.py --self-test
python3 -m py_compile scripts/validate_cuda_build_boundary_manifest.py
python3 scripts/validate_cuda_build_boundary_manifest.py --self-test
python3 -m py_compile scripts/m3_cuda_build_boundary_probe.py
python3 scripts/m3_cuda_build_boundary_probe.py --self-test
python3 -m py_compile scripts/check_fa2_source_native.py
python3 scripts/check_fa2_source_native.py --self-test
python3 scripts/check_fa2_source_native.py
git diff --check
```

Remote GPU pod after rsync to `/workspace/ferrum-codex-clean`:

```bash
cd /workspace/ferrum-codex-clean
cargo check -q -p ferrum-kernels \
  --features cuda,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source
```

Remote no-content touch validation:

```bash
cd /workspace/ferrum-codex-clean
touch crates/ferrum-kernels/kernels/paged_varlen_attention_vllm.cu
cargo check -p ferrum-kernels \
  --features cuda,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source -vv \
  > /workspace/m3-core-ptx-cache-summary-20260530.log 2>&1
```

Evidence:

- `scripts/validate_cuda_build_summary.py --self-test` passed, covering
  valid summary rows, required `cache_hit`/`built` enforcement, malformed row
  rejection, and file loading.
- `scripts/validate_cuda_build_boundary_manifest.py --self-test` passed,
  covering manifest schema validation, p50/p95 `limits_pass` consistency, and
  rejection of per-run CUDA summary validation failures.
- `scripts/m3_cuda_build_boundary_probe.py --self-test` passed with a fake
  cargo binary, covering manifest writing, per-run CUDA summary validation,
  manifest validation, required cache-hit artifacts, and timing limit checks.
- `/workspace/m3-core-ptx-cache-summary-20260530.log` contained `39`
  `[cuda-build-summary]` rows: `34` core PTX artifacts plus `5` static
  libraries.
- All `39` rows were `status=cache_hit`.
- The touched `core-ptx:kernels/paged_varlen_attention_vllm.cu` row was still
  `status=cache_hit reason=signature-match`, proving the PTX cache is
  content-hash based rather than mtime-only.
- `marlin`, `vllm_marlin`, `vllm_moe_marlin`, `vllm_paged_attn`, and
  `fa2_source` also reported `cache_hit`.

Remote content-change validation:

```bash
cd /workspace/ferrum-codex-clean
K=crates/ferrum-kernels/kernels/paged_varlen_attention_vllm.cu
cp "$K" /tmp/paged_varlen_attention_vllm.cu.codex.bak
printf "\n// codex content-hash build validation\n" >> "$K"
cargo check -p ferrum-kernels \
  --features cuda,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source -vv \
  > /workspace/m3-core-ptx-content-change-20260530.log 2>&1
mv /tmp/paged_varlen_attention_vllm.cu.codex.bak "$K"
```

Evidence:

- `/workspace/m3-core-ptx-content-change-20260530.log` contained exactly one
  built artifact:
  `core-ptx:kernels/paged_varlen_attention_vllm.cu status=built
  reason=signature-changed elapsed_ms=524`.
- The unrelated CUDA static libraries stayed on cache-hit:
  `marlin`, `vllm_marlin`, `vllm_moe_marlin`, `vllm_paged_attn`, and
  `fa2_source`.
- A follow-up CUDA `cargo check -q` after restoring the file succeeded,
  refreshing the PTX stamp back to the repository content.

Remote release build:

```bash
cd /workspace/ferrum-codex-clean
cargo build --release -p ferrum-cli \
  --features cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source \
  > /workspace/m3-core-ptx-release-build-20260530.log 2>&1
```

Evidence:

- The release build completed successfully in `693s`.
- The long duration was dominated by release Rust compilation/final LTO link;
  the CUDA artifact summary evidence comes from the `-vv` check logs above.

Remote release touch probe:

```bash
cd /workspace/ferrum-codex-clean
python3 scripts/m3_cuda_build_boundary_probe.py \
  --out /workspace/m3-release-touch-probe-20260530 \
  --iterations 5 \
  --fail-on-limit
```

Evidence:

- Legacy one-off `/workspace/m3-release-touch-probe-20260530.log` completed
  in `3m17s`.
- The ordinary cargo log showed the release chain recompiling
  `ferrum-kernels`, `ferrum-quantization`, `ferrum-models`,
  `ferrum-engine`, `ferrum-server`, and `ferrum-cli`.
- This does not meet the Milestone A `<= 90s` attention-only release rebuild
  target even though CUDA artifact rebuilds were avoided.

Restored-pod 5-run cache-hit release touch probe:

```bash
cd /workspace/ferrum-fa2-native-restore-git-ac3dfab
python3 scripts/m3_cuda_build_boundary_probe.py \
  --iterations 5 \
  --out /workspace/m3-release-touch-probe-cachehit-20260601-20260601_043825 \
  --fail-on-limit
```

Evidence:

- Artifact summary:
  `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-release-touch-probe-cachehit-20260601.md`.
- Remote manifest:
  `/workspace/m3-release-touch-probe-cachehit-20260601-20260601_043825/build_boundary_manifest.json`.
- Git head: `d93790393c04232e265abecfb32fdd81b3547e01`.
- All 5 runs exited `0`; per-run elapsed seconds were `231.517`, `229.133`,
  `230.500`, `234.102`, and `234.608`.
- The manifest timing gate failed: `p50_sec_nearest_rank=231.517`,
  `p95_sec_nearest_rank=234.608`, limits `75.0/90.0`,
  `limits_pass=false`.
- Every run had `status_counts {"cache_hit": 39}` and no summary validation
  error. This proves the current miss is not broad CUDA/nvcc rebuild; it is
  the remaining Rust/Cargo release dirtying and release link tail.

Remote package-scoped release dirty probe:

```bash
cd /workspace/ferrum-codex-clean
touch crates/ferrum-kernels/kernels/paged_varlen_attention_vllm.cu
cargo build --release -p ferrum-kernels \
  --features cuda,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source -vv \
  > /workspace/m3-release-kernels-touch-vv-20260530.log 2>&1
```

Evidence:

- `/workspace/m3-release-kernels-touch-vv-20260530.log` completed in `55s`.
- Cargo marked `ferrum-kernels` dirty because
  `crates/ferrum-kernels/kernels/paged_varlen_attention_vllm.cu` changed by
  mtime:
  `the file ... has changed`.
- The same log still showed the CUDA layer doing the right thing:
  `core-ptx:kernels/paged_varlen_attention_vllm.cu status=cache_hit
  reason=signature-match`, with all core PTX artifacts and cached static
  libraries on `cache_hit`.
- This isolates the remaining rebuild cost to Cargo/Rust release dirtying and
  downstream release/LTO work, not to nvcc work for the touched attention PTX.

Remote vLLM-MoE-Marlin source-change validation:

```bash
cd /workspace/ferrum-codex-clean
OUT_DIR_PATH=$(ls -td target/debug/build/ferrum-kernels-*/out | head -1)
K=crates/ferrum-kernels/kernels/vllm_marlin_moe/ops.cu
cp "$K" /tmp/ferrum_ops_cu_codex_bak2
cp "$OUT_DIR_PATH/libvllm_moe_marlin.a" /tmp/libvllm_moe_marlin_codex_bak2.a
cp "$OUT_DIR_PATH/libvllm_moe_marlin.stamp" /tmp/libvllm_moe_marlin_codex_bak2.stamp
printf "\n// codex vllm-moe content-hash build validation rerun2\n" >> "$K"
cargo check -p ferrum-kernels \
  --features cuda,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source -vv \
  > /workspace/m3-vllm-moe-source-change-rerun2-20260530.log 2>&1
mv /tmp/ferrum_ops_cu_codex_bak2 "$K"
cp /tmp/libvllm_moe_marlin_codex_bak2.a "$OUT_DIR_PATH/libvllm_moe_marlin.a"
cp /tmp/libvllm_moe_marlin_codex_bak2.stamp "$OUT_DIR_PATH/libvllm_moe_marlin.stamp"
```

Evidence:

- `/workspace/m3-vllm-moe-source-change-rerun2-20260530.log` contained `39`
  `[cuda-build-summary]` rows: `38` cache hits and exactly one built artifact.
- The only built artifact was
  `vllm_moe_marlin status=built reason=signature-changed elapsed_ms=474451`.
- The core PTX artifact for the attention kernel stayed cache-hit:
  `core-ptx:kernels/paged_varlen_attention_vllm.cu status=cache_hit`.
- Unrelated static libraries stayed cache-hit:
  `marlin`, `vllm_marlin`, `vllm_paged_attn`, and `fa2_source`.
- The source file, static archive, and stamp were restored after the temporary
  validation edit. A follow-up CUDA `cargo check -q` succeeded and the process
  scan found no live `target/release/ferrum`, `bench-serve`, `cargo`, `nvcc`,
  or `vllm` process.

## Remaining A Gaps

- The 5-run restored-pod release rebuild timing gate has been run and fails:
  `p50=231.517s`, `p95=234.608s`, required `<=75s/<=90s`.
- The earlier single release touch probe also failed the `<= 90s` target: the
  best measured full `ferrum-cli` rebuild after an attention-kernel touch is
  `3m17s`.
- The release build still spends substantial time in Rust release compilation
  and final link. The current checkpoint removes unnecessary CUDA rebuilds
  from narrow core-PTX edits, but it does not stop Cargo from dirtying
  `ferrum-kernels` on `.cu` mtime changes.
- A `CARGO_INCREMENTAL=1` release probe was stopped and discarded because it
  created a different release build-script output directory and began a cold
  vLLM-Marlin nvcc rebuild, so it is not a viable low-risk fix for this gate.
