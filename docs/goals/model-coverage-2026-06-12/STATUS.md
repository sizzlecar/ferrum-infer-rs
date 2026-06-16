# STATUS — model-coverage-2026-06-12

进度日志,倒序。

## 2026-06-16 XV — W2 CUDA checkpoint: Marlin evict-first product run/serve correctness passes

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_cache_policy_product_correctness_2026-06-16/`.
- Paid GPU lane:
  `W2 Marlin evict-first product correctness smoke` on the cached 1x RTX 4090
  Vast instance.
- Contract:
  - expected runtime/cost: 30-60 minutes, about USD 0.21-0.43 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/clean checkout/build/`ferrum run`/
    `ferrum serve` first failure, or run+serve correctness evidence collected;
  - correctness gate: product-default CUDA binary, `ferrum run` and
    `ferrum serve` both return expected `5`, usage present for serve, and log
    scan has no panic/error/NaN/`<unk>`/`[PAD]`/invalid UTF patterns;
  - performance command: none for this lane; correctness-only after the Marlin
    default-path source change.
- Evidence:
  - remote HEAD `212b2bf925c998062ef22767a1da41ba47ed5101`;
  - clean worktree: `remote/git_status_short.txt` has `0` lines;
  - CUDA release build rc `0`;
  - binary SHA256
    `d38caf704f252045c29bdfe02795606937f400ab00edef05647da74179b215d5`;
  - `ferrum run` rc `0`, JSONL assistant content `"5"`,
    `finish_reason=stop`, `n_tokens=3`;
  - `ferrum serve` chat rc `0`, response content `"5"`,
    `finish_reason=length`, usage `prompt_tokens=23`, `completion_tokens=1`,
    `total_tokens=24`;
  - `server/error_scan.txt` has `0` lines;
  - `correctness_check.json` reports `ok=true`;
  - Vast cleanup confirmed `stopped/exited`.
- Note:
  - the first background attempt failed before build because the script did not
    include `/root/.cargo/bin` in `PATH`; preserved under
    `build_initial_env_failure/`;
  - the retry used the same instance and clean worktree and passed.
- Interpretation:
  - the Marlin B-weight `L2::evict_first` default path has now cleared the
    required product-entrypoint correctness smoke for both `ferrum run` and
    `ferrum serve`;
  - this unlocks endpoint performance diagnostics for this source change, but
    it is not itself performance or release-grade evidence.
- Next:
  - run a focused same-dataset Ferrum diagnostic against the existing clean
    vLLM ShareGPT baseline before deciding whether the 1-2% MLP gain moves the
    endpoint ratio materially;
  - continue searching for a higher-return dense MLP `gate_up` /
    work-reduction lever.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XU — W2 native CUDA checkpoint: product-default Marlin evict-first validated

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_cache_policy_default_native_probe_2026-06-16/`.
- Paid GPU lane:
  `W2 Marlin cache-policy product-default native probe` on the cached 1x RTX
  4090 Vast instance.
- Contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/compile first failure, probe non-zero or
    timeout, or VERDICT plus artifact copyback;
  - correctness gate: native probe exit 0 and
    `VERDICT: gemma3 Marlin cache-policy native CUDA probe complete`;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_marlin_cache_policy_perf.sh`.
- Evidence:
  - remote HEAD `c76bfcfa2b00a73a816e6d44bbd999a621b12a49`;
  - probe rc `0`;
  - legacy plain binary SHA256
    `b0ee9ba92b2a3ab74c382273ea2fc82763277671b436581b5fc47e0d9b896e00`;
  - product default binary SHA256
    `82edfb8e6561f87eef067d3ea7fe5327b54f3cc9450d6c42cf63fe72963aec66`;
  - stdout contains
    `VERDICT: gemma3 Marlin cache-policy native CUDA probe complete`;
  - Vast cleanup confirmed `stopped/exited`.
- Key rows:
  - m16 product chain event: legacy plain `215.344us`, product default
    `211.791us` (`-1.6%`);
  - m16 product down: legacy plain `70.496us`, product default `68.852us`;
  - m32 product chain event: legacy plain `227.980us`, product default
    `225.103us` (`-1.3%`);
  - m32 product down: legacy plain `75.653us`, product default `75.414us`.
- Interpretation:
  - after `c76bfcfa`, the product-default Marlin path matches the previously
    validated evict-first variant;
  - this is a real default-path kernel improvement, but only a 1-2% MLP segment
    lever, so it does not close the W2 release-grade performance gap alone.
- Next:
  - validate `ferrum run` and `ferrum serve` correctness on a CUDA product
    binary before any endpoint performance claim;
  - continue searching for a larger dense MLP `gate_up` / work-reduction lever.
- Scope:
  - this is diagnostic native CUDA evidence, not release performance evidence;
  - the remote worktree had old tracked artifact-log modifications after
    syncing `.git`; those are recorded in `remote/git_status_short.txt` and are
    not used for release performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XT — W2 source checkpoint: productize Marlin B-weight evict-first cache policy

- Changed `crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu` so Marlin
  B-weight `cp.async` uses `L2::evict_first` by default for CUDA `sm_80` through
  pre-Blackwell architectures. Blackwell `sm_120` keeps the plain
  `cp.async.cg` fallback because the fractional L2 cache-policy syntax is not
  accepted there.
- Updated `scripts/microbenches/build_and_run_gemma3_marlin_cache_policy_perf.sh`
  to compare legacy plain `cp.async.cg` (`FERRUM_MARLIN_CP_ASYNC_PLAIN=1`)
  against the product-default path.
- Rationale:
  - XS native CUDA evidence showed this cache policy is a small positive
    product-shaped tail-MLP lever rather than a cost-shifting warmup trick;
  - the behavior is now a default CUDA build path, not a hidden user env
    combination.
- Local validation:
  - `bash -n scripts/microbenches/build_and_run_gemma3_marlin_cache_policy_perf.sh`;
  - `git diff --check -- crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu scripts/microbenches/build_and_run_gemma3_marlin_cache_policy_perf.sh scripts/microbenches/README.md`.
- Required next validation:
  - run the native cache-policy probe on 1x RTX 4090 to confirm the product
    default matches the previously measured evict-first path;
  - before endpoint performance claims, validate `ferrum run` and
    `ferrum serve` correctness on the product binary.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XS — W2 native CUDA checkpoint: Marlin B-weight evict-first is a small positive lever

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_cache_policy_native_probe_2026-06-16/`.
- Paid GPU lane:
  `W2 Gemma3 Marlin cache-policy native probe` on the cached 1x RTX 4090
  Vast instance.
- Contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/compile first failure, probe non-zero or
    timeout, or VERDICT plus artifact copyback;
  - correctness gate: native probe exit 0 and
    `VERDICT: gemma3 Marlin cache-policy native CUDA probe complete`;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_marlin_cache_policy_perf.sh`.
- Evidence:
  - remote HEAD `018ea7bce6494db5539ce32e22f104144fe87eba`;
  - probe rc `0`;
  - baseline binary SHA256
    `50e4ad67f5d79293da1d524eedcae2cde7edb71d7e6d85387e94b5b37cb0ca41`;
  - evict-first binary SHA256
    `69655f683cc80daf98737e290946ca69bbcec87d69c818deff5cf2038e8c8e41`;
  - stdout contains
    `VERDICT: gemma3 Marlin cache-policy native CUDA probe complete`;
  - Vast cleanup confirmed `stopped/exited`.
- Key rows:
  - m16 product chain event: baseline `215.580us`, evict-first `211.690us`
    (`-1.8%`);
  - m16 product down: baseline `70.549us`, evict-first `68.800us`;
  - m32 product chain event: baseline `227.722us`, evict-first `225.173us`
    (`-1.1%`);
  - m32 product down: baseline `75.659us`, evict-first `75.339us`.
- Interpretation:
  - `FERRUM_MARLIN_CP_ASYNC_EVICT_FIRST=1` compiles on CUDA 12.4 / Ada and
    produces a small positive tail-MLP segment gain;
  - unlike explicit down prefetch, this improves segment wall time rather than
    shifting cost into a warm kernel;
  - the gain is too small by itself to close the W2 gap, so this is a useful
    low-risk kernel lever but not the main missing performance breakthrough.
- Next:
  - either wire this as a typed/default CUDA build policy and validate
    `ferrum run` / `ferrum serve` correctness before endpoint performance, or
    keep searching for a higher-return dense MLP `gate_up` / work-reduction
    lever.
- Scope:
  - this is diagnostic native CUDA evidence, not release performance evidence;
  - the remote worktree had old tracked artifact-log modifications after
    syncing `.git`; those are recorded in `remote/git_status_short.txt` and are
    not used for release performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XR — W2 native CUDA checkpoint: overlap prefetch warms down but worsens wall time

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_down_prefetch_overlap_native_probe_2026-06-16/`.
- Paid GPU lane:
  `W2 Gemma3 down prefetch-overlap native probe` on the cached 1x RTX 4090
  Vast instance.
- Contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/compile first failure, probe non-zero or
    timeout, or VERDICT plus artifact copyback;
  - correctness gate: native probe exit 0 and
    `VERDICT: gemma3 down prefetch-overlap native CUDA probe complete`;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_down_prefetch_overlap_perf.sh`.
- Evidence:
  - remote HEAD `432e6588bac59902b7488484934494c751534221`;
  - probe rc `0`;
  - binary SHA256
    `58491a34483c8c4ba0ccbd4b1d9c9b127676b1f520a6ba42a0409daae5cc64bc`;
  - stdout contains
    `VERDICT: gemma3 down prefetch-overlap native CUDA probe complete`;
  - Vast cleanup confirmed `stopped/exited`.
- Key rows:
  - m16 no prefetch: down `69.744us`, segment `216.072us`;
  - m16 overlap qweight: down `34.063us`, segment `239.437us`;
  - m16 overlap qweight+scales: down `32.560us`, segment `238.979us`;
  - m32 no prefetch: down `74.849us`, segment `227.628us`;
  - m32 overlap qweight: down `53.836us`, segment `268.300us`;
  - m32 overlap qweight+scales: down `53.790us`, segment `269.916us`.
- Interpretation:
  - explicit warm/prefetch does make down fast under 8-layer rotation;
  - it increases end-to-end segment wall time, so the current warm kernel shifts
    cost rather than reducing wall time;
  - do not productize this cache-warm branch as W2 performance work unless a
    cheaper producer-integrated prefetch design is found.
- Next:
  - return to dense MLP work-reduction/kernel-design options rather than
    stream-level L2 policy or standalone warm kernels.
- Scope:
  - this is diagnostic native CUDA evidence, not release performance evidence;
  - the remote worktree had old tracked artifact-log modifications after
    syncing `.git`; those are recorded in `git_verify.txt` and are not used for
    release performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XQ — W2 source checkpoint: native down prefetch-overlap probe

- Added `scripts/microbenches/gemma3_down_prefetch_overlap_perf.cu` plus
  `scripts/microbenches/build_and_run_gemma3_down_prefetch_overlap_perf.sh`.
- Purpose:
  - XP showed simple L2 access-policy is not enough under 8-layer rotation;
  - explicit down-warm is an upper bound but adds an extra down read;
  - this probe launches a lightweight down qweight/scales read kernel on a
    second CUDA stream while the main stream runs gate_up+GeGLU, then measures
    both down kernel time and host-synchronized segment time.
- Expected GPU use:
  - run one native CUDA validation before touching product code;
  - if overlap prefetch reduces down time but increases segment time by a
    comparable amount, reject it as non-productizable;
  - if it reduces down time and keeps segment time flat or lower, evaluate a
    typed product prefetch policy and then validate `ferrum run`/`serve`
    correctness before endpoint performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XP — W2 native CUDA checkpoint: simple L2 policy fails under multi-layer weight rotation

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_down_l2_persist_cycle_native_probe_2026-06-16/`.
- Paid GPU lane:
  `W2 Gemma3 down L2 persistence cycle native probe` on the cached 1x RTX
  4090 Vast instance.
- Contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/compile first failure, probe non-zero or
    timeout, or VERDICT plus artifact copyback;
  - correctness gate: native probe exit 0 and
    `VERDICT: gemma3 down L2 persistence cycle native CUDA probe complete`;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_down_l2_persist_cycle_perf.sh`.
- Evidence:
  - remote HEAD `357a4b98a2eb80744b8beacf256b91bbff8ae0f2`;
  - probe rc `0`;
  - binary SHA256
    `f9c3e69f4407c4b4bd42b7f28593efcc7eb1c2bc81dff7c10ba98baf10b510f1`;
  - stdout contains
    `VERDICT: gemma3 down L2 persistence cycle native CUDA probe complete`;
  - Vast cleanup confirmed `stopped/exited`.
- Key rows:
  - m16 single-layer no-policy: `69.832us`;
  - m16 single-layer persist hit60: `34.493us`;
  - m16 8-layer cycle no-policy: `69.736us`;
  - m16 8-layer cycle persist hit60: `69.743us`;
  - m16 8-layer cycle persist plus explicit down-warm: `34.634us`;
  - m32 single-layer no-policy: `75.903us`;
  - m32 single-layer persist hit60: `58.984us`;
  - m32 8-layer cycle no-policy: `75.745us`;
  - m32 8-layer cycle persist hit60: `75.117us`;
  - m32 8-layer cycle persist plus explicit down-warm: `54.067us`.
- Interpretation:
  - XN's single-layer L2 persistence win is real but not sufficient for product
    decode because one layer's next down call is separated by many other layer
    weights;
  - simple per-layer stream access-policy does not improve 8-layer rotation;
  - explicit down-warm remains a useful upper bound but reads down weights an
    extra time, so it is not a free product fix;
  - do not productize simple access-policy alone as the W2 performance lever.
- Next:
  - if staying on this branch, test an overlap/prefetch strategy that can warm
    down qweight concurrently with gate_up work; otherwise return to another
    dense MLP reduction lever.
- Scope:
  - this is diagnostic native CUDA evidence, not release performance evidence;
  - the remote worktree had old tracked artifact-log modifications after
    syncing `.git`; those are recorded in `git_verify.txt` and are not used for
    release performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XO — W2 source checkpoint: native multi-layer L2 persistence cycle probe

- Added `scripts/microbenches/gemma3_down_l2_persist_cycle_perf.cu` plus
  `scripts/microbenches/build_and_run_gemma3_down_l2_persist_cycle_perf.sh`.
- Purpose:
  - XN proved stream access-policy can keep a single layer's down qweight hot
    across that same layer's `gate_up -> GeGLU` producer;
  - product decode revisits one layer only after many other layer weights run,
    so the single-layer loop may overstate productizable benefit;
  - this probe allocates 8 synthetic Gemma3 layer weight sets and compares
    single-layer no-policy/persist against 8-layer no-policy/persist and an
    explicit down-warm upper bound.
- Expected GPU use:
  - run one native CUDA validation before productizing L2 policy;
  - if 8-layer persist does not improve over no-policy, reject simple per-layer
    access-policy as insufficient for product performance;
  - if 8-layer persist still helps materially, implement a typed product
    policy and validate `ferrum run` / `ferrum serve` correctness before any
    endpoint performance claim.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XN — W2 native CUDA checkpoint: down qweight L2 persistence restores post-gate_up down speed

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_down_l2_persist_native_probe_2026-06-16/`.
- Paid GPU lane:
  `W2 Gemma3 down L2 persistence native probe` on the cached 1x RTX 4090 Vast
  instance.
- Contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/compile first failure, probe non-zero or
    timeout, or VERDICT plus artifact copyback;
  - correctness gate: native probe exit 0 and
    `VERDICT: gemma3 down L2 persistence native CUDA probe complete`;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_down_l2_persist_perf.sh`.
- Evidence:
  - remote HEAD `6cf26ca99f1958d2e326245bbe55fd8ed22c7e4a`;
  - probe rc `0`;
  - binary SHA256
    `c3fafa5657c5dbc1496f6a9790ffc4440cb4f17ddf01014f55df1212226826f3`;
  - stdout contains
    `VERDICT: gemma3 down L2 persistence native CUDA probe complete`;
  - Vast cleanup confirmed `stopped/exited`.
- Device/context:
  - RTX 4090 L2 cache: `75,497,472` bytes;
  - persisting L2 max: `51,904,512` bytes;
  - access window max: `134,213,632` bytes;
  - down qweight policy window: `57,802,752` bytes.
- Key rows:
  - m16 warm repeated baseline: `35.135us`;
  - m16 no-policy after gate_up+GeGLU: `70.342us`;
  - m16 down qweight full-window persist hit100: `35.088us`;
  - m16 down qweight full-window persist hit60: `33.158us`;
  - m32 warm repeated baseline: `55.127us`;
  - m32 no-policy after gate_up+GeGLU: `75.148us`;
  - m32 down qweight full-window persist hit100: `55.545us`;
  - m32 down qweight full-window persist hit60: `54.434us`.
- Interpretation:
  - simple CUDA stream access-policy on down qweight is a real W2 lever;
  - it restores down performance after the product-shaped `gate_up -> GeGLU`
    producer sequence instead of only improving isolated warm microbench rows;
  - expected product upside is bounded to the dense Marlin down component, so it
    will not by itself prove W2 release-grade, but it is the first currently
    measured lever with material tail-MLP savings.
- Next:
  - productize as a typed CUDA runtime/config policy, not a hidden env-only
    requirement;
  - validate `ferrum run` and `ferrum serve` correctness before performance;
  - only after correctness passes, run a focused c16/c32 diagnostic and then
    decide whether to promote to release evidence.
- Scope:
  - this is diagnostic native CUDA evidence, not release performance evidence;
  - the remote worktree had old tracked artifact-log modifications after
    syncing `.git`; those are recorded in `git_verify.txt` and are not used for
    release performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XM — W2 source checkpoint: native down L2 persistence probe

- Added `scripts/microbenches/gemma3_down_l2_persist_perf.cu` plus
  `scripts/microbenches/build_and_run_gemma3_down_l2_persist_perf.sh`.
- Purpose:
  - XL showed down is cold after product-shaped `gate_up -> GeGLU`, even when
    down reads a separate constant input;
  - this probe applies CUDA's stream access-policy window to down `qweight`
    and compares no-policy, full-window, half-window, lower hit-ratio, and
    explicit down-warm cases;
  - it is a native CUDA minimal verification of whether simple persisting L2
    hints are a productizable lever for the `gate_up -> down` sequence.
- Expected GPU use:
  - run one cached 1x4090 native probe, not a release sweep;
  - if no-policy and persisting modes match, reject stream-level L2 persistence
    as a W2 lever;
  - if persisting materially narrows the m16/m32 down gap, inspect whether the
    same policy can be represented as a typed CUDA runtime option without
    hidden env and then validate product correctness before performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XL — W2 native CUDA checkpoint: down slowdown is cache/producer-state, not GeGLU value sensitivity

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_down_input_source_native_probe_2026-06-16/`.
- Paid GPU lane:
  `W2 Gemma3 down input-source native probe` on the cached 1x RTX 4090 Vast
  instance.
- Contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/compile first failure, probe non-zero or
    timeout, or VERDICT plus artifact copyback;
  - correctness gate: native probe exit 0 and
    `VERDICT: gemma3 down input-source native CUDA probe complete`;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_down_input_source_perf.sh`.
- Evidence:
  - remote HEAD `cea63ef0b5c933a2a39802a82010b34eaa1a9d45`;
  - probe rc `0`;
  - binary SHA256
    `dd1a5ba3cd0f244603bc1fbebe8f2a6a224004f98943a5c31a21001e9aa7bfb0`;
  - stdout contains
    `VERDICT: gemma3 down input-source native CUDA probe complete`;
  - Vast cleanup confirmed `stopped/exited`.
- Key m16 rows:
  - constant input baseline: `32.606us`;
  - small constant input baseline: `32.670us`;
  - constant input after gate_up+GeGLU producer: `69.793us`;
  - GeGLU output immediate: `68.356us`;
  - GeGLU output after sync: `70.200us`;
  - copied GeGLU output after sync: `70.098us`;
  - constant input after L2 flush: `90.343us`.
- Interpretation:
  - the isolated Marlin down row is fast only when repeated on warm constant
    input;
  - running the product-shaped gate_up+GeGLU producer immediately before down
    makes down slow even when down reads a separate constant input;
  - small constant input is not slower, so this is not GeGLU numeric magnitude
    or subnormal value sensitivity;
  - the remaining W2 tail-MLP lever is cache/producer-state or weight residency
    around the `gate_up -> down` sequence, not the existing Triton W4A16 path
    or GeGLU data sensitivity.
- Scope:
  - this is diagnostic evidence, not release performance evidence;
  - the remote worktree had old tracked artifact-log modifications after
    syncing `.git`; those are recorded in `git_verify.txt` and are not used for
    release performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XK — W2 source checkpoint: native down input-source probe

- Added `scripts/microbenches/gemma3_down_input_source_perf.cu` plus
  `scripts/microbenches/build_and_run_gemma3_down_input_source_perf.sh`.
- Purpose:
  - XG measured the product-shaped tail MLP chain and found m16 `down_proj`
    around `68-71us` when it consumes GeGLU output;
  - XJ measured isolated Marlin down at the same Gemma3 shape around `30-33us`
    at m16 with synthetic constant input;
  - this probe keeps the Marlin down shape fixed and varies only input source /
    producer state: constant input, small constant input, constant after GeGLU,
    constant after L2 flush, immediate GeGLU output, synced GeGLU output, and
    device-copied GeGLU output.
- Expected GPU use:
  - run as a native CUDA minimal verification before any product change;
  - if GeGLU-derived input remains slow after sync/copy, inspect activation
    value range or down-kernel data sensitivity;
  - if constant input slows after preceding GeGLU/flush, inspect cache/producer
    state instead;
  - if the gap disappears, treat the previous difference as measurement setup
    and avoid this branch.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XJ — W2 native CUDA checkpoint: existing Triton W4A16 is slower than Marlin on Gemma3 MLP

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_triton_w4a16_native_probe_2026-06-16/`.
- Lane:
  `W2 Gemma3 dense Triton W4A16 vs Marlin native probe` on cached 1x RTX
  4090 Vast instance `40826362`.
- Paid GPU contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at prior
    USD 0.425/hr;
  - stop condition: startup/SSH/CUDA failure, nvcc compile failure, probe
    nonzero/timeout, or VERDICT line with artifacts copied back;
  - correctness gate: native probe exit 0 and VERDICT line;
  - performance command:
    `bash scripts/microbenches/build_and_run_dense_triton_w4a16_gemma3_perf.sh`.
- Lifecycle:
  - instance started from `stopped/exited`;
  - CUDA verified with 1x RTX 4090, driver 565.77, nvcc 12.4;
  - source synced to remote HEAD
    `2847822395e857cbe23196b9590b88479eadeb60`;
  - remote source status was clean after restoring tracked artifact files
    affected by the source rsync exclude;
  - artifact copied back;
  - instance stopped and final Vast poll confirmed `stopped/exited`.
- Probe result:
  - `probe.rc=0`;
  - stdout contains
    `VERDICT: dense Triton W4A16 Gemma3 native CUDA probe complete`;
  - binary SHA256:
    `83a8112f31951e930b90736fcc7a7a99db69936fdebfa1f92b17449159a6e77c`.
- Key timings:
  - m16 `gate_up`: Marlin product workspace-zero `137.111us`,
    Triton W4A16 `618.924us`, so Triton is `4.51x` slower;
  - m16 `down`: Marlin product workspace-zero `32.527us`, Triton W4A16
    `609.813us`, so Triton is `18.75x` slower;
  - m32 `gate_up`: Marlin `141.253us`, Triton `781.304us`;
  - m32 `down`: Marlin `54.504us`, Triton `749.147us`.
- Interpretation:
  - existing `w4a16_gptq_f16.ptx` is not a W2 dense MLP performance lever;
  - do not productize `FERRUM_TRITON_INT4=1` for W2 release-grade work;
  - any Triton direction would require a new kernel/tile design, not the
    currently committed dense W4A16 PTX.
- Next:
  - continue with levers that can reduce dense GPTQ MLP work or improve the
    Marlin path itself; the direct alternative backend path is now rejected.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XH — W2 source checkpoint: native Triton W4A16 vs Marlin dense MLP probe

- No new GPU run in this checkpoint; this adds the next minimal native CUDA
  probe for the dense MLP compute lever.
- Added `scripts/microbenches/dense_triton_w4a16_gemma3_perf.cu` plus
  `scripts/microbenches/build_and_run_dense_triton_w4a16_gemma3_perf.sh`.
- Probe scope:
  - loads the committed `w4a16_gptq_f16.ptx` through the CUDA Driver API,
    outside Cargo and outside product model loading;
  - compares the existing Marlin path against Triton W4A16 at Gemma3 W2
    `gate_up` (`k=5376,n=43008`) and `down` (`k=21504,n=5376`) shapes;
  - reports `m={1,10,16,23,32}` rows for product Marlin workspace-zero,
    Marlin kernel-only diagnostic, and Triton W4A16.
- Why this is aligned:
  - XG showed the measured product `tail_mlp` cost is explained by dense MLP
    compute across 62 layers, not by a hidden launch-chain overhead;
  - the next useful question is whether an existing alternative dense W4A16
    backend can materially beat Marlin on the exact Gemma3 MLP shapes before
    spending product-code effort on typed runtime wiring.
- Next:
  - run this probe once on the cached 1x4090 lane, capture stdout, binary
    SHA256, and cleanup evidence; if Triton is not faster enough or fails, do
    not wire it into product defaults.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XG — W2 native CUDA checkpoint: tail-MLP chain PASS, bottleneck is dense MLP compute across layers

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_tail_mlp_chain_native_probe_2026-06-16/`.
- Lane:
  `W2 Gemma3 native tail-MLP chain probe` on the cached 1x RTX 4090 Vast
  instance `40826362`.
- Paid GPU contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at prior
    USD 0.425/hr;
  - stop condition: nvcc compile failure, probe nonzero/timeout, or VERDICT
    line with artifacts copied back;
  - correctness gate: native probe exit 0 and VERDICT line;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_tail_mlp_chain_perf.sh`.
- Lifecycle:
  - instance started from `stopped/exited`;
  - CUDA verified with 1x RTX 4090, driver 565.77, nvcc 12.4;
  - source synced to remote HEAD
    `2c281e56557c11486cbdec5da9dae1234dcae78d`;
  - remote source status was clean after restoring tracked artifact files
    affected by the source rsync exclude;
  - artifact copied back;
  - instance stopped and final Vast poll confirmed `stopped/exited`.
- Probe result:
  - `probe.rc=0`;
  - stdout contains
    `VERDICT: gemma3 tail MLP chain native CUDA probe complete`;
  - binary SHA256:
    `7dd82cd65a02958533c65b45d018e0b49600b1a30d394c7fa567a41f0d4ccca7`.
- Key timings:
  - m16 `product_ws_zero`:
    `chain_event_us=215.750`, `chain_host_sync_us=217.782`;
  - m16 product phase split:
    `pre_norm=5.914us`, `gate_up=139.671us`, `geglu=4.680us`,
    `down=70.903us`, `final_norm=6.352us`;
  - m16 `kernel_only_ws_prezero_diagnostic`:
    `chain_event_us=212.986`.
- Interpretation:
  - the single-layer Gemma3 tail-MLP chain is about `216us`; multiplied by
    62 layers this is about `13.4ms`, matching the earlier product profile
    band where `tail_mlp` was about `13.6-14.9ms` per decode step;
  - this rejects the hypothesis that W2 c16 is blocked by a hidden multi-ms
    launch-chain overhead outside the measured kernels;
  - the remaining W2 performance gap is dense GPTQ MLP compute across layers,
    dominated by `gate_up` and `down`, not HTTP/scheduler/postprocess,
    legacy graph routing, Marlin block-policy, or direct dense Marlin kernel
    swap.
- Next:
  - choose an optimization that changes the dense MLP compute path itself
    rather than doing another scheduling/env sweep: viable candidates are a
    different W4A16 backend for dense Gemma3 MLP, layer/prompt-level MLP
    work reduction if correctness-safe, or a product-profile check that
    compares exact per-layer call counts with the native chain.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XF — W2 source checkpoint: native CUDA Gemma3 tail-MLP chain probe

- No new GPU run in this checkpoint; this adds a minimal native CUDA probe for
  the next paid diagnostic lane.
- Added `scripts/microbenches/gemma3_tail_mlp_chain_perf.cu` plus
  `scripts/microbenches/build_and_run_gemma3_tail_mlp_chain_perf.sh`.
- Probe scope:
  - models the product Gemma3 tail MLP sequence
    `rms_norm_f32_to_f16 -> Marlin gate_up -> GeGLU -> Marlin down ->
    rms_norm_f16_add_to_f32`;
  - uses product-shaped W2 dimensions (`h=5376`, `intermediate=21504`,
    `gate_up n=43008`, `down n=5376`) and calls the existing product CUDA
    kernels directly;
  - emits phase timing plus full-chain event and host-sync timing for
    `m={1,10,16,23,32}`;
  - keeps `product_ws_zero` as the primary row and labels the workspace-prezero
    kernel-only row as diagnostic only, so it cannot be mistaken for product
    evidence or for an unsafe skip-workspace-zero runtime mode.
- Why this checkpoint matters:
  - earlier evidence already ruled out legacy `--batched-graph`, Marlin block
    policy override, direct vLLM dense Marlin kernel swap, and scheduler/HTTP
    as first-order W2 c16 levers;
  - current c16 profiling points to model-side Gemma3 tail MLP / dense Marlin,
    so this probe is the next smallest native CUDA validation before any full
    product benchmark.
- Next:
  - run this build script once on the cached 1x4090 lane, capture stdout and
    binary SHA256 under a new artifact dir, then choose a concrete optimization
    from the phase split.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XE — W2 source guard: reject Gemma3 typed unified graph before CUDA runtime

- 本轮无新增 GPU run;这是对 `w2_unified_graph_typed_c16_2026-06-16`
  correctness failure 的 source guard checkpoint。
- Source change:
  - `FerrumConfigBuilder::validate_unified_graph` now rejects
    `FERRUM_UNIFIED_GRAPH=1` when model capabilities report
    `architecture=gemma3`;
  - failure message:`unified decode graph is disabled for Gemma3 sandwich-norm
    models`;
  - `docs/runtime-env-registry.tsv` notes that `FERRUM_UNIFIED_GRAPH` is typed
    but rejected for Gemma3 until graph replay is correctness-safe。
- Rationale:
  - typed unified graph passed one-shot `ferrum run`/`ferrum serve`,but c16
    bench hit `CUDA_ERROR_ILLEGAL_ADDRESS`;
  - since the flag is now product-visible,it must fail early rather than
    allowing users or release scripts to reach a CUDA illegal-address path。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-types unified_graph -- --nocapture` PASS;
  - `cargo test -p ferrum-types batched_graph_override_materializes_decode_graph_policy -- --nocapture` PASS;
  - `cargo test -p ferrum-cli run_effective_runtime_config_records_batched_graph_flag -- --nocapture` PASS。
- Release-grade status:
  - this closes a correctness hazard only;it does not improve W2 performance;
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 XD — W2 typed unified graph smoke passes but c16 bench hits CUDA illegal address

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_unified_graph_typed_c16_2026-06-16/`。
- Source/binary:
  - remote source head:`7f15a3ef9a57e2c23d889975ab629d25e8638803`;
  - source status clean for `crates/`,`scripts/`,`Cargo.toml`,`Cargo.lock`,
    and `ferrum.toml`;
  - release binary SHA256:
    `05f18a4cd8d8f34530758584122afad9e12f0bb929b450fc283449bb7d3180bd`。
- GPU 执行合同:
  - lane:`W2 Gemma3-27B CUDA typed unified graph c16 diagnostic`;
  - reused Vast/cache-retained instance `40826362`,1x RTX 4090,about
    USD `0.425/hr`;
  - stop condition:start/SSH/CUDA/source sync/build,`ferrum run
    --unified-graph`,`ferrum serve --unified-graph`,or c16 bench first
    failure;otherwise copy artifacts and stop the instance;
  - correctness gate:`ferrum run --unified-graph` known-answer smoke plus
    `ferrum serve --unified-graph` chat smoke with usage;
  - performance command:c16-only `bench-serve --fail-on-error --require-ci
    --seed 9271 --n-repeats 3`,diagnostic only。
- Runtime config evidence:
  - `serve_decision_trace.jsonl` selected `decode_graph_policy =
    unified_decode_graph` with `source=cli` and
    `source_key=FERRUM_UNIFIED_GRAPH`;
  - `serve_effective_config.json` selected
    `selected_graph_mode=unified_decode_graph`;
  - this run uses the typed CLI/config path,not a hidden env-only toggle。
- Correctness smoke:
  - `ferrum run --unified-graph` PASS:
    `RUN_SMOKE_PASS content='5' tokens=3`;
  - `ferrum serve --unified-graph` PASS:
    `SERVE_SMOKE_PASS content='5' completion_tokens=3`。
- c16 bench result:
  - repeat 1/3: `16 completed / 0 errored / 3.1s`;
  - repeat 2/3: `16 completed / 0 errored / 3.1s`;
  - repeat 3/3 started,then server hit CUDA illegal address;
  - no throughput result is valid,bench did not complete。
- Failure:
  - server log:
    `[unified-graph] replay err: Unsupported operation: post-launch sync:
    CUDA_ERROR_ILLEGAL_ADDRESS`;
  - follow-on panic:
    `CudaBackend: load_function(rms_norm_f32_to_f16):
    DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was
    encountered")`;
  - run was stopped per first-fail rule;GPU process list was clear afterward。
- Vast cleanup:
  - final poll verified `cur_state=stopped actual_status=exited`;
  - Vast JSON artifacts have `jupyter_token` redacted。
- Interpretation:
  - typed unified graph is now proven product-visible,but not product-safe for
    W2 performance work;
  - it passes one-shot run/serve smoke but fails under c16 bench,so it is a
    correctness blocker and cannot be used as performance evidence;
  - this reinforces the current path:avoid broad graph toggles,keep unified
    graph disabled for release evidence, and pursue the model-side
    Gemma3 MLP/Marlin bottleneck or a native CUDA graph-capture minimal repro
    before re-enabling this path.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 XC — W2 bottleneck narrowed after graph A/B: model-side Gemma3 MLP/Marlin dominates

- 本轮未新增 GPU run,没有新 release-grade artifact;这是基于最新
  `--batched-graph` A/B 与既有 profiler artifact 的 source/evidence
  checkpoint。
- 当前 head:`c5ff183f`。
- Evidence used:
  - `w2_paged_unified_default_path_cuda_smoke_2026-06-16`:
    default paged-unified product path `ferrum run` / `ferrum serve` correctness
    passes, c16 diagnostic `295.806 ± 5.211 tok/s`, health shows
    `decode_batch.calls=0`, `executor_model_lock.samples=4097`,
    `model_execution_time_ms=46.761`;
  - `w2_batched_graph_ab_cuda_diag_2026-06-16`: `--batched-graph`
    correctness passes, c16 diagnostic `287.117 ± 41.633 tok/s`,
    so `--batched-graph/default=0.9706` and graph toggle is not the current
    high-return lever;
  - `w2_typed_decode_profile_2026-06-16`: full `decode=16` iterations had
    mean total `23679.2us`, model time `23311.3us`, decode postprocess
    `347.9us`; model share `98.44%`, postprocess share `1.47%`;
  - `w2_profiler_graph_disabled_retry_2026-06-16` and
    `w2_marlin_typed_profile_2026-06-16`: c16 model-side profile repeatedly
    shows Gemma3 decode dominated by tail MLP / dense Marlin projections:
    `tail_mlp` around `13.6-14.9ms` per step, `matmul` around `7-8ms`,
    attention around `2.7ms`, QKV/RoPE around `0.7ms`; Marlin kernel aggregate
    around `16.5ms`, with `gate_up` around `8.7-9.5ms` and `down` around
    `4.3-5.3ms`.
- Updated bottleneck statement:
  - current c16 requests are reaching the paged unified model path; legacy
    `decode_batch` graph replay metrics stay at zero because that path is not
    serving the steady-state decode;
  - scheduler/HTTP/postprocess is not the primary c16 gap in the profiled
    steady-state path;
  - the highest-confidence bottleneck is Gemma3 model-side decode, especially
    tail MLP dense GPTQ/Marlin `gate_up` and `down` work;
  - dense Marlin grid/block-policy override and legacy batched graph toggle have
    already been falsified as main levers.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。
- Required next:
  - avoid more broad graph/scheduler sweeps;
  - use native CUDA or a very small product profiler cell to test one concrete
    Gemma3 MLP/Marlin lever at a time;
  - after any source change, validate correctness first with product
    `ferrum run` and `ferrum serve`, then a minimal c16 diagnostic before any
    broader performance run。

## 2026-06-16 LXXXIX — W2 CUDA A/B: `--batched-graph` correct but not a c16 performance lever

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_batched_graph_ab_cuda_diag_2026-06-16/`。
- Source/binary:
  - local head at launch:`0adb292a`;
  - remote reused clean source `d6d872c1e12fc364886117b0431aec752b2d78ac`;
  - reused binary SHA256
    `11b26df2b8dccf3138b2fe294e80ef618cc6255e56af626213e6aaabe8b2e48f`;
  - no rebuild/reinstall performed,只复用上一轮环境和模型缓存。
- GPU 执行合同:
  - lane:`W2 batched-graph default-path A/B diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - stop condition:启动/SSH/binary check/run/serve 首败即停,或 run+serve
    correctness + c16 diagnostic 后停止实例;
  - correctness gate:`ferrum run --batched-graph` 与
    `ferrum serve --batched-graph`;
  - performance command:`bench-serve --fail-on-error --require-ci` c16
    diagnostic,非 release evidence。
- Correctness evidence:
  - `ferrum run --batched-graph` rc `0`,output content `"5"`,
    `finish_reason=stop`;
  - `ferrum serve --batched-graph` readiness poll `8`,chat rc `0`,response
    content `"5"`,`finish_reason=length`,`completion_tokens=1`;
  - health after bench:`successful_requests=331`,`failed_requests=0`;
  - server log scan file has `0` lines for panic/error/NaN/`<unk>`/`[PAD]`/
    invalid UTF/fallback/graph-failed/capture-failed patterns;
  - server stopped cleanly,Vast shutdown verified
    `cur_state=stopped actual_status=exited`。
- Effective server config:
  - `selected_graph_mode=legacy_batched_decode_graph`;
  - `selected_kv_layout=paged`;
  - `selected_attention_impl=legacy_paged_decode`;
  - `selected_max_sequences=16`,`selected_kv_capacity=512`,
    `selected_max_batched_tokens=2048`。
- c16 diagnostic performance:
  - command shape:`bench-serve --random-input-len 256 --random-output-len 128
    --concurrency-sweep 16 --num-prompts 100 --n-repeats 3 --fail-on-error
    --require-ci --seed 9271`;
  - rc `0`,completed per run `[100,100,100]`,errored per run `[0,0,0]`,
    output token count source `usage`;
  - output throughput `287.1167006548677 ± 41.632552793935645 tok/s`;
  - goodput `2.251751298484382 ± 0.3173552733445153 req/s`;
  - TTFT p50 `798.304ms`,TPOT p50 `46.950ms`。
- Interpretation:
  - previous same-binary default-path c16 was
    `295.8064415567493 ± 5.210666937312439 tok/s`;
  - `--batched-graph/default = 0.970624`,so graph replay is not the current
    W2-P2 throughput lever;
  - vs direct random-prompt vLLM diagnostic baseline
    `381.3929242134927 tok/s`,this is `75.2811%`,still below 80%;
  - remaining bottleneck likely sits in decode cadence, scheduler/admission,
    or per-token tail work above/beside graph replay。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。
- Required next:
  - stop spending full sweeps on graph toggle;
  - inspect c16 execution cadence and tail latency with the smallest profiler
    that does not perturb correctness more than necessary;
  - consider comparing default vs graph profile traces only if trace overhead is
    bounded and the hypothesis is specific。

## 2026-06-16 LXXXVIII — W2 CUDA checkpoint: default paged-unified run/serve correctness passes, c16 still below target

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_paged_unified_default_path_cuda_smoke_2026-06-16/`。
- Source checkpoint:
  `d6d872c1e12fc364886117b0431aec752b2d78ac`,远端通过 git bundle clone,
  `git status --short` clean,无 remote diagnostic source patch。
- GPU 执行合同:
  - lane:`W2 default-path paged-unified correctness smoke`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - stop condition:启动/SSH/CUDA/source sync/build/run/serve 任一失败即收集
    日志并停止;run+serve 正确性通过后同次启动内跑一个 c16 diagnostic;
  - correctness gate:默认产品路径 `ferrum run` 和 `ferrum serve`;
  - performance command:correctness clean 后 `bench-serve --fail-on-error
    --require-ci` c16 diagnostic,非 release evidence。
- Build evidence:
  - CUDA release build rc `0`;
  - binary SHA256
    `11b26df2b8dccf3138b2fe294e80ef618cc6255e56af626213e6aaabe8b2e48f`;
  - CUDA environment:driver `565.77`,runtime CUDA `12.7`,`nvcc 12.4.131`。
- Correctness evidence:
  - `ferrum run` rc `0`,one-shot JSONL output content `"5"`,
    `finish_reason=stop`;
  - `ferrum serve` readiness poll `8`,chat rc `0`,response content `"5"`,
    `finish_reason=length`,`usage.prompt_tokens=23`,`completion_tokens=1`;
  - health after bench:`successful_requests=331`,`failed_requests=0`;
  - server log scan file has `0` lines for panic/error/NaN/`<unk>`/`[PAD]`/
    invalid UTF patterns used in this artifact;
  - server stopped cleanly,post-stop `nvidia-smi` shows no running GPU
    processes;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Effective server config:
  - `selected_kv_layout=paged`;
  - `selected_attention_impl=legacy_paged_decode`;
  - `selected_graph_mode=graph_disabled`;
  - `selected_max_sequences=16`,`selected_kv_capacity=512`,
    `selected_max_batched_tokens=2048`。
- c16 diagnostic performance:
  - command shape:`bench-serve --random-input-len 256 --random-output-len 128
    --concurrency-sweep 16 --num-prompts 100 --n-repeats 3 --fail-on-error
    --require-ci --seed 9271`;
  - rc `0`,completed per run `[100,100,100]`,errored per run `[0,0,0]`,
    output token count source `usage`;
  - output throughput `295.8064415567493 ± 5.210666937312439 tok/s`;
  - goodput `2.3204614024423846 ± 0.031239672060048216 req/s`;
  - TTFT p50 `798.748ms`,TPOT p50 `45.528ms`。
- Performance interpretation:
  - 直接同形状 random-prompt vLLM artifact 约 `381.5 tok/s`,但该 vLLM
    run 有 `1` 个 bad output/errored request,所以只能做 diagnostic;
    按该不干净 baseline 计算,Ferrum 约 `77.6%`;
  - 更干净的 same-instance vLLM ShareGPT baseline 是 `518.796 tok/s`,但本轮
    Ferrum 没有复跑同一 ShareGPT dataset,不能拿它做严格当前同数据集比例;
  - 因此本轮仍不能证明 `>=80%` mainstream-engine target。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。
- Required next:
  - 先补一个干净 same-dataset current Ferrum/vLLM comparison,再更新性能结论;
  - 当前 correctness blocker 已解除,剩余问题回到性能瓶颈,重点看
    graph-disabled/default runtime policy、decode cadence、scheduler/admission
    与 per-token tail latency。

## 2026-06-16 LXXXVII — W2 source checkpoint: allow paged KV for windowed Gemma3 on varlen backends

- 本轮源码修复:
  - `paged_kv_allowed_for_layer_schedule(...)` 不再对所有
    `sliding_window_pattern != 0` 模型一刀切禁用 paged KV;
  - 新规则: paged enabled 且 (`sliding_window_pattern == 0` 或 backend
    supports varlen QKV);
  - 注释更新为:windowed Gemma3 只有在后端 varlen QKV 路径能接收 per-layer
    sliding-window schedule 时才允许 paged KV;
  - 单测改名并覆盖正反例:
    `paged_kv_layer_schedule_allows_windowed_models_with_varlen_backend`。
- Why:
  - LXXXVI 已用同形状 CUDA product diagnostic 证明,在放开 paged KV guard
    且应用 embed-scale 修复后,`ferrum serve` 最小 chat smoke 从空输出
    恢复为 expected first token `"5"`;
  - LXXXIII 的 native CUDA probe 已排除 split-QKV + paged-varlen attention
    pair 的独立正确性问题;
  - 因此可以把远端 diagnostic guard override 提升为受测试覆盖的源码逻辑,
    但仍需要无 dirty patch 的默认产品路径验证。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models paged_kv_layer_schedule_allows_windowed_models_with_varlen_backend -- --nocapture`
    PASS;
  - `cargo test -p ferrum-models unified_varlen_qkv_requires_gemma_sandwich_prerequisites -- --nocapture`
    PASS;
  - `cargo test -p ferrum-models llm_executor -- --nocapture` PASS
    (13 tests)。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。
- Required next:
  - CUDA default-path correctness smoke without any remote diagnostic source
    patch;
  - because this changes product behavior, validate both `ferrum run` and
    `ferrum serve` before using performance numbers as evidence;
  - only after default-path correctness is clean should c16/c32 same-hardware
    performance comparison resume。

## 2026-06-16 LXXXVI — W2 CUDA diagnostic: embed-scale fix restores first-token correctness

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_paged_unified_embed_scale_fix_cuda_smoke_2026-06-16/`。
- Source checkpoint:
  `fb6789c7f99cc08f05842503846ea42af2be842d` plus a remote-only
  diagnostic patch that temporarily allowed paged KV for windowed Gemma3 when
  CUDA supports varlen QKV。默认 checked-in guard 仍未放开。
- GPU 执行合同:
  - lane:`W2 paged-unified embed-scale fix product smoke`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - stop condition:启动/SSH/source sync/diagnostic guard patch/build/serve/chat
    任一失败,或 fixed-path `[unified-logits]` 与 response evidence collected
    后停机;
  - correctness command:`ferrum serve` + one non-stream chat request with
    `max_tokens=1`;
  - performance command:none。
- Execution evidence:
  - CUDA release build rc `0`;
  - binary SHA256
    `e131ce885efb3f8aeb6049a9181f646638c4c8f81d0c993cfb33da29a4d7bc65`;
  - response content `"5"`,`finish_reason=length`,`completion_tokens=1`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`;
  - `nvidia_smi_after_stop.txt` shows no running GPU processes。
- Key result:
  - `[unified-decode] call#0 items=1 prefill=1 decode=0 total_q=23
    attempted_unified=true fallback=false fallback_reason=none elapsed=141708us`;
  - `[unified-logits] call#0 row=0 orig_idx=0 global=22 finite=262208
    nan=0 pos_inf=0 neg_inf=0
    top=[236810:42.031250,239374:20.453125,247918:20.453125,239341:20.187500,242323:20.015625]`。
- Interpretation:
  - LXXXIV 的 pre-fix 同形状 smoke 首 token 直接 EOS/stop;本轮同形状
    smoke 返回 expected first token `"5"`;
  - logits row 仍全 finite,且 EOS token id `106` 不再是 top-1;
  - 这证明 `unified_forward_internal` 漏乘 Gemma3 `embed_scale` 是一个真实
    paged-unified 正确性 bug,不是单纯性能测量噪声。
- Release-grade status:
  - 这是 diagnostic evidence,不是 release evidence,因为远端用了临时
    paged-KV guard override;
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。
- Required next:
  - 将 paged-KV guard 放开逻辑以源码形式提交并更新单测;
  - 之后不用远端 dirty patch,按默认产品路径最小验证 `ferrum run` 和
    `ferrum serve`;
  - 默认路径 correctness 过之前不跑 c16/c32 performance。

## 2026-06-16 LXXXV — W2 source checkpoint: apply Gemma embed scale in unified forward

- 本轮源码修复:
  - `unified_forward_internal` 在 `embedding_lookup` 后补上
    `cfg.embed_scale` 的 `B::scale_inplace`;
  - 缩放发生在 `activation_to_f32_shadow` 之前,因此 Gemma3 CUDA 的 F32
    residual shadow 也接收缩放后的 residual;
  - 这与 legacy `decode_batch_internal`,`prefill_internal`,`decode_internal`
    等路径保持一致。
- Why:
  - LXXXIV 的 product logits diagnostic 证明 paged-unified 首步 logits 全
    finite,但 eos/stop token id `106` 排 top;
  - native split-QKV + paged-varlen attention 已在 LXXXIII 通过,所以问题
    更可能在 product unified path 的模型语义差异;
  - 源码对比发现 unified embedding path 漏掉 Gemma3 的
    `embed_scale = sqrt(hidden_size)` 语义,这是与已正确 legacy path 的
    直接差异。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models llm_executor -- --nocapture` PASS
    (13 tests);
  - `cargo test -p ferrum-models unified_varlen_qkv_requires_gemma_sandwich_prerequisites -- --nocapture`
    PASS。
- Required next validation:
  - rerun only the LXXXIV minimal CUDA product smoke shape with the same
    diagnostic paged-KV guard override;
  - success criterion for this diagnostic: chat response content should no
    longer be empty/stop-at-first-token, and `[unified-logits]` should no
    longer rank eos token id `106` as top-1;
  - do not run c16/c32 performance until this correctness smoke is clean。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXXIV — W2 CUDA product diagnostic: paged-unified logits rank eos top-1

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_paged_unified_logits_product_diag_2026-06-16/`。
- Source checkpoint:
  `b768073a80c8a7519c1107083f5a10b478d0fe1a` plus a remote-only
  diagnostic patch that temporarily allowed paged KV for windowed Gemma3
  when CUDA supports varlen QKV。默认 checked-in path remains protected。
- GPU 执行合同:
  - lane:`W2 paged-unified product logits diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - stop condition:启动/SSH/CUDA/source sync/diagnostic patch/build/serve/chat
    任一失败,或 `[unified-logits]` evidence collected 后停机;
  - correctness command:`ferrum serve` + one non-stream chat smoke with
    `max_tokens=1`;
  - performance command:none。
- Execution evidence:
  - CUDA release build rc `0`;
  - binary SHA256
    `1d046a81f5194f80a946b2c0e2f37f1de97fdde69668ad359135f032e32af5d9`;
  - response empty,`finish_reason=stop`,`completion_tokens=1`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Key result:
  - `[unified-decode] call#0 items=1 prefill=1 decode=0 total_q=23
    attempted_unified=true fallback=false fallback_reason=none elapsed=136978us`;
  - `[unified-logits] call#0 row=0 orig_idx=0 global=22 finite=262208
    nan=0 pos_inf=0 neg_inf=0
    top=[106:11.039062,108:9.445312,107:8.882812,245526:8.460938,236743:8.304688]`;
  - tokenizer/generation metadata confirms eos token ids `[1,106]`,and token
    id `106` is `<end_of_turn>`。
- Interpretation:
  - paged-unified product path is not producing NaN/Inf or uninitialized
    sampled logits in this repro;
  - the wrong behavior is specifically that the first sampled logits row ranks
    the stop token highest;
  - after LXXXIII ruled out the standalone split-QKV + paged-varlen attention
    chain, the next source diff to fix is product unified model semantics
    above those kernels。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXXIII — W2 native checkpoint: split-QKV + paged-varlen combo probe PASS

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_paged_varlen_split_qkv_native_probe_2026-06-16/`。
- Source checkpoint:
  `7dc711ef817af737903098f14c068852c04d7dbf`。
- GPU 执行合同:
  - lane:`W2 paged-varlen split-QKV native correctness probe`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - stop condition:启动/SSH/CUDA/source sync/compile/probe 任一失败,
    或 probe PASS 后复制 artifact 并停机;
  - correctness command:
    `bash scripts/microbenches/build_and_run_paged_varlen_split_qkv_correctness.sh`;
  - performance command:none。
- Execution evidence:
  - remote/source head `7dc711ef817af737903098f14c068852c04d7dbf`;
  - CUDA environment:driver `565.77`,runtime-reported CUDA `12.7`,
    `nvcc 12.4.131`;
  - `probe/paged_varlen_split_qkv_correctness.rc` = `0`;
  - stdout contains
    `VERDICT: paged varlen split-qkv correctness PASS`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Native CUDA result:
  - `qk_mode=1 sliding_window=0`:q/k/v err all `0`,
    attention err `0.00012147`;
  - `qk_mode=1 sliding_window=3`:q/k/v err all `0`,
    attention err `0.00012141`;
  - `qk_mode=2 sliding_window=3`:q/k/v err all `0`,
    attention err `0.00011945`;
  - `qk_mode=3 sliding_window=3`:q/k/v err all `0`,
    attention err `0.00011978`;
  - `qk_mode=1 semantic_delta_full_vs_window=0.06742159`,证明 full
    causal 和 sliding-window 语义确实不同,不是等价空测。
- Interpretation:
  - standalone native CUDA chain
    `split_qkv_norm_rope_into_paged_cache_varlen_f16` ->
    `paged_varlen_attn_f16` 在合成非零 historical KV、当前 varlen 写入、
    QK-norm/RoPE modes 和 sliding-window attention 上对齐 CPU reference;
  - LXXX 的 Gemma3 paged-unified empty output 更可能在这对 kernels 之上
    或依赖真实 product/model state,例如后续 residual/tail/lm_head、sampled
    logits/stop token,或真实形状/权重数据;
  - 下一步应使用 LXXXI 的 `[unified-logits]` 做一次最小 product smoke,
    不跑 c16/c32,以判断首 token 是否 EOS/stop、logits row 错位或数值异常。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXXII — W2 source checkpoint: native split-QKV + paged-varlen combo probe

- 本轮没有启动 GPU,没有新的性能数字,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source/tooling change:
  - 新增
    `scripts/microbenches/paged_varlen_split_qkv_correctness.cu`;
  - 新增
    `scripts/microbenches/build_and_run_paged_varlen_split_qkv_correctness.sh`;
  - probe 直接用 `nvcc` 链接
    `split_qkv_norm_rope_into_paged_cache.cu` 和
    `paged_varlen_attention.cu`,不走 Cargo,不加载模型;
  - 覆盖 `qk_mode=1`(QK-norm + half-split RoPE, Gemma/Qwen-style),
    `qk_mode=2`,`qk_mode=3`,以及 full causal / sliding-window;
  - cache_k/cache_v 预置非零 historical KV,再由 varlen split 覆盖当前
    q tokens,最后由 paged-varlen attention 消费同一套 Q/K/V buffers。
- Why:
  - XLIII 已证明孤立 `paged_varlen_attention` 的 sliding-window 语义正确;
  - LXXX 失败发生在 product 链路
    `split_qkv_norm_rope_paged` -> `paged_varlen_attention` 之后,所以需要
    native CUDA 组合 probe,而不是继续跑 c16/c32 或只测 attention。
- Local validation:
  - `bash -n scripts/microbenches/build_and_run_paged_varlen_split_qkv_correctness.sh`
    PASS;
  - `git diff --check -- scripts/microbenches/paged_varlen_split_qkv_correctness.cu scripts/microbenches/build_and_run_paged_varlen_split_qkv_correctness.sh`
    PASS;
  - local machine has no `nvcc`,so native CUDA compile/run is pending on the
    cached 1x4090 instance。
- Required next validation:
  - run only:
    `bash scripts/microbenches/build_and_run_paged_varlen_split_qkv_correctness.sh`;
  - expected PASS line:
    `VERDICT: paged varlen split-qkv correctness PASS`;
  - if it fails, fix the native kernel issue before any product smoke or
    performance sweep;
  - if it passes, use the `[unified-logits]` product smoke from LXXXI to
    classify the remaining empty-output cause。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXXI — W2 source checkpoint: unified logits diagnostic for paged-varlen failure

- 本轮没有启动 GPU,没有新的性能数字,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - 在 `FERRUM_DECODE_OP_PROFILE` 已启用时,`unified_forward_internal`
    读回 sampled logits 后打印 `[unified-logits]`;
  - 每行日志包含前两条 sampled row 的 `orig_idx/global`,
    finite/NaN/+Inf/-Inf 计数,以及 top-5 token id/logit;
  - 诊断采样规则为前 8 次 unified logits readback 必打,之后每 64
    次打一条,避免长 bench 日志爆量;
  - 默认产品路径不启用该日志,不改变 scheduler/KV/sampling 行为。
- Why:
  - LXXX 证明 paged-unified Gemma3 可以去掉
    `fallback_reason=paged_kv_required`,但首个 chat smoke 变成空输出
    且 `finish_reason=stop`;
  - 下一次 CUDA 应该先跑 `max_tokens=1` 的最小 correctness smoke,
    用 `[unified-logits]` 判断是 EOS/stop token 排在 top,logits row
    错位,还是 NaN/Inf/未初始化等数值问题;
  - 在这个结果出来前,不要再跑 c16/c32 性能 sweep。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models logit_row_diagnostics_counts_and_sorts_top_values -- --nocapture`
    PASS;
  - `cargo test -p ferrum-models unified_logits_diag_uses_front_loaded_sampling -- --nocapture`
    PASS;
  - `cargo test -p ferrum-models llm_executor -- --nocapture` PASS
    (13 tests)。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXX — W2 CUDA checkpoint: paged unified removes fallback but fails chat correctness

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_paged_kv_unified_cuda_smoke_2026-06-16/`。
- Source checkpoint tested:
  `103c7013e849b198cabaa7ad47cd45063bf21e6d`。
- GPU 执行合同:
  - lane:`W2 CUDA paged-KV unified smoke after guard fix`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - stop condition:build + serve/chat smoke + c16 profiler evidence,或首个失败;
  - correctness gate:build rc `0`,serve ready,chat smoke content `5`,
    bench rc `0`;
  - performance command:c16/n=16/n_repeats=1 `bench-serve --fail-on-error
    --seed 9271`,但本轮因 correctness failure 未进入 bench。
- Execution evidence:
  - remote git head `103c7013e849b198cabaa7ad47cd45063bf21e6d`,
    remote source status clean;
  - binary SHA256
    `0d4595b6dbb6f4920ec5ed4af286ce7fbd89ad936d8dba1c76ad99d15806ac70`;
  - `build/build.rc=0`,serve ready poll `61`;
  - chat smoke failed:content empty,`completion_tokens=1`,
    `finish_reason=stop`;
  - `run_profile.rc=1`;bench did not start;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Key result:
  - `[unified-decode] call#0 items=1 prefill=1 decode=0 total_q=23
    attempted_unified=true fallback=false fallback_reason=none elapsed=336326us`;
  - `[unified-op]` confirms `split_qkv_norm_rope_paged` and
    `paged_varlen_attention` executed;
  - so `paged_kv_required` was removed, but the paged-unified Gemma3 path
    is not product-correct yet。
- Source follow-up in current HEAD:
  - keep windowed Gemma3 on contiguous KV by default;
  - update the helper/test so `sliding_window_pattern != 0` remains
    non-paged until paged-varlen correctness is fixed;
  - this prevents `103c7013` from leaving a product correctness regression
    active at HEAD。
- Required next work:
  - do not run another c16 perf bench until a smaller correctness repro
    isolates paged-varlen wrong output;
  - preferred next probe: native CUDA/minimal kernel or model-layer smoke for
    `split_qkv_norm_rope_paged` + `paged_varlen_attention` against the
    contiguous Gemma3 path, then re-enable only after chat smoke passes。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXIX — W2 source checkpoint: allow CUDA paged KV for Gemma3 windowed unified path

- 本轮没有再次启动 GPU,不产生新的性能结论,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - `ensure_kv` 不再用 `sliding_window_pattern == 0` 一刀切禁用 paged KV;
  - 新增 `paged_kv_allowed_for_layer_schedule(...)`,规则是:
    paged enabled 且 (`sliding_window_pattern == 0` 或后端支持
    `varlen_qkv`);
  - 结果:CUDA 这类已有 paged-varlen/sliding-window 参数的后端可以给
    Gemma3 windowed layers 分配 paged pools,从而满足 `unified_forward`
    的 paged KV 前置条件;
  - Metal paged decode 仍未暴露 per-layer window 到 paged dispatch,所以
    Gemma3/windowed family 仍保持 contiguous KV 保护。
- Why:
  - LXXVIII CUDA artifact 显示所有 observed `unified_decode` 都
    `attempted_unified=true`,但全部
    `fallback_reason=paged_kv_required`;
  - 源码确认 `ensure_kv` 里 Gemma3 因 `sliding_window_pattern != 0`
    禁用 paged pools,而 `unified_forward` 又硬要求 `paged_pools`,
    形成真实设计矛盾。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models paged_kv_layer_schedule_allows_windowed_models_only_with_varlen_backend -- --nocapture`
    PASS;
  - `cargo test -p ferrum-models unified_varlen_qkv_requires_gemma_sandwich_prerequisites -- --nocapture`
    PASS;
  - `cargo test -p ferrum-models llm_executor -- --nocapture` PASS
    (13 tests);
  - `git diff --check -- crates/ferrum-models/src/models/llama_family.rs ...`
    PASS。
- Required next validation:
  - one minimal CUDA smoke only: build current checkpoint, serve/chat,
    c16 profiler-on bench;
  - expected first check is that `[unified-decode]` prefill lines no longer
    report `fallback_reason=paged_kv_required`;
  - if correctness fails, stop and inspect token/logit/KV state before any
    performance measurement。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXVIII — W2 CUDA checkpoint: unified_decode fallback is paged_kv_required

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_unified_decode_fallback_reason_cuda_diag_2026-06-16/`。
- Source checkpoint:
  `5e9ae1514247e1ea7c34459dae3e2d1198c5e77d`。
- GPU 执行合同:
  - lane:`W2 unified_decode fallback_reason CUDA diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:10-20min;实际包含 release relink;
  - stop condition:build + serve/chat smoke + c16 profiler fallback_reason
    evidence complete,或首个失败;
  - correctness gate:build rc `0`,serve ready,chat smoke pass,bench rc `0`;
  - performance command:c16/n=16/n_repeats=1 `bench-serve --fail-on-error
    --seed 9271`,diagnostic only。
- Execution evidence:
  - remote git head `5e9ae1514247e1ea7c34459dae3e2d1198c5e77d`,
    remote source status clean;
  - binary SHA256
    `9ef89f43cc5f8675f85aaa32811ba2a3ee9ea704f79f2f87fd250a913363913e`;
  - `build/build.rc=0`,`run_profile.rc=0`,
    `bench/bench_sharegpt_c16.rc=0`;
  - serve ready poll `60`;chat response content `5`,usage present,
    bad_output false;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Key result:
  - c16 diagnostic:16 completed / 0 errored,`output_token_count_source=usage`;
  - throughput `315.39451845233344 tok/s`;
  - orientation-only vLLM baseline `518.7959572662905 tok/s`,Ferrum/vLLM
    ratio `0.6079355747378077`;
  - `[unified-decode]` line count `131`,prefill line count `3`;
  - all observed fallback reasons:
    `{"paged_kv_required": 131}`;
  - c16 prefill cohort:
    `call#3 items=10 prefill=10 decode=0 total_q=1220 attempted_unified=true fallback=true fallback_reason=paged_kv_required elapsed=890406us`;
  - later c16 cohort:
    `call#67 items=16 prefill=16 decode=0 total_q=1952 attempted_unified=true fallback=true fallback_reason=paged_kv_required elapsed=1396576us`。
- Interpretation:
  - LXXV full-logits fix was not the main bottleneck;
  - `LlmExecutor::unified_decode` now proves the model unified path is
    attempted,then falls back because Gemma3 CUDA lacks paged pools;
  - source trace confirms `ensure_kv` disables paged KV for
    `sliding_window_pattern != 0`,while `unified_forward` requires
    `paged_pools`。This is the next real bottleneck。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXVII — W2 source checkpoint: expose unified_decode fallback reason

- 本轮没有启动 GPU,不产生性能结论,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - `LlmExecutor::unified_decode` 在现有 typed profiler 开关
    `FERRUM_BATCH_PREFILL_PROF` / `FERRUM_BATCH_DECODE_PROF` 下新增
    `[unified-decode]` 结构化日志;
  - 日志记录 `items`、`prefill`、`decode`、`total_q`、
    `attempted_unified`、`fallback`、`fallback_reason` 和 elapsed time;
  - full-logits 不可用时稳定输出
    `fallback_reason=requires_full_logits_unavailable`;
  - `model.unified_forward` 返回 Unsupported 时复用既有短码分类,例如
    `unified_varlen_qkv_disabled`、`sandwich_f32_shadow_required`、
    `paged_kv_required`、`active_lora_adapter`。
- Why:
  - LXXVI 证明 full-logits guard 修复只带来约 `+4.7%`,并且
    `prefill-profile tokens=122` 仍重复出现,说明 Gemma3 c16 prefill cohort
    仍在 `unified_decode` 内部回落到 serial prefill;
  - 继续做 full c16 sweep 前,需要一次最小 CUDA 验证直接读出 fallback reason,
    避免继续猜测瓶颈。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models unified_decode_prof_logs_prefill_fallback_and_sampled_decode -- --nocapture`
    PASS;
  - `cargo test -p ferrum-models llm_executor -- --nocapture` PASS
    (13 tests);
  - `git diff --check -- crates/ferrum-models/src/executor/llm_executor.rs`
    PASS。
- Required next CUDA validation:
  - reuse cached 4090 lane,run minimal serve/chat+c16 diagnostic with profiler on;
  - target evidence is the first `[unified-decode]` prefill line and its
    `fallback_reason`,not a repeated full performance sweep;
  - if it reports a source guard such as `unified_varlen_qkv_disabled` or
    `sandwich_f32_shadow_required`,fix that exact guard before measuring again。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXVI — W2 CUDA checkpoint: full-logits unified prefill fix is insufficient

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_full_logits_unified_prefill_cuda_diag_2026-06-16/`。
- Source checkpoint:
  `40186c75e393ef58e81b9f5acfe529186505a0bc`。
- GPU 执行合同:
  - lane:`W2 full-logits unified-prefill CUDA diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:15-30min,约 USD `0.11-0.22`;
  - stop condition:CUDA build + serve/chat smoke + c16 diagnostic bench +
    log evidence complete,或首个失败;
  - correctness gate:build rc `0`,serve ready,deterministic chat smoke pass,
    bench rc `0`;
  - performance command:c16/n=16/n_repeats=1 `bench-serve --fail-on-error
    --seed 9271`,diagnostic only。
- Execution evidence:
  - `build/build.rc=0`,`run_profile.rc=0`,
    `bench/bench_sharegpt_c16.rc=0`;
  - smoke response content `5`,usage present,bad_output false;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`;
  - remote source dirty includes the 3 synced source files for
    `40186c75`,so this is not release performance evidence。
- Key result:
  - c16 diagnostic:16 completed / 0 errored,`output_token_count_source=usage`;
  - throughput `298.24957600538823 tok/s` vs previous `284.90049780836483`,
    about `+4.7%`;
  - orientation-only vLLM baseline `518.7959572662905 tok/s`,Ferrum/vLLM
    ratio `0.5748880110341743`;
  - `prefill-profile tokens=122` lines remain `26`;
  - first c16 prefill cohort still shows
    `iter#3 items=10 prefill=10 total=927027us model=924576us` plus repeated
    serial 122-token `prefill-profile` rows。
- Interpretation:
  - LXXV full-logits guard fix was locally correct but insufficient for W2-P2;
  - main TTFT/prefill wall time remains,so do not run another CUDA c16
    diagnostic until `LlmExecutor::unified_decode` records why
    `model.unified_forward` still falls back;
  - next source step: add unified_decode fallback reason observability
    analogous to `batch_prefill`。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXV — W2 source checkpoint: allow full-logits unified prefill

- 本轮没有启动 GPU,不产生性能结论,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - `DecoderOnlyLLM` 新增 `unified_forward_can_return_full_logits()`,默认
    `true`;
  - `LlmExecutor::unified_decode` 不再因为 batch 中存在
    `ferrum_require_full_logits` 就无条件跳过 `model.unified_forward`;
  - 只有模型声明 unified path 不能返回 full logits 时才保留旧 fallback;
  - Qwen3 MoE 在 `unified_greedy_argmax` sentinel 路径开启时返回 `false`,
    保持 full-logits correctness 保护。
- Why:
  - LXXIV artifact 里的 engine `[unified-prof] items=10 prefill=10` 只能证明
    engine 构造了 unified batch,不能证明 model 层走了 true unified prefill;
  - 同一日志中 `prefill-profile tokens=122` 重复 10 次,与
    `LlmExecutor::unified_decode` 的 full-logits guard 对应:普通请求带
    tokenizer/sampling mask 时会设置 `ferrum_require_full_logits`,从而把
    Gemma3 c16 prefill cohort 退回 serial `model.prefill`;
  - 此改动把产品路径从“full-logits 必定 serial fallback”改为“模型能返回
    full logits 时仍用 unified prefill”,直接对齐 W2-P2 的 batched/unified
    Gemma3 fast path 目标。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models llm_executor -- --nocapture` PASS
    (12 tests);
  - targeted tests PASS:
    `unified_decode_uses_unified_forward_when_full_logits_supported`,
    `unified_decode_skips_unified_forward_when_full_logits_unsupported`,
    `unified_decode_full_logits_prefill_uses_unified_forward_and_prepares_kv_capacity_hint`,
    `batch_prefill_falls_back_after_unified_unsupported`。
- Required next CUDA validation:
  - build CUDA release binary on the cached 4090 lane;
  - rerun minimal c16 serve/chat/bench diagnostic;
  - verify `prefill-profile tokens=122` no longer repeats serially before
    the c16 prefill batch, and compare throughput/TTFT against LXXIV。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXIV — W2 CUDA checkpoint: batch-prefill fallback hypothesis rejected

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_batch_prefill_fallback_reason_diag_2026-06-16/`。
- GPU 执行合同:
  - lane:`W2 batch-prefill fallback-reason diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:15-25min,约 USD `0.11-0.18`;
  - stop condition:CUDA build + serve/chat smoke + c16 diagnostic bench +
    captured prefill/fallback evidence,或首个失败;
  - correctness gate:build rc `0`,serve ready,deterministic chat smoke pass,
    bench rc `0`;
  - performance command:c16/n=16/n_repeats=1 `bench-serve --fail-on-error
    --seed 9271`,diagnostic only。
- Execution evidence:
  - local source checkpoint:`7eb1747703e63ba7ac58ef2133a991f98c21e413`;
  - remote base HEAD:`935777e9feb8c1606631761ec8e0fb6c3f3f0a06`,只同步本轮
    instrumentation diff,因此不作为 release performance evidence;
  - `build/build.rc=0`,`run_profile.rc=0`,
    `bench/bench_sharegpt_c16.rc=0`;
  - smoke response content `5`,usage present,bad_output false;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Key result:
  - `FERRUM_BATCH_PREFILL_PROF` 来自 typed config,但完整 server/profile 日志中
    `[batch-prefill]`/`fallback_reason=` 行数为 `0`;
  - continuous unified path 明确已跑 batch prefill:
    `iter#3 items=10 prefill=10 total=946123us model=943620us`;
  - 因此此前“c16 TTFT 主要是 `LlmExecutor::batch_prefill` serial fallback”的
    假设被本轮产品路径证据推翻。
- Diagnostic performance shape:
  - c16 diagnostic:16 completed / 0 errored,`output_token_count_source=usage`;
  - throughput `284.90049780836483 tok/s`;
  - orientation-only vLLM baseline `518.7959572662905 tok/s`,Ferrum/vLLM
    ratio `0.549157127803387`;
  - 该数字为 single-run diagnostic,不是 release 或最终性能声明。
- Interpretation:
  - 下一步不要继续围绕 `LlmExecutor::batch_prefill` fallback 猜测消耗;
  - 真瓶颈转向 unified prefill wall time 本身,并结合已确认的 dense GPTQ
    Marlin MLP kernel 热点和 prefill attention 成本做最小验证;
  - dense Marlin block-policy native probe 已排除 grid override,不能作为
    产品优化杠杆。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXIII — W2 source checkpoint: expose batch-prefill fallback reason

- 本轮没有启动 GPU,不产生性能结论,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - `LlmExecutor::batch_prefill` 在 `FERRUM_BATCH_PREFILL_PROF` profiler 行中新增
    `fallback_reason=<code>`;
  - unified prefill `Unsupported` 文本会被归类为稳定短码,包括
    `unified_varlen_qkv_disabled`,`sandwich_f32_shadow_required`,
    `paged_kv_required`,`active_lora_adapter`,`requires_full_logits` 等;
  - 行为不变:unsupported 时仍 fallback 到 serial per-item `model.prefill`。
- Why:
  - W2 c16 TTFT 侧已定位为 Gemma3 serial prefill fallback;
  - 之前 profiler 只能看到 `fallback=true`,不能结构化证明是 varlen/Gemma3
    guard、prefix cache、LoRA、paged KV 还是 full-logits 触发;
  - 下一次 CUDA prefill diagnostic 可以直接验证产品路径的 fallback reason,
    为 narrow Gemma3 cohort-prefill 实现提供可回归证据。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models unified_fallback_reason_code -- --nocapture` PASS;
  - `cargo test -p ferrum-models batch_prefill_falls_back_after_unified_unsupported -- --nocapture`
    PASS;
  - `git diff --check -- crates/ferrum-models/src/executor/llm_executor.rs` PASS。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXII — W2 native checkpoint: dense Marlin block-policy probe rejects grid override

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_marlin_block_policy_probe_2026-06-16/`。
- Source checkpoint:
  `da8d8b25 test(cuda): probe dense marlin block policy`。
- GPU 执行合同:
  - lane:`W2 dense Marlin block-policy native probe`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:8-15min,约 USD `0.06-0.12`;
  - stop condition:nvcc 编译失败或 native probe 打出 `VERDICT` 后复制
    artifact 并停机;
  - correctness gate:native CUDA compile rc `0`,probe rc `0`;
  - performance command:
    `timeout 1800 bash scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh`。
- Execution evidence:
  - remote base HEAD:`935777e9feb8c1606631761ec8e0fb6c3f3f0a06`;
  - local source checkpoint:`da8d8b25a3f0aa28e826cfd75f3bcfae7b70ea3e`;
  - 本轮为节省时间只同步 native microbench 相关 dirty diff,不作为
    release performance evidence;
  - `probe/dense_marlin_gemma3_perf.rc=0`;
  - stdout 包含 `VERDICT: dense Marlin native CUDA probe complete`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Key result:
  - `gate_up m=16 auto` weight-cycle default `133.956us`,
    `blocks_n_tiles` `134.284us`,`blocks_2sms` `134.647us`;
  - `down m=16 auto` weight-cycle default `68.689us`,
    `blocks_n_tiles` `74.203us`,`blocks_2sms` `74.354us`;
  - `m=23/32` 上 `blocks_n_tiles`/`2sms` 对 `gate_up/down` 更差。
- Interpretation:
  - dense Marlin `gridDim.x`/block policy override 不是当前 W2 产品优化杠杆;
  - 不应把 `blocks=n_tiles` 或 `2sms` 推进到产品内核;
  - 下一步回到 decode integration / non-Marlin scheduling / prefill TTFT,
    或做更窄的 launch-count/overlap native probe。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXI — W2 CUDA checkpoint: profiler path passes with graph capture disabled

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_profiler_graph_disabled_retry_2026-06-16/`。
- GPU 执行合同:
  - lane:`W2 profiler graph-disabled retry`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:8-15min,hard cap 25min;
  - stop condition:build/server/chat/bench/profile 首败,或最小 c16
    profile 完成后复制 artifact 并停机;
  - correctness gate:CUDA release build,`ferrum serve` ready,chat smoke,
    `bench-serve --fail-on-error` c16 single cell;
  - performance command:diagnostic-only c16 ShareGPT single-run profile,
    no CI/no `--require-ci`。
- Execution evidence:
  - remote HEAD:`f7612c3a2a17c7e051f326ed7bac54484b25eb3a`;
  - non-artifact source status clean;
  - CUDA release build rc `0`,binary SHA256:
    `2a2ed419f3e80ede06ceaf54ba4495b66265c5ce2ba14b66dc39a35257cb6844`;
  - `ferrum serve` ready after poll `62`,chat smoke passed with content
    `5` and `completion_tokens=3`;
  - `bench_sharegpt_c16.rc=0`,`run_profile.rc=0`,
    `run_remote_profile.outer.rc=0`;
  - c16 diagnostic profile completed `16/16`,errors `0`,
    `output_token_count_source=usage`;
  - `capture_unsupported_panic=false`,`graph_capture_line_count=0`;
  - profile lines: `prefill-profile=297`,`batched-op-profile=128`,
    `unified-prof=67`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Diagnostic performance,not release evidence:
  - profile/eager path output throughput mean:`312.22668693855985 tok/s`;
  - same-artifact orientation vs prior vLLM c16 baseline:
    `60.18294525342617%`;
  - because this is `n_repeats=1`,profiler/eager path,no CI/no
    `--require-ci`,it is diagnostic only。
- Bottleneck signal:
  - c16 decode profile is now stable enough to use;
  - repeated `batched-op-profile m=16` shows `tail_mlp` around
    `13.7ms` of `27.5-28.1ms` total per decode iteration;
  - `tail_gate_up` around `9.0ms`, `tail_down` around `4.7ms`,
    matmul bucket around `7.0ms`, attention around `2.1-2.7ms`;
  - next performance lever should focus on Gemma3 tail MLP / GeGLU
    projection path before broad scheduler or engine work。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXX — W2 source checkpoint: disable graph capture for syncing diagnostics

- 本轮没有启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - `LlamaFamilyRuntimeEnv::graph_capture_allowed()` 新增 single decode
    graph capture 保护;
  - `LlamaBatchedRuntimeConfig::graph_capture_allowed()` 新增 batched/unified
    graph capture 保护;
  - single decode CUDA graph 在 decode/prefill/layer profile、nan trace、
    op dump、layer dump 任一诊断开关启用时不再 capture;
  - batched/unified graph 在 `decode_op_profile`、`unified_profile`、
    `batched_trace` 任一同步型诊断开关启用时不再 capture。
- Why:
  - LXIX 证明 profile+batched graph 组合仍能在 capture window 内走到
    普通 `B::sync`;
  - profiler/trace 本质上依赖同步计时边界,不应和 CUDA graph capture 同时
    生效;
  - 正常产品 graph path 保持可用,diagnostic profile path 改为 eager,
    用于稳定采集热点分解。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models llama_batched_runtime_config -- --nocapture`
    PASS,2/2;
  - `cargo test -p ferrum-models llama_family_runtime_env -- --nocapture`
    PASS,2/2;
  - `cargo test -p ferrum-models batched_graph_capture_is_disabled_by_syncing_diagnostics -- --nocapture`
    PASS,1/1;
  - `git diff --check` PASS。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXIX — W2 CUDA checkpoint: capture-lifecycle retry still fails under profile

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_typed_prefill_profile_capture_retry_2026-06-16/`。
- GPU 执行合同:
  - lane:`W2 typed prefill profile capture-lifecycle retry`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:8-15min,hard cap 25min;
  - stop condition:build/server/chat/bench/profile 首败,或最小 c16
    profile 完成后复制 artifact 并停机;
  - correctness gate:CUDA release build,`ferrum serve` ready,chat smoke,
    `bench-serve --fail-on-error` c16 single cell;
  - performance command:diagnostic-only c16 ShareGPT single-run profile,
    no CI/no `--require-ci`。
- Execution evidence:
  - remote HEAD:`a9d8b439097f89011fb02dc78e1046ddb07d73e6`;
  - non-artifact source status clean;
  - CUDA release build rc `0`,binary SHA256:
    `e111e6ec9653fd141ad5eb8ed504f18997a7d29244dbbe685be9955d2277a350`;
  - `ferrum serve` ready after poll `58`,chat smoke passed with content
    `5` and `completion_tokens=3`;
  - typed profiler config came from config file for
    `FERRUM_BATCH_DECODE_PROF`,`FERRUM_BATCH_PREFILL_PROF`,
    `FERRUM_DECODE_OP_PROFILE`,`FERRUM_PREFILL_OP_PROFILE`,
    `FERRUM_NEXT_BATCH_PROF`,`FERRUM_UNIFIED_POST_PROF`;
  - c16 profile emitted usable partial profile lines before failure:
    `prefill-profile` lines `121`,`batched-op-profile` lines `3`,
    `unified-prof` lines `7`;
  - failure remained:
    `CudaBackend: stream sync: DriverError(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "operation not permitted when stream is capturing")`;
  - run was stopped at first failure;`run_profile.rc=143`,
    `bench_sharegpt_c16.rc=143`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Interpretation:
  - LXVIII fixed one capture-end condition, but this retry proves the
    profile+batched-graph path can still reach a normal `B::sync` while CUDA
    capture is in flight;
  - this is still a correctness blocker for profiler-backed performance
    diagnosis, even though product chat smoke passed;
  - partial profiles still narrow the hot region: c16 prefill sample shows
    `tail_mlp` about `37-41ms/62 layers`, and batched decode sample shows
    `tail_mlp` about `13.5ms` per m=10 decode iteration。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXVIII — W2 source checkpoint: end capture based on active capture state

- 本轮没有启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - legacy single decode、unified graph、batched graph 的 capture-end 条件
    从 `should_capture && !*_graph_failed` 改成
    `should_capture && B::graph_capture_in_flight(&ctx)`;
  - 如果 begin capture 成功,即使后续 failure flag 已被置位,也会尝试
    `end_graph_capture` 收口,避免 capture window 泄漏到后续普通
    `B::sync`。
- Why:
  - LXVII retry 说明第一刀 guarded profiler sync 还不够;
  - 失败形态更像 capture window 没有正常结束,导致后续正常同步命中
    `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`;
  - 以 backend active-capture state 作为 end 条件比 failure flag 更直接。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models llama_batched_runtime_config -- --nocapture`
    PASS,2/2。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXVII — W2 CUDA checkpoint: first graph-safe profiler fix still leaves capture open

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_typed_prefill_profile_retry_2026-06-16/`。
- GPU 执行合同:
  - lane:`W2 typed prefill profile graph-safe retry`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:8-15min,hard cap 25min;
  - stop condition:启动/SSH/source sync/build/server/smoke/bench 首败,或
    retry artifact 完成后复制 artifact 并停机;
  - correctness gate:CUDA release build,`ferrum serve` ready,chat smoke,
    `bench-serve --fail-on-error` 零错误;
  - performance command:diagnostic-only c16 ShareGPT single-run profile,
    no CI/no `--require-ci`。
- Execution evidence:
  - remote HEAD:`f352ff3f6b596418659ff5912995d07f5e9fc1fc`;
  - non-artifact source status clean;
  - CUDA release build rc `0`,binary SHA256:
    `f3742953afabfa1ad3ac99d58978d0508825085b4a2b706e4f7e508a1a1944f7`;
  - `ferrum serve` ready after poll `59`,chat smoke passed with content
    `5` and `completion_tokens=3`;
  - retry still hit
    `CudaBackend: stream sync: DriverError(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "operation not permitted when stream is capturing")`;
  - bench was manually stopped after panic;`run_profile.rc=143`,
    `bench_sharegpt_c16.rc=143`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Interpretation:
  - first source fix guarded profile sync calls, but did not fully fix capture
    lifecycle;
  - evidence points to a capture window remaining open when a later normal
    `B::sync` runs;
  - next source fix should end graph capture whenever
    `B::graph_capture_in_flight(&ctx)` is true, instead of relying on
    `!*_graph_failed` flags。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXVI — W2 source checkpoint: make batched op profiler graph-capture safe

- 本轮没有启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - `Backend` 新增 `graph_capture_in_flight(&Context) -> bool`,默认返回
    `false`;
  - CUDA backend 返回 `CudaState.capture_in_flight`;
  - Llama/Gemma batched/unified decode op profiler 在 graph-capture window
    自动跳过 `B::sync` 型计时边界,避免
    `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`。
- Why:
  - LXV 失败证据显示 typed `decode_op_profile/prefill_op_profile` 与
    batched CUDA graph capture 同时开启时,profiler 的同步型计时触发
    CUDA stream-capture 不支持错误;
  - 这不是默认产品 graph 路径的性能问题,而是诊断观测路径的正确性问题;
  - 修复保持 graph 开启,只让 graph capture iteration 不执行 graph-unsafe
    sync profiler。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-kernels cpu_timer -- --nocapture` PASS,2/2;
  - `cargo test -p ferrum-models llama_batched_runtime_config -- --nocapture`
    PASS,2/2。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXV — W2 CUDA checkpoint: typed prefill profile exposes graph-capture profiler correctness bug

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_typed_prefill_profile_2026-06-16/`。
- GPU 执行合同:
  - lane:`W2 Gemma3 typed-config ShareGPT prefill profile`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:10-20min,hard cap 30min;
  - stop condition:启动/SSH/source sync/build/server/smoke/bench 首败,或
    profile artifact 完成后复制 artifact 并停机;
  - correctness gate:CUDA release build,`ferrum serve` ready,chat smoke,
    `bench-serve --fail-on-error` 零错误;
  - performance command:diagnostic-only c16 ShareGPT
    `bench-serve --fail-on-error --seed 9271 --dataset sharegpt --num-prompts 16 --n-repeats 1`,
    no CI/no `--require-ci`。
- Execution evidence:
  - remote HEAD:`353c1eb2521118c37342def279fe3c22b2715e20`;
  - non-artifact source status clean;
  - CUDA release build rc `0`,binary SHA256:
    `138ff2e0000947dafb7299b74d96397fa300b5eb61cc168305fae160d06deeff`;
  - profiler flags came from config-file entries:
    `FERRUM_BATCH_DECODE_PROF`,`FERRUM_BATCH_PREFILL_PROF`,
    `FERRUM_DECODE_OP_PROFILE`,`FERRUM_PREFILL_OP_PROFILE`,
    `FERRUM_NEXT_BATCH_PROF`,`FERRUM_UNIFIED_POST_PROF`;
  - `ferrum serve` ready,chat smoke passed with content `5` and
    `completion_tokens=3`;
  - bench was manually stopped after server panic;`run_profile.rc=143`,
    `bench_sharegpt_c16.rc=143`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Failure:
  - server panic:
    `CudaBackend: stream sync: DriverError(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "operation not permitted when stream is capturing")`;
  - panic occurred after `batched-op-profile` emitted under
    `FERRUM_DECODE_OP_PROFILE` while batched graph was active;
  - this is a correctness issue in diagnostic/profile instrumentation under
    CUDA graph capture,not a release performance data point。
- Interpretation:
  - no W2 performance conclusion from this run;
  - profiler path must become graph-safe before using typed prefill/decode op
    profile to locate the remaining ShareGPT c16 TTFT/decode gap;
  - next source fix should avoid stream synchronization inside CUDA graph
    capture,or automatically disable the graph-unsafe op profiler when graph
    capture is active。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXIV — W2 source checkpoint: expose prefill profiler knobs through runtime config

- 本轮没有启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - config-file `runtime` 现在可显式开启:
    - `batch_prefill_prof` -> `FERRUM_BATCH_PREFILL_PROF`;
    - `decode_op_profile` -> `FERRUM_DECODE_OP_PROFILE`;
    - `prefill_op_profile` -> `FERRUM_PREFILL_OP_PROFILE`;
  - `LlmExecutorRuntimeEnv` 不再从 `std::env::vars()` 直接冻结 profiler
    flags,改为读取 composition root 安装的 `RuntimeConfigSnapshot`;
  - profiler flags 仍保持 presence-flag 语义:只有 config `true` 才
    materialize runtime entry,`false`/unset 不会生成 `FERRUM_*_PROF=0`。
- Why:
  - 上一轮 typed decode integration profile 已排除 scheduler/postprocess/普通
    host loop gap 为主因;
  - W2 c16 剩余风险集中在 Gemma3 模型侧 decode/prefill,尤其 ShareGPT TTFT
    和 batched prefill 是否仍有 fallback;
  - 下一次 CUDA 只需用 typed config 打开 prefill/decode profiler 做最小
    ShareGPT c16 诊断,不需要隐藏 env 组合或重复 full sweep。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-cli runtime_cli_config -- --nocapture` PASS,3/3;
  - `cargo test -p ferrum-models llm_executor_runtime_env -- --nocapture`
    PASS,3/3;
  - `git diff --check` PASS。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXIII — W2 CUDA checkpoint: typed-config decode integration profile points back to model-side decode

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_typed_decode_profile_2026-06-16/`。
- GPU 执行合同:
  - lane:`W2 Gemma3 typed-config decode integration profile`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:10-20min,hard cap 30min;
  - stop condition:启动/SSH/source sync/build/server/smoke/bench 首败,或 profile
    artifact 完成后复制 artifact 并停机;
  - correctness gate:CUDA release build,`ferrum serve` ready,chat smoke,
    `bench-serve --fail-on-error` 零错误;
  - performance command:diagnostic-only c16
    `bench-serve --fail-on-error --seed 9271 --num-prompts 16 --random-input-len 32 --random-output-len 32`,
    `n_repeats=1`,no CI/no `--require-ci`。
- Execution evidence:
  - remote HEAD:`4fea56ec79d0c8a9edcf99dd90b3889d422869e9`;
  - non-artifact source status clean;
  - CUDA release build rc `0`,binary SHA256:
    `23f04a49e361c836ab6a8afb125d68e771df361219013bee0d32ecf630a2559d`;
  - profiler flags came from typed config-file entries:
    `FERRUM_BATCH_DECODE_PROF`,`FERRUM_NEXT_BATCH_PROF`,
    `FERRUM_UNIFIED_POST_PROF`;
  - `ferrum serve` ready after poll `62`,selected graph mode
    `legacy_batched_decode_graph`;
  - chat smoke content `5`,usage completion_tokens `3`;
  - bench rc `0`,`completed_per_run=[16]`,`errored_per_run=[0]`,
    `bad_output_per_run=[0]`,`zero_output_tokens_per_run=[0]`,
    `output_token_count_source=usage`;
  - c16 diagnostic throughput `380.492 tok/s`,TTFT p50 `587.854ms`,
    TPOT p50 `23.822ms`;
  - `output_tokens_per_request`:
    `[[32,32,32,28,32,32,32,31,32,32,32,32,32,30,32,32]]`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Profile result:
  - full `decode=16` iterations:`27`;
  - mean full decode iteration total:`23679.2us`;
  - mean model time:`23311.3us`,mean model share:`98.44%`;
  - mean decode postprocess time:`347.9us`,mean postprocess share:`1.47%`;
  - `bg-loop-gap` mostly single-digit microseconds,scheduler/process loop gap is
    not the main bottleneck。
- Interpretation:
  - 这次最小 profile 没有发现新的正确性问题;
  - engine scheduler、postprocess、streaming、普通 host loop gap 都不足以解释
    W2 c16 与 vLLM 的差距;
  - 继续扫 admission/开关的收益很低,下一步应集中在 Gemma3 模型侧 decode,
    尤其 tail/Marlin/projection behavior 与 weight-residency 类问题。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXII — W2 source checkpoint: expose decode profiler knobs through runtime config

- 本轮没有启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - config-file `runtime` 现在可显式开启:
    - `batch_decode_prof` → `FERRUM_BATCH_DECODE_PROF`;
    - `next_batch_prof` → `FERRUM_NEXT_BATCH_PROF`;
    - `rbd_prof` → `FERRUM_RBD_PROF`;
    - `unified_post_prof` → `FERRUM_UNIFIED_POST_PROF`;
  - 这些 profiler 是 presence flags,所以 config `true` 才 materialize runtime
    entry;`false`/unset 都不会生成 `FERRUM_*_PROF=0`,避免误触发。
- Why:
  - 最新 c16 诊断排除了稳定 `m=15` underfill 与输出早停主因;
  - 下一步需要量化 engine/process_batch/model/decode_post/scheduler 开销;
  - 通过 typed config-file 暴露现有 profiler 后,CUDA profile 诊断不需要隐藏
    env 组合,更符合 release-grade 证据路径。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-cli runtime_cli_config -- --nocapture` PASS,3/3。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 LXI — W2 CUDA checkpoint: c16 output-token/batch-shape diagnostic narrows remaining bottleneck

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_output_token_c16_diag_2026-06-15/`。
- GPU 执行合同:
  - lane:`W2 Gemma3 c16 output-token/batch-shape diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:10-20min,hard cap 30min;
  - stop condition:启动/SSH/build/server/smoke/bench 首败,或 c16 诊断完成后
    复制 artifact 并停机;
  - correctness gate:CUDA release build,`ferrum serve` ready,chat smoke,
    `bench-serve --fail-on-error` 零错误;
  - performance command:diagnostic-only c16
    `bench-serve --fail-on-error --seed 9271 --num-prompts 16 --random-input-len 32 --random-output-len 32`,
    `n_repeats=1`,no CI/no `--require-ci`。
- Execution evidence:
  - remote HEAD:`25c32dac9305eb62acd733bd491b2d1294a3ba64`;
  - non-artifact source status clean;
  - CUDA release build rc `0`,binary SHA256:
    `a7f561a5f49a6858e8a63040a595143395f369ab7da1a5983d94927469b3861a`;
  - chat smoke content `5`,usage completion_tokens `3`;
  - bench rc `0`,`completed=[16]`,`errored=[0]`,
    `output_token_count_source=usage`;
  - new `output_tokens_per_request`:
    `[[32,32,32,28,32,32,32,32,32,30,32,32,32,32,32,32]]`;
  - c16 diagnostic throughput `363.087 tok/s`,TTFT p50 `496.040ms`,
    TPOT p50 `27.226ms`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Interpretation:
  - short outputs are only `6` tokens below the `16*32=512` cap,so early stop
    cannot explain the remaining W2 performance gap;
  - first batched decode trace saw `m=13`,but the main captured/replayed graph
    was `m=16 m_padded=16`;the previous `m=15` graph capture is not a stable
    sustained-decode bottleneck;
  - drain shape captured at `m=3 m_padded=4`,consistent with the two shorter
    requests finishing before the rest;
  - remaining high-probability bottleneck is still decode integration/host
    scheduling + tail MLP/Marlin projection/weight-residency,not graph replay
    absence,not stable c16 underfill,not output-token early stop。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 LX — W2 source checkpoint: bench-serve records per-request output token counts

- 本轮没有启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - `BenchReport` 增加 `output_tokens_per_request`;
  - `compute_metrics` 从每个 `RunRecord` 写出每次 repeat 的 measured request
    output token 数,失败请求保留记录值(通常为 0);
  - `bench-serve` 失败策略测试现在断言 fail-on-error 路径仍会先写出该字段。
- Why:
  - 最新 batched graph artifact 已证明 graph replay 真实发生,但主 replay shape
    是 `m=15` 而不是满 `m=16`;
  - 旧 report 只有总吞吐,缺少每请求 output token 分布,无法快速区分
    早停/尾部 drain 与持续调度不满批;
  - 下一次 CUDA 最小验证可直接用该字段判断 c16 诊断是否被某些请求短输出
    拉低,避免再做无信息 full sweep。
- Local validation:
  - `cargo test -p ferrum-bench-core` PASS,45/45;
  - `cargo test -p ferrum-cli fail_on_error_still_writes_json_report -- --nocapture`
    PASS;
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check` PASS。
- Bottleneck status:
  - 已有证据仍指向 decode integration + tail MLP/Marlin projection + batch
    cadence 的组合问题;
  - workspace zero 和直接 dense vLLM Marlin kernel swap 均不是当前第一杠杆;
  - W2 仍不是 release-grade。

## 2026-06-15 LIX — W2 CUDA checkpoint: batched graph replay confirmed, not sufficient for 80%

- 本轮源码 checkpoint:
  `22f92677 test(cuda): log batched graph replay progress`。
- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_batched_graph_replay_observability_2026-06-15/`。
- GPU 执行合同:
  - lane:`W2 Gemma3 batched graph replay observability smoke`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:10-15min,hard cap 20min;
  - stop condition:收集 capture/replay 日志 + serve correctness smoke,或首败并
    复制 artifact,或 20min hard cap;
  - correctness gate:远端 HEAD `22f92677`,CUDA release build rc `0`,
    `ferrum serve` ready,OpenAI chat smoke 输出 `5`,日志无实际 panic/CUDA
    error/`<unk>`/`[PAD]`;
  - performance command:diagnostic-only c16
    `bench-serve --fail-on-error --seed 9271 --num-prompts 16 --random-output-len 32`,
    `n_repeats=1`,不是 release 性能证据。
- 源码改动:
  - capture 成功时输出 `[batched-graph-capture]`;
  - post-capture replay 与 pure replay 成功时输出低频
    `[batched-graph-replay]`;
  - post-capture replay 现在也计入 `BATCHED_GRAPH_REPLAY_COUNT`;
  - replay 日志按 1/2/4/8/... 次数打印,避免长 bench 刷屏。
- 本地验证:
  - `cargo test -p ferrum-models batched_decode_graph --lib -- --nocapture`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`。
- CUDA result:
  - remote HEAD:`22f92677b34bab932407215fcb8c11dd0b372faf`;
  - binary SHA256:
    `f6d6828290c330749f1523c191c3e4034759f97d7c53e0ad4948d7e786995b1b`;
  - `serve_selected_graph_mode=legacy_batched_decode_graph`,
    `serve_selected_max_sequences=16`;
  - chat smoke content `5`;
  - c16 diagnostic bench rc `0`,`completed=16`,`errored=0`,
    output throughput mean `348.0 tok/s`;
  - replay evidence in server log:
    - capture `m=15 m_padded=16 device_shadow=true`;
    - post-capture replay count `1`;
    - pure replay counts `2,4,8,16` on the same `m_padded=16` graph;
    - an additional drain-shape capture `m=7 m_padded=8 device_shadow=true`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Interpretation:
  - 现在已证明 Gemma3 device-shadow product path 真实进入 legacy batched
    CUDA graph capture/replay;
  - 单次 c16 throughput 没有稳定越过 80%,说明 graph launch overhead 不是
    剩余唯一主瓶颈;
  - 观测到 replay 主 shape 是 `m=15` 而非满 `m=16`,且 drain shape 会额外
    capture;下一步应回到 W2-P2 的剩余两条:batch cadence/TTFT 与
    sustained decode tail MLP/Marlin 投影成本。
- Release status:
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍不是 release-grade。

## 2026-06-15 LVIII — W2 CUDA checkpoint: batched graph product path enabled, c16 diagnostic improves but remains below 80%

- 本轮源码 checkpoint:
  `2b3b5891 perf(cuda): expose batched decode graph policy`。
- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_batched_graph_product_smoke_2026-06-15/`。
- GPU 执行合同:
  - lane:`W2 Gemma3 batched decode graph product correctness smoke`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:15-25min,hard cap 30min;
  - stop condition:`ferrum run` + `ferrum serve` smoke 通过、首个明确失败并收集
    artifact、或 30min hard cap;
  - correctness gate:CUDA release build,`ferrum run` known-answer,
    `ferrum serve` OpenAI chat smoke,effective config/decision trace 证明
    `decode_graph_policy=legacy_batched_decode_graph`;
  - performance command:diagnostic-only c16 `bench-serve --fail-on-error --seed 9271`,
    `n_repeats=1`,不是 release 性能证据。
- 源码/产品路径:
  - `ferrum run` 与 `ferrum serve` 均新增公开
    `--batched-graph/--disable-batched-graph`;
  - config file 支持 `runtime.batched_graph`;
  - auto-config 新增 `decode_graph_policy` decision,并 materialize
    `FERRUM_BATCHED_GRAPH`;
  - legacy batched decode graph 仍禁止 host residual shadow,但允许 Gemma3
    device residual shadow,并使用独立 graph key namespace。
- 本地验证:
  - `cargo test -p ferrum-types batched_graph -- --nocapture`;
  - `cargo test -p ferrum-cli batched_graph -- --nocapture`;
  - `cargo test -p ferrum-cli runtime_cli_config_emits_config_file_source_entries -- --nocapture`;
  - `cargo test -p ferrum-cli batched_graph_cli_override_records_flag_state -- --nocapture`;
  - `cargo test -p ferrum-models batched_decode_graph --lib -- --nocapture`;
  - `cargo test -p ferrum-types m3_preset_selects_current_safe_fast_path_without_fa2 -- --nocapture`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`。
- CUDA product smoke:
  - remote HEAD:`2b3b5891ff94d6a4d793bb39bd6cab148af49588`;
  - binary SHA256:
    `c31d8b4af03f4669f7fac4fc49035adff97ca4d680d80775703aff99474b3d33`;
  - model path:
    `/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2`;
  - `ferrum run` rc `0`,content `5`,
    `selected_graph_mode=legacy_batched_decode_graph`;
  - `ferrum serve` ready,OpenAI chat content `5`,
    `selected_graph_mode=legacy_batched_decode_graph`,
    `selected_max_sequences=16`;
  - c16 diagnostic bench rc `0`,`16 completed / 0 errored`,
    `372.3 tok/s`;
  - log scan found no actual panic/CUDA error/illegal address/OOM/`<unk>`/`[PAD]`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Interpretation:
  - 产品入口已能用公开 typed path 打开 batched decode graph policy;
  - c16 diagnostic 从 prompt-token admission checkpoint 的 `344.7 tok/s`
    提升到 `372.3 tok/s`,但与同机 vLLM c16 `518.8 tok/s` 相比仍约
    `71.8%`,没有达到 80%;
  - 当前 artifact 证明 policy 生效和产品 correctness smoke 通过,但没有明确
    记录 graph replay counter;下一步应补最小 replay 可观测性,再用同一 c16
    diagnostic 验证 replay 是否确实发生。
- Release status:
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍不是 release-grade。

## 2026-06-15 LVII — W2 native checkpoint: Gemma3 shadow graph native probe PASS

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_gemma3_shadow_graph_native_probe_2026-06-15/`。
- 源码 checkpoints:
  - `23d8569b test(cuda): add gemma3 shadow graph probe`;
  - `e927e4c1 test(cuda): make shadow graph probe relocatable`;
  - `c46d9540 test(cuda): keep shadow graph probe finite`。
- GPU 执行合同:
  - lane:`W2 Gemma3 shadow graph native CUDA probe`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:5-10min,hard cap 20min;
  - stop condition:Vast start/SSH/CUDA/nvcc compile/probe 首败,或 probe
    artifact 复制完成后立即停机;
  - correctness gate:`nvcc` 编译 rc `0`,probe rc `0`,checksum finite,
    stdout 含 `VERDICT: Gemma3 shadow graph native CUDA probe complete`;
  - performance command:`bash scripts/microbenches/build_and_run_gemma3_shadow_graph_bench.sh`,
    diagnostic-only,不是 release 性能证据。
- 执行环境:
  - remote HEAD:`c46d95408d12c8c1e177145f7c4c217a34080e62`;
  - GPU:`NVIDIA GeForce RTX 4090`,24564 MiB,driver `565.77`;
  - CUDA compiler:`Build cuda_12.4.r12.4/compiler.34097967_0`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Native probe result:
  - simulated shape:Gemma3-style device F32 residual shadow decode,
    `62` layers,`batch=16`,`hidden=5376`,`498` kernel launches/step;
  - eager ordered state upload:`1.143 ms/step`;
  - eager pre-sync state upload:`1.137 ms/step`;
  - graph ordered state upload:`0.565 ms/step`,`2.02x` vs eager ordered;
  - graph pre-sync state upload:`0.568 ms/step`,`2.00x` vs eager pre-sync;
  - checksum16:`127.94618988`;
  - verdict line present。
- Interpretation:
  - native CUDA 层面已证明 Gemma3 device-shadow-like decode step 可以稳定
    graph capture/replay,并且 launch-overhead headroom 约 `2x`;
  - 这不是产品性能修复;当前 Ferrum product path 仍在 host/device residual
    shadow 路径禁用 legacy batched graph;
  - 下一步应做窄产品改动:shadow-safe graph eligibility/guard,然后先跑
    `ferrum run` + `ferrum serve` correctness smoke,再做 c16 diagnostic。
- Release status:
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍不是 release-grade。

## 2026-06-15 LVI — W2 source checkpoint: add Gemma3 shadow graph native probe

- 本轮没有启动 GPU,不产生性能结论;这是针对 decode integration/graph
  瓶颈的源码和诊断工具 checkpoint。
- 实质定位:
  - Gemma3 sandwich-norm CUDA 正确性路径依赖 device F32 residual shadow;
  - legacy batched decode graph 入口当前要求
    `FERRUM_BATCHED_GRAPH && !host_residual_shadow && !device_residual_shadow`,
    因此目标 Gemma3 路径即使设置 `FERRUM_BATCHED_GRAPH=1` 也不会进入
    batched graph replay;
  - 这解释了为什么继续扫 graph env 开关不能作为有效性能定位手段。
- 新增最小 native CUDA probe:
  - `scripts/microbenches/gemma3_shadow_graph_bench.cu`;
  - `scripts/microbenches/build_and_run_gemma3_shadow_graph_bench.sh`;
  - probe 模拟 Gemma3-style `62` 层、`batch=16`、device F32 residual
    shadow update 的 decode step,比较 eager launch 与 graph replay;
  - 不加载模型,不跑产品 entrypoint,不改变默认产品路径。
- 本地验证:
  - `git diff --check -- scripts/microbenches/gemma3_shadow_graph_bench.cu scripts/microbenches/build_and_run_gemma3_shadow_graph_bench.sh scripts/microbenches/README.md` PASS;
  - `bash -n scripts/microbenches/build_and_run_gemma3_shadow_graph_bench.sh`
    PASS;
  - 本地无 `nvcc`,CUDA compile/run 待在已有 1x4090 cache-retained instance 上
    执行单 probe。
- 下一步:
  - 只启动已有 4090 实例跑该 native probe,保存 `VERDICT: Gemma3 shadow graph
    native CUDA probe complete`、stdout、GPU metadata 和 shutdown 记录;
  - 若 native graph replay 稳定且有足够 headroom,再设计产品侧 shadow-safe
    graph eligibility/guard;若不稳定,转向 tail MLP launch/copy fusion,不做
    产品默认 graph 改动。
- Release status:
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍不是 release-grade。

## 2026-06-15 LV — W2 CUDA checkpoint: prompt-token admission 默认路径正确但不是主瓶颈

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_prompt_token_admission_c16_ab_2026-06-15/`。
- 源码 checkpoint:
  `2f732131 perf(scheduler): default to prompt-token admission`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA prompt-token admission c16 A/B diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD 0.425/hr;
  - expected runtime/cost:10-20min,hard cap 30min;
  - stop condition:start/SSH/CUDA/clean worktree/build/product smoke/c16
    bench 首败,或 artifact 复制完成后停机;
  - correctness gate:`ferrum run` known-answer smoke plus `ferrum serve`
    chat smoke with usage and zero benchmark errors;
  - performance command:diagnostic-only `bench-serve` ShareGPT c16,
    `--num-prompts 16 --n-repeats 1 --fail-on-error --seed 9271`。
- 执行环境:
  - remote clean worktree HEAD
    `2f73213181475ba4bdff3e907e45182c24981a0e`;
  - remote `git status --short` 为空;
  - binary SHA256
    `551f83921ea1fb6eb0cfb75170fc2325e31d887530ba084ab72ef77b238ebaf0`;
  - Vast shutdown verified:`cur_state=stopped`,
    `actual_status=exited`。
- Correctness:
  - CUDA release build rc `0`;
  - `ferrum run` rc `0`,answer `5`,n_tokens `3`;
  - `ferrum serve` chat smoke rc `0`,content `5`,usage
    `prompt_tokens=23`,`completion_tokens=3`;
  - `bench-serve --fail-on-error` rc `0`,c16 completed `16/16`,
    `0` errored,`0` bad_output,output token count source `usage`;
  - both run and serve decision traces selected
    `scheduler_admission_policy=prompt_token_estimate` from default。
- Performance diagnostic:
  - prior same-host Ferrum c16 natural ShareGPT baseline:
    `340.003 tok/s`,p50 TTFT `887.683ms`,p50 TPOT `32.817ms`;
  - new default prompt-token admission c16:
    `344.714 tok/s`,p50 TTFT `931.776ms`,p50 TPOT `31.592ms`;
  - delta vs Ferrum baseline:`+1.39%`,single-run diagnostic only;
  - ratio vs same-host vLLM c16 `518.796 tok/s`:`66.4%`,
    still well below W2 80% line。
- Conclusion:
  - 默认 prompt-token admission 是正确的产品默认修复,并由 decision trace
    证明已生效;
  - 但它不是当前 c16 性能主瓶颈;不要继续围绕 admission/env flip
    做 sweep;
  - 下一步回到已经定位的 decode/Marlin tail MLP,尤其是 gate_up/down
    投影与每步 integration 开销。
- Artifact note:
  - remote driver 的最后 summary helper 对 single-c `bench-serve` JSON schema
    假设错误,benchmark 完成后触发 `KeyError: 0`;
  - build/run/smoke/bench rc 均已是 `0`,`summary.json` 由 bench JSON
    重新生成,`run.status` 记录为
    `PASS_CORE_WITH_POSTPROCESS_WARNING`。
- Release status:
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍不是 release-grade。

## 2026-06-15 LIV — W2 source checkpoint: default scheduler admission uses prompt-token metadata

- 本轮没有启动 GPU,不产生性能结论;这是针对 c16 TTFT/prefill/scheduler
  半边差距的源码 checkpoint。
- 源码改动:
  - `SchedulerConfig::default()` 现在默认
    `prompt_token_estimate=true`;
  - `#[serde(default)]` 改为显式 true default,避免配置文件省略字段时回到
    bool 的 false;
  - auto-config 默认 decision trace 现在选择
    `scheduler_admission_policy=prompt_token_estimate`;
  - 显式 `FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE=0` 仍可回到
    `continuous_default`,且 trace 保留该 source key;
  - scheduler 测试保留显式禁用路径,验证旧 admission 行为仍可复现。
- 为什么这个 checkpoint 有意义:
  - 产品 `ferrum run` / `ferrum serve` 已在提交 scheduler 前写入真实
    `ferrum_prompt_tokens`;
  - 旧默认 false 会让初始 prefill admission 用 `prefill_chunk_size=512`
    粗估,在 `max_num_batched_tokens=2048` 下 c16 短 prompt 首批只能进
    约 4 个请求;
  - 默认 true 后,短 prompt admission 会按真实 prompt token 计入预算,目标是
    降低 c16 TTFT/prefill 排队部分,不是直接改 decode kernel。
- 本地验证:
  - `cargo fmt --all -- --check`;
  - `cargo test -p ferrum-types scheduler_config -- --nocapture`;
  - `cargo test -p ferrum-types scheduler_prompt_token_estimate -- --nocapture`;
  - `cargo test -p ferrum-scheduler prompt_token_metadata -- --nocapture`;
  - `cargo test -p ferrum-types engine_config_default_sane -- --nocapture`。
- 下一步:
  - 只做最小 CUDA c16 A/B/product smoke,验证默认 prompt-token admission
    是否实际降低 TTFT/提升 c16 ratio;
  - 若没有明显收益,继续回到已定位的 decode/Marlin tail MLP 半边。
- Correctness/performance status:
  - 本地配置和 scheduler 单元测试通过,没有发现新的 correctness blocker;
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`,W2 仍不是
    release-grade。

## 2026-06-15 LIII — W2 bottleneck synthesis: c16 gap is split between TTFT/prefill and sustained decode

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_bottleneck_synthesis_2026-06-15/summary.md`。
- 本轮没有新增 GPU run;这是对 2026-06-15 已有 c16 evidence 的综合。
- 使用的主要证据:
  - Ferrum/vLLM ShareGPT c16/c32 baseline:
    `w2_vllm_sharegpt_baseline_probe_2026-06-15`;
  - Ferrum c16 `[batched-op-profile]`:
    `w2_tail_profile_buckets_2026-06-15`;
  - Ferrum Marlin projection split:
    `w2_marlin_projection_profile_2026-06-15`;
  - native Ferrum/vLLM Marlin weight-cycle:
    `w2_dense_vllm_marlin_weight_cycle_probe_2026-06-15`;
  - product Marlin shape trace:
    `w2_marlin_shape_trace_probe_2026-06-15`。
- Same-hardware c16 端到端差距:
  - Ferrum:`340.003 tok/s`,`5.328 req/s`;
  - vLLM:`518.796 tok/s`,`8.106 req/s`;
  - ratio:`65.5%`,仍低于 W2 80% 目标。
- latency 拆分:
  - Ferrum p50 TTFT `887.683ms`,vLLM p50 TTFT `411.903ms`,
    差 `+475.780ms`;
  - Ferrum p50 TPOT `32.817ms`,vLLM p50 TPOT `24.789ms`,
    差 `+8.027ms/token`;
  - 以约 63 个 inter-token gap 估算,TPOT 差距约 `506ms`;
  - c16 batch wall-time 差约 `16/5.328 - 16/8.106 = 1.03s`;
  - TTFT + TPOT 基本解释该差距,说明剩余性能问题不是一个单点内核。
- batch/cadence 结论:
  - 既有 decode batch stats:c16 段 `calls=391`,
    `total_items=5334`,`avg_m=13.642`,`max_m=16`;
  - `w2_tail_profile_buckets` 中 batch `m=16` 有 `118` 行;
  - 因此当前主问题不是“c16 batch 完全没有形成”。
- decode breakdown:
  - `w2_tail_profile_buckets` batch `m=16` mean decode step
    `28.037ms`;
  - `tail_mlp` `13.744ms` (`49.0%`),`matmul` `6.971ms`
    (`24.9%`),`attention` `2.406ms` (`8.6%`),
    `unwrapped` `0.649ms` (`2.3%`);
  - `w2_marlin_projection_profile` batch `m=16` with Marlin profiling:
    Marlin kernels `16.548ms` (`55.0%`),其中 `gate_up`
    `8.728ms`,`down` `4.352ms`,`qkv` `2.132ms`,`o_proj`
    `1.336ms`。
- Current bottleneck statement:
  - c16 batching 大体有效,但平均 batch 仍低于 16,尾段会 drain;
  - 端到端差距约一半来自 TTFT/prefill/scheduling,一半来自持续
    TPOT/decode;
  - decode 内部主要是 Gemma3 tail MLP / Marlin projection 时间;
  - native Ferrum/vLLM Marlin weight-cycle 已经排除“直接换 dense Marlin
    单核即可解决”的主假设。
- 下一步:
  - first-token 侧:查 Ferrum c16 ShareGPT 是否串行/弱批处理 prefill,
    以及 chunked/batched prefill 是否能降低 TTFT;
  - decode 侧:继续从 native CUDA 最小 probe 入手,针对 Gemma3
    tail MLP 的 gate_up/down 调度、activation、residual/norm 边界找可
    落地优化。
- Correctness/performance status:
  - 没有新的 correctness blocker;
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`,W2 仍不是
    release-grade。

## 2026-06-15 LII — W2 CUDA checkpoint: product Marlin shape trace wired and single-request decode is m=1

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_shape_trace_probe_2026-06-15/`。
- 源码 checkpoint:
  `b3403dd5 test(cuda): add marlin shape trace probe`。
- GPU 执行合同:
  - lane:`W2 Marlin shape-trace compile/product smoke`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD 0.425/hr;
  - expected runtime/cost:10-20min,hard cap 30min;
  - stop condition:start/SSH/CUDA/source sync/CUDA feature test/build/server/chat
    smoke 首败,或 trace 输出 + artifact 复制后停机;
  - correctness gate:CUDA/marlin feature compile retry rc `0`,release build rc
    `0`,`ferrum serve` ready,non-stream chat smoke 有内容和 usage;
  - performance command:无 release perf;本轮只做 diagnostic product trace。
- 执行结果:
  - remote HEAD:`b3403dd5394bb044690c918535a71ccc202cd3e7`;
  - release CUDA build rc `0`,binary SHA256
    `730df7d84ede559b7ace54abcf1a6c16a3a81e55113789c5e3bc37c9f3844b8f`;
  - `run.status=PASS`;
  - chat smoke 返回 `5`,usage 为 `prompt_tokens=23`,
    `completion_tokens=3`;
  - artifact 已复制回本地;`vast_shutdown/stopped.json` 记录
    `cur_state=stopped actual_status=exited`。
- shape trace 结果:
  - `shape_trace_lines=256`,全部可解析;
  - calls `0..247`:prefill,`m=23`,62 层 × 4 个 Marlin projection;
  - calls `248..255`:decode,`m=1`,trace cap 前 2 层 × 4 个 projection;
  - projection shapes:
    `qkv n=8192 k=5376`,`o n=5376 k=4096`,
    `gate_up n=43008 k=5376`,`down n=5376 k=21504`。
- Interpretation:
  - trace 已接入真实 `ferrum serve` 产品路径,不是 standalone
    microbench;
  - 单请求 decode 明确是 `m=1`;这本身符合预期,但还不能解释 c16
    端到端差距;
  - 结合上一轮 native Ferrum/vLLM Marlin weight-cycle probe,当前主假设应
    继续收敛到 c16 decode integration/cadence:并发 decode 是否稳定形成
    `m≈16`,是否存在 scheduler gap、非 Marlin op 或 per-step sync。
- Correctness/performance status:
  - 该窄 smoke 没有发现新的 correctness blocker;
  - `decode_op_profile.log` 本轮为空,不能据此做非 Marlin 时间结论;
  - 没有生成 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`,W2 仍不是
    release-grade;
  - 最新可引用性能仍是 Ferrum c16 `341.24 tok/s` vs vLLM c16
    `518.80 tok/s`,约 `65.8%`。

## 2026-06-15 LI — W2 source checkpoint: add Marlin shape trace for decode integration probe

- 本轮代码改动:
  `crates/ferrum-kernels/src/backend/cuda/marlin.rs`。
- 背景:
  - dense Marlin kernel swap 已被 native hot/weight-cycle A/B 排除;
  - W2 c16 剩余差距应继续追 decode integration/scheduling,而不是继续
    跑 full sweep 或 blind kernel flip;
  - 已有 `FERRUM_MARLIN_PROFILE=1` 能聚合 projection bucket 时间,但缺少
    每次 Marlin dispatch 的 shape/pointer/label trace。
- 改动:
  - 新增默认关闭的 `FERRUM_MARLIN_TRACE_SHAPES=1`;
  - 新增 `FERRUM_MARLIN_TRACE_SHAPES_MAX=<N>`,默认最多 `256` 条;
  - trace 行记录 call id,当前 CUDA alloc label,bucket,m/n/k,group size,
    qweight/scales/workspace len,以及 A/B/C/scales/workspace device pointer;
  - 仅在显式 env 打开时输出,不改变默认产品路径。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-kernels cuda_marlin_runtime_config -- --nocapture`
    PASS,但默认 feature 下未命中 CUDA-gated Marlin tests;
  - `cargo test -p ferrum-kernels --features cuda,marlin cuda_marlin_runtime_config -- --nocapture`
    在本机失败,原因是本地无 `nvcc`/`nvidia-smi`,不是代码测试失败。
- 下一步:
  - 在 1x4090 上做最小 CUDA 编译验证,并用
    `FERRUM_MARLIN_TRACE_SHAPES=1 FERRUM_MARLIN_TRACE_SHAPES_MAX=128`
    跑短 product diagnostic,确认 trace 输出可解析且不引入 correctness
    回归。

## 2026-06-15 L — W2 CUDA checkpoint: vLLM dense Marlin weight-cycle 也落在同一瓶颈带

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_vllm_marlin_weight_cycle_probe_2026-06-15/`。
- 源码 checkpoint:
  `2b6e1922 test(cuda): add vllm marlin weight cycle probe`。
- GPU 执行合同:
  - lane:`W2 dense vLLM Marlin weight-cycle native probe`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD 0.425/hr;
  - expected runtime/cost:5-12min,hard cap 20min;
  - stop condition:start/SSH/CUDA/source sync/compile/probe 首败,或 verdict
    后复制 artifact 并停机;
  - correctness gate:native compile success + probe rc `0` + verdict line;
  - performance command:diagnostic-only
    `bash scripts/microbenches/build_and_run_dense_vllm_marlin_gemma3_perf.sh`,
    只比较 hot vs weight-cycle,不产生 release 性能声明。
- 执行结果:
  - remote HEAD:`2b6e192205f01ef9106a7c12dbce38198c2584a3`;
  - probe rc `0`,`run.status=PASS`;
  - stdout 含
    `VERDICT: dense vLLM Marlin native CUDA probe complete`;
  - artifact 复制回本地后已停机;`vast_shutdown/cleanup_check.txt`
    记录 `cur_state=stopped actual_status=exited`。
- m=16 A/B/weight-cycle 关键数据:
  - qkv: Ferrum hot `17.005us`,Ferrum weight-cycle `30.278us`,
    vLLM hot `18.315us`,vLLM weight-cycle `30.950us`;
  - gate_up: Ferrum hot `133.715us`,Ferrum weight-cycle `133.985us`,
    vLLM hot `136.988us`,vLLM weight-cycle `137.524us`;
  - down: Ferrum hot `30.356us`,Ferrum weight-cycle `68.651us`,
    vLLM hot `36.027us`,vLLM weight-cycle `69.268us`。
- Interpretation:
  - down projection 的 vLLM dense Marlin hot 看起来快,但一旦模拟 product
    跨层权重轮换,它同样落到 `~69us`,与 Ferrum default weight-cycle
    只差约 `1%`;
  - gate_up 在 hot/weight-cycle 下基本不变,说明它是 compute-bound 大投影;
  - 因此当前剩余 c16 端到端差距不应继续押注 dense Marlin kernel swap;
  - 下一步应转向 decode integration/scheduling:每 token launch 数、非
    Marlin 时间、batch cadence、host/device sync、以及 vLLM 是否在请求调度
    或 decode loop 层面减少了空转/间隙。
- Correctness/performance status:
  - 当前窄产品 correctness 仍无新增 blocker;
  - Ferrum c16 最新 product diagnostic 仍约 `341.24 tok/s`,vLLM same-hardware
    c16 `518.80 tok/s`,约 `65.8%`;
  - W2 仍无 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。

## 2026-06-15 XLIX — W2 source checkpoint: extend vLLM dense Marlin probe with weight-cycle mode

- 本轮代码改动:
  `scripts/microbenches/dense_vllm_marlin_gemma3_perf.cu`。
- 背景:
  - XLVIII 的 native vLLM dense Marlin probe 已经排除 gate_up 上的
    kernel-swap 主假设;
  - 但 down projection 的 product profile 接近 Ferrum default
    weight-cycle,而 XLVIII 只测了 vLLM hot repeated weight;
  - 因此不能只用 vLLM hot down `36.277us` 与 Ferrum weight-cycle
    `68.651us` 做最终判断。
- 改动:
  - 为每个 Gemma3 qkv/gate_up/down shape 和 `m=16/23/32` 同时输出
    `hot` 与 `weight_cycle`;
  - `weight_cycle` 使用 8 组 synthetic qweight/scales/workspace 轮换,
    模拟 product decode 跨层权重切换;
  - 仍是 native CUDA probe,不加载模型,不改变 product dense GPTQ routing。
- 验证状态:
  - 本 checkpoint 未启动 GPU,不产生性能结论;
  - 下一次 paid CUDA 只跑同一 build script,用于确认 vLLM dense Marlin
    down projection 在 weight-cycle 下是否仍明显优于 Ferrum default。

## 2026-06-15 XLVIII — W2 CUDA checkpoint: native vLLM dense Marlin A/B 排除 kernel-swap 主假设

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_vllm_marlin_native_probe_retry_2026-06-15/`。
- 源码 checkpoint:
  - `09734267 test(cuda): add dense vllm marlin gemma probe`;
  - `5ce9299e fix(cuda): enable dense vllm marlin probe selector`。
- GPU 执行合同:
  - lane:`W2 dense vLLM Marlin native same-shape A/B probe`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD 0.425/hr;
  - expected runtime/cost:5-12min,hard cap 20min;
  - stop condition:start/SSH/CUDA/source sync/compile/probe 首败,或
    `VERDICT: dense vLLM Marlin native CUDA probe complete` 后复制 artifact
    并停机;
  - correctness gate:native compile success + probe rc `0` + verdict line;
  - performance command:diagnostic-only
    `bash scripts/microbenches/build_and_run_dense_vllm_marlin_gemma3_perf.sh`,
    不产生 release 性能声明。
- 执行结果:
  - first attempt artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_vllm_marlin_native_probe_2026-06-15/`;
  - first attempt rc `134`,原因是 build script 的 temporary `perl`
    selector include 替换没有命中,临时 `marlin.cu` 仍保留注释 include;
  - 修正脚本后 retry rc `0`,`run.status=PASS`;
  - stdout 含
    `VERDICT: dense vLLM Marlin native CUDA probe complete`;
  - artifact 复制回本地后已停机;`vast_shutdown/cleanup_check.txt`
    记录 `cur_state=stopped actual_status=exited`。
- m=16 native A/B 关键数据:
  - qkv: Ferrum hot `17.005us`,Ferrum weight-cycle `30.278us`,
    vLLM dense Marlin `18.354us`;
  - gate_up: Ferrum hot `133.715us`,Ferrum weight-cycle `133.985us`,
    vLLM dense Marlin `136.581us`;
  - down: Ferrum hot `30.356us`,Ferrum weight-cycle `68.651us`,
    vLLM dense Marlin `36.277us`。
- Product profile 对照:
  - c16 profile batch `16`;
  - product per-layer gate_up kernel 约 `140.77us`,与 native Ferrum/vLLM
    gate_up 基本同量级;
  - product per-layer down kernel 约 `70.20us`,接近 Ferrum weight-cycle
    `68.651us`,而不是 Ferrum hot `30.356us`。
- Interpretation:
  - “直接换 vLLM dense Marlin kernel 能补齐 c16 14 个百分点差距”这个
    主假设已被本轮 native A/B 排除;
  - 当前更可信的瓶颈方向是 product 集成侧的 weight residency/cache-cycle,
    down projection 的权重访问状态,以及 decode launch/host scheduling
    组合开销;
  - 下一步不跑 full sweep,应做最小产品/原生关联 probe:在 decode loop
    里记录 projection 权重地址/调用顺序/stream sync 与 down projection
    cache-cycle 状态,确认为什么 product down 落在 weight-cycle 而不是 hot
    kernel 轨道。
- Release-grade status:
  - 这是诊断证据,不是 release gate;
  - W2 仍无 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。

## 2026-06-15 XLVII — W2 source checkpoint: add dense vLLM Marlin native A/B probe

- 本轮代码改动:
  - `scripts/microbenches/dense_vllm_marlin_gemma3_perf.cu`;
  - `scripts/microbenches/build_and_run_dense_vllm_marlin_gemma3_perf.sh`;
  - `scripts/microbenches/README.md`。
- 目的:
  - 当前 c16 产品路径正确性窄门干净,但 unified prefill 与 vLLM paged
    attention 诊断均没有提升吞吐;
  - 已有 op profile 将主要热区压缩到 dense GPTQ decode MLP/Marlin,
    尤其 gate_up/down;
  - 下一步用 native CUDA 同形状 A/B 直接测 vendored vLLM dense GPTQ-Marlin
    在 Gemma3 qkv/gate_up/down 形状上的核耗时,避免继续启动整套
    Cargo/product gate 做盲目验证。
- Probe 设计:
  - 直接调用 vendored `ferrum_marlin_mm_f16_u4b8` C ABI;
  - 覆盖 Gemma3-27B GPTQ 关键 dense 形状:
    `qkv k=5376 n=8192`,`gate_up k=5376 n=43008`,
    `down k=21504 n=5376`;
  - 覆盖 decode 常见 m 值 `16/23/32`;
  - companion build script 只在 `/tmp` 临时副本中打开 minimal
    `kernel_selector.h`,不改变产品 dense GPTQ routing。
- 验证状态:
  - 本 checkpoint 未启动 GPU,不产生性能结论;
  - 下一次 paid CUDA 只需运行
    `bash scripts/microbenches/build_and_run_dense_vllm_marlin_gemma3_perf.sh`
    并保存 `VERDICT: dense vLLM Marlin native CUDA probe complete`;
  - 如果同形状 vLLM dense Marlin 明显快于当前 `dense_marlin_gemma3_perf`
    的 hot/weight-cycle 数据,再考虑产品侧 selector/核接入;否则继续转向
    host scheduling/weight residency/launch overhead。

## 2026-06-15 XLVI — W2 CUDA checkpoint: Gemma3 unified prefill c16 诊断通过但性能仍约 65.8% vLLM

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_unified_prefill_c16_diag_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090,
  约 USD 0.425/hr。artifact 复制回本地后已停机;`vast_shutdown/cleanup_check.txt`
  记录 `cur_state=stopped actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA unified-prefill c16 diagnostic`;
  - expected runtime/cost:10-20min,hard cap 30min,约 USD 0.425/hr;
  - stop condition:instance start/SSH/source sync/server readiness 首败,
    chat smoke 首败,`bench-serve` 首败,malformed/missing usage 首败,或
    c16 diagnostic 完成后复制 artifact 并停机;
  - correctness gate:`ferrum serve` readiness + non-stream chat smoke +
    `bench-serve --fail-on-error`;
  - performance command:diagnostic-only
    `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --concurrency-sweep 16 --num-prompts 16`,
    no `--require-ci`,不产出 release 性能结论。
- Build/runtime evidence:
  - remote HEAD:`d5f82822b56527b47a9d3884639fe737cbb37570`;
  - CUDA release build rc `0`;
  - binary SHA256:
    `4ebf50b5c64a5f72d929e1aeaefde61b0f1bb9ec6fed9dd0d84596f5b803be89`;
  - server log contains
    `Gemma3 family: legacy batched_decode=true varlen_unified=true`;
  - remote git status contains only historical artifact deletion noise from
    rsync artifact exclusion;no non-artifact source dirty rows were present.
- Correctness:
  - `run.status=PASS`,`bench-serve.rc=0`;
  - non-stream chat smoke returned content `"5\n"`,finish_reason `stop`,
    completion_tokens `3`,bad_output `false`;
  - c16 ShareGPT diagnostic reported `completed_per_run=[16]`,
    `errored_per_run=[0]`,`bad_output_per_run=[0]`,
    `zero_output_tokens_per_run=[0]`,`malformed_stream_per_run=[0]`,
    `missing_done_per_run=[0]`,`duplicate_done_per_run=[0]`,
    `panic_per_run=[0]`,`http_500_per_run=[0]`;
  - `output_token_count_source="usage"`;
  - log/bad-marker scan found no product correctness blocker;this checkpoint
    does not replace the full W2 L0-L5 release gate.
- Performance diagnostic:
  - Ferrum unified-prefill c16: `341.2397 tok/s`;
  - same-hardware vLLM c16 baseline:
    `518.7960 tok/s`;
  - Ferrum/vLLM ratio: `65.78%`;
  - previous Ferrum c16 baseline was `340.0029 tok/s`,so enabling
    Gemma3 unified path did not materially close the gap.
- Interpretation:
  - current known correctness state is clean for this narrow product-path
    diagnostic;
  - the main W2 blocker is now performance:still about `14.2` percentage
    points below the 80% vLLM target at c16;
  - next high-value step is not another full sweep;use a minimal targeted
    profile/native CUDA probe to locate the remaining bottleneck, likely in
    decode/projection/attention scheduling rather than the unified-prefill
    enablement guard itself.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 XLV — W2 CUDA checkpoint: Gemma3 unified tail 产品 smoke 通过

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_gemma_unified_tail_cuda_smoke_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;`vast_shutdown/cleanup_check.txt` 记录
  `cur_state=stopped actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA unified-tail product correctness smoke`;
  - expected runtime/cost:10-25min,hard cap 35min,约 USD 0.425/hr;
  - stop condition:instance start/SSH/CUDA/source sync/build 首败、`ferrum run`
    首败、`ferrum serve` smoke 首败、乱码/缺 usage/stream `[DONE]` 等首败,
    或 smoke PASS 后复制 artifact 并停机;
  - correctness gate:CUDA release build + `ferrum run` + `scripts/model_coverage_smoke.sh`;
  - performance command:none;本轮不产出性能结论。
- Build:
  - remote HEAD:`ab0dc99cdc71345e236513dbe0300ce52b162416`;
  - `cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`
    PASS,`cargo_build.rc=0`;
  - binary SHA256:
    `4ebf50b5c64a5f72d929e1aeaefde61b0f1bb9ec6fed9dd0d84596f5b803be89`;
  - remote git status has only expected artifact noise from rsync artifact exclusion。
- Product correctness:
  - `run.status=PASS`;
  - `ferrum run gemma3:27b-gptq --backend cuda ...` rc `0`;
  - run validation:assistant content `"5"`,finish_reason `stop`,n_tokens `3`,
    bad_output `false`;
  - run stderr contains
    `Gemma3 family: legacy batched_decode=true varlen_unified=true`;
  - `scripts/model_coverage_smoke.sh gemma3:27b-gptq --port 8491 --kv-capacity 2560 --max-seqs 2`
    rc `0`;
  - serve stdout contains `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`;
  - serve log contains
    `Gemma3 family: legacy batched_decode=true varlen_unified=true`;
  - validation scan did not find `panic`,`CUDA_ERROR`,`<unk>`,or `[PAD]` markers.
- Interpretation:
  - Gemma3 unified tail source change is now product-smoke validated on CUDA
    for both `ferrum run` and `ferrum serve`;
  - this is still a smoke checkpoint,not full L0-L5 release evidence and not
    a performance claim;
  - next high-value correctness/perf checkpoint is a tiny c16 ShareGPT
    diagnostic with `--fail-on-error`,checking that fresh prefill no longer
    falls back to serial per-item profiles and measuring the new gap vs vLLM。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 XLIV — W2 source checkpoint: Gemma3 unified tail 接入 sandwich/F32 residual 语义

- 本轮没有启动 GPU;在上一个 native CUDA window probe 通过后,继续补
  Gemma3 unified prefill 的源码正确性前置件。
- Source change:
  - unified mixed-batch scratch 增加 `unified_sandwich_tmp`,
    `unified_residual_f32_shadow`, `unified_sandwich_branch_f32`;
  - `ensure_unified_scratch` 会在 sandwich-norm family 且 backend 支持
    device-side F32 residual shadow 时分配 unified F32 residual/branch buffer;
  - `unified_forward_internal` 在 embedding 后把 activation residual 写入
    F32 residual shadow,并在返回前恢复 shadow scratch;
  - `unified_forward_layer` 现在对 sandwich layer:
    - input RMSNorm 从 F32 residual shadow 读;
    - post-attn path 执行 `rms_norm(o_proj_out, post_attn_ln_w)` 后加到
      F32 residual,再对 F32 residual 做 `post_ln_w` pre-MLP norm;
    - gated activation 按 `Activation::GeluTanh` 走 GeGLU,否则走 SwiGLU;
    - post-ffn path 执行 `rms_norm(mlp_out, post_ffn_ln_w)` 后加到
      F32 residual;
    - final norm 从 F32 residual shadow 读;
  - `unified_varlen_qkv_unsupported_reason` 现在把 Gemma3 unified prereq
    明确为:backend varlen QKV + local/global layer pattern +
    device-side F32 residual shadow。
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo fmt --all -- --check` PASS;
  - `cargo check -p ferrum-models` PASS;
  - `cargo test -p ferrum-models unified_varlen_qkv_requires_gemma_sandwich_prerequisites --lib`
    PASS;
  - `cargo test -p ferrum-models llama_attention_semantics_cover_qk_mode_and_layer_windows --lib`
    PASS;
  - `git diff --check` PASS。
- Correctness status:
  - this is not release evidence and not a product correctness PASS;
  - source now has the missing Gemma3 unified tail semantics,so the next
    checkpoint must run CUDA product smoke (`ferrum run` and `ferrum serve`)
    before trusting the newly-enabled unified path;
  - if CUDA smoke fails, stop at the failing artifact and do minimal triage,
    not a full perf sweep。
- Performance status:
  - no performance command in this checkpoint;
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains functional / known-gap,not release-grade。
- Next step:
  - reuse the cached 1x4090 only once for a minimal CUDA correctness smoke:
    CUDA build, `ferrum run` Paris/multi-turn smoke, `ferrum serve` non-stream
    and streaming smoke, then a small c16 diagnostic only if correctness is clean。

## 2026-06-15 XLIII — W2 native checkpoint: paged varlen sliding-window CUDA probe 通过

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_paged_varlen_window_native_probe_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090,
  `nvidia/cuda:12.4.0-devel-ubuntu22.04`,driver `565.77`,CUDA compiler
  `12.4`。验证结束后已复制 artifact 并停机;`vast_shutdown/cleanup_check.txt`
  记录 `cur_state=stopped actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA paged-varlen sliding-window native correctness probe`
    plus `W2 CUDA product build smoke after varlen-window ABI change`;
  - expected runtime/cost:5-15min native probe,额外 3-8min build smoke,
    hard cap 25min,约 USD 0.425/hr;
  - stop condition:instance start/SSH/CUDA/source sync/compile/probe 首败,
    CUDA release build 首败,或 PASS 后复制 artifact 并停机;
  - correctness command:
    `bash scripts/microbenches/build_and_run_paged_varlen_window_correctness.sh`;
  - product build smoke:
    `cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`;
  - performance command: none;本轮不产出性能结论。
- Source/tooling change:
  - 新增 `scripts/microbenches/paged_varlen_window_correctness.cu`;
  - 新增 `scripts/microbenches/build_and_run_paged_varlen_window_correctness.sh`;
  - README 记录该 probe 用于 Gemma3 unified prefill 前验证 paged-varlen
    one-pass 和 split-K sliding-window 语义。
- Native CUDA result:
  - `paged_varlen_window_correctness.rc` = `0`;
  - stdout contains `VERDICT: paged varlen window correctness PASS`;
  - `sliding_window=0`: one-pass/split-K max abs err both `0.00003045`;
  - `sliding_window=3`: one-pass/split-K max abs err both `0.00002996`;
  - full causal vs window CPU reference semantic delta `0.02593978`,证明
    probe 实际覆盖了不同语义,不是只测等价路径。
- CUDA product build smoke:
  - `cargo_cuda_build.rc` = `0`;
  - build finished release profile in `3m 46s`;
  - binary SHA256:
    `3d5ce5a0dd931a88f26d2d3e23c27805deeb91c77f804e140dff3304e7afcc4a`;
  - first build attempt failed with rc `127` only because non-login SSH PATH
    lacked `/root/.cargo/bin`;rerun after `source /root/.cargo/env` passed.
- Evidence caveat:
  - remote HEAD was `aa741f90b5a135d974fbb824283252a6b66d5857`;
  - remote git status is dirty because this checkpoint's new microbench files
    were not yet committed at sync time and rsync deliberately excluded local
    historical artifact directories;this is diagnostic correctness evidence,
    not a performance claim.
- Correctness/performance status:
  - the varlen-window CUDA ABI/semantics checkpoint is now validated by native
    CUDA and product CUDA build smoke;
  - Gemma3 unified prefill guard is still not relaxed;W2 still requires
    sandwich norm/GeGLU unified-tail semantics and product `ferrum run`/`serve`
    smoke before performance evidence;
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains functional / known-gap,not release-grade。
- Next step:
  - implement the next Gemma3 unified prefill correctness piece: device-side
    unified tail support for Gemma3 sandwich post-attn/post-ffn norms and GeGLU,
    keeping the guard closed until Paris/chat smoke passes。

## 2026-06-15 XLII — W2 source checkpoint: varlen attention 接入 per-layer sliding-window 语义

- 本轮没有启动 GPU;按最小源码验证推进,避免重复开关机器、重装环境和完整 sweep。
- Source change:
  - `BackendPagedKv::paged_varlen_attention` 增加显式 `sliding_window`
    参数;`0` 表示 full causal,非 0 表示只看最近窗口;
  - CUDA `paged_varlen_attn_f16` 和 split-K phase1 都按
    `attend_start..valid_kv_len` 计算 QK、softmax 和 V 汇聚;
  - Llama/Gemma attention 语义抽成 `llama_qk_mode` 和
    `llama_layer_attention_schedule`,统一 single path、legacy batched
    decode 和 unified varlen path 的 q/k norm、interleaved rope、本地/全局
    layer window 决策;
  - Qwen3 MoE unified callsite 显式传 `0`,保持既有 full causal 行为。
- Why:
  - 上一个 checkpoint 证明 Gemma3 W2 的 c16 fresh prefill cohort 仍走
    serial fallback;直接放开 `supports_varlen_qkv` guard 会绕过 Gemma3
    sandwich norm、GeGLU、local/global window 和 dual RoPE correctness 保护;
  - 本 checkpoint 先补齐 backend varlen attention 的 window 语义,为后续
    Gemma3 unified prefill correctness 做前置切点。
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models llama_attention_semantics_cover_qk_mode_and_layer_windows --lib`
    PASS;
  - `cargo test -p ferrum-models unified_varlen_qkv_rejects_sandwich_configs_even_when_backend_supports_varlen --lib`
    PASS;
  - `cargo check -p ferrum-kernels -p ferrum-models` PASS;
  - `git diff --check` PASS;
  - local Mac 没有 `nvcc`,因此 CUDA kernel 编译/运行验证尚未在本 checkpoint
    本地完成。
- Correctness status:
  - 没有放开 Gemma3 unified varlen guard,所以默认产品路径不应因本轮改变而
    启用未验证 Gemma3 unified prefill;
  - 目前没有新增已知产品 correctness blocker,但 CUDA 编译和产品 smoke 仍是
    下一 checkpoint 必须验证项。
- Performance status:
  - 本轮不是性能证据;Ferrum W2 Gemma3 27B GPTQ 仍约为同机 vLLM c16 baseline
    的 `~65%`,距离 `80%` 目标约 `14-15` percentage points;
  - W2 仍没有 `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Next step:
  - 先做 native CUDA 最小编译/attention-window probe 或产品 CUDA smoke,确认
    新 kernel 参数和窗口语义没有破坏现有 path;
  - 再补 Gemma3 unified tail 的 sandwich norm/GeGLU 语义,通过 Paris/chat smoke
    后才考虑放开 Gemma3 unified prefill guard。

## 2026-06-15 XLI — W2 native checkpoint: dense Marlin 多权重轮转 probe 完成

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_marlin_weight_cycle_probe_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;`vast_shutdown/cleanup_check.txt` 记录
  `cur_state=stopped actual_status=exited`。
- Source/tooling change:
  - `scripts/microbenches/dense_marlin_gemma3_perf.cu` 增加
    `weight_cycle_kernel` 与 `weight_cycle_ws_plus_kernel` 输出行;
  - 关键 auto-tile `m=16/23/32` 会轮转 `8` 份独立 synthetic
    qweight/scales/workspace,用于判断 Gemma3 gate/up/down Marlin shape 的
    产品侧表现更接近热缓存、冷缓存还是多权重轮转;
  - 更新 `build_and_run_dense_marlin_gemma3_perf.sh` 与 microbench README。
- Why:
  - 既有 native probe 的 repeated-hot timing 对小投影过于乐观,而真实 W2
    Gemma3 27B decode 会在 62 层和多投影权重间切换;
  - 本改动提供比完整 Ferrum release build/bench 更便宜的 native CUDA
    最小验证入口,用于选择下一步是否值得改 Marlin tile/grid/repack path。
- Validation:
  - `git diff --check -- scripts/microbenches/dense_marlin_gemma3_perf.cu scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh scripts/microbenches/README.md`
    PASS;
  - `bash -n scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh`
    PASS;
  - remote native CUDA command:
    `timeout 1800 bash scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh`;
  - `probe/dense_marlin_gemma3_perf.rc` = `0`;
  - `probe/dense_marlin_gemma3_perf.stdout` contains
    `VERDICT: dense Marlin native CUDA probe complete`;
  - remote HEAD `82fb3272451083bc7f79c7aeca4610793ef579aa`;
  - remote git status is dirty only as diagnostic evidence because rsync excluded
    local artifact directories and the remote checkout reports 812 artifact deletes.
- Key auto-tile results, kernel-only us:
  - `gate_up`: hot/weight-cycle/cold at m16 `133.715/133.985/176.844`,
    m23 `137.396/136.962/181.151`,m32 `138.025/138.386/181.254`;
  - `down`: hot/weight-cycle/cold at m16 `30.356/68.651/93.560`,
    m23 `52.520/72.835/98.045`,m32 `53.017/73.524/99.045`;
  - `qkv` and `o_proj` also show cache sensitivity on small m, but they are not the
    dominant W2 decode bucket.
- Interpretation:
  - `gate_up` does not move under 8-weight cycling,so the large Gemma3 dense GPTQ
    gate/up bucket is compute/path-bound rather than a simple weight-cache artifact;
  - `down` is materially cache sensitive,so product-side timing should be compared
    against weight-cycle/cold-cache brackets rather than repeated-hot microbench rows;
  - this narrows the next useful native lever to shape-specific gate/up Marlin path
    review, while the higher-level W2 gap still also includes Gemma3 serial prefill
    fallback from the previous checkpoint.
- Next step:
  - 不跑新的完整 sweep;先做 `gate_up` Marlin shape-specific source review或更小的
    native CUDA A/B;
  - 真正改产品路径后再按顺序跑 Paris/chat smoke、产品 `ferrum run`/`serve` quick
    regression,最后才进入 W2 release-grade gate。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 XL — W2 source trace:Gemma3 prefill 是 serial fallback,不是 unified cohort prefill

- 本轮没有再开 GPU;基于已提交 artifact
  `w2_prefill_bucket_profile_2026-06-15/` 和源码追踪完成定位。
- Evidence:
  - artifact 中 26 个 ShareGPT 测量请求对应 26 条
    `[prefill-profile] tokens=122` row;如果 cohort prefill 真实合批,不应表现为
    每个请求一条 `prefill_internal` profile;
  - `LlamaFamilyModel::load` 对 sandwich-norm families(Gemma 3)设置
    `supports_varlen_qkv = B::supports_varlen_qkv() && !cfg.sandwich_norms`,
    并记录 `Gemma3 family: legacy batched_decode=... varlen_unified=false`;
  - `LlamaFamilyModel::unified_forward` 在 `!supports_varlen_qkv` 时返回
    `Unsupported`,原因是 Gemma3 unified/paged attention 尚未支持 per-layer
    local/global window semantics;
  - `LlmExecutor::batch_prefill` 在 `model.unified_forward(...)` 返回
    `Unsupported` 后,在同一个 model lock 下循环 `model.prefill(cid,toks)`,
    即 serial prefill fallback。
- Interpretation:
  - c16 TTFT p50 约 `0.9s` 与单个 122-token prefill 约 `84ms`
    一致:初始 cohort 大概率被 serial full-prefill 队列放大;
  - 这比单个 Marlin GEMM 微优化更能解释 Ferrum vs vLLM 的 14-15 percentage
    points 缺口;
  - `tail_mlp`/`flash_attn` 仍是单次 prefill 的局部热区,但高杠杆方向是
    Gemma3 unified/batched prefill 或等价 cohort prefill path,而不是继续扫
    `FERRUM_MARLIN_SKIP_WS_ZERO` 这类历史上已显示收益接近 0 的开关。
- Next implementation direction:
  - 不直接打开现有 `supports_varlen_qkv` guard;该 guard 保护 Gemma3 的 sandwich
    norms、dual rope、per-layer local/global attention correctness;
  - 先做一个小型设计/代码切点:复用 Llama unified scaffolding,给 Gemma3 增加
    explicit unsupported reason/profile 或受测的 narrow cohort-prefill path;
  - 任何真正启用 Gemma3 unified prefill 的改动都必须先过 Paris/chat smoke,
    再跑 native CUDA c16 最小验证。
- Release-grade status:
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未达到 release-grade。

## 2026-06-15 XXXIX — W2 prefill bucket profile: profiler 修复有效,瓶颈落在 tail MLP + prefill attention

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_prefill_bucket_profile_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;Vast shutdown poll 3 记录 `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA prefill profile buckets validation`;
  - expected runtime/cost:15-35min,hard cap 45min,约 USD 0.425/hr;
  - stop condition:instance start/SSH/CUDA/source sync/build 首败、serve readiness
    首败、chat smoke 首败、c16 diagnostic 完成,或 45min cap;
  - correctness gate:CUDA release build,serve readiness,non-stream chat smoke,
    then `bench-serve --fail-on-error`;
  - performance command:diagnostic-only natural ASCII ShareGPT c16,
    `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --num-prompts 16`;
  - profile scope:`FERRUM_PREFILL_OP_PROFILE=1`,只作诊断。
- Build/evidence:
  - remote source commit:`3c407faf25eed833fbb785057c6a7f39d0578e5b`;
  - binary SHA256:
    `5873e674ed0aff9a301af532e0f38c898595d02fd12441125240cf24abea9403`;
  - `cargo_build.rc=0`,release build 用时 `3m27s`;
  - remote full `git status --short` 不干净,原因是本轮为最小源码同步,
    有意没有同步历史 docs artifact;构建相关源码已同步到上述 commit。
- Correctness/perf diagnostic:
  - `run.status=PASS`,`bench-serve.rc=0`;
  - chat smoke content `5`,usage `completion_tokens=3`;
  - c16:`16 completed / 0 errored`,bad_output `[0]`;
  - c16 with profiler:`321.551 tok/s`,request throughput `5.024 req/s`,
    TTFT p50 `925.570ms`,TTFT p95 `1516.331ms`,TPOT p50 `35.035ms`,
    ITL p50 `26.578ms`,ITL p99 `295.802ms`,
    `output_token_count_source=usage`;
  - 因为本轮开启 profiler,吞吐只用于诊断,不作为正式性能 claim。
- Prefill bucket evidence:
  - captured 27 prefill profiles;其中 ShareGPT `tokens=122` 有 26 个;
  - ShareGPT prefill total mean `83.577ms`,range `83-92ms`;
  - `tail_mlp` mean `37.654ms`,约 `45.1%`;
  - `flash_attn` mean `30.192ms`,约 `36.1%`;
  - ordinary `matmuls` mean `6.000ms`,约 `7.2%`;
  - `qk_norm_rope` mean `1.000ms`,约 `1.2%`;
  - `tail_mlp` 内部:`tail_gate_up` mean `23.115ms`,
    `tail_down` mean `13.115ms`。
- Interpretation:
  - profiler source fix 有效;prefill bucket 不再为空;
  - prefill/TTFT 侧的主热区是 Gemma GPTQ MLP tail 和 prefill attention,
    不是普通 QKV/O matmul;
  - typed vLLM paged attention 已经验证无 end-to-end 收益,下一步应优先看
    Gemma GPTQ MLP tail 的 prefill/decode 共享实现与 kernel launch/Marlin 调度,
    再看 prefill attention 是否仍有可替换路径。
- Release-grade status:
  - 本轮是 N=1 diagnostic,没有 `--require-ci`,没有
    `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未达到 release-grade。

## 2026-06-15 XXXVIII — W2 prefill/TTFT first profile:正确性干净,发现 prefill profiler bucket 缺口并修复源码

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_prefill_ttft_profile_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;Vast shutdown poll 1 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA prefill/TTFT profile diagnostic`;
  - expected runtime/cost:8-20min,hard cap 30min,约 USD 0.425/hr;
  - stop condition:启动/SSH/CUDA/server readiness 首败、chat smoke 首败、
    c16 ShareGPT diagnostic 完成并复制 artifact,或 30min cap;
  - correctness gate:`ferrum serve` readiness plus non-stream chat smoke before
    `bench-serve`;`bench-serve` 使用 `--fail-on-error`;
  - performance command:diagnostic-only natural ASCII ShareGPT c16,
    `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --num-prompts 16`;
  - profile scope:server 使用 `FERRUM_PREFILL_OP_PROFILE=1`,只作诊断。
- Correctness/perf diagnostic:
  - `run.status=PASS`,`bench-serve.rc=0`;
  - chat smoke content `5`,usage `completion_tokens=3`;
  - c16:`16 completed / 0 errored`,bad_output `[0]`;
  - c16 throughput `340.882 tok/s`,TTFT p50 `889.558ms`,
    TTFT p95 `1452.948ms`,TPOT p50 `32.804ms`,ITL p50 `24.678ms`,
    ITL p99 `281.837ms`;
  - ratio vs clean vLLM c16 baseline:`340.882 / 518.796 = 0.657`,
    距 80% 线约 `14.3` percentage points。
- Prefill observation:
  - captured 27 `[prefill-profile]` total rows;
  - smoke prefill:`tokens=23,total=29ms`;
  - ShareGPT prefills:`tokens=122,total=80-88ms`,median `80ms`;
  - bucket breakdown 为空。
- Source fix in this checkpoint:
  - prefill profile now enables ordinary op timers for `tokens > 1`
    (`decode_op_profile || prefill_op_profile`);
  - prefill start clears stale op/tail counters before timing;
  - prefill summary now drains and prints tail buckets:
    `tail_norm`,`tail_gate_up`,`tail_act`,`tail_down`,`tail_mlp`,
    `tail_resid`;
  - default product path is unchanged; this only affects diagnostic runs with
    `FERRUM_PREFILL_OP_PROFILE=1`。
- Local validation after source fix:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check -- crates/ferrum-models/src/models/llama_family.rs
    docs/goals/model-coverage-2026-06-12/artifacts/w2_prefill_ttft_profile_2026-06-15`
    PASS;
  - `cargo check -q -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-models
    llama_family_runtime_env_parses_startup_knobs --lib` PASS。
- Next step:
  - rerun the same native CUDA prefill profile after rebuilding on
    `40826362`,then use bucket evidence to choose the next small source lever;
  - do not treat this first profile as release evidence or as proof W2 is
    release-grade。
- Release-grade status:
  - 本轮是 N=1 diagnostic,没有 `--require-ci`,没有
    `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未达到 release-grade。

## 2026-06-15 XXXVII — W2 typed vLLM paged-attn diagnostic:正确性通过,性能无改善

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_vllm_paged_attn_gemma_diag_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;Vast shutdown poll 2 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA typed vLLM paged-attn ShareGPT diagnostic`;
  - expected runtime/cost:10-25min,hard cap 35min,约 USD 0.425/hr;
  - stop condition:启动/SSH/CUDA/server readiness 首败、typed attention-selection
    assertion 首败、chat smoke 首败、c16/c32 ShareGPT diagnostic 完成并复制
    artifact,或 35min cap;
  - correctness gate:artifact-local `ferrum.toml` 设置
    `runtime.use_vllm_paged_attn=true` 后 `ferrum serve` readiness、
    decision-trace assertion、non-stream chat smoke,之后才跑 `bench-serve`;
  - performance command:diagnostic-only natural ASCII ShareGPT c16/c32,
    `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --num-prompts 16`。
- Correctness:
  - `run.status=PASS`,`bench-serve.rc=0`;
  - decision trace 明确 `attention_prefill_mixed_backend=vllm_paged_varlen`,
    source `config_file`,key `FERRUM_USE_VLLM_PAGED_ATTN`;
  - decode selected `vllm_paged_attn_v1_short`;
  - chat smoke content `5`,usage `completion_tokens=3`;
  - c16:`16 completed / 0 errored`,bad_output `[0]`;
  - c32 diagnostic cell:`16 completed / 0 errored`,bad_output `[0]`;
  - 本轮没有发现新的 Ferrum product correctness 问题。
- Diagnostic bench(非 release evidence,N=1,无 CI):
  - c16:`340.443 tok/s`,TTFT p50 `890.332ms`,TTFT p95 `1453.858ms`;
  - c32 diagnostic cell:`341.419 tok/s`,TTFT p50 `889.279ms`,
    TTFT p95 `1440.689ms`。
- Diagnostic ratio vs clean vLLM ShareGPT baseline:
  - c16:`340.443 / 518.796 = 0.656`,差距约 `34.4%`,
    距 80% 线约 `14.4` percentage points;
  - c32 diagnostic cell:`341.419 / 524.128 = 0.651`,差距约 `34.9%`,
    距 80% 线约 `14.9` percentage points。
- Interpretation:
  - typed config VPA 路径已经生效,不是 hidden env 组合;
  - 相比 no-VPA Ferrum ShareGPT,c16 只 `+0.13%`,c32 diagnostic cell
    `-0.25%`,没有性能收益;
  - VPA 不是当前缺失的 14-15 percentage points 的主要杠杆,下一步继续回到
    已定位的 Gemma tail/GEMM 热点,尤其 `tail_gate_up` 与 `tail_down`。
- Release-grade status:
  - 本轮是 N=1 diagnostic,没有 `--require-ci`,没有
    `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未达到 release-grade。

## 2026-06-15 XXXVI — W2 typed prefix-cache ShareGPT diagnostic: 正确性干净,0 hit,性能无改善

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_prefix_cache_sharegpt_diag_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;Vast poll 1 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA typed prefix-cache ShareGPT diagnostic`;
  - expected runtime/cost:10-25min,hard cap 35min,约 USD 0.425/hr;
  - stop condition:启动/SSH/CUDA/server readiness 首败、chat smoke 首败、
    c16/c32 ShareGPT diagnostic 完成并复制 artifact,或 35min cap;
  - correctness gate:`ferrum serve --enable-prefix-cache` readiness plus
    non-stream chat smoke before `bench-serve`;
  - performance command:diagnostic-only natural ASCII ShareGPT c16/c32,
    `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --num-prompts 16`。
- Correctness:
  - `run.status=PASS`,`bench-serve.rc=0`;
  - chat smoke content `5`,usage `completion_tokens=3`;
  - c16:`16 completed / 0 errored`,bad_output `[0]`;
  - c32:`16 completed / 0 errored`,bad_output `[0]`;
  - 本轮没有发现新的 Ferrum product correctness 问题。
- Prefix-cache observation:
  - decision trace 显示 `prefix_cache_policy=prefix_cache_enabled`,
    source `cli`;
  - health after 显示 `enabled=true`,`hits=0`,`misses=53`,
    `saved_prefill_tokens=0`,`entries=0`;
  - 结论:typed product prefix cache 已打开,但没有命中这个 repeated-prompt
    ShareGPT 场景。
- Diagnostic bench(非 release evidence,N=1,无 CI):
  - c16:`340.618 tok/s`,TTFT p50 `889.469ms`,TTFT p95 `1453.788ms`;
  - c32:`342.350 tok/s`,TTFT p50 `887.527ms`,TTFT p95 `1438.820ms`。
- Diagnostic ratio vs clean vLLM ShareGPT baseline:
  - c16:`340.618 / 518.796 = 0.657`,差距约 `34.3%`,
    距 80% 线约 `14.3` percentage points;
  - c32:`342.350 / 524.128 = 0.653`,差距约 `34.7%`,
    距 80% 线约 `14.7` percentage points。
- Interpretation:
  - prefix cache 不是当前差距的现成解;c16 相比 no-prefix 只 `+0.18%`,
    c32 只 `+0.02%`;
  - 下一步若追 prefix-cache,应查 why zero hits/entries,不要重复 full sweep;
  - 否则继续回到已定位的 Gemma tail/GEMM 热点,尤其 `tail_gate_up` 与
    `tail_down`。
- Release-grade status:
  - 本轮是 N=1 diagnostic,没有 `--require-ci`,没有
    `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未达到 release-grade。

## 2026-06-15 XXXV — W2 vLLM natural ShareGPT baseline clean;Ferrum c16/c32 约 65%

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_vllm_sharegpt_baseline_probe_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;Vast poll 2 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA vLLM ShareGPT baseline-cleanliness probe`;
  - expected runtime/cost:20-45min,hard cap 60min,约 USD 0.425/hr;
  - stop condition:启动/SSH/CUDA/vLLM server 首败、baseline smoke 首败、
    c16/c32 ShareGPT diagnostic 完成并复制 artifact,或 60min cap;
  - correctness gate:vLLM `/v1/models` + 非流式 chat smoke;
  - performance command:diagnostic-only natural ASCII ShareGPT c16/c32,
    `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --num-prompts 16`。
- vLLM baseline diagnostic:
  - engine:`vllm 0.10.1.1`,GPTQ Marlin,同一 HF/safetensors GPTQ model;
  - `/v1/models` ready,poll 33;
  - chat smoke rc=0,content `5`,usage `completion_tokens=3`;
  - c16:`16 completed / 0 errored`,bad_output `[0]`,
    `518.796 tok/s`;
  - c32:`16 completed / 0 errored`,bad_output `[0]`,
    `524.128 tok/s`。
- Ferrum same-dataset no-profile compare:
  - binary SHA256:
    `3e28a4cf37b2e25b127dbd591e8891b141863d8082d1757486707c785e6869ce`;
  - `ferrum serve --model gemma3:27b-gptq --kv-capacity 512 --max-num-seqs 16`
    ready,poll 29;
  - c16:`16 completed / 0 errored`,bad_output `[0]`,
    `340.003 tok/s`;
  - c32:`16 completed / 0 errored`,bad_output `[0]`,
    `342.284 tok/s`。
- Diagnostic ratio:
  - c16:`340.003 / 518.796 = 0.655`,差距约 `34.5%`;
  - c32:`342.284 / 524.128 = 0.653`,差距约 `34.7%`;
  - 距 80% release-grade 线仍差约 `14.5` percentage points。
- Interpretation:
  - 本轮没有发现新的 Ferrum product correctness 问题;
  - 之前 vLLM random-prompt c16 baseline 自身 invalid-UTF8,不能做 final
    baseline;本轮 natural ShareGPT vLLM c16/c32 是 zero-error,说明 baseline
    路线可以考虑改成自然 prompt 数据集并正式 N>=3 化;
  - 但按这个 clean baseline,Ferrum c16/c32 仍显著低于 80%,W2-P2 仍要继续
    优先优化 Gemma tail/GEMM 路径。
- Release-grade status:
  - 本轮是 N=1 diagnostic,没有 `--require-ci`,没有
    `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未达到 release-grade。

## 2026-06-15 XXXIV — W2 fused sandwich residual-add: native CUDA minimal validation PASS,收益有限

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_fused_sandwich_residual_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;Vast API poll 5 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA fused sandwich residual-add minimal validation`;
  - expected runtime/cost:15-35min,hard cap 45min,约 USD 0.425/hr;
  - stop condition:启动/SSH/CUDA/source sync/build 首败、`ferrum run`
    correctness 首败、serve/bench diagnostic 完成并复制 artifact,或 45min cap;
  - correctness gate:CUDA release build + product `ferrum run`,通过后才进入
    `serve/bench-serve --fail-on-error`;
  - performance command:diagnostic-only natural ASCII ShareGPT c16/c32 小样本,
    `n_repeats=1`,seed 9271,`FERRUM_DECODE_OP_PROFILE=1`。
- 远端源码/构建:
  - git HEAD:`4eeea0ba76a2ac8b0671941bcba0d66020c31ed4`;
  - 本轮 rsync 为减少付费 GPU 空转排除了历史 artifacts,远端 git status
    因旧 artifact 缺失显示 docs 删除;本轮只作为 diagnostic evidence;
  - `CUDA_COMPUTE_CAP=89 cargo build --release -p ferrum-cli --bin ferrum
    --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source` PASS;
  - binary SHA256:
    `3e28a4cf37b2e25b127dbd591e8891b141863d8082d1757486707c785e6869ce`。
- Product correctness:
  - `ferrum run gemma3:27b-gptq --backend cuda --max-tokens 64 --temperature 0
    --kv-capacity 2560 --max-num-seqs 2 --output-format jsonl`
    rc=0,assistant content `5`,finish_reason `stop`;
  - `ferrum serve` readiness PASS,poll 29;
  - `bench-serve --fail-on-error` rc=0。
- Diagnostic bench(非 release evidence,N=1,无 CI):
  - c=16:`16 completed / 0 errored`,output token count source `usage`,
    throughput `306.061 tok/s`;
  - c=32:`16 completed / 0 errored`,output token count source `usage`,
    throughput `307.373 tok/s`。
- Profile interpretation:
  - 相比 `w2_tail_gate_down_profile_2026-06-15`,batch=16
    `tail_norm_us_mean` 约 `806.5us -> 685.2us`,
    `tail_resid_us_mean` 约 `567.0us -> 494.8us`;
  - batch=16 total decode step 约 `28.08ms -> 27.82ms`,诊断收益约 `0.9%`;
  - 最大热点仍是 `tail_gate_up` 约 `9.01ms` 与 `tail_down` 约 `4.70ms`,
    因此下一步不应继续围绕 residual add 小项重复验证,应转向 gate/up/down
    或更高收益的 Gemma tail/GEMM 路径。
- Correctness status:
  - 本轮未发现新的 product correctness 问题;
  - 但没有 `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未达到 release-grade。

## 2026-06-15 XXXIII — W2 source checkpoint: fuse Gemma sandwich branch norm into F32 residual add

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - added backend trait method `rms_norm_activation_add_to_f32` with a safe
    fallback that materializes the F32 branch then calls `add_inplace`;
  - added CUDA kernel `rms_norm_f16_add_to_f32` in `sandwich_norm.cu`;
  - CUDA backend now launches the fused kernel for F16 activation +
    F16 norm weight + F32 residual shadow;
  - Gemma sandwich device-shadow path now uses the fused helper for
    post-attention and post-FFN residual updates;
  - `nan_trace` keeps the old two-step path so `post_attn_norm` /
    `post_ffn_norm` intermediate dumps remain available for diagnostics.
- Expected effect:
  - removes one F32 residual-add kernel launch and one F32 branch scratch
    write/read at each Gemma sandwich branch residual update;
  - affects default CUDA Gemma3 device-shadow path only,not CPU/Metal
    fallback semantics.
- Local validation:
  - `cargo fmt --all` PASS;
  - `git diff --check -- crates/ferrum-kernels/src/backend/traits.rs
    crates/ferrum-kernels/kernels/sandwich_norm.cu
    crates/ferrum-kernels/src/backend/cuda/mod.rs
    crates/ferrum-models/src/models/llama_family.rs` PASS;
  - `cargo check -q -p ferrum-kernels -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-kernels --lib` PASS,8/8 tests;
  - `cargo test -q -p ferrum-models --lib` PASS,124/124 tests.
- Validation still required:
  - CUDA build must compile the new `sandwich_norm.cu` symbol;
  - run a minimal product correctness check on the same cache-retained 4090
    before any performance measurement;
  - if correctness passes, run a small same-dataset diagnostic to see whether
    the fused branch update moves W2 throughput or profile buckets.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 XXXII — W2 native CUDA dense Marlin probe: gate/up still top target, no tile-default change

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_marlin_native_probe_2026-06-15/`。
- Scope:
  - 这是 native CUDA kernel-ceiling diagnostic,不是 release-grade gate;
  - 没有运行 `ferrum run` 或 `ferrum serve`;
  - 没有生成 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`,W2 仍不是
    release-grade。
- GPU execution:
  - reused Vast instance `40826362`,1x RTX 4090,约 USD 0.425/hr;
  - source sync 后远端 HEAD:`951348b23956caab8c459823708ddc4b63b90a8e`;
  - first native probe rc=`0`,printed
    `VERDICT: dense Marlin native CUDA probe complete`;
  - host-sync/cold-cache probe rc=`0`,printed the same `VERDICT`;
  - artifact copied back locally;
  - Vast stop poll verified `cur_state=stopped`,`actual_status=exited` at
    `2026-06-15T06:52:02Z`。
- Source/tooling change in this checkpoint:
  - extended `scripts/microbenches/dense_marlin_gemma3_perf.cu` with
    product-profile-style `host_sync_kernel` / `host_sync_ws_plus_kernel`
    modes;
  - added limited `cold_cache_kernel` rows for auto-tile `m=16/23/32`
    by flushing a 256MiB scratch buffer before timing;
  - updated `scripts/microbenches/README.md`.
- Key m=16 auto-tile native timings:
  - `qkv`: hot event `17.207 us`,host-sync `18.887 us`,cold-cache
    `39.929 us`;
  - `o_proj`: hot event `12.058 us`,host-sync `13.695 us`,cold-cache
    `24.447 us`;
  - `gate_up`: hot event `133.650 us`,host-sync `135.924 us`,cold-cache
    `177.144 us`;
  - `down`: hot event `30.395 us`,host-sync `32.049 us`,cold-cache
    `93.558 us`.
- Interpretation:
  - host-sync overhead alone does not explain product-profile `qkv/o/down`
    time; repeated-hot native timing was too optimistic for smaller
    projections because the same synthetic weight buffer is reused;
  - forced cold-cache timing is too pessimistic but brackets product behavior,
    confirming cache residency is a major measurement variable;
  - tile override evidence is weak: `64x256` only marginally improves hot
    `gate_up` and regresses `down`,so this checkpoint does not justify a
    default tile change;
  - no new product correctness issue was found in this diagnostic.
- Local validation:
  - `git diff --check -- scripts/microbenches/dense_marlin_gemma3_perf.cu`
    PASS;
  - `bash -n scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh`
    PASS.
- Next step:
  - continue from the correct default path, but choose the next lever based on
    product-representative weight/cache behavior rather than repeated-hot
    synthetic kernel timings alone;
  - do not claim performance or release readiness from this native diagnostic.

## 2026-06-15 XXXI — W2 source checkpoint: add native CUDA dense Marlin Gemma3 probe

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source/tooling change:
  - added `scripts/microbenches/dense_marlin_gemma3_perf.cu`;
  - added `scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh`;
  - updated `scripts/microbenches/README.md`.
- Probe purpose:
  - bypass Cargo, model loading, tokenizer, server, and bench client;
  - compile only the probe plus `crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu`
    with `nvcc`;
  - call `marlin_cuda` directly on synthetic buffers for Gemma3-27B GPTQ
    `qkv`, `o_proj`, `gate_up`, and `down` shapes;
  - report `kernel_only` and `ws_plus_kernel` µs/call plus useful and padded
    TFLOPS for `m={1,3,6,9,12,16,23,32}` and tile choices
    `auto`, `128x128`, `64x256`.
- Why this checkpoint matters:
  - before changing dense Marlin tile selection or grid policy, we can now get
    a native CUDA kernel-ceiling result in minutes on the cached 4090 instead
    of paying a full Ferrum release build/product run for each hypothesis.
- Local validation:
  - `git diff --check -- scripts/microbenches/dense_marlin_gemma3_perf.cu
    scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh
    scripts/microbenches/README.md` PASS;
  - `bash -n scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh`
    PASS;
  - local machine has no `nvcc`,so native CUDA compile/run is pending on the
    CUDA host.
- Next step:
  - on the same cache-retained 4090, run only this native CUDA probe first;
  - keep the machine running during tight source/probe iterations; stop only
    after artifacts are copied or the iteration is no longer active.

## 2026-06-15 XXX — W2 source checkpoint: restore dense vLLM Marlin guard after first-fail evidence

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - restored `reject_dense_vllm_marlin_if_requested`;
  - dense GPTQ load now rejects `FERRUM_VLLM_MARLIN=1` before building a
    dense vLLM Marlin store;
  - removed the diagnostic dense vLLM load path added in `ce960292`,because
    first-fail evidence showed it reaches vendored vLLM Marlin `abort()`
    before generation;
  - updated the unsupported message with the real blocker: vendored dense
    vLLM Marlin currently compiles with `kernel_selector.h` disabled for the
    CUDA hidden-symbol workaround,so it cannot select a real GEMM kernel
    safely;
  - default dense Marlin remains unchanged; vLLM Marlin MoE remains behind
    `FERRUM_VLLM_MOE` for stacked MoE weights.
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo check -q -p ferrum-kernels --tests` PASS;
  - `cargo test -q -p ferrum-kernels --lib` PASS,8/8 tests;
  - `git diff --check -- crates/ferrum-kernels/src/backend/cuda/quant.rs`
    PASS.
- Next step:
  - do not spend more time on dense vLLM Marlin unless explicitly taking on
    the broader `kernel_selector.h` / CUDA hidden-symbol linker problem;
  - continue W2 performance work on the existing correct default path with
    minimal same-pod validation.

## 2026-06-15 XXIX — W2 dense vLLM Marlin first-fail: prefill launch config aborts before generation

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_vllm_marlin_diag_2026-06-15/`。
- Scope:
  - 这不是 release-grade gate,也没有生成 `MODEL_RELEASE_GRADE_W2 PASS:
    <out_dir>`;
  - 失败只覆盖新接入的诊断路径 `FERRUM_VLLM_MARLIN=1`,默认 dense GPTQ
    Marlin 路径未因本轮诊断改判为失败;
  - remote git HEAD:`ce960292cf3132b982770a4cc727a9a6b19d2f4e`;
  - remote git status 因 artifacts 目录 rsync 排除显示旧 artifact 删除,所以本轮
    只能作为 first-fail/debug evidence,不是 clean performance evidence。
- GPU execution:
  - lane:`W2 Gemma3 CUDA dense vLLM Marlin first-fail diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD 0.425/hr;
  - release CUDA build PASS in `3m 28s`;
  - binary SHA256:
    `abd576f024776ed6df39c9e4c939b28344d93e6e69429cb663a749de28a1f3c8`;
  - sensitive scan of copied artifact: no `VAST_API_KEY`,`HF_TOKEN`,
    private-key,`jupyter_token`,or startup-script hits.
- Product-path result:
  - command: `FERRUM_VLLM_MARLIN=1 target/release/ferrum run
    gemma3:27b-gptq --backend cuda --prompt "What is 2+3? Answer with just
    the number." --max-tokens 64 --temperature 0 --kv-capacity 2560
    --max-num-seqs 2 --output-format jsonl`;
  - `run.status=FAIL`;
  - `correctness/run.rc=134`,`nohup.rc=134`;
  - model load completed,then the first dense vLLM Marlin launch aborted before
    token generation:
    `m=23 n=8192 k=5376 group_size=128`;
  - vLLM Marlin error:
    `Invalid thread config: thread_m_blocks = 1, thread_k = -1,
    thread_n = -1, num_threads = -1 for MKN = [23, 5376, 8192] and
    num_bits = 4, prob_m_split = 16, group_size = 128`;
  - server readiness and `bench-serve` did not run because the correctness
    first-fail stop condition triggered.
- Interpretation:
  - dense vLLM Marlin load/repack path is wired far enough to reach the kernel
    launch;
  - the current launch path is not safe for the skinny prefill shape seen by
    `ferrum run`,so it is a blocker for using `FERRUM_VLLM_MARLIN=1` as a
    product path;
  - this does not invalidate the previously collected default-path Ferrum
    zero-error diagnostics, but W2 still remains not release-grade until the
    final validator prints the required PASS line.
- GPU cleanup:
  - artifact copied locally before shutdown;
  - stop poll reached `cur_state=stopped actual_status=exited`;
  - stopped timestamp:`2026-06-15T06:23:32Z`.
- Next step:
  - inspect vLLM Marlin shape/config constraints and make the smallest safe
    source change: either route unsupported skinny prefill shapes to the
    existing IST-DASLab Marlin path,or correct the vLLM launch config;
  - validate on the same cache-retained 4090 with a minimal `ferrum run`
    first,then only run c16/c32 diagnostic if correctness passes.

## 2026-06-15 XXVIII — W2 source checkpoint: wire diagnostic dense vLLM Marlin GPTQ load path

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 代码侧推进:
  - `FERRUM_VLLM_MARLIN=1` 不再在 dense GPTQ dispatch 入口被提前拒绝;
  - 当二进制带 `vllm-marlin` feature 且设置 `FERRUM_VLLM_MARLIN=1` 时,
    dense `load_gptq` 会:
    - 上传 GPTQ qweight;
    - 通过已有 `ferrum_vllm_gptq_marlin_repack` FFI 生成
      vLLM Marlin tile qweight;
    - 使用与 vLLM stacked path 一致的 Marlin scale permutation;
    - 构造 `MarlinWeight` 后复用现有 `launch_vllm_marlin` dispatch;
  - 如果设置 `FERRUM_VLLM_MARLIN=1` 但未编译 `vllm-marlin`,现在会在
    load 阶段明确报错;
  - 默认路径不变:未设置 `FERRUM_VLLM_MARLIN=1` 时仍走
    IST-DASLab Marlin repack/dispatch。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check -- crates/ferrum-kernels/src/backend/cuda/quant.rs`
    PASS;
  - `cargo check -q -p ferrum-kernels --tests` PASS;
  - `cargo test -q -p ferrum-kernels cuda_quant_runtime_config_parses_marlin_and_moe_knobs --lib`
    PASS,0 tests matched in the non-CUDA local build.
- Evidence caveat:
  - 本机没有做 CUDA/vLLM feature build;该 checkpoint 必须通过下一轮
    4090 release CUDA build and diagnostic run 才能证明 dense vLLM Marlin
    path可加载、可正确生成、并有可比较性能。
- Next step:
  - 复用 1x4090 cache-retained instance 做 first-fail 小样本:
    release build with `cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`,
    再跑 `FERRUM_VLLM_MARLIN=1` 的 `ferrum run` smoke 和
    c16/c32 diagnostic;如果 load/correctness 失败,立刻拷回失败 artifact
    并停机。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 XXVII — W2 projection-level Marlin profile: gate/up kernel is the dominant dense GPTQ target

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_projection_profile_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=exited intended_status=stopped`;
  - stop timestamp: `2026-06-15T06:02:00Z`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA projection-level dense Marlin profile diagnostic`;
  - expected runtime/cost:15-35min,hard cap 45min,1x RTX 4090 instance
    `40826362` at about USD 0.425/hr;
  - stop condition:start/SSH/CUDA/source sync/build/server readiness first
    failure,projection-level dense Marlin profile c16/c32 small sample complete
    and copied,or 45min cap;
  - correctness gate:release build plus server readiness and
    `bench-serve --fail-on-error` zero-error diagnostic;
  - performance command:diagnostic-only natural ASCII ShareGPT dataset with
    `FERRUM_DECODE_OP_PROFILE=1` and `FERRUM_MARLIN_PROFILE=1`,
    `num_prompts=16`,`n_repeats=1`,`random-output-len=64`,`seed=9271`,
    c16/c32.
- Evidence scope:
  - release build PASS in `3m 28s`,binary SHA256:
    `0991e89489c205f6fdffec5dbf138923367e51c02f0356ddd8828c276003a950`;
  - remote git HEAD was `5fac46d8a45b99932d06c462d7be50d8825d9d55`;
  - remote git status had `337` lines because local `docs/.../artifacts/`
    was excluded from source rsync to avoid copying old evidence directories,
    so this remains profiling evidence only,not final clean performance
    evidence;
  - `FERRUM_MARLIN_PROFILE=1` adds per-Marlin-call syncs,so throughput in
    this artifact is profiling overhead and not a product performance claim.
- Build/bench result:
  - `run.status=PASS`,`nohup.rc=0`,`bench-serve.rc=0`;
  - server readiness poll: `29`;
  - c16:completed `[16]`,errored `[0]`,mean `290.042 tok/s`,p95 ITL
    `31.777 ms`,p95 TTFT `1567.437 ms`,output token source `usage`;
  - c32 client / active cap16:completed `[16]`,errored `[0]`,mean
    `290.221 tok/s`,p95 ITL `31.855 ms`,p95 TTFT `1561.008 ms`,
    output token source `usage`.
- Op-profile result:
  - server log captured `264` `[batched-op-profile]` rows with projection
    fields populated and `marlin_other_* = 0`,so the profile labels covered
    all dense Marlin calls in this path;
  - for batch `m=16` (`118` rows),mean total per decode step
    `30063 us`; aggregate dense Marlin kernel was `55.0%` (`16548 us`);
    projection kernel split:
    - gate/up `29.0%` (`8728 us`);
    - down `14.5%` (`4352 us`);
    - qkv `7.1%` (`2132 us`);
    - o_proj `4.4%` (`1336 us`);
    aggregate workspace zero was `3.8%` (`1137 us`);
  - for batch `m=10` (`123` rows),aggregate dense Marlin kernel was
    `55.6%` (`16349 us`); projection kernel split:
    gate/up `29.2%` (`8593 us`),down `14.8%` (`4344 us`),
    qkv `7.1%` (`2099 us`),o_proj `4.5%` (`1313 us`);
    workspace zero was `3.8%` (`1120 us`).
- Interpretation:
  - gate/up dense Marlin kernel alone is the largest single measured decode
    cost and is bigger than any other dense GPTQ projection bucket;
  - workspace zero remains measurable but small relative to kernel time;
  - next useful checkpoint should compare/alter the gate/up dense GPTQ kernel
    path itself: Triton INT4 diagnostic viability, vLLM dense GPTQ repack/path
    comparison, or a shape-specific Marlin lever. Do not change product
    defaults from this artifact alone.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 XXVI — W2 profile instrumentation: split dense Marlin counters by projection

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 代码侧推进:
  - `FERRUM_MARLIN_PROFILE=1` 的 dense Marlin nested counters 现在按
    projection label 细分输出:
    `qkv`,`o_proj`,`gate_up`,`down`,`lm_head`,`other`;
  - 每个 projection bucket 分别记录 `ws_zero` 与 `kernel` 的时间和调用数;
  - `[batched-op-profile]` 保留原有 aggregate
    `marlin_ws_zero`/`marlin_kernel`,并新增
    `marlin_qkv_*`,`marlin_o_*`,`marlin_gate_up_*`,
    `marlin_down_*`,`marlin_lm_head_*`,`marlin_other_*`;
  - 给 batched decode 的 `o_proj` 补上 CUDA alloc label,避免它在
    projection profile 中落入 `other`;
  - 默认路径不变;新增分桶只在 `FERRUM_MARLIN_PROFILE=1` 的诊断路径累加。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check -- crates/ferrum-kernels/src/backend/cuda/marlin.rs crates/ferrum-models/src/models/llama_family_forward_batched.rs`
    PASS;
  - `cargo check -q -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-models llama_batched_runtime_config_parses_startup_knobs --lib`
    PASS,1 test passed。
- Evidence caveat:
  - 本机未做 CUDA feature 编译;该 checkpoint 需要下一轮 4090 release
    build/profile artifact 来验证 CUDA profile fields 的实际日志输出。
- Next step:
  - 复用 1x4090 cache-retained instance 做一个小样本
    `FERRUM_DECODE_OP_PROFILE=1` + `FERRUM_MARLIN_PROFILE=1` profile,
    确认 gate/up Marlin kernel 是否确实主导 dense GPTQ time;如果成立,
    下一步再比较 Triton INT4 或 vLLM dense GPTQ packing path。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 XXV — W2 dense-Marlin nested profile: kernel dominates, workspace zero is not first lever

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_nested_profile_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=exited intended_status=stopped`;
  - stop timestamp: `2026-06-15T05:38:17Z`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA dense-Marlin nested profile diagnostic`;
  - expected runtime/cost:15-35min,hard cap 45min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:start/SSH/CUDA/source sync/build/server readiness first
    failure,dense Marlin nested profile c16/c32 small sample complete and
    copied,or 45min cap;
  - correctness gate:release build plus server readiness and
    `bench-serve --fail-on-error` zero-error diagnostic;
  - performance command:diagnostic-only natural ASCII ShareGPT dataset with
    `FERRUM_DECODE_OP_PROFILE=1` and `FERRUM_MARLIN_PROFILE=1`,
    `num_prompts=16`,`n_repeats=1`,`random-output-len=64`,`seed=9271`,
    c16/c32.
- Evidence scope:
  - release build PASS,binary SHA256:
    `3c503a7cabcc0acba90fd35ac40704c19f631f4f2a6f206f8a8374758b20a280`;
  - remote git HEAD was `95a27d7738d4834fa09b52ee5a86cf084c16de75`;
  - remote git status is dirty because local `docs/.../artifacts/` was
    excluded from source rsync to avoid copying old evidence directories,so
    this remains profiling evidence only,not final clean performance evidence;
  - `FERRUM_MARLIN_PROFILE=1` adds per-Marlin-call syncs,so the lower
    throughput in this artifact is profiling overhead and not a product
    performance claim.
- Build/bench result:
  - `run.status=PASS`,`nohup.rc=0`,`bench-serve.rc=0`;
  - build completed in `3m 29s`;
  - c16:completed `[16]`,errored `[0]`,mean `288.575 tok/s`,p95 ITL
    `32.181 ms`,p95 TTFT `1570.866 ms`;
  - c32 client / active cap16:completed `[16]`,errored `[0]`,mean
    `289.619 tok/s`,p95 ITL `32.077 ms`,p95 TTFT `1560.208 ms`.
- Op-profile result:
  - server log captured `264` `[batched-op-profile]` rows;
  - for batch `m=16` (`118` rows),mean total per decode step
    `30134 us`; nested dense Marlin kernel aggregate was `54.9%`
    (`16550 us`),workspace zero aggregate was `3.9%` (`1181 us`);
    `tail_gate_up` was `31.6%` (`9526 us`),`tail_down` `17.6%`
    (`5299 us`),combined `tail_mlp` `49.2%` (`14825 us`);
  - for batch `m=10` (`123` rows),nested dense Marlin kernel aggregate was
    `55.5%` (`16350 us`),workspace zero aggregate `3.9%` (`1145 us`);
    `tail_gate_up` was `31.8%` (`9379 us`),`tail_down` `17.7%`
    (`5212 us`).
- Interpretation:
  - workspace zero is measurable but not the first lever; it is roughly
    `1.1-1.2 ms` per decode step across all dense Marlin calls;
  - most of the remaining time is dense Marlin kernel work,with the fused
    Gemma3 gate/up projection still the largest projection-level bucket;
  - next useful checkpoint should focus on the gate/up Marlin shape itself:
    kernel launch/shape behavior, existing Triton INT4 diagnostic viability,
    or a source comparison against the vLLM dense GPTQ path. Avoid changing
    product defaults until correctness and same-dataset diagnostics support it.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.

## 2026-06-15 XXIV — W2 profile instrumentation: add dense Marlin nested counters

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 代码侧推进:
  - 新增 `FERRUM_MARLIN_PROFILE=1` profile-only 开关;
  - 在 dense Marlin `marlin_gemm_chunk` 内部新增 nested counters:
    `marlin_ws_zero` 和 `marlin_kernel`;
  - batched decode `[batched-op-profile]` 日志会输出这两个 nested
    字段,但不会把它们加入 `wrapped_us`,避免和 `tail_gate_up` /
    `tail_down` 双计;
  - 默认路径不变,`FERRUM_MARLIN_PROFILE` 未设置时不增加同步计时;
  - 未改变 `FERRUM_MARLIN_SKIP_WS_ZERO` 行为。该开关仍只用于已有的
    strided path,没有扩展到 dense path。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check -- crates/ferrum-kernels/src/backend/cuda/marlin.rs crates/ferrum-models/src/models/llama_family_forward_batched.rs`
    PASS;
  - `cargo check -q -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-models llama_batched_runtime_config_parses_startup_knobs --lib`
    PASS,1 test passed。
- Evidence caveat:
  - local non-CUDA checks passed, but the new CUDA-feature path still needs a
    4090 release build in the next diagnostic checkpoint before relying on the
    new Marlin nested fields.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - rerun the c16/c32 CUDA profile diagnostic with both
    `FERRUM_DECODE_OP_PROFILE=1` and `FERRUM_MARLIN_PROFILE=1` to split
    gate/up projection time into workspace-zero and Marlin kernel work.

## 2026-06-15 XXIII — W2 tail-gate/down profile: gate/up projection is the largest single decode bucket

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_tail_gate_down_profile_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=03 cur_state=stopped actual_status=exited intended_status=stopped`;
  - stop timestamp: `2026-06-15T05:20:40Z`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA tail-gate/down profile diagnostic`;
  - expected runtime/cost:15-35min,hard cap 45min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:start/SSH/CUDA/source sync/build/server readiness first
    failure,tail gate/down profile c16/c32 small sample complete and copied,
    or 45min cap;
  - correctness gate:release build plus server readiness and
    `bench-serve --fail-on-error` zero-error diagnostic;
  - performance command:diagnostic-only natural ASCII ShareGPT dataset,
    `num_prompts=16`,`n_repeats=1`,`random-output-len=64`,`seed=9271`,
    c16/c32.
- Evidence scope:
  - release build PASS,binary SHA256:
    `fb45a77d328c90233ffeb19cb4576bc12ef7079c2096632fe145431c83fcfe2a`;
  - remote git HEAD was `ccc58aba9f5333f1ecd258d841de0fd5ab40a379`;
  - remote git status is dirty because local `docs/.../artifacts/` was
    excluded from source rsync to avoid copying old evidence directories,so
    this remains profiling evidence only,not final clean performance evidence;
  - first remote tmux attempt failed before build because Rust was not on PATH
    in the non-login shell; runner was fixed to source `/root/.cargo/env`,
    the remote output directory was removed, and the corrected rerun produced
    the copied artifact.
- Build/bench result:
  - `run.status=PASS`,`nohup.rc=0`,`bench-serve.rc=0`;
  - build completed in `3m 25s`;
  - c16:completed `[16]`,errored `[0]`,mean `304.328 tok/s`,p95 ITL
    `30.068 ms`,p95 TTFT `1523.492 ms`;
  - c32 client / active cap16:completed `[16]`,errored `[0]`,mean
    `305.245 tok/s`,p95 ITL `29.929 ms`,p95 TTFT `1512.247 ms`.
- Op-profile result:
  - server log captured `264` `[batched-op-profile]` rows;
  - for batch `m=16` (`118` rows),mean total per decode step
    `28076 us`; `tail_gate_up` was `32.2%` (`9039 us`),`tail_down`
    `16.8%` (`4709 us`),combined `tail_mlp` `49.0%` (`13748 us`);
    remaining unwrapped was `2.3%` (`658 us`);
  - for batch `m=10` (`123` rows),`tail_gate_up` was `32.5%`
    (`8901 us`),`tail_down` `16.8%` (`4621 us`),combined `tail_mlp`
    `49.3%` (`13522 us`).
- Interpretation:
  - the largest single decode bucket is the fused Gemma3 gate/up GPTQ
    projection,not down projection,attention/QKR,or logits readback;
  - current next target is the dense GPTQ Marlin path for the fused
    `gate_up_proj` shape. A useful next checkpoint is to measure Marlin
    fixed overhead/workspace-zero and vLLM-Marlin/Triton alternatives as
    explicit diagnostics before changing product defaults;
  - do not run a full `--require-ci` release sweep until this gate/up
    projection bottleneck is reduced.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.

## 2026-06-15 XXII — W2 profile instrumentation: split tail MLP into gate/up and down

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 代码侧推进:
  - `FERRUM_DECODE_OP_PROFILE` 的 batched decode 日志继续保留
    `tail_mlp` 聚合字段;
  - 新增 `tail_gate_up` 和 `tail_down` 子字段,分别计时 Gemma3 tail
    的 fused gate/up projection 和 down projection;
  - `unwrapped` 现在扣除 `tail_gate_up + tail_down`,避免双计
    `tail_mlp` 聚合值;
  - 非 profile 路径不改变。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check -- crates/ferrum-models/src/models/llama_family.rs crates/ferrum-models/src/models/llama_family_forward_batched.rs`
    PASS;
  - `cargo check -q -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-models llama_batched_runtime_config_parses_startup_knobs --lib`
    PASS,1 test passed。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - rerun a small c16/c32 CUDA profile diagnostic to quantify
    `tail_gate_up` vs `tail_down` before choosing the MLP projection
    optimization target.

## 2026-06-15 XXI — W2 tail-profile bucket validation: Gemma3 MLP projections dominate decode

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_tail_profile_buckets_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=03 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=04 cur_state=stopped actual_status=exited intended_status=stopped`;
  - stop timestamp: `2026-06-15T05:00:20Z`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA tail-profile bucket validation`;
  - expected runtime/cost:15-35min,hard cap 45min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:start/SSH/CUDA/source sync/build/server readiness first
    failure,tail bucket profile c16/c32 small sample complete and copied,or
    45min cap;
  - correctness gate:release build plus server readiness and
    `bench-serve --fail-on-error` zero-error diagnostic;
  - performance command:diagnostic-only natural ASCII ShareGPT dataset,
    `num_prompts=16`,`n_repeats=1`,`random-output-len=64`,`seed=9271`,
    c16/c32.
- Evidence scope:
  - release build PASS,binary SHA256:
    `c32bb67c8ee9aee90b0054bcb0fb0eca0e1d127fad5c99929941b714bdf741ed`;
  - remote git HEAD remained `c51002b793f00c8345e160b99b6b74217ca273d9`
    with the profiling source files dirty-synced from the local checkpoint,so
    this is profiling evidence only,not final clean performance evidence;
  - decision trace selected `attention_decode_backend=legacy_paged_decode`
    and `sampling_readback_path=gpu_greedy_argmax`;
  - `bench-serve` rc=0,run status `PASS`,server log captured `264`
    `[batched-op-profile]` rows.
- Bench/profile result:
  - c16:completed `[16]`,errored `[0]`,mean `305.182 tok/s`;
  - c32 client / active cap16:completed `[16]`,errored `[0]`,mean
    `305.181 tok/s`;
  - for batch `m=16` (`118` rows),mean total per decode step
    `28037 us`; mean shares:tail_mlp `49.0%` (`13744 us`),matmul
    `24.9%` (`6971 us`),attention `8.6%` (`2406 us`),tail_norm `2.9%`,
    tail_resid `2.0%`,tail_act `1.5%`,QKR `2.3%`,norm `1.6%`,
    remaining unwrapped `2.3%` (`649 us`);
  - for batch `m=10` (`122` rows),tail_mlp was again the largest bucket:
    `49.3%` (`13516 us`),with remaining unwrapped down to `2.4%`
    (`663 us`).
- Interpretation:
  - the previous `unwrapped` bucket was mostly Gemma3 tail MLP projection work
    rather than attention/QKR or logits readback;
  - current top target is the Gemma3 tail gate/up/down GPTQ linear path. The
    next useful profiling checkpoint is to split `tail_mlp` into gate/up and
    down projection buckets before choosing an optimization patch;
  - do not run a full `--require-ci` release sweep until this c16/c32 decode
    bottleneck is reduced.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.

## 2026-06-15 XX — W2 profile instrumentation: split Gemma3 tail buckets

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 代码侧推进:
  - `FERRUM_DECODE_OP_PROFILE` 的 batched decode 输出新增 tail bucket:
    `tail_norm`,`tail_mlp`,`tail_act`,`tail_resid`;
  - 新 bucket 细分 Gemma3 sandwich tail 的 post-attn/post-ffn norms、
    gate/up/down projections、GeGLU/SwiGLU activation、residual add;
  - `unwrapped` 现在会扣除这些 tail bucket,用于下一轮 GPU diagnostic
    定位 2026-06-15 XIX 里约 `55.6%` 的未拆分 decode-step 时间;
  - 非 profile 路径不改变。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `cargo check -q -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-models llama_batched_runtime_config_parses_startup_knobs --lib`
    PASS,1 test passed。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - rerun a small c16/c32 decode-op-profile diagnostic on CUDA to quantify the
    new tail buckets before choosing an optimization target.

## 2026-06-15 XIX — W2 decode-op profile: bottleneck is mostly unwrapped decode work

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_decode_op_profile_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=03 cur_state=stopped actual_status=exited intended_status=stopped`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA decode-op-profile diagnostic`;
  - expected runtime/cost:10-25min,hard cap 40min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:start/SSH/CUDA/server readiness first failure,profile log
    captures c16/c32 small sample,or 40min cap;
  - correctness gate:server readiness plus first
    `bench-serve --fail-on-error` zero-error diagnostic;
  - performance command:diagnostic-only natural ASCII ShareGPT dataset,
    `num_prompts=16`,`n_repeats=1`,`random-output-len=64`,`seed=9271`,
    c16/c32.
- Evidence scope:
  - reused existing release binary SHA256:
    `a942a2e79880bbc821c26a1c60720fa753d6b8e66a62a73900a4592d123abb0e`;
  - remote git HEAD remained `c51002b793f00c8345e160b99b6b74217ca273d9`
    with `crates/ferrum-types/src/auto_config.rs` dirty-synced from the
    current checkpoint,so this is profiling evidence only,not final clean
    performance evidence;
  - decision trace still selected `attention_decode_backend=legacy_paged_decode`
    and `sampling_readback_path=gpu_greedy_argmax`.
- Bench/profile result:
  - `bench-serve` rc=0,run status `PASS`;
  - c16:completed `[16]`,errored `[0]`,mean `313.252 tok/s`;
  - c32 client / active cap16:completed `[16]`,errored `[0]`,mean
    `315.362 tok/s`;
  - server log captured `264` `[batched-op-profile]` rows.
- Op-profile summary:
  - for batch `m=16` (`117` rows),mean total per decode step
    `26785 us`,p95 `27077 us`;
  - mean shares:unwrapped `55.6%` (`14884 us`),matmul `26.1%`
    (`6980 us`),attention `9.0%` (`2414 us`),QKR `2.4%`
    (`636 us`),norm `1.9%` (`500 us`),other `5.1%` (`1371 us`);
  - for batch `m=10` (`119` rows),shares were similar:unwrapped `55.8%`,
    matmul `26.4%`,attention `8.3%`.
- Interpretation:
  - the current profile does not point first at attention kernels; more than
    half of the measured decode-step time is outside the existing op counters;
  - the next useful checkpoint is to split the `unwrapped` bucket into concrete
    sections,likely Gemma3 sandwich tail/GeGLU/projector glue, device-shadow
    handling, sync/copy, or uninstrumented linear/activation work;
  - avoid a full `--require-ci` performance sweep until that unwrapped bucket
    is explained and reduced.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.

## 2026-06-15 XVIII — W2 greedy-argmax default diagnostic: product default confirmed, performance still below 80%

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_greedy_argmax_default_diag_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=03 cur_state=stopped actual_status=exited intended_status=stopped`。
- Source checkpoint:
  - `9a338235 fix(types): enable greedy argmax for accelerator defaults`;
  - `FERRUM_GREEDY_ARGMAX` now auto-resolves to true on CUDA/Metal when the
    compiled accelerator supports greedy argmax, unless explicitly disabled.
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA greedy-argmax default validation`;
  - expected runtime/cost:20-45min,hard cap 60min,1x RTX 4090 instance
    `40826362` at about USD 0.425/hr;
  - stop condition:startup/SSH/CUDA/build/`ferrum run`/serve smoke first
    failure,decision trace missing `gpu_greedy_argmax`,c16/c32 diagnostic
    complete and artifact copied,or 60min cap;
  - correctness gate:`ferrum run` plus
    `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:diagnostic-only
    `ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --fail-on-error --seed 9271`
    for c16/c32 small sample first.
- Build/correctness:
  - release build PASS,binary SHA256:
    `a942a2e79880bbc821c26a1c60720fa753d6b8e66a62a73900a4592d123abb0e`;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke rc=0,PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Runtime default evidence:
  - both `ferrum run` and diagnostic `ferrum serve` decision traces selected
    `sampling_readback_path=gpu_greedy_argmax`;
  - selected source:`hardware_capability`;
  - `diagnostic_summary.json` reports
    `sampling_trace_has_gpu_greedy_argmax=true`.
- Diagnostic result(same natural ASCII ShareGPT dataset as the vLLM baseline
  and previous Ferrum diagnostic; `num_prompts=32`,`n_repeats=1`,zero errors,
  `output_token_count_source=usage`):
  - c16:completed `[32]`,errored `[0]`,mean `347.880 tok/s`,
    ratio vs vLLM natural baseline LCB `0.708`,ratio vs baseline mean
    `0.655`,required 80% of baseline LCB `392.920 tok/s`,p95 ITL
    `109.763 ms`;
  - c32 client / Ferrum active cap16:completed `[32]`,errored `[0]`,
    mean `356.835 tok/s`,ratio vs vLLM natural baseline LCB `0.650`,
    ratio vs baseline mean `0.634`,required 80% of baseline LCB
    `439.514 tok/s`,p95 ITL `109.657 ms`.
- Interpretation:
  - the typed default fix is product-visible and no hidden env var is needed;
  - performance did not materially improve versus the previous Ferrum natural
    diagnostic(c16 `350.868 tok/s`,c32 `354.291 tok/s`),so the remaining W2
    blocker is not an accidental logits-readback default;
  - continue with targeted decode/attention/batching evidence before any
    full `--require-ci` release sweep.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - inspect current CUDA decode path against the decision trace: notably the
    earlier diagnostic still selected `legacy_paged_decode`;
  - choose one targeted optimization/profiler step that can move c16/c32 tail
    ITL before rerunning the same natural dataset diagnostic.

## 2026-06-15 XVII — W2 Ferrum natural-prompt diagnostic: correctness clean, c16/c32 below 80%

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_ferrum_natural_prompt_diag_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=exited intended_status=stopped`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA natural-prompt Ferrum diagnostic`;
  - expected runtime/cost:20-45min,hard cap 60min,1x RTX 4090 instance
    `40826362` at about USD 0.425/hr;
  - stop condition:startup/SSH/CUDA/build/product-smoke first failure,c16/c32
    diagnostic complete and artifact copied,or 60min cap;
  - correctness gate:`ferrum run` plus
    `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:diagnostic-only
    `ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --fail-on-error --seed 9271`
    for c16/c32 small sample first.
- Build/correctness:
  - release build PASS,binary SHA256:
    `90a30cafef8ea1fe9f1edf3ea326d04dd2f0ca1b8226923ffec559d61d8c5d78`;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke rc=0,PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Dataset:
  - exact same `ascii_sharegpt.jsonl` as
    `w2_natural_prompt_baseline_probe_2026-06-15`;
  - diagnostic run used `num_prompts=32`,`n_repeats=1`,so it is not final
    release evidence and has no CI lower bound.
- Diagnostic result(all usage token counts,zero errors,bad_output `[0]`):
  - c16:completed `[32]`,errored `[0]`,mean `350.868 tok/s`,
    ratio vs vLLM natural baseline LCB `0.714`,ratio vs baseline mean `0.661`,
    required 80% of baseline LCB `392.920 tok/s`,p95 ITL `109.550 ms`
    vs baseline `28.130 ms`;
  - c32 client / Ferrum active cap16:completed `[32]`,errored `[0]`,
    mean `354.291 tok/s`,ratio vs vLLM natural baseline LCB `0.645`,
    ratio vs baseline mean `0.630`,required 80% of baseline LCB
    `439.514 tok/s`,p95 ITL `109.782 ms` vs baseline `27.716 ms`.
- Interpretation:
  - product-path correctness remains clean on the current build;
  - same natural prompt dataset removes the baseline correctness ambiguity and
    shows the remaining W2 blocker is performance,especially tail ITL and c32
    throughput under active cap16;
  - do not expand to a full `--require-ci` release sweep until a targeted
    optimization/profiler step moves c16/c32 close to the 80% thresholds.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - profile/optimize the decode path under natural prompts,with emphasis on
    high p95 ITL and the remaining c16/c32 throughput gap;
  - after a targeted fix, rerun the same natural dataset diagnostic before any
    full release-grade CI sweep.

## 2026-06-15 XVI — W2 natural-prompt baseline probe: vLLM c16/c32 zero-error

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_natural_prompt_baseline_probe_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=exited intended_status=stopped`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA natural-prompt baseline safety probe`;
  - expected runtime/cost:20-50min,hard cap 60min,1x RTX 4090 instance
    `40826362` at about USD 0.425/hr;
  - stop condition:startup/SSH/CUDA/vLLM smoke failure,ShareGPT-style ASCII
    dataset c16 nonzero,c16 pass then c32/cap16 nonzero,probe complete and
    artifact copied,or 60min cap;
  - correctness gate:torch CUDA smoke plus vLLM OpenAI smoke;
  - performance command:diagnostic-only
    `ferrum bench-serve --dataset sharegpt --sharegpt-path <artifact>/ascii_sharegpt.jsonl --fail-on-error --require-ci --seed 9271 --n-repeats 3 --num-prompts 100`
    for c16 then c32.
- Dataset:
  - generated artifact-local JSONL:
    `dataset/ascii_sharegpt.jsonl`;
  - actual tokenizer-counted input length:requested `256`,min/max/mean `112`.
- vLLM setup:
  - venv:`/workspace/vllm-venv-0101-cu126`;
  - server:vLLM OpenAI API,`v0.10.1.1`,`transformers==4.55.4`,
    `--max-model-len 512 --max-num-seqs 16 --gpu-memory-utilization 0.92`;
  - smoke request returned content `"5\n"` with usage.
- Probe result(all `n_repeats=3`, `num_prompts=100`,
  `output_token_count_source=usage`,zero errors,bad_output `[0,0,0]`):
  - c16:completed `[100,100,100]`,errored `[0,0,0]`,
    mean `530.829 tok/s`,ci95 half-width `39.679`,LCB `491.150`,
    p95 ITL `28.130 ms`;
  - c32 client / vLLM `--max-num-seqs 16`:completed `[100,100,100]`,
    errored `[0,0,0]`,mean `562.685 tok/s`,ci95 half-width `13.292`,
    LCB `549.393`,p95 ITL `27.716 ms`.
- Interpretation:
  - natural ASCII ShareGPT-style prompts avoid the vLLM invalid-UTF8 failure
    seen on random-token prompts,so this is a viable correctness-clean
    baseline dataset candidate;
  - the baseline is substantially faster than current Ferrum random-matrix
    c16/c32,so the final W2 80% line would be about c16 `392.9 tok/s`
    and c32 `439.5 tok/s` if this dataset is adopted;
  - no final claim yet: Ferrum must be rerun on the exact same JSONL dataset
    and c32 effective concurrency must be published as 16.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - use this dataset for targeted Ferrum diagnostics or optimization;
  - avoid another full release sweep until there is evidence that Ferrum c16/c32
    can approach the natural-prompt baseline 80% thresholds.

## 2026-06-15 XV — W2 baseline safety probe: vLLM c16 invalid-UTF8 reproduces

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_baseline_safety_probe_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=exited intended_status=stopped`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA mainstream baseline safety probe`;
  - expected runtime/cost:20-50min,hard cap 60min,1x RTX 4090 instance
    `40826362` at about USD 0.425/hr;
  - stop condition:startup/SSH/CUDA/vLLM server smoke failure,any probe cell
    nonzero,invalid-UTF8 reproduction,probe complete and artifact copied,or
    60min cap;
  - correctness gate:torch CUDA smoke plus vLLM OpenAI smoke;
  - performance command:
    `ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3 --num-prompts 100`
    for vLLM c16 first,then c32/cap16 only if c16 passes.
- vLLM setup:
  - venv:`/workspace/vllm-venv-0101-cu126`;
  - server:vLLM OpenAI API,`v0.10.1.1`,`transformers==4.55.4`,
    `--max-model-len 512 --max-num-seqs 16 --gpu-memory-utilization 0.92`;
  - smoke request returned content `"5\n"` with usage.
- Probe result:
  - c16 release-shape rerun reproduced the exact blocker:
    `[err] bad output invalid-utf8: �\"`;
  - repeats completed `[100,100,99]`,errored `[0,0,1]`,bad_output
    `[0,0,1]`,rc=`1`;
  - output token count source:`usage`;
  - diagnostic throughput mean:`385.332 tok/s`,ci95 half-width:`7.385`,LCB
    `377.947`,p95 ITL `27.353 ms`.
- Interpretation:
  - vLLM c16 is fast but not zero-error under the release-shape
    `bench-serve --fail-on-error --require-ci` contract;
  - this confirms the earlier `w2_vllm0101_cuda12_baseline_probe_2026-06-15`
    failure and means vLLM c16 cannot be used in the final W2 manifest as-is;
  - c32 was intentionally not run after the c16 first-fail stop.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - choose a final baseline strategy that can produce zero-error c16/c32
    evidence: either an alternate mainstream engine/config allowed by
    `RELEASE_GRADE_GOAL.md`,or a same-dataset rerun path where both Ferrum and
    baseline are release-clean.

## 2026-06-15 XIV — W2 sentinel-fix Ferrum release-shape matrix PASS;baseline still blocks release-grade

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_sentinel_fix_release_shape_ferrum_ci_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=exited intended_status=stopped`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA sentinel-fix release-shape Ferrum CI matrix`;
  - expected runtime/cost:1.5-3h,hard cap 3h,1x RTX 4090 instance
    `40826362` at about USD 0.425/hr;
  - stop condition:startup/SSH/CUDA/build/correctness first failure,any bench
    cell nonzero or blocker warning,full matrix artifact copied,or 3h cap;
  - correctness gate:CUDA `argmax_rows` test,`ferrum run`,
    `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:
    `ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3 --num-prompts 100`
    for c=1/4/16/32。
- Build/correctness:
  - CUDA `argmax_rows` test PASS,including sentinel case;
  - release build PASS/cache-hit,binary SHA256:
    `6883cc81f3c0a9e16c6c8d374cc98d5c154309e75bd1d7cac7cad832902cbcfb`;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Ferrum release-shape matrix result(all `n_repeats=3`, `num_prompts=100`,
  `output_token_count_source=usage`,zero errors,blocker warning count `0`):
  - c=1:`39.152 tok/s`,ci95 half-width `0.053`,LCB `39.099`,
    completed `[100,100,100]`,errored `[0,0,0]`,p95 ITL `24.618 ms`;
  - c=4:`125.981 tok/s`,ci95 half-width `2.397`,LCB `123.584`,
    completed `[100,100,100]`,errored `[0,0,0]`,p95 ITL `26.236 ms`;
  - c=16:`259.130 tok/s`,ci95 half-width `80.145`,LCB `178.985`,
    completed `[100,100,100]`,errored `[0,0,0]`,p95 ITL `38.334 ms`;
  - c=32 client / typed active cap16 (`--kv-capacity 400 --max-num-seqs 16`):
    `281.525 tok/s`,ci95 half-width `15.552`,LCB `265.973`,
    completed `[100,100,100]`,errored `[0,0,0]`,p95 ITL `39.561 ms`。
- Interpretation:
  - Ferrum product-path correctness and full release-shape matrix now pass on
    the sentinel-fix build;
  - c4 release-shape LCB `123.584` clears same-hardware vLLM c4 80% threshold
    `123.335` by a narrow margin, replacing the earlier 32-request pre-gate as
    better c4 evidence;
  - c16 remains release-grade risk:LCB `178.985` is far below 80% of the
    previous vLLM diagnostic c16 mean (`381.5 * 0.8 = 305.2`), though that vLLM
    c16 run itself had invalid UTF-8 and cannot be final baseline evidence;
  - c32 must be represented as requested c=32 with effective/published
    concurrency 16 unless true active c=32 is implemented.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - create a checkpoint commit for the sentinel-fix/full-matrix evidence;
  - resolve release-grade baseline coverage for c=16/c=32, then assemble
    `model_release_grade_manifest.json` and run
    `python3 scripts/release/model_release_grade_goal_gate.py w2 <out_dir>`。

## 2026-06-15 XIII — W2 c4 CI pre-gate: c4 lower bound clears 80%

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_c4_ci_pregate_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `10:38:02 cur_state=stopped actual_status=running`;
  - `10:38:13 cur_state=stopped actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA c4 release-grade confidence pre-gate`;
  - expected runtime/cost:25-55min,stop cap 75min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:startup/SSH/CUDA/build failure,CUDA argmax test failure,
    `ferrum run` failure,serve smoke failure,c4 `--require-ci --n-repeats 3`
    nonzero/error/warning,c4 CI evidence completes,or 75min cap;
  - correctness gate:CUDA `argmax_rows` test,`ferrum run`,
    `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:
    `ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`
    at c=4;
  - baseline:same-hardware vLLM 0.10.1.1 c4 baseline `154.169 tok/s`,
    80% threshold `123.335 tok/s`。
- Correctness:
  - CUDA `argmax_rows` test PASS,including sentinel case;
  - release build cache-hit PASS;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- c4 CI result:
  - repeats: `32/32/32 completed`, `0/0/0 errored`;
  - output token count source:`usage`;
  - greedy argmax warning count:`0`;
  - throughput mean:`128.988 tok/s`,stddev:`0.501`,ci95 half-width:`1.246`;
  - lower bound:`127.742 tok/s`,ratio to baseline:`0.829`;
  - mean ratio:`128.988 / 154.169 = 0.837`;
  - c4 p95 ITL mean:`25.582 ms`,ci95 half-width:`0.004 ms`;
  - c4 p95 TTFT mean:`814.284 ms`,ci95 half-width:`61.656 ms`;
  - c4 p95 TPOT mean:`27.212 ms`,ci95 half-width:`4.918 ms`。
- Interpretation:
  - c4 now has CI evidence clearing the 80% throughput line;
  - this is still a pre-gate,not final W2 release-grade, because required W2 cells
    c=1/16/32 and final manifest/validator are still missing, and c16/c32
    mainstream baseline handling must be resolved.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - collect Ferrum release-shape CI evidence for c=1/16/32 on the sentinel-fix
    build, then assemble/validate the W2 manifest;
  - in parallel, resolve release-grade baseline coverage for c=16/c=32, because
    previous vLLM c16 failed invalid UTF-8 and cannot be used as final baseline
    evidence as-is.

## 2026-06-15 XII — W2 sentinel-fix c4/c16 diagnostic: c4 mean crosses 80% line

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_sentinel_fix_c4_c16_diag_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `10:26:19 cur_state=stopped actual_status=running`;
  - `10:26:30 cur_state=stopped actual_status=running`;
  - `10:26:43 cur_state=stopped actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA sentinel-fix c4/c16 performance diagnostic`;
  - expected runtime/cost:20-50min,stop cap 70min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:startup/SSH/CUDA/build failure,CUDA argmax test failure,
    `ferrum run` failure,serve smoke failure,any bench cell nonzero error,
    greedy-argmax warning reproduces,c4/c16 diagnostic completes,or 70min cap;
  - correctness gate:CUDA `argmax_rows` test,`ferrum run`,
    `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:diagnostic-only
    `ferrum bench-serve --fail-on-error --seed 9271 --n-repeats 1`
    for c=4 and c=16;
  - baseline:reuse previously saved same-hardware vLLM 0.10.1.1 c4 baseline
    `154.169 tok/s`;baseline not rerun in this diagnostic.
- Correctness:
  - CUDA `argmax_rows` test PASS,including sentinel case;
  - release build cache-hit PASS;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Diagnostic bench result(非 release evidence,N=1,无 CI):
  - c4:`32 completed / 0 errored`, `125.057 tok/s`;
  - c16:`32 completed / 0 errored`, `305.287 tok/s`;
  - output token count source:`usage`;
  - greedy argmax warning count:`0`;
  - c4 p95 ITL:`25.562 ms`,p95 TTFT:`828.434 ms`,p95 TPOT:`30.289 ms`;
  - c16 p95 ITL:`28.539 ms`,p95 TTFT:`3251.742 ms`,p95 TPOT:`47.267 ms`;
  - health after c16:`force_full_logits_calls=0`,`calls=1805`,
    `total_items=10570`,`avg_items_per_call=5.856`,`max_items=16`,
    buckets `m3_4=1271`,`m9_16=379`。
- Interpretation:
  - c4 now crosses the 80% mean line versus same-hardware vLLM baseline:
    `125.057 / 154.169 = 0.811`;80% threshold is `123.335 tok/s`;
  - this is still only N=1 diagnostic evidence,not release-grade performance
    evidence under `RELEASE_GRADE_GOAL.md`;
  - next step should be release-grade performance collection with
    `--fail-on-error --require-ci --seed 9271 --n-repeats 3` for the required
    cells, plus manifest/validator wiring.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade until CI/variance evidence and the final
    validator pass.

## 2026-06-15 XI — W2 masked-argmax sentinel fix: CUDA c16 diagnostic clean

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_masked_argmax_sentinel_fix_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `10:13:50 cur_state=stopped actual_status=running`;
  - `10:14:01 cur_state=stopped actual_status=running`;
  - `10:14:13 cur_state=stopped actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA masked-argmax sentinel-fix validation`;
  - expected runtime/cost:20-45min,stop cap 60min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:startup/SSH/CUDA/build failure,CUDA argmax test failure,
    `ferrum run` failure,serve smoke failure,c16 repeats forbidden-token diagnostic,
    c16 diagnostic clean completes,or 60min cap;
  - correctness gate:CUDA `argmax_rows` masked test including sentinel case,
    `ferrum run`,`scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:diagnostic-only
    `ferrum bench-serve --fail-on-error --seed 9271 --n-repeats 1` at c=16;
  - baseline:reuse previously saved same-hardware vLLM 0.10.1.1 c4 baseline
    `154.169 tok/s`;baseline not rerun in this diagnostic.
- CUDA validation:
  - `argmax_rows_f16_masked_skips_invalid_tokens` PASS;
  - `argmax_rows_f16_masked_returns_sentinel_without_finite_valid_token` PASS;
  - release build PASS:`cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Diagnostic bench result(非 release evidence,N=1,无 CI):
  - c16:`32 completed / 0 errored`, `305.275 tok/s`;
  - output token count source:`usage`;
  - greedy argmax warning count:`0`;
  - decode stats:`force_full_logits_calls=0`,`calls=392`,`total_items=5334`,
    `avg_items_per_call=13.607`,`max_items=16`,bucket `m9_16=379`;
  - run status:`diagnostic_clean`,run rc:`0`。
- Interpretation:
  - sentinel fix removed the reproduced c16 forbidden-token failure in this
    diagnostic shape;
  - c16 diagnostic throughput improved relative to masked-argmax retry
    (`300.242 -> 305.275 tok/s`),but this is still diagnostic and not a
    release-grade performance claim;
  - c4 remains the known release-grade bottleneck:latest valid diagnostic is
    still `120.056 tok/s` vs same-hardware vLLM c4 baseline `154.169 tok/s`,
    ratio `0.779`,below 80% (`123.335 tok/s`).
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - run a targeted c4/c16 diagnostic on the sentinel-fix build,then either:
    c4 clears the 80% mean line and we move to release-grade N>=3/CI evidence,
    or c4 remains below target and we move to the next performance lever;
  - likely next lever remains model hot path/kernel profiling,not scheduler
    formation, because c16 batches are already reaching `avg_m≈13.6`.

## 2026-06-15 X — W2 masked-argmax mask diagnostic:定位到 GPU masked argmax 返回被 mask token

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_masked_argmax_maskdiag_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `09:37:08 cur_state=stopped actual_status=running`;
  - `09:37:19 cur_state=stopped actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA masked-argmax forbidden-token diagnostic`;
  - expected runtime/cost:20-45min,stop cap 60min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:startup/SSH/CUDA/build failure,`ferrum run` failure,serve
    smoke failure,c16 diagnostic completes,forbidden-token diagnostic is captured,or
    60min cap;
  - correctness gate:CUDA `argmax_rows` masked test,`ferrum run`,
    `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:diagnostic-only
    `ferrum bench-serve --fail-on-error --seed 9271 --n-repeats 1` at c=16;
  - baseline:reuse previously saved same-hardware vLLM 0.10.1.1 c4 baseline
    `154.169 tok/s`;baseline not rerun in this diagnostic.
- Correctness before diagnostic bench:
  - CUDA `argmax_rows` masked test PASS;
  - release build PASS:`cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Diagnostic result:
  - c16 diagnostic was manually stopped after the first forbidden-token warning;
  - server warning:
    `token_id=0`, `token_text="<pad>"`, `generated_tokens=126`,
    `forbidden_count=6380`, `base_vocab_size=Some(262144)`,
    `argmax_mask=...len=262144,value=0`;
  - health at stop confirms this was the typed masked-argmax path:
    `force_full_logits_calls=0`, `calls=391`, `total_items=5336`,
    `avg_items_per_call=13.647`, `max_items=16`, bucket `m9_16=379`;
  - GPU memory returned to 1 MiB after manual stop.
- Conclusion:
  - engine-side mask construction was correct for the returned token (`value=0`);
  - the remaining correctness bug is in the CUDA/model masked-argmax path returning
    a token that should have been excluded, specifically the no-finite-valid-token
    fallback/default behavior returning index 0.
- Follow-up source fix:
  - CUDA `argmax_rows_f16_masked` now ignores non-finite logits and returns
    sentinel `u32::MAX` (`-1` as i32) when no finite valid token exists;
  - `LlamaFamilyModel` falls back to full logits if any row returns the sentinel,
    preserving correctness instead of emitting a masked token id;
  - CUDA test added:
    `argmax_rows_f16_masked_returns_sentinel_without_finite_valid_token`;
  - local validation:
    `cargo fmt --all -- --check`,
    `cargo check -q -p ferrum-models --tests`,
    `cargo check -q -p ferrum-kernels --tests`,
    `cargo test -q -p ferrum-engine model_greedy_argmax_sentinel -- --nocapture`,
    `cargo test -q -p ferrum-engine model_decode_logits_policy -- --nocapture`,
    `git diff --check`。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - run a minimal CUDA validation of the sentinel fix: CUDA `argmax_rows` test,
    `ferrum run`,serve smoke,and c16 diagnostic;
  - only after c16 stays clean should we resume c4/c16 performance work toward
    the 80% line.

## 2026-06-15 IX — W2 masked GPU argmax probe: c4 小幅改善,但 c16 暴露 forbidden-token 风险

- 本轮 artifacts:
  - 首次探针:
    `docs/goals/model-coverage-2026-06-12/artifacts/w2_masked_argmax_probe_2026-06-15/`;
  - sentinel 修正后重试:
    `docs/goals/model-coverage-2026-06-12/artifacts/w2_masked_argmax_retry_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。两轮结束后均已同步
  artifact 并停机;stop poll 分别记录:
  - `08:53:13 cur_state=stopped actual_status=exited`;
  - `09:09:09 cur_state=stopped actual_status=exited`。
- 改动:
  - 引入 typed `LogitsReturnPolicy::GreedyArgmax` 与 `TokenSelectionMask`;
  - CUDA `argmax_rows_f16_masked` 支持 GPU 侧 masked greedy argmax;
  - Gemma/Llama-family batched decode 在确定性 greedy/text 输出路径上可避免
    full-logits readback;
  - engine 仍保留 product-side forbidden/initial/extended-vocab/output-quality 校验,
    不允许模型侧 argmax 绕过采样 mask。
- 本地验证:
  - `cargo fmt --all -- --check`;
  - `cargo check -q -p ferrum-interfaces --tests`;
  - `cargo check -q -p ferrum-engine --tests`;
  - `cargo check -q -p ferrum-models --tests`;
  - `cargo check -q -p ferrum-kernels --tests`;
  - targeted engine/model tests for logits policy, sentinel acceptance, decode stats;
  - `git diff --check`。
- GPU correctness:
  - CUDA `argmax_rows` masked test PASS;
  - CUDA `flash_attn_batched_eq` tests PASS;
  - retry build:`CUDA_COMPUTE_CAP=89 cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`;
  - retry `ferrum run` rc=0,content `"5"`;
  - retry serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- 首次探针结果:
  - `ferrum run` 与 serve smoke 通过后,c=4 bench 启动即出现
    `model returned greedy token sentinel for request requiring full logits`;
  - 随后 server log 记录 CUDA illegal-address failure,本轮停止并复制 artifact;
  - 结论是 sentinel 接受条件仍按旧 `requires_full_logits_for_sampling()` 判断,
    未识别 typed masked-greedy product policy。
- 重试结果(非 release evidence,N=1,无 CI):
  - c=4:`32 completed / 0 errored`, `120.056 tok/s`,
    output token count source:`usage`;
  - c=4 health stats:`force_full_logits_calls=0`,
    `avg_items_per_call=3.595`,`max_items=4`;
  - 相比 sliding-window probe c=4 `117.172 tok/s`,增量约 `+2.5%`;
  - 相比 same-hardware vLLM c=4 baseline `154.169 tok/s`,ratio 约 `0.779`,
    仍低于 W2 80% 目标。
- c16 风险:
  - retry 进入 c=16 后 server log 出现 124 次
    `model greedy argmax returned a forbidden token`;
  - 虽然 artifact 中有 c=16 诊断 JSON,本轮已按 first-triage 原则停止,不把
    c=16 作为有效性能证据;
  - 下一步必须先定位 forbidden token id/mask 来源,再继续性能验证。
- 本地 follow-up:
  - `accept_model_greedy_argmax_token` 错误已补充 token id、token text、
    decoded delta、generated token 数、forbidden/initial-forbidden 数量、
    base vocab size 和 allowed-extended 数量;
  - 新增 targeted test 断言 forbidden-token 错误包含关键诊断字段;
  - 验证命令:
    `cargo fmt --all -- --check`,
    `cargo check -q -p ferrum-engine --tests`,
    `cargo test -q -p ferrum-engine model_greedy_argmax_sentinel -- --nocapture`,
    `cargo test -q -p ferrum-engine model_decode_logits_policy -- --nocapture`,
    `git diff --check`。
- 发布级判定:
  - 未生成 `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未完成。
- 下一步:
  - 用新增诊断跑一次小 CUDA 探针,确认 masked argmax 是否仍返回被 mask token,
    或是 per-sequence mask/fingerprint 选择错误;
  - 正确性修复后还需要继续找至少约 `3.3 tok/s` 的 c=4 增量,才能越过
    `0.80 * 154.169 = 123.335 tok/s`。

## 2026-06-15 VIII — W2 sliding-window batched attention: c16 明显改善,c4 仍未达 80%

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_sliding_batched_attn_probe_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。结束后已同步
  artifact 并停机;Vast API poll 记录 `cur_state=stopped`,
  `actual_status=exited`。停止前 `nvidia-smi` 显示无运行进程,GPU memory 1 MiB。
- 改动:
  - CUDA head-major single decode attention 增加 `sliding_window` 实现;
  - CUDA `flash_attention_batched_per_cache` 增加 common `sliding_window` 参数;
  - Gemma3 local-window 层不再强制 per-item attention fallback,而是通过 batched
    attention kernel 处理 local window;
  - 新增 CUDA test:
    `flash_attn_batched_sliding_window_one_selects_latest_v`。
- 本地验证:
  - `cargo fmt --all -- --check`;
  - `cargo check -q -p ferrum-models --tests`;
  - `cargo test -q -p ferrum-models decode_batch_stats_snapshot_records_shape_and_fallbacks -- --nocapture`;
  - `cargo check -q -p ferrum-kernels --tests`;
  - `git diff --check`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA sliding-window batched-attn probe`;
  - expected runtime/cost:45-75min,stop cap 90min / 约 USD 0.65;
  - stop condition:CUDA kernel test 失败、`ferrum run`/serve smoke 失败、c=4/c=16
    诊断完成,或达到 90min;
  - correctness gate:
    `cargo test --release -p ferrum-kernels --features cuda --test flash_attn_batched_eq -- --nocapture`,
    再跑 `ferrum run` 与 `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:诊断型
    `ferrum bench-serve --fail-on-error --seed 9271 --n-repeats 1`
    分别跑 c=4 与 c=16。
- CUDA kernel test:
  - `flash_attn_batched_matches_per_item`:max diff `3.052e-5`;
  - `flash_attn_batched_sliding_window_one_selects_latest_v`:max diff `1.206e-4`;
  - `test result: ok. 2 passed`。
- Product correctness:
  - release build:`CUDA_COMPUTE_CAP=89 cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Diagnostic bench 结果(非 release evidence,N=1,无 CI):
  - c=4:`32 completed / 0 errored`, `117.172 tok/s`;
  - c=16:`32 completed / 0 errored`, `245.801 tok/s`;
  - output token count source:`usage`。
- 对比上一轮 stats probe:
  - c=4:`105.050 -> 117.172 tok/s`,约 `+11.5%`;
  - c=16:`177.364 -> 245.801 tok/s`,约 `+38.6%`;
  - c=16 段增量仍形成大 batch:`avg_m=13.677`,bucket `m9_16=379/390`。
- 发布级判定:
  - 同硬件 vLLM c=4 baseline 为 `154.169 tok/s`;本轮 Ferrum c=4 诊断 ratio
    约 `0.760`,仍低于 80%;
  - c=16 vLLM baseline 仍因 invalid UTF-8 不能作为 release-grade evidence,但
    诊断 ratio 约 `245.8 / 381.4 = 0.645`;
  - 未生成 `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未完成。
- 下一步:
  - 继续一个窄性能 lever,优先用 `FERRUM_DECODE_OP_PROFILE` 或等价可记录 artifact
    确认剩余 c=4 gap 是否主要在 full-logits/lm_head readback、qkr/kv append、
    MLP/GEMM 小 m 效率,或 attention 本身;
  - 不重复 full sweep,直到 c=4 diagnostic ratio 明确越过 80% 或定位出下一个
    高收益修复。

## 2026-06-15 VII — W2 decode batch stats probe: c16 已形成大 batch,瓶颈转向模型/kernel 路径

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_decode_batch_stats_probe_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。结束后已同步
  artifact 并停机;Vast API poll 记录 `cur_state=stopped`,
  `actual_status=exited`。停止前 `nvidia-smi` 显示无运行进程,GPU memory 1 MiB。
- 本轮新增 source instrumentation:
  - `LlamaFamilyModel` 通过 `/health.cache.prefix_cache.decode_batch` 暴露
    decode_batch 调用数、total rows、max m、m bucket、fallback 计数;
  - 只记录 metrics,不改变 scheduler/model/kernel/sampling 行为;
  - 本地验证:
    `cargo test -q -p ferrum-models decode_batch_stats_snapshot_records_shape_and_fallbacks -- --nocapture`,
    `cargo fmt --all -- --check`,
    `cargo check -q -p ferrum-models --tests`,
    `git diff --check`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA batched-decode stats probe`;
  - expected runtime/cost:35-60min,stop cap 90min / 约 USD 0.60;
  - stop condition:启动/SSH/构建失败、`ferrum run`/serve smoke 失败、c=16
    bench 出错、或采集完 health stats;
  - correctness gate:`ferrum run` + `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:诊断型
    `ferrum bench-serve --fail-on-error --seed 9271 --n-repeats 1`
    分别跑 c=4 与 c=16。
- Correctness:
  - release build:`CUDA_COMPUTE_CAP=89 cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`;
  - `ferrum run` rc=0;
  - serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Diagnostic bench 结果(非 release evidence,N=1,无 CI):
  - c=4:`32 completed / 0 errored`, `105.050 tok/s`;
  - c=16:`32 completed / 0 errored`, `177.364 tok/s`;
  - output token count source:`usage`。
- Decode batch stats:
  - c=4 段后累计:`calls=1422,total_items=5334,avg_m=3.751,max_m=4`;
    bucket:`m1=13,m2=151,m3_4=1258`;
  - c=16 段增量:`calls=391,total_items=5334,avg_m=13.642,max_m=16`;
    bucket:`m1=1,m3_4=4,m5_8=8,m9_16=378`;
  - fallback:`unsupported_fallback_calls=0,lora_fallback_calls=0`;
  - server log 首个 batched decode:`m=4 use_batched_qkr=true`,
    `batched-kv-append ok=true`, `batched-attn ok=true`。
- 结论:
  - c=16 扩展性差不是 scheduler 没形成大 batch;闭环 c=16 段实际平均 m≈13.6,
    绝大多数调用在 m=9..16;
  - 下一步应转向模型/kernel hot path,尤其是 Gemma local-window 层是否仍大量
    per-item attention、full-logits readback/sampling、以及 per-layer qkr/attention/MLP
    profile;
  - W2 仍未完成:没有 `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。

## 2026-06-15 VI — W2 CUDA12 vLLM baseline probe:server 可用,但 release-grade 失败

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_vllm0101_cuda12_baseline_probe_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。结束后已同步
  artifact 并停机;Vast API poll 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3-27B CUDA vLLM 0.10.1.1/CUDA12 baseline probe`;
  - expected runtime/cost:45-120min,约 USD 0.32-0.85;
  - stop condition:torch CUDA smoke 失败、vLLM Gemma3/GPTQ 不支持并保存日志、
    vLLM OpenAI smoke + baseline 完成,或任一 baseline cell 非零错误;
  - correctness gate:`torch.cuda` smoke,再走 vLLM OpenAI
    `/v1/chat/completions` 非空内容 + usage;
  - performance command:`ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`。
- 环境与依赖:
  - GPU:`NVIDIA GeForce RTX 4090,24564 MiB,driver 565.77`;
  - `vllm==0.10.1.1`, `torch==2.7.1+cu126`;
  - 初始 pip 解析到 `transformers 5.12.0`,导致 Gemma3TextConfig 的 nested
    rope_scaling 与 vLLM 0.10.1.1 不兼容:
    `rope_scaling should have a 'rope_type' key`;
  - pin `transformers==4.55.4` 后模型 config smoke 通过;
  - 初始 pip 解析到 `fastapi 0.137.0` / `starlette 1.3.1` /
    `prometheus-fastapi-instrumentator 8.0.0`, `/v1/models` 触发
    `'_IncludedRouter' object has no attribute 'path'`;
  - pin `fastapi==0.116.1`, `starlette==0.47.2`,
    `prometheus-fastapi-instrumentator==7.1.0` 后 `pip check` clean。
- vLLM product-path smoke:
  - server 成功加载同一 HF/safetensors GPTQ model,日志显示
    `Resolved architecture: Gemma3ForCausalLM` 与
    `Using gptq_marlin kernel`;
  - `/v1/models` rc=0;
  - `/v1/chat/completions` 非流式 smoke rc=0,返回非空 content 且 usage 含
    completion tokens。
- Baseline 结果:
  - c=1: vLLM `43.486 tok/s`,Ferrum `40.021 tok/s`,mean ratio `0.920`;
  - c=4: vLLM `154.169 tok/s`,Ferrum `105.158 tok/s`,mean ratio `0.682`;
  - c=16: vLLM 两次 N=3 rerun 都在第三轮产生
    `bad output invalid-utf8: �"`;`--fail-on-error` 非零,因此 c=16 不能作为
    release-grade baseline evidence。诊断均值约 `381.4 tok/s`,Ferrum c=16
    `165.469 tok/s`,mean ratio 约 `0.434`。
- 发布级判定:
  - 有效 c=4 same-hardware mainstream baseline 已证明当前 Ferrum 低于 80%;
  - vLLM c=16 baseline 自身没有零错误,仍不能进入 final manifest;
  - 未生成 `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 release-grade 仍未完成。
- 下一步:
  - 继续 W2-P2 性能修复,重点缩小 c=4/c=16 gap,不能用 hidden env 或 sampler
    参数绕过输出问题;
  - 优先审计 Gemma3 batched decode 是否实际形成 m=4 以上的 decode batch、
    local/sliding attention 是否仍强制小 batch fallback、以及 per-layer residual/
    norm/GeGLU 是否还有 host sync 或重复 materialize;
  - 下一轮 CUDA 只跑 targeted A/B 或 smoke,不要重复 full sweep,直到有明确
    高收益改动。

## 2026-06-15 V — W2 baseline probe:latest vLLM 0.23.0 安装成功但 CUDA13/driver565 不可用

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_vllm_baseline_probe_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。验证结束后已
  复制 artifact 并停机;Vast API poll 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3-27B CUDA vLLM/GPTQ baseline probe`;
  - expected runtime/cost:30-90min,约 USD 0.21-0.64;
  - stop condition:vLLM 明确不支持该模型/量化并保存日志、vLLM smoke 通过并完成
    baseline、安装/启动超过 90min 无进展,或任一 baseline cell 非零错误;
  - correctness gate:vLLM OpenAI `/v1/chat/completions` 简单问题返回有效非空内容;
  - performance command:通过 smoke 后用
    `ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`。
- 环境:
  - GPU:`NVIDIA GeForce RTX 4090,24564 MiB,driver 565.77`;
  - CUDA compiler:`cuda_12.4.r12.4`;
  - Python:`3.10.12`;
  - 初始环境无 `vllm`/`torch`,Gemma3 GPTQ cache 存在。
- 安装过程:
  - 初次 `python3 -m venv /workspace/vllm-venv` 失败,因为镜像缺
    `python3.10-venv`;
  - 安装 `python3.10-venv` 后重建 venv;
  - `pip install vllm` 成功,解析到 `vllm 0.23.0` 与 `torch 2.11.0`;
  - 该 torch wheel 依赖 CUDA13 运行时包,包括 `cuda-toolkit 13.0.2`,
    `nvidia-cublas 13.1.0.3`, `nvidia-cudnn-cu13 9.19.0.56`,
    `nvidia-nccl-cu13 2.28.9` 等。
- CUDA smoke:
  - 命令:`/workspace/vllm-venv/bin/python import_smoke`;
  - 失败位置:`torch.cuda.get_device_name(0)` 触发 `_cuda_init()`;
  - 错误:
    `RuntimeError: The NVIDIA driver on your system is too old (found version 12070)`;
  - 因此 latest vLLM 0.23.0/CUDA13 wheel 栈不能在当前 driver 565.77 机器上作为
    W2 same-hardware baseline。
- 发布级判定:
  - 本轮没有生成 baseline throughput,没有 `model_release_grade_manifest.json`,
    没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 release-grade 仍未完成。
- 下一步:
  - 不再继续 latest vLLM/CUDA13 路线,除非先更换/升级同硬件 driver;
  - 在同一 4090/cache-retained instance 上尝试 CUDA12 兼容的 vLLM 版本
    (独立 venv,先 `torch.cuda` smoke,再 vLLM server smoke);
  - 若 CUDA12-compatible vLLM 也不支持 Gemma3 GPTQ,保存明确模型/量化不支持证据,
    再选择 `RELEASE_GRADE_GOAL.md` 允许的最快同模型同格式 mainstream engine。

## 2026-06-15 IV — W2 Ferrum release-shape 全矩阵 PASS,release-grade 仍缺 mainstream baseline

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_release_shape_ferrum_cuda_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。验证结束后已
  复制 artifact 并停机;Vast API poll 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3-27B CUDA Ferrum release-shape matrix`;
  - expected runtime/cost:1.5-3h,约 USD 0.64-1.28;
  - stop condition:correctness 首败、任一 release-shape cell 非零错误、全矩阵
    完成并回收 artifact,或 3h;
  - correctness gate:`ferrum run` + `ferrum serve` smoke;
  - performance command:`bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`
    覆盖 c=1/4/16/32。
- 远端源码/硬件:
  - git SHA `2656cc1a4c1b4f722f14700a5e50d4e0af37db14`;
  - 远端 dirty status 保存于 `remote_metadata.txt`;
  - GPU:`NVIDIA GeForce RTX 4090,24564 MiB,driver 565.77`;
  - CUDA compiler:`cuda_12.4.r12.4`;
  - Rust:`cargo 1.96.0`, `rustc 1.96.0`。
- CUDA release build:
  - `CUDA_COMPUTE_CAP=89 cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`
    PASS;
  - binary SHA256:
    `5fa06ab8dc93285bccca692702d5386bfbb39a8a6ba3e8e6b66a2467ee99c6b8`。
- Correctness/product path:
  - `ferrum run gemma3:27b-gptq --backend cuda ... --kv-capacity 2560 --max-num-seqs 2`
    rc=0,输出 `content:"5"`, `finish_reason:"stop"`;
  - `scripts/model_coverage_smoke.sh gemma3:27b-gptq --port 8401 --kv-capacity 2560 --max-seqs 2`
    rc=0,stdout 打印 `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Ferrum release-shape L5:
  - c=1:`40.0214 tok/s`,completed `[100,100,100]`,errored `[0,0,0]`,
    usage count,n_repeats=3;
  - c=4:`105.1577 tok/s`,completed `[100,100,100]`,errored `[0,0,0]`,
    usage count,n_repeats=3;
  - c=16:`165.4689 tok/s`,completed `[100,100,100]`,errored `[0,0,0]`,
    usage count,n_repeats=3;
  - c=32 typed cap16 (`--kv-capacity 400 --max-num-seqs 16`):
    `169.4372 tok/s`,completed `[100,100,100]`,errored `[0,0,0]`,
    usage count,n_repeats=3;
  - c=32 server log 首个并发 decode 为
    `[batched-qkr] first batched_decode call: m=4 use_batched_qkr=true`,
    证明 release-shape 并发路径实际触发 legacy batched decode。
- 发布级判定:
  - Ferrum 侧 release-shape correctness/perf matrix 已干净,且相对上一轮
    flat 40 tok/s 有明确 c=4/16/32 提升;
  - 仍未补同硬件、同模型、同量化/格式的 mainstream baseline,也未生成
    `model_release_grade_manifest.json` 与最终
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - 因此 W2 仍不能宣称 release-grade。
- 下一步:
  - 在同一 RTX 4090 上优先尝试 vLLM/GPTQ baseline;若当前 vLLM 不支持
    Gemma3 27B GPTQ,必须保存不支持证据并选择目标文档允许的最快同模型同格式
    mainstream engine;
  - baseline 必须用同一 prompt/cell、同一 effective active cap(c=32 cap16 时),
    并保存 engine version/build/runtime config;
  - 之后生成 `model_release_grade_manifest.json` 并运行
    `python3 scripts/release/model_release_grade_goal_gate.py w2 <out_dir>`。

## 2026-06-15 III — W2-P2 legacy batched decode CUDA 诊断:correctness PASS,并发路径已触发

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_legacy_batched_cuda_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。验证结束后已
  复制 artifact 并停机;Vast API poll 记录 `cur_state=stopped`,
  `actual_status=exited`。
- 远端源码状态:
  - git SHA `2656cc1a4c1b4f722f14700a5e50d4e0af37db14`;
  - 远端 `git status --short` 保存于 `remote_metadata.txt`,包含本轮源码改动与既有
    未跟踪 W2 artifacts;
  - 为减少付费 GPU 空转,第二次 rsync 排除了历史 `docs/.../artifacts/`;
    本轮是 diagnostic,不是最终 release-grade artifact collection。
- CUDA release build:
  - `CUDA_COMPUTE_CAP=89 cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`
    PASS;
  - binary SHA256:
    `5fa06ab8dc93285bccca692702d5386bfbb39a8a6ba3e8e6b66a2467ee99c6b8`。
- Correctness/product path:
  - `ferrum run gemma3:27b-gptq --backend cuda ... --kv-capacity 2560 --max-num-seqs 2`
    rc=0,输出 `content:"5"`, `finish_reason:"stop"`;
  - `scripts/model_coverage_smoke.sh gemma3:27b-gptq --port 8401 --kv-capacity 2560 --max-seqs 2`
    rc=0,stdout 打印 `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`,覆盖
    known-answer 10/10,multi-turn,stream==non-stream,custom stop,tool-call,
    strict json_schema;
  - run/serve 日志均显示 `Gemma3 family: legacy batched_decode=true varlen_unified=false`。
- Diagnostic perf/correctness:
  - typed serve:`--kv-capacity 512 --max-num-seqs 16`;
  - `bench-serve --random-input-len 256 --random-output-len 128 --concurrency-sweep 4,16 --num-prompts 32 --n-repeats 1 --fail-on-error --seed 9271`;
  - c=4:completed `[32]`,errored `[0]`,usage count,105.5 tok/s;
  - c=16:completed `[32]`,errored `[0]`,usage count,177.3 tok/s;
  - server log 首个并发 decode 为
    `[batched-qkr] first batched_decode call: m=4 use_batched_qkr=true`,证明本轮
    legacy batched decode 窄开关在 CUDA 并发路径上实际触发。
- 发布级判定:
  - 未运行 `--require-ci --n-repeats 3` 全矩阵,未跑 c=32,未补同硬件主流引擎
    80% baseline,未生成 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - 因此 W2 仍是 functional/diagnostic 进展,不能宣称 release-grade。
- 下一步:
  - 在相同路径上跑 release-grade 形状的 c=1/4/16/32 correctness/perf,并补 baseline;
  - 若 c=32 仍需 active admission cap,release-grade manifest 必须把 effective
    concurrency/cap 与 baseline 对齐;
  - 继续审计 `batched-attn m=4 ok=true` 是否在 local-window 层走了预期 per-item
    fallback,以及是否还有 Gemma3 tail/attention 可融合热点。

## 2026-06-15 II — W2-P2 legacy batched decode 窄开关:CUDA 候选,待 GPU 验证

- 本轮仍未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 代码侧推进:
  - Gemma/sandwich family 的 `supports_batched_decode` 不再无条件禁用;
  - 窄开关条件为: sandwich norms + 非零 `sliding_window_pattern` + 后端支持
    device-side F32 residual shadow。当前目标是只让 CUDA Gemma3 进入 legacy
    contiguous batched decode 候选路径;
  - `supports_varlen_qkv` 对 sandwich 仍禁用,因为 paged/unified attention 还没有
    per-layer local-window 语义;
  - active LoRA cache 继续 fallback 到 per-item decode,因为 legacy batched qkv/o/gate/down
    仍不携带 per-cache LoRA adapter;
  - layer-split pipeline 的 batch stage 对 sandwich family 继续 fallback,避免 full-model
    decode_batch capability 影响未验证的 pipeline hidden path。
- 新增/更新测试:
  - `sandwich_legacy_batched_decode_requires_device_shadow_and_layer_schedule` 锁定
    capability 条件,避免后续扩大到无 device shadow 或无 Gemma layer schedule。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check` PASS;
  - `cargo check -q -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-models sandwich_legacy_batched_decode_requires_device_shadow_and_layer_schedule -- --nocapture` PASS;
  - `cargo test -q -p ferrum-models --lib` PASS,123 tests passed;
  - `cargo check -q -p ferrum-cli --all-targets` PASS.
- 下一步:
  - 必须在 1x RTX 4090 上跑 CUDA correctness smoke,确认 `ferrum run` 和
    `ferrum serve` 在新 legacy batched decode 路径下无 `<unk>`/`[PAD]`/
    NaN/stream DONE 问题;
  - 通过 correctness 后再跑短 c=4/16 diagnostic,观察吞吐是否不再完全平坦。

## 2026-06-15 — W2-P2 batched decode 语义铺垫:共享 Gemma tail,fast path 仍禁用

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 代码侧推进:
  - `LlamaFamilyModel` 抽出 `forward_layer_post_o_proj_with_residual_shadow`,
    让单序列 post-attn tail 与 legacy batched decode 在 o_proj 后共享同一套
    sandwich norm / GeGLU / device-F32 residual shadow 语义;
  - legacy batched decode layer 现在可从 host/device residual shadow 做 input
    RMSNorm,并把 Gemma embedding scale、shadow 初始化、final norm、shadow 归还
    接入 `decode_batch_internal_with_full_logits`;
  - legacy contig batched decode attention 现在按 source layer 选择 Gemma3
    local/global rope 与 `layer_window`;由于 single-launch batched attention
    kernel 尚无 sliding-window 参数,local-window 层会强制走 per-item attention
    fallback,保留正确性语义;
  - batched CUDA graph 在 host/device residual shadow 路径下保持禁用,避免 graph
    replay 在未验证的 Gemma shadow 状态上成为隐式产品路径。
- 仍未完成:
  - `supports_batched_decode` / `supports_varlen_qkv` 对 `sandwich_norms` 仍保持
    false,所以用户路径仍走已验证的 per-item decode;
  - paged/unified attention kernel API 仍未完整支持 Gemma3 per-layer local-window
    语义;unified tail 仍需单独接入 sandwich/shadow 语义;
  - 因此这只是 W2-P2 的语义地基,不是性能通过声明。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `cargo check -q -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-models --lib` PASS,122 tests passed。

## 2026-06-14(发布级 II)— W2 device-side F32 shadow:正确性 PASS,诊断 L5 提升,仍非 release-grade

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_release_grade_device_shadow_cuda_2026-06-14/`。
- 复用 Vast/cache-retained CUDA instance `40826362`。验证结束后已复制 artifact 并
  停机;Vast API poll 记录 `cur_state=stopped`,`actual_status=exited`。
- 代码修正范围:
  - Gemma3 CUDA sandwich-norm 路径改为 device-side F32 residual shadow,避免每层
    host F32 shadow readback/copy;
  - `common.cuh` 的 block reduce helper 修正 `blockDim < 32` 时的
    `num_warps=(blockDim.x+31)/32`,否则小尺寸 CUDA precision tests 会把 variance
    归零并放大到 `rsqrt(eps)`;
  - `cuda/quant.rs` 的空 env-var test fixture 加显式类型,只影响测试编译。
- CUDA feature build/test:
  - `CUDA_COMPUTE_CAP=89 cargo check -q -p ferrum-kernels --tests --features cuda`
    PASS;
  - `CUDA_COMPUTE_CAP=89 cargo test -q -p ferrum-kernels --test cuda_activation_precision --features cuda`
    PASS,4 tests passed;
  - release binary command:
    `CUDA_COMPUTE_CAP=89 cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`;
  - binary SHA256:
    `3af53becc860a5e038cda486da69de4fc5aa6e8d81543d04aba0dbebbe6a393f`.
- 产品 correctness:
  - `ferrum run gemma3:27b-gptq --backend cuda --prompt ... --kv-capacity 2560 --max-num-seqs 2 --output-format jsonl`
    PASS:assistant content `5`,finish_reason `stop`,n_tokens `3`;
  - `ferrum serve` smoke PASS:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`,覆盖 known-answer 10/10,
    natural EOS,multi-turn,stream/non-stream,custom stop,max_tokens,
    tool-call 10/10,strict json_schema 20/20。
- Ferrum post-device-shadow diagnostic L5:
  - c=1:`40.5367 tok/s`,100/100/100 completed,0 errors;
  - c=4:`40.4754 tok/s`,100/100/100 completed,0 errors;
  - c=16:`40.3856 tok/s`,100/100/100 completed,0 errors;
  - c=32 typed cap16 (`--kv-capacity 400 --max-num-seqs 16`):
    `40.3111 tok/s`,100/100/100 completed,0 errors.
- 对旧同卡 llama.cpp GGUF sanity baseline `50.478 tok/s` 的诊断 ratio:
  c=1 `0.8031`,c=4 `0.8018`,c=16 `0.8001`,c=32 `0.7986`。
  这些数字只能说明 device-side F32 shadow 把旧 host-shadow 的约 `25 tok/s`
  提升到约 `40 tok/s`;按 `RELEASE_GRADE_GOAL.md`,跨格式 llama.cpp 不能作为
  CUDA GPTQ 正式 80% baseline。
- 当前结论:
  - W2 Gemma3 CUDA GPTQ correctness/product path 已保持 PASS;
  - c=1/4/16/32 Ferrum diagnostic L5 已干净,但并发吞吐仍基本平坦,说明
    batched/varlen Gemma3 fast path 仍是 release-grade 性能工作项;
  - 未生成 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`,W2 仍不得宣称
    release-grade。

## 2026-06-14(发布级 I)— release-grade gate 与发布口径收窄

- 新增发布级目标文档 `RELEASE_GRADE_GOAL.md`:明确 W2 coverage PASS 不等于
  release-grade,后续 README/release notes/性能宣传以 80% 主流引擎 baseline
  为硬门。
- 新增 validator:
  `scripts/release/model_release_grade_goal_gate.py`。
  - 命令:`python3 scripts/release/model_release_grade_goal_gate.py w2 <out_dir>`
    或 `w3 <out_dir>`;
  - 输入:`<out_dir>/model_release_grade_manifest.json`;
  - 输出:`model_release_grade_goal_gate.manifest.json`;
  - 只有 stdout 打印 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` 或
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` 才能宣称本文目标完成。
- validator self-test 已覆盖:
  - 合格 W2 manifest 可 PASS;
  - ratio `0.79` 会 FAIL;
  - `runtime_config.hidden_env` 非空会 FAIL。
- README/README_zh 已把 Gemma 3 27B CUDA 口径从 "certified/release-like"
  收窄为 "functional / known-gap":当前同卡 llama.cpp ratio `0.500260x`,
  低于 release-grade `0.8x`,因此不能写成 release-grade 支持。
- W2-P2 预备修正:
  - CUDA `Backend::rms_norm` 现在按 buffer dtype 分派 `rms_norm_f16` /
    `rms_norm_f32`,不再无条件调用 F16 kernel;
  - 新增 `cuda_activation_precision::f32_rms_norm_uses_f32_kernel`,为后续
    device-side F32 residual shadow / sandwich-norm 路径铺路;
  - 这不是 release-grade 性能修复,也没有启用 Gemma3 batched/varlen fast path。
- 本轮本地验证:
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py`
    PASS;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS:`MODEL RELEASE GRADE GOAL SELFTEST PASS`;
  - `python3 scripts/release/selftest_g0_validators.py`
    PASS:`G0 VALIDATOR SELFTEST PASS`;
  - `cargo fmt --all -- --check` PASS;
  - `cargo check -q -p ferrum-kernels --tests` PASS;
  - `cargo test -q -p ferrum-kernels --test cuda_activation_precision`
    PASS locally with 0 tests because CUDA feature is not enabled on this host.

## 2026-06-14(早 VI)— W2 L5/perf 打通:客户端 c=32 + admission cap 16 PASS

- 继续复用 stopped/cache-retained native CUDA instance `40826362`,没有重新租
  pod 或重装环境。全部验证结束后已复制 artifact 并停机,Vast API 确认
  `cur_state=stopped`,`actual_status=exited`。
- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_c32_admission16_l5_pass_2026-06-14/`。
- L5 c=32 最终产品合同:
  - 客户端并发仍是 W2 要求的 `bench-serve --concurrency-sweep 32`;
  - 服务端使用 typed product CLI `--max-num-seqs 16 --kv-capacity 400`;
  - 这是产品入口参数,不是隐藏 env。health/auto_config 记录
    `selected_admission_limit=16`。
- 先跑小探针 `num_prompts=32,n_repeats=1`:32/32 completed,0 errored,
  throughput 24.9 tok/s。随后跑正式门:
  `num_prompts=100,n_repeats=3,--fail-on-error,--require-ci,--seed 9271`。
- 正式 c=32 结果:
  - repeat 1:`100 completed / 0 errored / 521.3s`
  - repeat 2:`100 completed / 0 errored / 515.1s`
  - repeat 3:`100 completed / 0 errored / 514.0s`
  - JSON: `full/full/l5_gemma3-27b-gptq_cuda_c32_admission16.json`;
  - merged L5 JSON:
    `l5_gemma3-27b-gptq_cuda.json`,覆盖 c=1/4/16/32。
- llama.cpp same-card perf:
  - 远端缺 `llama-bench` binary,保留源码/缓存;为避免全架构 CUDA 编译浪费,
    中断了误启动的全架构 build,重新配置 `build-sm89` 只编译
    `CMAKE_CUDA_ARCHITECTURES=89`;
  - 命令:`llama-bench -m <Gemma3-27B-Q4_K_M.gguf> -ngl 999 -p 0 -n 128 -r 3 -o json`;
  - llama.cpp tg128:50.478285 tok/s;
  - Ferrum c=1 decode:25.252275 tok/s;
  - ratio:0.500260 PASS,刚过 0.5 floor,应记录为 known-gap,不是性能优化完成声明。
- 当前结论:
  - W2 矩阵 8/8 可满足;待 validator exact PASS 作为最终 W2 完成证据。
  - W3 仍只交付立项合同,不在本目标内宣称 W3 完成。

## 2026-06-14(凌晨 V)— W2 c=32 admission cap 31/30 验证:排队不足以解除 24GB OOM

- 按用户建议继续复用 stopped/cache-retained native CUDA instance
  `40826362`,没有重新开机器或重装环境。验证/诊断完成后已复制 artifact 并停机,
  Vast API 确认 `cur_state=stopped`,`actual_status=exited`。
- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_c32_admission_cap_diagnostics_2026-06-14/`。
- 产品路径 admission-cap 复测:
  - 客户端仍保持 W2 L5 要求的 `bench-serve random 256/128 c=32`
    三轮命令形态;
  - 服务端分别使用 typed CLI 参数 `--max-num-seqs 31` 和
    `--max-num-seqs 30`,并保持 `--kv-capacity 400`;
  - 两轮均 health 第 60 秒通过,进入
    `closed_loop c=32 — repeat 1/3`,随后 OOM。
- 两个 cap 的 OOM 同形:
  `CudaBackend::alloc failed: dtype=F16 elements=804864 bytes=1609728 free=17498112 total=25278087168`.
  也就是 unified product-path KV hint 已把单 cache 物理容量降到
  `393 * 16 * 128`,但 24GB 总显存仍不足。
- 附加 scheduler 诊断(非 PASS 证据):开启 `FERRUM_SCHED_NONE_PROF=1`
  做短 c=32 trace,拿到第一条
  `[sched-some] n=0 returning_batch=4 | decode_queue=0 prefill_queue=6 waiting_queue=0`,
  说明首个 iteration 不是直接一次性塞满 30/32 个请求;当前 blocker 更像是
  活跃请求累计 + Gemma3-27B fp16 contiguous KV 生命周期/总峰值问题,不是简单的
  首批 admission 数过大。
- 当前结论:
  - W2 仍不是 PASS。矩阵保持 l0/l1/l2_gptq/l3/l4 pass,
    l2_gguf waived,l5_concurrency fail,perf_vs_llamacpp pending。
  - `--max-num-seqs 31/30` 不能作为 L5 绕过;继续往下盲调 cap 价值低。
  - 下一步应在本地先定位 KV 生命周期/释放/复用或 Gemma3 paged-window 结构方案,
    再用 GPU 做最小验证;否则需要目标合同明确修订 L5。

## 2026-06-14(凌晨 IV)— W2 c=32 KV-hint 产品路径验证:hint 生效但 24GB 仍不足

- 按用户建议继续复用同一台 stopped/cache-retained native CUDA instance
  `40826362`,没有重新开机器或重装环境。验证结束后已复制 artifact 并停机,
  Vast API 确认 `cur_state=stopped`,`actual_status=exited`。
- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_c32_kv_hint_diagnostics_2026-06-14/`。
- 代码侧保留的产品路径改动:
  - engine 在请求 metadata 中写入 `ferrum_kv_capacity_hint`;
  - `LlmExecutor` 的 `prefill`/`batch_prefill`/`unified_decode` 在 fresh
    prefill 前调用 `prepare_kv_capacity`;
  - contiguous fp16 KV 支持按 hint 分配物理容量,逻辑 `kv_capacity`
    仍用于上下文校验;
  - hint 使用实际会写入 KV 的上限:`input_tokens + max_tokens - 1`
    (prefill 后采样出的首 token 不写入 KV)。
- native CUDA c=32 结果:
  1. 初版只覆盖 `prefill`/`batch_prefill`,但 W2 serve 走
     `unified_decode`;c=32 仍按 400 分配并 OOM:
     `elements=819200 = 400 * 16 * 128`。
  2. 补上 unified prefill hook 后,hint 生效;失败分配降为
     `elements=806912 = 394 * 16 * 128`,但仍 OOM。
  3. 修正 off-by-one 后,失败分配继续降为
     `elements=804864 = 393 * 16 * 128`,但仍 OOM;OOM 时 `free`
     仍约 17.5MiB,说明不是 prompt inflation 或 hook 缺失,而是 fp16
     contiguous KV c=32 峰值在 24GB 4090 上仍压线失败。
- 当前结论:
  - W2 仍不是 PASS。矩阵保持 l0/l1/l2_gptq/l3/l4 pass,
    l2_gguf waived,l5_concurrency fail,perf_vs_llamacpp pending。
  - `--kv-dtype int8` 已在上一轮证明 correctness 不可接受;不能作为绕过。
  - 后续若继续推进,需要结构性内存方案(例如可验证的 Gemma3 paged-window
    支持、正确的 KV quant、或经目标文档批准的 L5 合同调整),不是继续重复
    同一 c=32 fp16 KV 命令。

### 本轮验证

- Local:
  - `cargo fmt --all`
  - `cargo test -q -p ferrum-engine model_decode_metadata_marks_structured_requests_for_full_logits -- --nocapture`
  - `cargo test -q -p ferrum-models unified_decode_prepares_fresh_prefill_kv_capacity_hint -- --nocapture`
  - `cargo test -q -p ferrum-models unified_decode_full_logits_prefill_prepares_kv_capacity_hint -- --nocapture`
  - `cargo test -q -p ferrum-models contiguous_kv_capacity_hint_sizes_physical_cache -- --nocapture`
  - `cargo check -q -p ferrum-models --tests`
  - `git diff --check`
- Remote CUDA:
  - `kv_hint_c32_initial/build.log`: release CUDA build PASS in 3m26s.
  - `kv_hint_unified_c32/build.log`: release CUDA build PASS in 3m26s.
  - `kv_hint_actual_c32/build.log`: release CUDA build PASS in 3m12s.

## 2026-06-14(凌晨 III)— W2 c=32 根因收敛:Gemma3-27B fp16 KV 在 24GB 4090 上无法满足 c=32/400

- 按用户建议继续复用 stopped/cache-retained native CUDA instance `40826362`,
  未新租机器、未重装环境。全部诊断完成后已复制 artifact 并停机,Vast API
  确认 `cur_state=stopped`,`actual_status=exited`。
- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_c32_kv_allocator_diagnostics_2026-06-14/`。
- 代码侧做了两个诊断/减峰尝试并本地验证:
  - CUDA allocator OOM 日志增加 thread-local allocation label;
  - `batch_logits` 从 prefill scratch 中拆成 lazy grow,避免无谓保留
    `seq_len * vocab` logits;本地
    `prefill_scratch_keeps_batch_logits_lazy` PASS。
- native CUDA c=32 复测结论:
  1. `kv-capacity=400/max-num-seqs=32` 仍 OOM,失败 allocation
     `elements=819200 = 400 * 16 kv_heads * 128 head_dim`,确认是
     Gemma3 contiguous KV cache 单个 K/V layer buffer,不是 scratch/logits。
  2. `kv-capacity=396` 仍 OOM,失败 allocation
     `elements=811008 = 396 * 16 * 128`;`kv=384` 已在上一轮证明会
     context validation fail,所以单靠继续压 KV capacity 不可行。
  3. prefill scratch 收缩实验仍在同一 KV allocation OOM,已撤回,不作为产品改动保留。
  4. 短窗口 paged fp16 实验把失败转移为单个 paged pool allocation
     `elements=26214400`/约 52MiB,仍因 free 约 23MiB OOM;该实验也已撤回。
  5. `--kv-dtype int8` 能启动 CUDA paged INT8 KV,但 sanity known-answer
     第一题输出乱码(`'ㄝ Task sera exquisiteFolrbatovski'`),correctness 不可接受,
     不能作为 W2 L5 绕过方案。
- 当前结论:
  - W2 仍不是 PASS。矩阵保持 l0/l1/l2_gptq/l3/l4 pass,
    l2_gguf waived,l5_concurrency fail,perf_vs_llamacpp pending。
  - c=32 blocker 已从"未知 OOM"收敛为"Gemma3-27B GPTQ + fp16 KV +
    c=32/400 在 24GB 4090 上总显存不足";可选后续是结构性 KV 内存方案
    (正确的 Gemma3 paged-window 支持、可验证的 KV quant correctness,或
    合同调整),不是继续重跑同一 L5。

### 本轮验证

- Local:
  - `cargo fmt --all`
  - `cargo check -q -p ferrum-models --tests`
  - `cargo check -q -p ferrum-cli --tests`
  - `cargo test -q -p ferrum-models prefill_scratch_keeps_batch_logits_lazy -- --nocapture`
  - `git diff --check`
- Remote CUDA:
  - `build.log`: release CUDA build PASS in 3m26s.
  - `build_shrink.log`: release CUDA build PASS in 3m25s (failed experiment, not retained).
  - `build_paged_window.log`: release CUDA build PASS in 3m24s (failed experiment, not retained).

## 2026-06-14(凌晨 II)— W2 c=32 allocator 诊断:OOM 是整体显存峰值贴满,不是单个大分配

- 继续按 native CUDA 最小验证策略复用 stopped/cache-retained instance
  `40826362`。没有新租机器;诊断完成后已停机,Vast API 确认
  `cur_state=stopped`,`actual_status=exited`。
- 本轮 paid GPU lane 合同:
  - lane: W2 Gemma3-27B CUDA L5 c=32 allocator-diagnostic rerun,existing 1x RTX 4090;
  - expected runtime/cost:15-45min,约 $0.10-$0.30 at ~$0.402/hr plus storage;
  - stop condition:first OOM backtrace artifact collected / c=32 unexpected PASS / 45min no progress;
  - correctness/performance command:同一条 c=32
    `bench-serve random 256/128 --concurrency-sweep 32 --fail-on-error --require-ci --seed 9271`。
- 为定位 OOM,增强 CUDA allocator 失败日志:
  `CudaBackend::alloc` / `alloc_typed` 现在在失败时打印 dtype、元素数、
  字节数、`cuMemGetInfo` free/total,以及强制 backtrace。该改动只改善失败
  诊断,不改变产品行为。
- 远端增量 release build 通过:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_c32_alloc_diag_oom_2026-06-14/build.log`
  记录 `Finished release profile [optimized] target(s) in 28m 55s`。
- 第一次诊断运行 `c32/` 未进入模型请求:脚本把 `--tokenizer` 误传成
  `tokenizer.json` 文件路径,CLI 又追加 `/tokenizer.json`,因此 bench 立即报
  `Not a directory`。该运行只保留为脚本错误证据,不计入 W2。
- 第二次诊断运行 `c32_retry/` 使用 tokenizer snapshot 目录,服务成功启动并进入
  `closed_loop c=32 — repeat 1/3`。关键 OOM 行:
  `CudaBackend::alloc failed: dtype=F16 elements=819200 bytes=1638400 free=17498112 total=25278087168: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`。
- bench 侧写出失败 JSON
  `c32_retry/l5_gemma3-27b-gptq_cuda_c32_alloc_diag.json`;
  `bench.log` 记录 `0 completed / 100 errored` 和
  `bench-serve error rate 1.0000 exceeds max 0.0000`。
- 最新结论:
  1. c=32 OOM 不是某个巨型 allocation;失败 allocation 只有约 1.56MiB;
  2. OOM 时整卡只剩约 16.7MiB free,说明 c=32 的稳态/瞬态总峰值贴满 24GB;
  3. backtrace 因 release strip 显示 `<unknown>`,下一步要加 shape/caller label
     或做非 strip diagnostic,而不是继续盲目调 `kv-capacity`;
  4. W2 仍是 l0/l1/l2_gptq/l3/l4 pass,l2_gguf waived,
     l5_concurrency fail,perf_vs_llamacpp pending;**不宣称 W2 PASS**。

### 下一步

- 本地先沿 `elements=819200` 反查可能 shape/callsite,优先看 Gemma3
  c=32 单序 forward 下每步 activation/logits/scratch 分配。
- 下一次 GPU 只跑更小诊断:给 allocator callsite 加标签或禁用 strip 的 backtrace,
  目标是拿到"哪个张量/阶段"把 free 压到 17MB,而不是重复完整 L5。
- 修复方向要减少 c=32 总峰值,例如缩小可配置 batch token/scratch 峰值、释放可复用
  scratch、或修正 per-request retained buffer 生命周期;若改变 gate 合同必须写入
  W2 文档/矩阵,不能靠隐藏 env。

## 2026-06-14(凌晨 I)— W2 c=32 最小复测:prompt 长度问题已排除,运行期 OOM 仍阻塞

- 按用户建议复用现有 native CUDA GPU 机器,没有重新租新 pod/重装环境:
  instance `40826362`,1x RTX 4090,Iceland host 1647。复测完成后立即停机,
  Vast API 已确认 `cur_state=stopped`,`actual_status=exited`。
- 本轮 paid GPU lane 合同:
  - lane: W2 Gemma3-27B CUDA L5 c=32 minimal regression,existing 1x RTX 4090;
  - expected runtime/cost:30-90min,约 $0.20-$0.60 at $0.402/hr;
  - stop condition:c=32 PASS artifact / c=32 OOM or context failure artifact / 90min no progress;
  - correctness/performance command:`bench-serve random 256/128 --concurrency-sweep 32 --fail-on-error --require-ci --seed 9271`。
- 先修正并本地验证 `bench-serve` random prompt 生成:
  `--random-input-len 256` 现在按 tokenizer 重新编码后的实际 token 数逼近目标,
  避免 Gemma tokenizer 把随机 token 文本重编码成 270+ tokens。
  本地用 `unsloth/gemma-3-1b-it` tokenizer fixture 跑
  `random_prompt_generation_targets_reencoded_length_when_fixture_is_set` PASS。
- 远端只同步最小改动并 release build:
  `target/release/ferrum` build finished in 3m26s;远端 `.env.local` 在一次误同步后已立即删除,
  本轮正式运行目录的 `c32/ferrum_env.txt` 为空,没有隐藏 `FERRUM_` env 修复。
- c=32 复测命令使用产品 CLI 参数:
  `ferrum serve --model gemma3:27b-gptq --port 8402 --kv-capacity 400 --max-num-seqs 32`
  加 `ferrum bench-serve --random-input-len 256 --random-output-len 128 --concurrency-sweep 32 --num-prompts 100 --n-repeats 3 --fail-on-error --require-ci --seed 9271`。
- 新证据:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_c32_prompt_exact_oom_2026-06-14/`。
  `serve.log` 证明 `kv-capacity=400` 能启动 800 blocks/32 seqs 并进入服务;
  `bench.log` 进入 `closed_loop c=32 — repeat 1/3`;
  随后服务端在 `crates/ferrum-kernels/src/backend/cuda/mod.rs:774`
  报 `CUDA_ERROR_OUT_OF_MEMORY`。
- 因此最新结论是:
  1. 之前 `kv=400` 的"上下文不足"分支已被 prompt 精确化排除;
  2. c=32 仍在运行期 OOM,不是 KV pool 启动失败;
  3. W2 L5 仍 FAIL,`perf_vs_llamacpp` 仍 pending,**不宣称 W2 PASS**。

### 下一步

- 保持同一台 stopped/cache-retained 4090 作为后续最小验证目标。
- 下一轮不要再跑完整 L5。先定位运行期 CUDA transient/scratch/请求调度内存峰值:
  优先用 c=32、kv=400、100 prompts 的同命令加最小显存 instrumentation,
  看 OOM 前最后一次 allocator 请求大小和调用路径。
- 如果需要改变 batch/scratch 上限,必须走 typed CLI/config/default 或目标合同修订,
  不能靠用户不可见的隐藏 env 当作通过证据。

## 2026-06-13(晚 XIX)— W2 correctness 打通,L5 c=32 首个内存阻塞点落证

- 采用当前 1x RTX 4090 pod 原生 CUDA 最小验证,不重建环境:
  instance `40826362`,Iceland host 1647,`/workspace/ferrum-infer-rs`。
  本轮目标从"反复开机装环境"改为"同机快速定位,拿到证据后停机"。
- 根因定位:Gemma3 sandwich-norm CUDA 路径把 residual/activation 存成 f16,
  在中层出现合法的大幅值 norm 输出与 residual 相加后超过 f16 上限
  65504,导致后续 logits NaN。Gemma3 27B 的 post-ffn norm 权重可到
  700 级,该路径必须保留 f32/bf16 语义,不能把 residual shadow 降成 f16。
- 修复方式:CUDA f16 activation 后端为 Gemma3 sandwich-norm 路径启用
  host/F32 residual shadow;prefill/decode 的 norm、residual add、final norm
  通过 f32 shadow 保持有限值。该行为走产品默认路径,不是隐藏 env 修复。
- 最小 CUDA 验证:
  - `cargo test -p ferrum-kernels --features cuda --release --test cuda_activation_precision -- --nocapture`
    PASS,2 tests;
  - `ferrum run` 一 token smoke 输出 `content:"5"`,layer dump/logits 全 finite;
  - layer dump summary:64 entries,`first_nonfinite=None`,logits
    `262208/262208` finite,`maxabs=41.84375`。
- W2 L2/L3/L4 smoke 已过:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_gemma3_cuda_host_shadow_l5_fail_2026-06-13/gates/smoke_gemma3-27b-gptq.log`
  记录 `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`:
  known-answer 10/10,natural EOS,multi-turn Bob,stream==non-stream,
  custom stop,max_tokens,required tool-call 10/10,strict json_schema 20/20。
- W2 L5 未过,所以**不宣称 W2 PASS**:
  - `gates/l5_gemma3-27b-gptq_cuda_1_4_16.json` 已记录
    c=1/4/16 三档均 `completed_per_run=[100,100,100]`,
    `errored_per_run=[0,0,0]`,`output_token_count_source=usage`,
    decode throughput 约 25.1 tok/s;
  - required c=32 在 `--kv-capacity 448 --max-num-seqs 32` 触发
    `CUDA_ERROR_OUT_OF_MEMORY`,日志见
    `gates/serve_l5_gemma3-27b-gptq_32.log`;
  - targeted diagnostics:kv-capacity 400 可启动但上下文不足
    (`278 input + 128 output > 400`),kv-capacity 408/416 因 KV block
    rounding 仍 OOM;
  - 因 L5 首败即停,未进入 llama.cpp same-card ratio,`perf_vs_llamacpp`
    仍 pending。
- 当前矩阵结论:
  W1 PASS;W2 现在是 l0/l1/l2_gptq/l3/l4 pass,l2_gguf waived,
  l5_concurrency fail,perf_vs_llamacpp pending;W3 仍只有 charter 草案。

### 下一步

- 不再跑完整重装 pod。优先在现有 CUDA 内存模型上做一个小改动:
  让 c=32 gate 真正降低 per-seq/context 占用或改为可解释的 W2 合同修订;
  不能只把 kv-capacity 降到 400,因为该设置已被实测证明不满足
  256/128 bench prompt 的上下文需求。

## 2026-06-13(晚 XVIII)— W2 pod 脚本固化 parity-first:失败即停,通过才跑 Gemma3 early-smoke

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- `scripts/pod_w2_gemma3.sh` 已把下一次 1x4090 执行顺序固化:
  - `build.ok` 后立即运行 synthetic CUDA desc_act parity,不等待 27B 权重下载:
    1. `cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference`;
    2. `cuda_desc_act_gemma3_attention_shapes_vs_cpu_reference`;
  - parity 日志写入 `$W/diagnostics/desc_act_parity.log`,
    return code 写入 `$W/diagnostics/desc_act_parity.rc`;
  - parity fail 会写 `$W/desc_act_parity.fail`,停止 HF 下载/llama.cpp build,
    touch `$W/w2_session.done`,不进入 Gemma3 smoke/L5/perf;
  - parity pass 才写 `$W/desc_act_parity.ok`,随后 early-smoke 等待
    `build.ok + dl_gptq.ok + desc_act_parity.ok`。
- early-smoke 默认诊断也更新:
  - `FERRUM_SMOKE_NAN_TRACE="${FERRUM_W2_NAN_TRACE:-layer0}"`;
  - `FERRUM_SMOKE_OP_DUMP_DIR="${FERRUM_W2_OP_DUMP_DIR:-/workspace/w2/gates_early/op_dump_layer0}"`;
  - 因晚 XVII 已支持 `layer0` 选择语法,下一次默认只写首层 op dump,
    而不是全模型 `all` dump。
- 下一次 paid GPU lane 合同(尚未启动):
  - lane: W2 Gemma3-27B CUDA GPTQ correctness micro-diagnostic,1x RTX 4090;
  - expected runtime/cost:约 1-2h,按最近 Vast $0.35-$0.45/hr 估算约 $0.35-$0.90,
    若需要继续 early-smoke/op-dump 可能到 3h/$1.35;
  - stop condition: parity fail / early-smoke fail artifact collected / first
    correctness PASS artifact collected / 3h 无进展;
  - correctness gate:两个 CUDA desc_act parity 先 PASS;若 PASS,再跑
    Gemma3 early-smoke L2 known-answer;
  - performance command:不跑性能。只有 L2/L3/L4 correctness PASS 后,才恢复
    `pod_w2_gates.sh` 的 L5 `bench-serve` 和 llama.cpp ratio。
- 本地验证:
  - `bash -n scripts/pod_w2_gemma3.sh scripts/pod_w2_gates.sh scripts/model_coverage_smoke.sh` PASS(本机 locale warning only)。
  - `cargo fmt --all -- --check` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test` PASS。
  - `cargo test -q -p ferrum-models nan_trace_selector_accepts_all_or_layer_lists` PASS。
  - `python3 -m py_compile scripts/analyze_layer_dump.py scripts/inspect_hf_gptq_tensor.py scripts/w1_goal_validator.py scripts/w2_goal_validator.py` PASS。

### 下一步

- 下一次 GPU 只跑 `scripts/pod_w2_gemma3.sh` 的新 parity-first 流程;
  若 `desc_act_parity.fail`,直接修 CUDA Marlin desc_act/repack/scale layout;
  若 parity 过而 early-smoke 仍失败,读取
  `gates_early/op_dump_layer0/summary.jsonl` 的 `maxabs_row/maxabs_col`
  定位首个产生 `row=0,col=104` 爆炸的 op。

## 2026-06-13(晚 XVII)— W2 nan-trace 坐标化:下次 op 日志直接报 maxabs row/col

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 基于晚 XVI 的 `layer_00 row=0,col=104` 爆炸证据,增强
  `crates/ferrum-models/src/models/llama_family.rs` 的诊断输出:
  - `DumpStats` 现在记录 `maxabs_index`;
  - `FERRUM_NAN_TRACE` 日志现在在知道 row width 时打印
    `maxidx,row_width,row,col`;
  - `FERRUM_OP_DUMP` 的 `summary.jsonl` 同步写入 `maxabs_index`,
    且对有 row width 的 op 写入 `maxabs_row/maxabs_col`;
  - `FERRUM_NAN_TRACE` 保留 `0` 作为关闭,新增显式 `layer0`/`l0`/
    `source0` 语法,允许只抓首层 op 而不是 `all` 全模型 dump;
  - 新增 `down_proj` 和 `resid_ffn` 两个 trace 点,补齐 MLP 输出与最终
    residual add,避免下一次只看到 layer dump 爆炸却不知道 down-proj
    之前/之后的边界。
- row width 绑定:
  - token-major hidden/residual/norm/o_proj/down_proj/resid_ffn → `hidden_size`;
  - `qkv_proj` → `q_dim + 2*kv_dim`;
  - head-major q/attention → `head_dim`;
  - `gate_up` → `2*intermediate_size`;
  - `act_mul` → `intermediate_size`。
  这让下一次 Gemma3 early-smoke 日志能直接回答首层异常是否已经出现在
  `qkv_proj`,还是 attention/o_proj/MLP 后才出现。
- 本地验证:
  - `python3 -m py_compile scripts/analyze_layer_dump.py scripts/inspect_hf_gptq_tensor.py` PASS。
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -q -p ferrum-models --tests` PASS。
  - `cargo test -q -p ferrum-models dump_stats_counts_nonfinite_values` PASS。
  - `cargo test -q -p ferrum-models nan_trace_selector_accepts_all_or_layer_lists` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test` PASS
    (5 default tests;CUDA ignored tests未在本机执行)。
  - `git diff --check` PASS。
  - `python3 scripts/w1_goal_validator.py` PASS:
    `MODEL_COVERAGE_W1 GOAL PASS: docs/goals/model-coverage-2026-06-12`。
  - `python3 scripts/w2_goal_validator.py` 仍 3/8,5 blocking cells(预期)。

### 下一步

- 下一次可靠 1x4090 不跑完整 W2。先跑 CUDA desc_act parity;若过,再跑
  Gemma3 early-smoke with `FERRUM_W2_NAN_TRACE=layer0` +
  `FERRUM_W2_OP_DUMP_DIR=<artifact-dir>`,日志里重点看 layer0 每个 op 的
  `row=0,col=104` 是否在 qkv/o_proj/MLP 哪一步首次出现。

## 2026-06-13(晚 XVI)— W2 NaN artifact 机器化复盘:异常从 layer_00 内部开始,不是 logits-only

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 新增 `scripts/analyze_layer_dump.py`:无第三方依赖读取
  `FERRUM_LAYER_DUMP` f32 `.bin`,统计 finite/nonfinite、max_abs、first
  threshold crossing,并可用 `--last-dim` 把 flat index 标注为
  `[row,col]` 坐标。
- 用该脚本复盘既有
  `artifacts/w2_gemma3_cuda_nan_logits_2026-06-13/gates_early/logit_dump_smoke`
  并生成:
  `artifacts/w2_gemma3_cuda_nan_logits_2026-06-13/gates_early/layer_dump_summary.json`。
- 机器化 summary 结论:
  - `embed.bin`:shape `[24,5376]`,all finite,`max_abs=17.65625`;
  - `layer_00.bin`:all finite,但 `max_abs=23056.0` 出现在
    `row=0,col=104`;首个 `abs>100` 在 `row=0,col=61`,首个
    `abs>1000/10000` 同在 `row=0,col=104`;
  - `layer_01..layer_07`:仍 all finite,但异常值持续存在,`layer_07`
    `max_abs=42432.0` 仍在 `row=0,col=104`;
  - `logits.bin`:262208/262208 全 NaN,首个 nonfinite index 0。
- 这把定位从"最终 logits 全 NaN"收紧为"首层输出已经爆炸,后续层仍 finite,
  到 lm_head/logits 阶段才变全 NaN"。下一次 GPU op dump 不需要先扫全模型,
  应优先捕获 layer0 的 qkv、attention score/output、o_proj、post-attn norm、
  gate/up/down/activation/down_proj 输入输出,特别关注 token row 0、hidden col
  61/104 附近。
- 本轮新证据与晚 XV 的真实 GPTQ CPU dequant 结合后,当前最高概率分支仍是
  CUDA Marlin desc_act/scale layout/act-order 或 Gemma3 layer0 CUDA forward
  内某个 op,不是 HF 源权重本身、qzeros 形态、g_idx balance 或最终 logits
  读回单点问题。

### 下一步

- 下一次可靠 1x4090 的执行顺序:
  1. `cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference`;
  2. 若过,跑 `cuda_desc_act_gemma3_attention_shapes_vs_cpu_reference`;
  3. 若仍过,跑 Gemma3 early-smoke + `FERRUM_W2_OP_DUMP_DIR`,优先 layer0
     op dump,用本轮 `row=0,col=61/104` 作为检查坐标。

## 2026-06-13(晚 XV)— W2 真实 layer0 attention 全投影 + MLP 采样:源权重继续排除,下轮直指 CUDA parity/op dump

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 新增可复用诊断脚本 `scripts/inspect_hf_gptq_tensor.py`:通过 HF
  `model.safetensors.index.json` 定位 shard,用 HTTP Range 读取单个 GPTQ
  tensor prefix 的 `qweight/scales/qzeros/g_idx`,输出 JSON summary。脚本默认
  8MiB 分块读取,避免 50MB+ range 被远端断连;无第三方依赖。
- 继续读取真实 `circulus/gemma-3-27b-it-gptq`
  commit `70d89a3a6b401b5f56558cb5d4c0f1fd158980b2` 的 layer0 权重,生成:
  - `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/layer0_kproj_cpu_dequant_report.json`
  - `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/layer0_vproj_cpu_dequant_report.json`
  - `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/layer0_oproj_cpu_dequant_report.json`
  - `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/layer0_gateproj_cpu_dequant_sample_report.json`
  - `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/layer0_upproj_cpu_dequant_sample_report.json`
  - `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/layer0_downproj_cpu_dequant_sample_report.json`
- layer0 self-attn 的 k/v/o 三个 projection 做完整列 CPU dequant + deterministic
  matmul probe(晚 XIV 已覆盖 q_proj):
  - k_proj: `K=5376,N=2048`,all finite,`max_abs=0.1303`;
  - v_proj: `K=5376,N=2048`,all finite,`max_abs=0.2098`;
  - o_proj: `K=4096,N=5376`,all finite,`max_abs=0.1708`;
  - 三者均 `g_idx_balanced_full_groups=true`,
    `g_idx_sequential_non_desc_act=false`,`qzeros_all_code7=true`,
    `scales_all_finite=true`。
- layer0 MLP 的 gate/up/down 三个 projection 做 512 个均匀输出列采样
  (不是完整 MLP 证明,但覆盖真实 qweight/scales/qzeros/g_idx 读取和每个 K row):
  - gate_proj: sampled all finite,`max_abs=0.1321`;
  - up_proj: sampled all finite,`max_abs=0.2784`;
  - down_proj: sampled all finite,`max_abs=0.0411`;
  - 三者同样满足 balanced non-trivial `g_idx`、全 code7 `qzeros`、finite scales。
- `crates/ferrum-quantization/tests/gptq_parity_test.rs` 新增 ignored CUDA
  micro-diagnostic:
  `cuda_desc_act_gemma3_attention_shapes_vs_cpu_reference`。它顺序覆盖真实
  layer0 attention 的 `k_proj K5376/N2048`,`v_proj K5376/N2048`,
  `o_proj K4096/N5376`,补足 q_proj-only parity 可能漏掉 tile/shape 问题。
- 结合晚 XII-XIV,真实 Gemma3 GPTQ 源权重的基本形态、scale 有限性、
  qzero 对称性、desc_act group balance、layer0 attention 全投影 dequant
  都不像首层 2e4 级爆炸和最终全 NaN logits 的源头。当前优先级进一步收敛到:
  1. CUDA Marlin desc_act/Gemma3 真实形状 parity;
  2. 若 parity 过,用 `FERRUM_W2_OP_DUMP_DIR` 捕获 Gemma3 early-smoke
     首层 qkv/attn/o_proj/MLP op 输入输出,定位 CUDA forward 内首个爆炸点。
- 本地验证:
  - `python3 -m py_compile scripts/inspect_hf_gptq_tensor.py scripts/w1_goal_validator.py scripts/w2_goal_validator.py` PASS。
  - `git diff --check` PASS。
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -q -p ferrum-quantization --tests` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test` PASS
    (5 default tests;CUDA ignored tests未在本机执行)。
  - `python3 scripts/w1_goal_validator.py` PASS:
    `MODEL_COVERAGE_W1 GOAL PASS: docs/goals/model-coverage-2026-06-12`。
  - `python3 scripts/w2_goal_validator.py` 仍 3/8,5 blocking cells(预期)。

### 下一步

- 不跑完整 W2。下一次可靠 1x4090 先跑:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference -- --ignored --nocapture`。
- 若 q_proj shape 过,同一 pod 继续跑:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_gemma3_attention_shapes_vs_cpu_reference -- --ignored --nocapture`。
- 若 Gemma3-shape parity 失败,修 CUDA Marlin repack/scale layout/act-order;
  若通过,再跑 Gemma3 early-smoke + op dump。只有首步 logits finite 且 smoke
  L2 过,才恢复 L3/L4/L5/perf。

## 2026-06-13(晚 XIV)— W2 真实 layer0 q_proj CPU dequant:权重合成有限,不像首层爆炸源

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 用 HF Range 读取真实 `circulus/gemma-3-27b-it-gptq`
  layer0 `self_attn.q_proj` 的完整 GPTQ 四件套:
  `qweight [672,4096]`, `scales [42,4096]`, `qzeros [42,512]`,
  `g_idx [5376]`,生成:
  `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/layer0_qproj_cpu_dequant_report.json`。
- 本地 CPU 按 `g_idx` scale lookup 完整 dequant 该 projection:
  - `g_idx_balanced_full_groups=true`;
  - `qzeros_all_code7=true`;
  - `scales_all_finite=true`;
  - `dequant_weight_all_finite=true`;
  - `dequant_weight_max_abs=0.1151123046875`。
- 额外做两个 deterministic matmul probe:
  - `x[k]=f16(sin(k*0.0041))` 时输出 `max_abs≈2.00`;
  - 同一输入乘以 17.7(上一轮 embed dump maxabs 量级)时输出
    `max_abs≈35.41`。
  这不能替代 CUDA parity,但能排除"真实 layer0 q_proj 的 GPTQ
  dequant 结果本身非 finite 或自然产生 2e4 级输出"这个分支。
- 因此当前最高价值 GPU micro-diagnostic 仍是 CUDA Marlin vs CPU
  reference,尤其是 Gemma3 q_proj 真实形状;若它过,再从 q_proj 之外的
  首层 op dump 查爆炸边界。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test`
    PASS(5 default tests;CUDA ignored tests未在本机执行)。
  - artifact sanity PASS:报告 summary 七项均符合预期。

### 下一步

- 下一次可靠 1x4090 先跑:
  `cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference`。若失败,优先修
  CUDA Marlin repack/kernel/scale layout;若通过,用 `FERRUM_W2_OP_DUMP_DIR`
  捕获 Gemma3 early-smoke 首层 qkv/attn/mlp op 输入输出,定位 q_proj
  之外的首个爆炸点。

## 2026-06-13(晚 XIII)— W2 真实 Gemma3 scales 采样:源 scales 有限且量级正常

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 延续晚 XII 的 HF Range 采样,对同一 layers 0 / 31 / 61、同一 7 个
  projection(`q/k/v/o/gate/up/down`)读取 `*.scales` 小样本,生成:
  `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/gptq_scale_sample_report.json`。
- 第一次 scales 读取误按 F32 解码,发现异常小量级后已废弃并用
  safetensors header 的 dtype-aware 解码覆盖报告。有效报告显示:
  - `dtypes=["F16"]`;
  - `sampled_scales_count=21`;
  - `all_finite=true`;
  - `bad_nonfinite_or_gt10_count=0`;
  - `max_abs_overall=0.0723876953125`。
  这排除了"真实 Gemma3 GPTQ 源 scales 本身非 finite 或异常大"这一分支。
- 结合晚 XII:
  - sampled `g_idx` 均 balanced full-group 且 non-trivial desc_act;
  - sampled `qzeros` 均 code7;
  - sampled `scales` 为有限 F16 且量级正常。
  因此下一步仍应集中在 CUDA Marlin repack/kernel/scale layout 与
  Gemma3 层内数值路径,不是权重 metadata 基本形态。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -q -p ferrum-quantization --tests` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test`
    PASS(5 default tests;CUDA ignored tests未在本机执行)。
  - artifact sanity PASS:layout summary 两项 true,scale summary all_finite
    true且 dtype 为 F16。

### 下一步

- 下一次可靠 1x4090 的第一条仍是两条 CUDA ignored parity:
  `cuda_desc_act_vs_cpu_reference` 与
  `cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference`。
- 若 Gemma3-shape parity 仍过,就把 focus 从 GPTQ metadata/repack 前置条件
  移到 Gemma3 forward 内的首层 op dump:全层 nan-trace 已默认打开,
  必要时显式 `FERRUM_W2_OP_DUMP_DIR` 捕获首个爆炸算子输入输出。

## 2026-06-13(晚 XII)— W2 真实 Gemma3 GPTQ layout 采样:guard 预计不拦截,下一步查 CUDA Marlin 数值

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 用 Hugging Face resolve HTTP Range 只读 `circulus/gemma-3-27b-it-gptq`
  commit `70d89a3a6b401b5f56558cb5d4c0f1fd158980b2` 的 safetensors
  header 与小 tensor 样本,未下载整模型权重:
  `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/gptq_layout_sample_report.json`。
- 采样范围:layers 0 / 31 / 61 的 7 个 projection
  (`q/k/v/o/gate/up/down`),共 21 个 `g_idx` + 21 个 `qzeros`,
  payload 约 5.47MB。一次全量 `g_idx` Range 尝试因远端 HTTP 408 停止,
  没有作为证据使用。
- 采样结论:
  - `sampled_g_idx_all_balanced_full_groups=true`;
  - `sampled_g_idx_sequential_non_desc_act_count=0`,确认样本确实是
    non-trivial desc_act,不是顺序 g_idx;
  - `sampled_qzeros_all_code7=true`。
  因此晚 X/XI 新增的 qzeros / balanced-g_idx guard 在这些真实
  Gemma3 GPTQ 样本上预计不会拦截;当前 L2 NaN blocker 更可能还在
  CUDA Marlin repack/kernel/scale layout 或 Gemma3 层内数值路径。
- `crates/ferrum-quantization/tests/gptq_parity_test.rs` 新增 ignored
  CUDA micro-diagnostic:
  `cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference`。它使用真实
  Gemma3 q_proj 形状 `K=5376,N=4096,group_size=128` 的 synthetic
  desc_act/sym GPTQ,对比 CUDA Marlin 与 CPU `g_idx` reference;用于补足
  旧 `K=512,N=256` 小形状 parity 不能覆盖真实 tile/scale 布局的缺口。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -q -p ferrum-quantization --tests` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test` PASS
    (5 default tests;CUDA ignored tests未在本机执行)。
  - `cargo test -q -p ferrum-quantization validate_cuda_marlin_desc_act`
    PASS(3 tests)。
  - artifact sanity PASS:`sampled_g_idx_count=21`,
    `sampled_qzeros_count=21`,两项兼容性 summary 均为 true。
  - `python3 scripts/w1_goal_validator.py` PASS:
    `MODEL_COVERAGE_W1 GOAL PASS: docs/goals/model-coverage-2026-06-12`。
  - `python3 scripts/w2_goal_validator.py` 仍 3/8,5 blocking cells(预期)。

### 下一步

- 下一次可靠 1x4090 仍不跑完整 W2。先跑:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_vs_cpu_reference -- --ignored --nocapture`
  和
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference -- --ignored --nocapture`。
- 若 small 过而 Gemma3-shape 失败,优先查 Marlin repack/scale layout 的
  shape-specific 假设;若两者都过,再跑 Gemma3 early-smoke,使用全层
  nan-trace 定位首个爆炸/非 finite 算子。

## 2026-06-13(晚 XI)— W2 desc_act 前提硬化:CUDA Marlin 只接受 balanced full-group g_idx

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 继续审计 `desc_act=true/static_groups=false` GPTQ 与当前 CUDA Marlin
  策略的等价前提。Ferrum 现策略是 load-time `argsort(g_idx)` 重排
  qweight row,运行时按同一 perm gather A,再交给固定 group-boundary 的
  IST-DASLab Marlin kernel。因此它只在每个 quant group 恰好有
  `group_size` 个 K row 时成立;若真实 `g_idx` 某 group 多/少,row 排序后
  的固定 `j/group_size` scale lookup 会错位。
- `crates/ferrum-quantization/src/native_safetensors.rs` 新增
  `validate_cuda_marlin_desc_act_g_idx()`:
  - CUDA build 下,普通 GPTQ linear、fused qkv/gate_up linear、stacked GPTQ
    experts 都会在 load 阶段校验 balanced full-group;
  - `quantize_config desc_act=true` 但缺 `g_idx` 的 stacked GPTQ 现在也会
    和普通 linear 一样显式报错,不再把 `None` 交给 backend 静默错跑;
  - 非 CUDA 的 CPU/Metal desc_act dequant 仍保留按 `g_idx` lookup 的通用
    fallback,不套 Marlin 前提。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `bash -n scripts/pod_w2_gemma3.sh scripts/model_coverage_smoke.sh scripts/pod_w2_gates.sh`
    PASS(本机 locale warning only)。
  - `cargo check -q -p ferrum-quantization --tests` PASS。
  - `cargo check -q -p ferrum-kernels --tests` PASS。
  - `cargo test -q -p ferrum-quantization validate_cuda_marlin_desc_act`
    PASS(3 tests)。
  - `cargo test -q -p ferrum-quantization validate_gptq_g_idx` PASS
    (4 tests)。
  - `cargo test -q -p ferrum-quantization qzero_stats` PASS(2 tests)。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test`
    PASS(5 tests)。
  - `python3 scripts/w1_goal_validator.py` PASS:
    `MODEL_COVERAGE_W1 GOAL PASS: docs/goals/model-coverage-2026-06-12`。
  - `python3 scripts/w2_goal_validator.py` 仍 3/8,5 blocking cells(预期)。

### 下一步

- 下一次可靠 1x4090 仍只跑 micro-diagnostic first:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_vs_cpu_reference -- --ignored --nocapture`。
- 若 parity 过,再跑 Gemma3 early-smoke。新的 loader guard 会把真实
  Gemma3 GPTQ 的 `g_idx`/`qzeros` 兼容性问题转成 load-stage 明确错误;
  若两项兼容性 guard 都过但仍 NaN,全层 nan-trace 才是下一步定位依据。

## 2026-06-13(晚 X)— W2 CUDA GPTQ guard 收紧:拒绝不安全 Marlin 路径,避免假诊断

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- `crates/ferrum-kernels/src/backend/cuda/quant.rs` 收紧默认 CUDA Marlin:
  - dense `FERRUM_VLLM_MARLIN=1` 现在显式 `unsupported`。原因:dense
    `load_gptq` 保存的是 IST-DASLab Marlin tile,而 vLLM Marlin kernel
    需要 vLLM-repacked weights;此前这个实验 env 可能让同一权重走错
    kernel,污染 W2 诊断。vLLM-repacked Marlin 仍只保留在 stacked MoE
    的 `FERRUM_VLLM_MOE` 路径。
  - 默认 dense/stacked Marlin 在忽略 `qzeros` 前,现在要求所有 GPTQ
    `qzeros` nibble 都是 code 7(GPTQ zero-1 编码下的对称 zero point 8)。
    若真实 Gemma3 GPTQ 不是该形态,下一次 early-smoke 会在 load 阶段
    明确失败并给出首个 bad code 位置,而不是继续进入可能全 NaN 的推理。
- 同步 `docs/runtime-env-registry.tsv`:`FERRUM_VLLM_MARLIN` 标记为
  dense GPTQ rejected,除非未来补 dense vLLM repack 证据,否则不能作为
  产品验证或诊断开关。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -q -p ferrum-kernels --tests` PASS(默认 feature;CUDA 模块
    不会在本机编译进来)。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test` PASS
    (5 tests)。
  - `python3 scripts/w1_goal_validator.py` PASS:
    `MODEL_COVERAGE_W1 GOAL PASS: docs/goals/model-coverage-2026-06-12`。
  - `python3 scripts/w2_goal_validator.py` 仍 3/8,5 blocking cells(预期)。
  - `cargo check -q -p ferrum-kernels --features cuda --tests` 未跑到 Rust
    语义编译:本机缺 `nvcc`/`nvidia-smi`,失败在 CUDA build scripts。

### 下一步

- 下一次可靠 1x4090 的第一项仍是最小 CUDA parity:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_vs_cpu_reference -- --ignored --nocapture`。
- parity 通过后再跑 Gemma3 early-smoke。若新 qzeros guard 拦截真实权重,
  W2 的 L2 blocker 变成"当前 Marlin 不支持该 GPTQ zero-point 形态";
  若 qzeros 通过但仍 NaN,读取全层 nan-trace 定位首个非 finite op。

## 2026-06-13(晚 IX)— W2 early-smoke 诊断默认升级:下轮直接收全层 nan-trace

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 复核上轮 NaN artifact:`FERRUM_LAYER_DUMP` 只给出 embed、layer_00..07
  和最终 logits;虽然 layer_00 起 maxabs 已爆炸,但缺少每层关键算子
  的 non-finite 边界,下一轮若仍失败会继续需要二次上卡定位。
- `scripts/pod_w2_gemma3.sh` 的 early-smoke 默认改为
  `FERRUM_SMOKE_NAN_TRACE="${FERRUM_W2_NAN_TRACE:-all}"`。这会让
  `ferrum serve` 日志在每一层的 qkv、attn、o_proj、post_attn_norm、
  gate_up、activation、down_proj 等关键点打印 finite/nan/inf/maxabs。
  `FERRUM_W2_NAN_TRACE` 仍可覆盖成单层列表;`FERRUM_W2_OP_DUMP_DIR`
  仍保持显式 opt-in,避免默认写出巨量 op dump。
- 本地验证:
  - `bash -n scripts/pod_w2_gemma3.sh` PASS(本机 locale warning only)。
  - `cargo test -q -p ferrum-models nan_trace_selector_accepts_all_or_layer_lists`
    PASS。
  - `cargo test -q -p ferrum-models dump_stats_counts_nonfinite_values` PASS。

### 下一步

- 下一次可靠 1x4090 仍先跑最小 CUDA parity;若需要直接跑 Gemma3
  early-smoke,它现在会一次性产出 g_idx/qzeros trace + 全层 nan-trace,
  足以定位首个非 finite op 或确认只是数值爆炸但未 NaN。

## 2026-06-13(晚 VIII)— W2 本地 qzeros/sym 审计:补真实模型 trace,CUDA parity fixture 收紧

- 没有新开 GPU。W2 仍停在 Gemma3-27B GPTQ CUDA correctness:
  首步 logits 全 NaN,不得进入 L5/perf,也不得宣称 W2 PASS。
- 本地审计确认 Ferrum 默认 CUDA Marlin 路径仍不读取 `qzeros`:
  Marlin no-zp/sym 路径隐含 int4 zero point = 8;GPTQ `qzeros`
  按 zero-1 存储时,对称量化应表现为 nibble code 7。
- `crates/ferrum-quantization/tests/gptq_parity_test.rs` 的 CUDA ignored
  parity fixture 已改为 `sym=true` 代表性 qzeros:
  `make_synthetic_symmetric()` 将所有 qzeros word 置为 `0x77777777`,
  避免随机 qzeros 让 Marlin no-zp 路径产生非代表性失败。
- `crates/ferrum-quantization/src/native_safetensors.rs` 增加
  `FERRUM_GPTQ_GIDX_TRACE=1` 下的 qzeros 统计:
  每个 GPTQ linear / fused GPTQ load 会打印 qzero nibble histogram、
  `min_code/max_code`、`code7/total` 与 `all_code7`。下一次 W2
  early-smoke artifact 可直接回答真实 Gemma3 GPTQ 是否满足
  sym=true/no-zp Marlin 假设。
- 复核加载路径:`ferrum run/serve` 的主 GPTQ product path 由
  `NativeSafetensorsLoader::<B>::open()` + `WeightLoader::load_linear()`
  承载;旧日志里的 `ferrum_models::loader::gptq_loader` 来自 registry
  的兼容 `QuantizeConfig` probe,不能单独代表实际 linear loader。
  因此本轮 trace 加在 `NativeSafetensorsLoader` 上,覆盖下一次 W2
  early-smoke 的真实 GPTQ linear/fused-linear load。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test`
    PASS(5 tests)。
  - `cargo test -q -p ferrum-quantization qzero_stats` PASS(2 tests)。
  - `cargo test -q -p ferrum-quantization validate_gptq_g_idx` PASS(4 tests)。
  - `cargo check -q -p ferrum-quantization --tests` PASS。

### 下一步

- 仍不跑完整 W2。下一次有可靠 1x4090 通道时,先跑最小 CUDA parity:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_vs_cpu_reference -- --ignored --nocapture`。
- 若 parity 通过,再跑 Gemma3 early-smoke,读取新增 qzeros/g_idx trace 与
  layer/op dump 来定位首个非 finite 算子;若 parity 失败,先修 Marlin
  repack/perm/scales/qzeros 假设,不要进入 L5/perf。

## 2026-06-13(晚 VII)— W2 CUDA parity 租机未跑到:Vast SSH/proxy 失败,实例归零

- 目标只是一条 W2 micro-diagnostic,不是完整 W2 gate:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_vs_cpu_reference -- --ignored --nocapture`。
  correctness-only,无性能命令、无 W2 pass 声明。
- Vast 四次尝试均未跑到测试:
  - `40812709`(Vietnam host 55116,$0.356/hr):卡在
    `actual_status=loading`,未 SSH,已销毁。
  - `40813345`(Iceland host 1647,$0.402/hr):Vast reported running/onstart
    success,但 SSH publickey 认证失败,已销毁。
  - `40813765`(Iceland host 1647,$0.402/hr):按 Vast 文档改用
    `runtype=ssh` + attach-key API 后进入 running;但 container log 显示
    `remote port forwarding failed for listen port 13764`,proxy SSH 被关;
    direct SSH 也 rejected attached key;已销毁。
  - `40814425`(Ukraine host 103274,$0.401/hr):按下一步改用
    `runtype=ssh_direct`,实例进入 running 且 direct port 打开;但
    `root/ubuntu/vastai/user` 均 rejected associated public key;已销毁。
- Artifact/证据:
  `artifacts/w2_desc_act_cuda_parity_2026-06-13/` 保存 offer/创建响应
  (instance key 已脱敏)、Vast logs、destroy responses。未产生
  `cuda_desc_act_vs_cpu_reference` 输出。
- Vast API 已确认 `instances_found: 0`;`ACTIVE_PODS.md` 已标记四台均
  DESTROYED/ZERO-VERIFIED。

### 下一步

- Vast 通道暂停。下一步回到本地源码审计,优先核查 CUDA GPTQ Marlin
  对 `qzeros`/sym 的假设与 CPU reference 是否一致;GPU 通道恢复前不再
  循环租 offers。

## 2026-06-13(晚 VI)— W2 desc_act 诊断收窄:补本地 parity,下一轮只测 CUDA repack/kernel

- 没有新开 GPU。当前本机是 `Darwin arm64` 且无 `nvcc`,因此
  `cuda_desc_act_vs_cpu_reference` 只能作为下一轮 4090 ignored test,
  不能在本地执行。
- `crates/ferrum-quantization/tests/gptq_parity_test.rs` 新增/修正
  desc_act 合成诊断:
  - 合成 `g_idx` 改为 balanced full-K 形态:每个 group 正好
    `group_size` 个元素,但 K 轴交错,避免非现实非均匀 group 误报。
  - `desc_act_reference_uses_g_idx_for_scale_lookup` 证明该 fixture
    确实会区分顺序 group lookup 与 `g_idx[k]` lookup。
  - `desc_act_perm_gather_is_equivalent_to_g_idx_reference` 证明在
    balanced full-K 前提下,Ferrum 当前 host-level
    `argsort(g_idx)` qweight 重排 + activation gather 的代数结果等价于
    `g_idx` CPU reference。也就是说,晚 V 的“vLLM wrapper 未传 g_idx
    必然不等价”表述过强;vendored vLLM 在 `has_act_order && is_k_full`
    时同样会 permute A 后把 `has_act_order` 降为 false。
- 新增可选诊断 `FERRUM_GPTQ_GIDX_TRACE=1`:GPTQ loader 会打印真实
  `g_idx` 的 group count min/max、nonzero group 数、unbalanced group 数
  和前 16 项。`scripts/pod_w2_gemma3.sh` 的 early-smoke 已打开该开关,
  下一轮 artifact 能直接确认 Gemma3 GPTQ 的真实 g_idx 分布是否满足
  balanced full-K 前提。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -q -p ferrum-quantization --tests` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test desc_act`
    PASS(2 tests)。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test cpu_selfcheck`
    PASS。
  - `bash -n scripts/pod_w2_gemma3.sh` PASS(本机 locale warning only)。
  - `scripts/w2_goal_validator.py` 仍 3/8,5 blocking cells(预期)。

### 下一步

- 下一次付费 GPU 不跑完整 W2。先跑最小 CUDA parity:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_vs_cpu_reference -- --ignored --nocapture`。
- 若 parity 失败,优先查 CUDA qweight repack/perm 方向、Marlin scale
  repack、qzeros/sym 假设和 real `g_idx` 分布;若 parity 通过,再用
  `FERRUM_GPTQ_GIDX_TRACE=1` + early-smoke 的 layer/op dump 定位
  Gemma3 层内首个爆炸算子。

## 2026-06-13(晚 V)— W2 本地定位:desc_act GPTQ/Marlin 成为主嫌,固化 early-smoke 止损

- 基于 `w2_gemma3_cuda_nan_logits_2026-06-13` artifact 继续本地定位:
  已回收层 dump 的 `embed.bin` 与 `layer_00..07.bin` 全 finite,但数值幅度
  从 `embed maxabs=17.7` 到 `layer_00 maxabs=23056` 已明显爆炸,
  `layer_07 maxabs=42432`,最终 `logits.bin` 为 262208/262208 NaN。
  这说明不是最终 tokenizer/stop/template 问题,也不是 final softcap 缺失;
  数值从首层 GPTQ transformer 路径开始异常。
- HF config 落证:
  - `circulus/gemma-3-27b-it-gptq`: `desc_act=true`, `sym=true`,
    `group_size=128`, `static_groups=false`, `lm_head=false`;
    `final_logit_softcapping=null`。
  - `ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g` 是
    `compressed-tensors/pack-quantized`,不是当前 GPTQ loader 可用的
    Marlin-clean 格式,不能直接替换成 W2 GPTQ 载体。
  - `orvp/gemma-3-27b-it-gptq` 与 `circulus` 同样是
    `desc_act=true/static_groups=false` GPTQModel 路径。
- 对照 vLLM current path:其 GPTQ-Marlin apply 路径把 `g_idx` 与
  `g_idx_sort_indices` 传入 Marlin op;ferrum 当前 CUDA act-order 路径是
  load-time qweight permute + runtime gather A,并在 vendored vLLM Marlin
  wrapper 中把 `g_idx/perm/a_tmp` 传 null、`has_act_order=false`。这与
  vLLM current path 不等价,是当前 NaN 的最高优先级嫌疑。
- 代码/脚本加固(不改变默认产品路径):
  - `FERRUM_LAYER_DUMP` 现在写 `summary.jsonl`,记录每个 dump 的
    finite/nan/inf/maxabs;smoke 会打印首个 non-finite entry。
  - `FERRUM_NAN_TRACE` 从只跟 layer 0 改为支持 `all` 或逗号分隔层号;
    `FERRUM_OP_DUMP` 输出文件名带 `layer_NN_` 前缀。
  - `scripts/model_coverage_smoke.sh` 新增
    `FERRUM_SMOKE_NAN_TRACE` / `FERRUM_SMOKE_OP_DUMP_DIR` 可选透传。
  - `scripts/pod_w2_gemma3.sh` 固化 early-smoke:build+GPTQ 下载完成后
    先跑 correctness smoke;若失败,写 `early_smoke.fail` 并停止 GGUF /
    llama.cpp 后台工作,不再等待 perf 前置项。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `bash -n scripts/model_coverage_smoke.sh scripts/pod_w2_gates.sh scripts/pod_w2_gemma3.sh`
    PASS。
  - `cargo test -q -p ferrum-models dump_stats_counts_nonfinite_values` PASS。
  - `cargo test -q -p ferrum-models nan_trace_selector_accepts_all_or_layer_lists` PASS。
  - `cargo test -q -p ferrum-models gemma3` PASS。

### 下一步

- 不开完整 W2 gate。下一次 GPU 只做小诊断:
  1. 先跑合成 `desc_act=true` GPTQ CUDA parity(补/跑 ignored test),证明
     ferrum act-order Marlin 是否已偏离 CPU reference;
  2. 若 parity 失败,修 `MarlinWeight` 保存 `g_idx`/sort indices,让
     vendored vLLM Marlin wrapper 使用原生 act-order(`a_tmp` scratch +
     `has_act_order=true`),或实现等价 kernel-side g_idx 路径;
  3. 只有合成 parity 过、Gemma3 early-smoke 首步 logits finite 后,才恢复
     W2 L2-L5 正式 gate。

## 2026-06-13(晚 IV)— W2 CUDA 早停:首步 logits 全 NaN,实例销毁

- Vast 实例 `40806710`(Iceland host 1647,1×RTX 4090,120GB,$0.402/hr)
  用于 W2 Gemma3-27B CUDA retry。Ferrum release CUDA build 完成,
  GPTQ/GGUF 下载完成;在 llama.cpp 仍编译时提前并行启动同参数
  `model_coverage_smoke` early smoke,避免等待 perf 前置项。
- 结果:Gemma3-27B GPTQ 仍不能转绿。服务能加载并进入首个 known-answer
  prefill,但 `logit_dump_smoke/logits.bin` 为 **262208/262208 全 NaN**:
  `finite=0, nan=262208, posinf=0, neginf=0`。`early_smoke.log` 记录
  `logits-topk known-answer-0-prefill: n=262208 finite=0 nonfinite=262208`,
  随后 known-answer 输出仍为空。
- 按 correctness first-stop,未进入 L5/perf。artifact 已回收到
  `artifacts/w2_gemma3_cuda_nan_logits_2026-06-13/`;核心证据包括
  `early_smoke.log`、`gates_early/nan_logits_summary.json`、
  `gates_early/logit_dump_smoke/logits.bin` 与 serve/build/download 日志。
- 已回收的 partial layer dump 显示 `embed.bin` 与 `layer_00.bin` 到
  `layer_07.bin` 全部 finite,但最终 `logits.bin` 全 NaN。NaN 不是
  tokenizer/embedding 入口即炸,下一步应定位 layer 8+ 或 final norm /
  tied lm_head / logit softcap 边界。
- 实例已销毁;Vast API 验证 `count: 0`。`w2_matrix.json` 仅更新
  `l2_gptq_cuda` 的失败证据指向本轮 NaN artifact;`l3_behavior`/
  `l4_agent` 仍引用上一轮完整 smoke 失败,`l5_concurrency` 与
  `perf_vs_llamacpp` 继续 pending。

### 下一步

- 不再重跑整条 W2 gate。先本地/小远端定位 Gemma3 GPTQ CUDA NaN:
  1. 用已有 layer dump 找首个 NaN 层/算子边界;
  2. 优先查 GPTQ desc_act permutation、Gemma3 tied lm_head / embed scale、
     `final_logit_softcapping` 和 CUDA dtype/scale 路径;
  3. 修到首步 logits finite 后,再开 1×4090 smoke;只有 smoke PASS 才跑
     L5/perf。

## 2026-06-13(晚 III)— W2 本地诊断加固:修一个采样 mask 缺口,下轮收 top-k

- 没有重开 GPU,先基于 `w2_gemma3_cuda_failure_2026-06-13` 失败证据做本地收口。
  W2 仍是 correctness blocking,不得转 pass。
- 代码修复:`SequenceState::requires_full_logits_for_sampling()` 现在会在 tokenizer
  暴露 base vocab 之外的可生成 control token 时强制 full logits。此前若
  `FERRUM_GREEDY_ARGMAX=1` 且模型 argmax 落在扩展/保留区,可能绕过
  `sample_with_processors` 的 extended-vocab mask。Gemma3 的
  model vocab/tokenizer base vocab 形态正好需要防这类风险。
- 诊断加固:`scripts/model_coverage_smoke.sh` 新增可选
  `FERRUM_SMOKE_LOGIT_DUMP_DIR`。设置后复用既有 `FERRUM_LAYER_DUMP`,
  在首个 known-answer 请求后打印 prefill `logits.bin` 的 finite/nonfinite
  统计与 top10 token id/logit。`scripts/pod_w2_gates.sh` 已为 W2 smoke
  开启该 dump,并在 smoke 首败时复制 `/tmp/ferrum_w1_smoke_8400.log`
  到 gate artifact 目录。
- 本地验证:
  - `cargo test -q -p ferrum-engine sample_allows_generated_control_tokens_above_base_vocab`
    PASS。
  - `cargo test -q -p ferrum-engine continuous_engine` PASS(18 tests)。
  - `bash -n scripts/model_coverage_smoke.sh scripts/pod_w2_gates.sh scripts/pod_w2_gemma3.sh`
    PASS。
  - `cargo fmt --all -- --check` PASS。
  - `scripts/w1_goal_validator.py` 仍 `MODEL_COVERAGE_W1 GOAL PASS`。
  - `scripts/w2_goal_validator.py` 仍 3/8,5 blocking cells(预期)。

### 下一步

- 下一次 1×4090 W2 重跑前先看 smoke top-k:
  - 若 top-k 集中在 `>= tokenizer_base_vocab_size` 或控制 token,优先验证本次
    full-logits/extended-mask 修复是否已把输出拉回正常文本。
  - 若 top-k 已是正常文本 token 但 decode 仍空,继续查 tokenizer decode /
    streaming delta。
  - 若 top-k 本身全异常/非数/同一保留 token,转向 GPTQ loader 或 Gemma3
    lm_head/logit softcap/量化 scale 路径。

## 2026-06-13(晚 II)— W2 Gemma3 CUDA gate 首败:加载成功,正确性失败,实例销毁

- Vast 实例 `40798977`(Iceland host 1647,1×RTX 4090,120GB,$0.402/hr)完成
  W2 重试并已销毁;API 验证 0 实例。artifact 已回收到
  `artifacts/w2_gemma3_cuda_failure_2026-06-13/`。
- 远端证据:
  - `build.ok`:Ferrum release CUDA build 完成。
  - `dl_gptq.ok` / `dl_gguf.ok`:circulus GPTQ 与 unsloth Q4_K_M 下载完成。
  - `llamacpp.ok`:llama.cpp `llama-bench` 构建完成。
  - `gates/session_metadata.json`:git SHA `86633c2d...`,dirty files,RTX 4090
    24GB,driver 565.77,nvcc 12.4 已记录。
- 环境修复:复现 W1 的 GeForce forward-compat `libcuda` 问题
  (`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`);已在远端移走 `/usr/local/cuda/compat/libcuda*`,
  并固化进 `scripts/pod_w2_gemma3.sh`。
- Gate 结果:Gemma3-27B GPTQ **能加载并服务**,但 smoke 正确性首败:
  - known-answer **0/10**,所有答案为空字符串,`finish_reason=length`。
  - natural EOS / custom stop / multi-turn 失败;stream identity 与 max_tokens
    mechanics 本身通过。
  - required tool-call **0/10**(HTTP 400),strict json_schema **0/20**(HTTP 500)。
  - 按 GOAL 首败即停,**未进入 L5/perf**。
- 矩阵更新:`w2_matrix.json` 将 `l2_gptq_cuda`、`l3_behavior`、`l4_agent`
  标为 fail 并引用 smoke artifact;`l5_concurrency` 与 `perf_vs_llamacpp`
  仍 pending(正确性未过,不得 bench)。

### 下一步

- 不再重复整条 W2 gate。先本地/小远端诊断 Gemma3 GPTQ 空输出:
  1. dump prompt token ids + first-step logits/top-k/EOS/PAD ids,判断是模板/EOS
     还是 logits/quant 退化;
  2. 用 1B BF16/GGUF 已绿路径对比同 prompt,确认 CUDA GPTQ-only 问题;
  3. 重点查 desc_act GPTQ permutation、Gemma3 `final_logit_softcapping`、
     tied lm_head / embed scale、logit mask/EOS stop。

## 2026-06-13(晚)— W2 pod 停止后本地收口:证据仍缺 5 格,gate 脚本加固

- 已按 `ACTIVE_PODS.md` 记录:W2 Gemma3 pod `40770078` 在 run 4 中途按用户
  "pod 销毁停止" 指令销毁,API 归零验证 0 实例。该 pod 已把三个修复提交回
  当前分支:smoke 阶梯不再钉死 KV、27B 16-kv-head capacity math、CUDA
  `scale_inplace` 保持 device dtype。
- 当前权威验证状态:
  - `scripts/w1_goal_validator.py` → `MODEL_COVERAGE_W1 GOAL PASS`
    (72/72)。
  - `scripts/w2_goal_validator.py` → 3/8 satisfied,仍阻塞在
    `l2_gptq_cuda`、`l3_behavior`、`l4_agent`、`l5_concurrency`、
    `perf_vs_llamacpp`。27B GPTQ 已能加载并服务单请求,但没有完整 L2-L5
    gate artifact,不得转 pass。
- 本地加固:W2 pod gate 脚本改为原始判据对齐:
  - `bench-serve` 强制 `--fail-on-error --require-ci --seed 9271`
    且显式 `random 256/128`。
  - c=32 不再 best-effort;`l5_gemma3-27b-gptq_cuda.json` 必须包含
    c=1/4/16/32 全 cell,每 run 100/100 完成、零错误。
  - llama.cpp 同卡比值 `<0.5x` 会让脚本失败,不只打印 FAIL。
- 本地加固:W2 validator 不再只检查 artifact 存在。若矩阵 cell 标为 pass,
  validator 会检查 smoke log 的 `SMOKE PASS`,L5 JSON 的
  c=1/4/16/32、N=3、usage token count、零错误,以及 perf ratio ≥0.5。

### 下一步

- 若继续 W2,重新开 1×4090 pod 前沿用下方 W2 pod 执行合同;跑
  `scripts/pod_w2_gemma3.sh`,回收 `/workspace/w2/gates/` 后只按
  validator 可证明的 artifact 更新 `w2_matrix.json`。
- 不得在缺少 `MODEL_COVERAGE_W2 GOAL PASS` 前启动 W3 实现;W3 当前只保持
  `W3_CHARTER.md` 草案交付物。

## 2026-06-13(凌晨)— CUDA pod 批次收官:实例归零,9 个产品 bug,4 模型 CUDA 认证

**pod 全部销毁,API 验证 0 实例**(单卡 ~9h + 双卡 ~5h + 两台坏机即弃,
总花费 ≈ $7)。夜班战果:

- **CUDA 侧转绿**:R1-8B BF16 smoke 12/12 + L5(54.5/163.7/382.5);
  R1-Distill-70B 双卡 L2/L3 + L5(21.4/67.3/68.1);R1-Distill-32B GPTQ
  L2/L3(known-answer 10/10 @ kv8192×1);Qwen3-32B GPTQ L2(10/10 +
  tools 10/10);Qwen2.5-Coder-32B GPTQ 全梯一次过(含 schema 20/20)。
  M3 同 pod 基线锚点 c=32 556.5±84。
- **修复并验证的引擎 bug**(夜班新增,均已推送):流式 think 泄漏
  (distill 全家,70B 上 E2E 验证);Marlin n%256≠0 在 m>16 崩溃 →
  m≤16 分块复用已证 128×128 路径(dense-dequant 首版方案引入新问题已
  撤);GGUF 多卡 layers=auto 未物化;serve/pull 过期 alias 副本;
  pod 构建缺 vllm-paged-attn-v2 的硬报错;smoke harness 两处
  (异常计为 miss、KV 钉死与 autosizer 叠加)。
- **L1 终局(等批提案 #3)**:20 分叉点中位 logit 间距 0.75/min 0 →
  BF16 平票翻转,非数值缺陷;跨实现逐位一致原则上不可达。
- **OPEN ISSUES(下一会话,按正确性优先排序)**:
  1. strict json_schema 在 32B-GPTQ 上间歇 500(~25-30% 请求,
     R1-32B 15/20、Qwen3-32B 14/20;R1-8B-BF16 与 Qwen2.5-Coder-32B
     同路径 20/20)+ 500 不落引擎日志的可观测性缺口。
  2. Coder-30B jart25-GPTQ 的 CUDA chat 首 token 即 EOS(Metal GGUF
     同 prompt 正常、CUDA 随机上下文 L5 正常);prime suspect =
     仓库自带魔改模板(sha 30b8ba8f ≠ 官方 5a38bfa0);待 prompt-ids
     双端 dump 对照。
  3. 同构 perf:Coder-30B c=32 411.6 vs M3 556.5(0.74×,宿主机噪声
     stddev 15% 需复核)— 超 ≤10% 判据,按 GOAL 记接入问题待查。
  4. CUDA autosizer:reasoning/32B-GPTQ 的 (seqs×capacity) 联合推导
     (512/seq 400、0-blocks 报错两态)。
- distill 系 tools-in-think 行为差异落档:32B 注入式 tools 10/10 可用,
  70B 把调用写进未闭合 think(0528 系 10/10)→ README 按"agent 分级"
  如实标注。

## 2026-06-12(夜)— 用户决策落地:32B Metal 收束 + CUDA pod 批次启动

- **用户指令**:32B 稠密不再在 32GB Mac 上折腾("同架构已证即可"——
  Qwen3-14B/R1-8B 的 Metal pass 即同架构证明);Vast 已充值,批准开
  GPU;**严格要求高效利用 + 异步并行 + 空闲即毁 + 结束全毁**(用户刚
  手动清理了数台未销毁实例)。API 已核实当前 0 实例。
- 矩阵落实:R1-Distill-32B 与 Qwen2.5-Coder-32B 的 `l2_gguf_metal`
  waived(同架构证明 + 部署无场景);Qwen3 dense 行按 14B 证据 pass。
- L5 Metal 批次进行中:Coder-30B ✅(c1/4/16/32 全零错)、R1-8B ✅
  (22.9/23.4/54.2 tok/s)、14B/Mistral-Small/Magistral 排队自动跑。

### CUDA pod 批量执行合同(开 pod 前置,GOAL 模板)

```text
Lever: W1 CUDA gate 批量 —— 单卡 4090:L1 代表(R1-8B BF16 byte-equal
  N≥20 vs transformers)+ L2-GPTQ smoke(R1-Distill-32B/OPEA、
  Qwen3-32B/JunHowie、Qwen2.5-Coder-32B/官方、Qwen3-Coder-30B/jart25)
  + R1-8B CUDA BF16 smoke + 各模型 L5(c=1/4/16,30B 级补 32)
  + C7/G0 存量回归(M2 Llama-8B-INT4、M3 Qwen3-30B-A3B-GPTQ floor)。
  双卡 2×4090:R1-Distill-Llama-70B GGUF 4bit layer-split smoke + L5。
Expected gain: ~18-20 个 gate cell 转绿,W1 除 README 外收口
Files: scripts/model_coverage_smoke.sh(复用)+ pod 上逐步驱动
Correctness gate: 每模型 smoke 全绿;首败即停该模型并记录
Benchmark gate: L5 全 cell 100%/零错误;同构 ≤10%;C7/G0 不回退
Budget cap: ≤2 pod-day;预计单卡 ~$0.35-0.5/hr + 双卡 ~$0.7-1/hr,
  目标一晚收口(~$10-25)
Stop condition: 单模型卡壳 >4h 降级记录;pod 空闲即毁;
  结束后 API 验证实例数 = 0(用户硬性要求)
```

## 2026-06-12(晚 II)— Mistral 线 2/3 收口;Devstral 2 降级(mistral3);[THINK] 修复

- ✅ **Mistral-Small-3.2 全过**(10/10,首个满足 L4 schema 20/20 新判据的
  模型);✅ **Magistral 12/12 全过**——其 reasoning 走 `[THINK]` 特殊
  token,暴露并修复了一个普适 bug:**skip-special 解码会吞掉标 special
  的 think 标记**,思考文本漏进 content(Qwen3 标 special 的 `<think>`
  同样潜伏)。修复:tokenizer 解码按标记 id 分段、规范化为
  `<think>/</think>` 再拼接,下游零改动,带单测。
- 🔻 **Devstral 2 按 GOAL 卡壳规则降级到 W2 末尾**:GGUF arch 是
  **mistral3**(YaRN factor 48 / 原窗 8192 / `attention.temperature_scale
  0.1`,全在 `mistral3.*` 命名空间)。loader 此前静默走 llama-family
  路径 → 退化输出(known-answer 3/10、重复循环)。已加**未知架构硬报错**
  守卫(带单测)——明确不支持好过悄悄输出垃圾。实现 mistral3 = 新
  rope/注意力数学,超出 W1 SMALL 预算;W2 与 Gemma 3 异构注意力地基
  一并评估。
- 验证器 19/63 → **30/72**(Devstral 拆分出独立降级行)。
- Mistral 线剩余 cell:L5 并发 + perf(pod)+ README。

## 2026-06-12(傍晚)— 修订批准落实;32B 稠密 Metal 诊断(需重启)

- **两个 GOAL 修订经用户批准并写入 GOAL.md 修订记录**:L1 按代码路径
  代表执行(5 个不可行 cell 转 waived,R1-8B 是 dense 路径代表);
  L3/L4 逐模型载体改为 smoke 阶梯(判据数字不变,schema 升 20/20)。
  R1-8B 与 Coder-30B 的 L3 cell 凭既有扩展 smoke 证据转 pass。
  验证器 12/63 → **19/63**。
- **Qwen3-14B Metal smoke 10/10 过**(cell 与 32B 同行,等 32B)。
- **32B 稠密 Metal 诊断**:R1-Distill-32B smoke 两次超时后实测解码
  **0.14 tok/s**(TTFT 5.2s 正常)——每 token 把被驱逐的 18GB mmap 权重
  从 SSD 重读(~2.6GB/s = SSD 速度)。llama.cpp 同文件对照**同样卡死**
  (22 CPU 分钟未完成加载)→ 非 ferrum 接入 bug。根因:早上 KV 池
  thrash 事故在压缩器里留下 ~9GB 系统级残留,可用内存 < 模型工作集。
  **需要用户重启后公平复测**;若干净 32GB Mac 仍装不下 32B 稠密 + 服务
  开销,则 32B 级稠密(R1-32B / Qwen3-32B / Qwen2.5-Coder-32B)的
  Metal cell 按修订精神 waive 给 CUDA GPTQ lane。
- 教训入库:32GB 机器一次只跑一个重负载(13GB 下载 + 18GB 常驻模型的
  page-cache 互相驱逐就是第一次超时的原因)。

## 2026-06-12(午后 II)— R1-8B L2-Metal cell 转绿;HF_ENDPOINT 落地

- ✅ **R1-8B 扩展阶梯 12/12 全过**(known-answer 10/10 语义正确 + stop
  不漏 + max_tokens 守预算 + reasoning/stream/tools/schema 机制),
  `l2_gguf_metal` cell 转 pass,验证器 11/63。
- **`HF_ENDPOINT` 支持落地**(huggingface_hub 同约定):本网络实测
  hf-mirror 直连 2.08MB/s vs 代理 0.156MB/s(**13×**)。Coder-30B 下载
  已切镜像直连续传(ETag 与 hub 一致,blob 无缝续);预计 ~1h 内落盘。
- R1-Distill-Llama-70B 模板与 R1-Distill-Qwen-32B fixture 同 hash 同
  EOS(`56a1447ad31926fd`),L0 模板面由现有 fixture 覆盖。

## 2026-06-12(午后)— gate 矩阵 + 验证器落地;两个 GOAL 修订提案待批

- **`w1_matrix.json` + `scripts/w1_goal_validator.py` 落地**:7 个模型 ×
  9 个 cell = 63 cell,当前 10/63 满足(L0 ×6 + waived ×4)。验证器是唯一
  允许打印 `MODEL_COVERAGE_W1 GOAL PASS` 的程序;cell 必须 pass(带
  artifact)或 waived(带理由),引用的 artifact 必须存在。
- **L0 完成度**:43/43 golden 全过(9 个 fixture 模型,新增 Mistral 线
  ×3 + Llama-3.1;`strftime_now` 时钟注入 + `tojson(indent=N)` 两个真实
  渲染缺口由 L0 抓出并已修,commit `c8f3703e`)。
- smoke 阶梯扩充:known-answer 1x→10x(对齐 L2 判据),新增自定义 stop
  机制断言 + max_tokens 截断断言(L3 缺口)。
- **修订提案 #1(L1,需用户决定)**:L1 BF16 byte-equal 对 14B+ 在现有
  硬件上物理不可行(14B BF16=28GB>24GB 单卡;32B=64GB;70B=140GB)。
  提案:L1 按"代码路径代表"执行——每条代码路径取硬件放得下的最大代表
  (Qwen3 dense → 8B/0.6B 已有 reference_match;Qwen3-MoE → 30B-A3B 需
  pod 上 BF16?同样放不下,24GB 单卡上 MoE BF16 60GB 也不可行 → MoE 路径
  L1 只能 waive 到"Mac/CPU 逐层激活对照"或双卡)。大尺寸模型靠"同代码
  路径 + L2 行为对照"传递。**未批前 5 个 l1_bf16 cell 保持 pending。**
- **修订提案 #2(L3,需用户决定)**:blast-radius 套件断言对 0.6B 哨兵
  模型定制(canonical id、即答行为),对 8B-32B reasoning 模型强行参数化
  会又重又脆。提案:L3 判据改为"model_coverage_smoke 的 L3 段全绿"
  (多轮/stream/自然 EOS/自定义 stop/max_tokens/reasoning 提取,行为
  断言与套件同源),blast-radius 套件保持小模型哨兵职责(引擎级回归)。
  **未批前 L3 cell 不以 smoke 结果记 pass。**

## 2026-06-12(午前)— L0 扩面:模板同一性 + Mistral/Llama golden

- **模板同一性(HF raw tokenizer_config,sha256 前 16 位)**:
  Qwen3-0.6B / 14B / 32B 模板逐字节同一份(`a55ee1b1660128b7`,EOS
  `<|im_end|>`);R1-Distill-Qwen-14B / 32B 同一份(`56a1447ad31926fd`,
  EOS `<｜end▁of▁sentence｜>`)。**结论:Qwen3-14B/32B 与 R1-Distill-14B
  的 L0 由现有 golden fixture 直接覆盖**,各自只剩 per-model
  EOS/generation_config 断言(T4 机制已通用)。
- Mistral 24B 线 + Llama-3.1 golden fixture 生成中(来源 = serve 实际用的
  tokenizer 仓库:unsloth 镜像 ×2 + mistralai 上游 ×1 + unsloth Llama)。
  环境坑:huggingface_hub 新版走 httpx,SOCKS 代理需要 `httpx[socks]`
  (socksio),`pysocks` 只管 requests。
- Coder-30B GGUF(17.28GB)断点续传循环推进中(代理频繁断流,每次
  尝试落 1–3GB,`.incomplete` blob 在涨)。

## 2026-06-12(深夜 III)— ✅ R1-0528-Qwen3-8B GGUF Metal smoke 全绿

**W1 第一个模型过本地阶梯**:`FERRUM W1 SMOKE PASS: deepseek-r1:8b-q4_k_m`
(8/8:known-answer、自然 EOS、reasoning 提取、think 不漏入 content、
多轮记忆、stream==non-stream、required tool 10/10、strict json_schema
10/10)。证据:`artifacts/smoke_deepseek-r1-8b-q4_k_m_metal_2026-06-12.txt`。
serve 参数:`--kv-capacity 8192 --max-num-seqs 4`(见下条 thrash 诊断)。
注意:这是 L2/L3/L4 的可跑子集;最终认证仍需完整套件
(json_schema 20/20 走 server_structured_output)+ CUDA 侧 gate。

## 2026-06-12(深夜 II)— GGUF pull 产品缺口修复 + KV 池 thrash 诊断

R1-8B GGUF smoke 调试中钉死三个真实产品问题(全部影响 W1 每个 GGUF alias):

1. **pull sidecar 全量下载 bug(已修)**:GGUF 仓库缺 tokenizer.json 时,
   兜底走 `HfDownloader::download(sibling)` —— 会把 sibling 的 **safetensors
   权重(8B≈16GB)整库拉下来**,只为拿 tokenizer。磁盘紧张时必死,这就是
   此前需要手工拷 tokenizer 的根因。新增
   `HfDownloader::download_sidecar_files`(只拉指定小文件),pull 改用之,
   清单补上 `generation_config.json`(EOS 解析第一优先级)+
   `chat_template.jinja`。
2. **bartowski 系 sibling 映射全断(已修)**:HF API 实测 9 个 W1 GGUF 仓库
   **全部不带 tokenizer.json**,sibling 兜底是必经之路;而 strip `-GGUF`
   约定对 bartowski/*(无 safetensors 镜像)全部失效。
   `tokenizer_sibling_repo` 加显式映射(2026-06-12 HF API 逐个核实
   tokenizer.json 存在):Qwen2.5-Coder→Qwen 官方;Mistral-Small-3.2 /
   Magistral→unsloth 镜像(**mistralai 上游只有 tekken 格式,无 HF
   tokenizer.json**);Devstral 2→mistralai 上游;Llama 系→unsloth 镜像
   (meta-llama 上游 gated)。
3. **`--kv-capacity` 单独抬高 = 32GB Mac 内存灾难(smoke 已加防护)**:
   KV 池 = `max_num_seqs × kv_capacity`。autosizer server 档默认
   (32, 512)≈2GB;只把 capacity 提到 8192 会得到 32×8192≈36GB 池
   (8B/36 层/8KV头/128hd),Metal 分配直接把机器打进内存压缩 thrash
   (实测:health 能过、首个请求触发 `ensure_kv` 后 600s 超时,压缩器
   存页 38GB)。smoke 的 reasoning 档改为
   `--kv-capacity 8192 --max-num-seqs 4`(池 32K token,与默认同量级)。
   **autosizer 产品缺口升级**:reasoning 模型需要的不是"调大 capacity",
   而是 (seqs × capacity) 在显存预算内的联合推导 + 长上下文低并发档位;
   `--kv-capacity` 作为独立产品 flag 缺少联动护栏。

## 2026-06-12(深夜)— 本地验证推进与环境修正

- **修正**:HF 缓存里的 R1-0528-8B / R1-Distill-32B / Qwen3-Coder-30B /
  Qwen2.5-Coder-32B 仅为 6–11MB 元数据壳(config/tokenizer),**无权重**。
  W1 端到端一律需要下载。
- 磁盘:删除 target/debug(15GB)后约 16GB 可用;R1-8B Q4_K_M GGUF(~5GB)
  下载中(第一次因网络/代理 "error decoding response body" 失败,重试中);
  Qwen3-Coder-30B Q4_K_M(~18.6GB)需要更多空间——待用户清理或换机。
- 新增 `scripts/model_coverage_smoke.sh <alias> [--reasoning]`:
  L2/L3/L4 阶梯(known-answer + 自然 EOS / 多轮 / stream==non-stream /
  reasoning 提取 / required tool 10x / strict schema 10x),所有 W1 模型复用。
- 下一步(按序):R1-8B GGUF smoke(--reasoning)→ 视磁盘跑
  qwen3-coder:30b-q4_k_m → W1 收尾(README 矩阵 + 验证器)→ pod 合同。

## 2026-06-12(深夜)— blast-radius 存量回归结果

T3/T4/T5 处于 EOS/stop/模板爆炸半径,全套件(release + Metal,真模型)结果:

- ✅ chat_smoke 13 / server_smoke 10 / chat_pty 3 / chat_stress 2 / server_stress 2
- ✅ server_openai_compat 7/7 — 其中两处修复:
  - `test_python_openai_sdk_*`:本机环境缺 `openai`/`socksio`(SOCKS 代理),
    已 pip --user 安装,非代码问题。
  - `test_openai_client_tools_stream_*`:模板修正后 prompt 与 transformers
    字节一致(差 1 token),0.6B 贪心解码改为真的调用工具——服务器输出了
    规范的 tool_calls delta + finish=tool_calls + usage。测试断言改为
    "文本 XOR 合法工具调用"(7c69e2a7),钉住流式机制而非模型选择。
- ⏸ reference_match:1 行 drift **等用户审核后 re-baseline**(分类器按
  CLAUDE.md 拦截了自动重置,正确):case `qwen3-0.6b-arith-2-plus-3`
  内容与 token 数完全一致,仅 `finish_reason: length → stop` ——
  这是 EOS 修复的直接证据(此前 tokenizer 探测不到 Qwen EOS,自然停止
  被误归因为 budget 耗尽)。审核通过后执行:
  `FERRUM_UPDATE_FIXTURES=1 cargo test --release -p ferrum-cli --features metal --test reference_match -- --ignored --test-threads=1`

## 2026-06-12(晚)

- **T5 完成:L0 golden 基建落地并修出 7 处真实偏差**(PR #234,auto-merge):
  - `scripts/gen_chat_template_goldens.py` + 5 模型 23 用例 fixture 入库,
    `chat_template_golden` 测试 23/23 与 transformers 字节级一致。
  - 修复项:trim_blocks/lstrip_blocks 对齐 transformers;tojson 改 Python
    json.dumps 风格(自定义 filter);minijinja+serde_json 双 preserve_order
    (minijinja 对 Rust struct 字段强制字母序,tools 改为有序 JSON 值进模板);
    `PromptMessage::new` 不再急切剥离 assistant 历史的 `<think>`
    (剥不剥是模板的政策:DeepSeek 剥、Qwen3-Coder 保留)。
- **W1 全模型 alias 配齐**(均经 HF API 核实文件名):safetensors/GPTQ/GGUF
  三组,含 deepseek-r1:8b/14b/32b、qwen3-coder:30b、qwen3:14b/32b、
  qwen2.5-coder、mistral-small/devstral/magistral 24b 线。
- **YaRN clamp 落地**:不支持的 rope_scaling → `max_seq_len` clamp 到
  `original_max_position_embeddings` + 启动警告(R1-0528 由 131072 clamp 到
  32768),含单测。
- **环境约束发现**:本机磁盘 100%(HF 缓存 42GB);已清理 target/debug/
  incremental 释放 7.3GB。**新模型权重无法下载**,但缓存中已有
  R1-0528-Qwen3-8B、R1-Distill-32B、Qwen3-Coder-30B、Qwen2.5-Coder-32B 的
  safetensors + blast-radius 三小模型 → 本地验证用缓存模型推进。
- blast-radius 套件(chat_smoke/pty/stress + server 三件 + reference_match)
  在后台执行中——T3/T4/T5 改动处于 EOS/stop/模板爆炸半径,存量回归必须绿。

## 2026-06-12(下午)

- **T3 完成并提交(`778082a6`)**:minijinja-contrib pycompat 接入;模板渲染失败/
  渲染为空改为硬错误(消灭静默 fallback);tools-unaware 模板改为"注入工具 spec 后
  仍走模型模板渲染"(此前会静默丢弃工具定义)。新增 5 条防回归测试,
  workspace 测试全绿。
- **T2 完成(外部核查,逐仓库 config 原文)**,三个重要修正:
  - **GLM-4.7-Flash 是 MLA 注意力**(q_lora 768/kv_lora 512)+ noaux_tc 路由 + MTP,
    接入成本 MEDIUM→LARGE,已从 W2 移出(W2 只剩 Gemma 3 27B)。
  - **R1-0528-Qwen3-8B 无 Marlin-clean GPTQ**(QuantTrio 版 sym=false+4/8 混合)
    → CUDA 走 BF16 + GGUF。**Devstral 2 同样无 Marlin GPTQ** → GGUF/BF16。
  - W1 各模型的 Marlin-clean GPTQ 仓库已逐一锁定(jart25 / OPEA / JunHowie /
    Qwen 官方 / Intel AutoRound),写入 GOAL.md UNVERIFIED #4。
  - Qwen3.6 无官方 GPTQ-Int4(仅 FP8);官方 GPTQ 停在 Qwen3.5 代。

## 2026-06-12

- GOAL.md 建立并提交(分支 `goal/model-coverage-20260612`)。
- 验收 gate 定义(L0–L5 正确性 + 分类性能门槛)写入 GOAL.md。
- UNVERIFIED 落证(本地 4 项,全部完成):
  - #1 YaRN:不支持(仅 Llama3 变体);发现 max_seq_len 不 clamp 的隐患,
    已追加为 W1 公共工程项。
  - #2 AWQ:无 loader,纯 Future 注释;维持 defer。
  - #3 gguf arch 白名单:`qwen3|qwen3moe|qwen2|qwen|llama|mistral`,
    W1 够用,GLM(W2)需新增。
  - #8(新增)模板渲染失败静默 fallback 实锤(`chat_template.rs:226/488`),
    待 T3 消灭。
- UNVERIFIED #4/#5/#7(GPTQ group size / GLM config / Qwen3.6 官方 GPTQ)
  由后台 web 核查进行中。
- 任务分解:12 个任务建于会话任务系统(T1–T12),T1 完成。

### 下一步

- T3:模板引擎改造(minijinja pycompat 路线 + 渲染失败显式报错 + 防回归测试)。
- T4:EOS/BOS generation_config 审计。
- T5:L0 golden 测试基建(需本机 Python transformers 生成 fixture)。

### 阻塞项(预先声明)

- CUDA 侧 gate(L2-GPTQ / L5 / C7 回归)需要 4090 pod:开 pod 前按 GOAL
  执行合同填表并征得用户预算批准(CLAUDE.md 要求)。当前无可用 pod
  (上一台 38237968 已失;见 memory)。本地(Metal/CPU)可推进项先行。

## 2026-06-13 15:25 — W1 GOAL PASS

- `scripts/w1_goal_validator.py`: **72/72 cells satisfied →
  `MODEL_COVERAGE_W1 GOAL PASS`**。
- 最后 6 cell(32B 三连 l5_concurrency + perf_same_arch)由冰岛 pod
  40751023 一次干净会话收齐:
  - `l5f_r1-32b_cuda.json` c=1/4/16/32 = 40.8/116.6/248.9/300.6 tok/s,
    1200 请求 0 错误。
  - `l5f_qwen3-32b_cuda.json` c=32 = 273.6 tok/s,0 错误。
  - `l5f_qwen25-coder-32b_cuda.json` c=32 = 257.1 tok/s,0 错误。
  - perf_same_arch(修订 #4 判据):三方互校最差偏差 8.5% ≤ 10% →
    `W1_PERF_SPREAD PASS`。
- GPU 纪律:三台问题宿主(台湾 docker_build 坏 / 阿根廷不开机 /
  冰岛 cuInit=804)处置后,**API 归零验证 0 实例**。本夜累计 GPU 支出
  约 $9。
- 新宿主病理学(已固化进 `pod_w1_final_armored.sh`):
  - cuInit=804 = 容器 compat libcuda(550)压住宿主驱动,GeForce 不在
    compat 支持表;删 compat so + ldconfig 即愈。
  - rsproxy.cn 在部分欧洲宿主被 TLS 劫持;脚本现在先探测 crates.io
    再选镜像。
  - hf xet/hf_transfer 在该宿主网络下饿死(0 MB/s);关 xet + 关
    hf_transfer 的普通 HTTP 路径反而跑满 3.6 Gbps。
- 待办移交(不阻塞 W1,记录在 GOAL.md 开放问题):schema-500、
  Coder jart25 CUDA chat、CUDA autosizer、Metal L5 复跑(等本机恢复)。

### 下一步

- W2:Gemma 3 27B 家族接入(SWA 5:1 / 双 rope / GeGLU / 三明治 norm /
  query_pre_attn_scalar),本地 Mac/CPU dump 对照先行,CUDA 验证晚开 pod。
- W3:DeltaNet 调查(W1+W2 后解锁)。

## 2026-06-13 — W2 Gemma3 接入(本地段完成)

- W2-1 实现:Gemma3 经 config 门控并入 LlamaFamilyModel(5:1 SWA 逐层
  调度 / 双 rope 表 + Linear scaling / GeGLU / 三明治 norm / (1+w) 与
  q_scalar 载入期折叠 / embed×√h)。batched/varlen/paged 快路对
  sandwich 家族构造期禁用(防静默错误),W2-3 再接。
- W2-2 验证:L0 golden 4 例字节相等;L1 dump 对照 CPU+Metal 双 PASS;
  greedy 18/20 byte-equal,2 例 HF top1-top2 gap=0.25(一个 bf16 ulp)
  平局翻转(修订 #3 方法)。证据 `artifacts/gemma3_l1/`。
- 顺带修复两个 Metal 内核潜伏 bug(Gemma3 首次踩出):
  flash_attn 简单核 acc[4] 在 head_dim=256 寄存器越界(→acc[8]);
  gelu_tanh fast-math 溢出 NaN(→clamp)。微基准 5 项钉死
  (`gemma3_metal_ops_test.rs`)。
- 27B 量化落证:ISTA-DASLab 是 compressed-tensors(不可直载);
  **circulus/gemma-3-27b-it-gptq = 经典 GPTQ 4b/g128/sym/desc_act=true,
  纯文本导出 `model.*` 命名** — ferrum perm-aware Marlin 已支持
  desc_act(quant.rs:151)。GOAL 的"ISTA GPTQ"假设据此修正。
- W2 矩阵 + 验证器就位(`w2_matrix.json` + `w2_goal_validator.py`),
  当前 2/8(l0/l1 pass)。

### W2 pod 执行合同(开 pod 前按 CLAUDE.md 填)

```text
Lever: Gemma3-27B CUDA gates(L2 GPTQ known-answer → L3 行为 → L4 agent
  → L5 bench)+ 同卡 llama.cpp Q4_K_M decode ≥0.5× 对照
Expected gain: w2_matrix 6 个 pending cell 出 pass/fail 结论
Files: scripts/pod_w2_gemma3.sh(armored 模式复用 W1 套件)、
  model_coverage_smoke.sh、bench-serve、pod 端构建 llama.cpp
Correctness gate: known-answer 10/10 + smoke 机制全绿,任一 rung 失败
  即停(不进 bench)
Benchmark gate: bench-serve c=1/4/16/32 零错误;llama.cpp 同卡比 ≥0.5
  (0.5–0.8 记 known-gap 不阻塞)
Budget cap: 1 pod-day 硬顶;目标 ≤6h(约 $2.5,单卡 4090)
Stop condition: 正确性 gate 失败 → 停手出报告;8h 无进展 → 销毁重估
```

## 2026-06-14 — 发布级推进 II:Gemma3 CUDA device F32 residual shadow

- 实现:Gemma3 sandwich-norm 残差流在 CUDA 上改为设备侧 F32 shadow,只把
  norm 后的投影输入物化回常规 activation dtype。覆盖 `prefill_internal`、
  `decode_internal`、speculative `forward_verify` 和 layer-split stage helper;
  CPU/Metal 保持原 host/default fallback。batched/varlen 快路仍在构造期对
  sandwich 家族禁用,避免未实现 Gemma 语义时静默走错路径。
- CUDA backend 增加 `sandwich_norm.cu` 三个 helper kernel:
  activation→F32 shadow、activation RMSNorm→F32 branch、F32 shadow
  RMSNorm→activation;同时 `rms_norm` 支持 F32 typed buffer,`copy_slice`
  支持 F32→F32。
- 目标意义:移除上一轮 W2 c=32 证据里的每层 D2H/F32 host shadow sync/copy
  热路径,为后续同卡 A/B 重新测 `Ferrum / llama.cpp >= 0.8x` 做源代码准备。
  这仍不是发布级通过声明;W2 release-grade 需要
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -q --workspace --all-targets` PASS。
  - `cargo test -q -p ferrum-kernels --tests` PASS。
  - `cargo test -q -p ferrum-kernels --test cuda_activation_precision` PASS
    (本机默认特性下 0 CUDA tests)。
  - `cargo test -q -p ferrum-models --tests` PASS。
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS:`MODEL RELEASE GRADE GOAL SELFTEST PASS`。
  - `python3 scripts/release/selftest_g0_validators.py`
    PASS:`G0 VALIDATOR SELFTEST PASS`。
  - `python3 scripts/w2_goal_validator.py`
    PASS:`MODEL_COVERAGE_W2 GOAL PASS: docs/goals/model-coverage-2026-06-12`。
- 本机 CUDA feature check 受环境阻塞:
  `cargo check -q -p ferrum-kernels --tests --features cuda` 在 build script
  阶段因缺少 `nvcc`/`nvidia-smi` 失败,未能在本机编译 CUDA 源码。下一步需
  4090 CUDA pod 运行 feature build、CUDA precision tests、Gemma3 smoke 和
  release-grade W2 manifest/gate。
