# S1 CUDA Qwen3.5-4B diagnostic

- Lane: `S1 CUDA / Qwen3.5-4B / ferrum run correctness diagnostic`
- Git SHA: `02b3f2d8e6edba9ecfba910e95245abe0c1142bd`
- Local dirty state at launch: artifact notes only
- Vast instance: `44965014`
- Hardware: `1x NVIDIA GeForce RTX 4090, 24564 MiB`
- Driver: `580.119.02`
- CUDA compiler: `12.4`
- Hourly rate: `$0.4166666667`
- Expected runtime/cost: `8-12 minutes / <= $0.09`
- Correctness prerequisite: dynamic pool tests `37/37 PASS`; Qwen3.5 conv layout contract `1/1 PASS`
- Stop condition: build failure, exact runtime failure, first valid token, or 90 seconds without a first token
- Performance command after correctness: `ferrum bench-serve --fail-on-error --seed 9271` (not part of this diagnostic)
- Model: `Qwen/Qwen3.5-4B`, cached HF snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Remote artifact: `/workspace/artifacts/s1_cuda_qwen35_4b_02b3f2d8_20260715T112109Z`

## Prediction

The sequence admission must no longer repeat one-MiB growth until the executor's fixed retry ceiling. A same-pool backing shortfall is now aggregated and grown in one maintenance operation. The next accepted signal is either a first output token or a new failure after sequence admission.

## Result

- Build: `PASS`, 4m15s, binary SHA256 `3cb183714f3aa2851f618136309d009e36fc5f7fb20eb031c26b88795ac69e4b`
- Product run: `REJECT`, deliberately terminated after the exact failure was observed (`run.exit=143`)
- Model load: 24.9s
- Failure class: logical fit capacity was deferred before the executor materialized growable backing capacity
- Exact gap: requested `1,572,864` bytes; current/available `1,048,576` bytes; maximum `15,576,584,095` bytes
- Next hypothesis: translate `AwaitBackingGrowth` through a typed capacity-domain-to-pool mapping and grow the live `requested-current_total` gap before retrying admission
