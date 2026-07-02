# W2 dynamic KV full matrix checkpoint

Date: 2026-06-17

Status: blocked by external GPU capacity/billing state. This is not a W2
release-grade artifact and did not produce `MODEL_RELEASE_GRADE_W2 PASS`.

## Progress

- Dynamic on-demand KV code was already validated locally and on a minimal CUDA
  c=32 `ferrum serve` smoke in
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dynamic_kv_c32_serve_smoke_cuda_2026-06-17/`.
- Full-matrix lane was started on Vast instance `41276321`, 1x RTX 4090 with
  49GB visible VRAM, at commit `75c0f71175da2190c151678c17cb47b109766e9d`.
- vLLM 0.10.1.1 installed successfully on that instance.
- The first Hugging Face snapshot download failed with an `IncompleteRead`.
- A low-concurrency retry completed the model snapshot download: 17/17 files,
  HF cache about 16GB.
- A retry CUDA release build resumed and was compiling `ferrum-kernels` when
  the instance became unavailable.

## Current blocker

- Instance `41276321` is now `stopped/exited`.
- Restarting `41276321` returned `resources_unavailable`; the state change is
  queued by Vast, but waiting would waste time.
- Attempting to create a replacement 49GB RTX 4090 instance from offer
  `40813991` returned `insufficient_credit`.
- A sanitized instance audit shows no Vast GPU instances currently running.
- After checkpoint commit `995c703f`, a fresh sanitized Vast status probe still
  showed no running instances and showed `credit=0` with the account balance
  below the enabled threshold. Higher-priced offers cannot be rented until the
  external credit state changes.

## Evidence

- GPU contract: `gpu_contract.md`
- W2 release-grade manifest generator checkpoint:
  `995c703f test(release): build W2 release-grade manifest`
- Sanitized restart response:
  `local_vast/restart_41276321_retry.sanitized.json`
- Sanitized create response:
  `local_vast/new_instance_2/create_response_40813991.sanitized.json`
- Sanitized instance audit:
  `local_vast/instances_audit_after_insufficient_credit.sanitized.json`
- Fresh sanitized account/instance probe:
  `local_vast/resume_probe_20260617T_after_checkpoint/vast_status.summary.json`

## Next step

After Vast credit is available, do not wait for the old stopped instance. Create
a new high-availability 49GB RTX 4090 instance and rerun the same full-matrix
command. Correctness must pass before performance is used as evidence. The final
W2 gate remains:

```bash
python3 scripts/release/model_release_grade_goal_gate.py w2 <out_dir>
```

Required PASS line:

```text
MODEL_RELEASE_GRADE_W2 PASS: <out_dir>
```
