# W3 Qwen35 9fda1101 GPU Diagnostic Start Block Retry

Date: 2026-06-25

Purpose: retry the c32 diagnostic for `9fda1101 fix(engine): narrow mixed prefill kv admission`.

Result: no GPU diagnostic ran.

Retained instance attempt:

- Instance id: `42216671`
- GPU: `1x RTX 4090`
- Start response: `Required resources are currently unavailable, state change queued.`
- Polls observed: 6
- Final observed state: `cur_state=stopped`, `actual_status=exited`, `intended_status=stopped`

Offer search:

- Queried Vast HTTP API endpoints: `/api/v0/bundles/` and `/api/v0/search/asks/`
- Required hardware: exact `1x RTX 4090`
- Required properties: verified, rentable, CUDA >= 12.4, sufficient disk for source/build/model/artifacts
- Result: no reliable offer was rented. Exact candidates were not rentable, and cheaper candidates were `deverified` or `unverified`.

No SSH/CUDA check, remote build, `ferrum run`, `ferrum serve`, c32 bench, or live vLLM ran.

Final instance state:

- Only retained instance `42216671` was listed.
- It remained `stopped/exited/intended_status=stopped`.
- No paid GPU was left running.
