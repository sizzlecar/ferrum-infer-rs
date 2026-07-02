# Vast Keep-One Cleanup

Date: 2026-06-17 05:56Z

User instruction: keep one usable Vast instance and destroy the rest.

Kept:

- `41241013` (`ferrum-w2-vllm-ferrum-c16-ab-20260617`)
- reason: recent CUDA devel RTX 4090 used for same-hardware Ferrum/vLLM W2
  follow-up, with 300GB disk and prior successful SSH/CUDA use
- final state after cleanup: `cur_state=stopped`, `actual_status=exited`,
  `gpuCostPerHour=0`

Destroyed:

- `41178475`
- `41187356`
- `41218739`
- `41230499`
- `41256521`
- `41276321`

Verification:

- all six destroy summary responses record `success=true`
- three final list polls show only `41241013`
- retained stopped cost is disk-only: `totalHour=0.11111111111111109`

No raw Vast API responses are intended for commit.
