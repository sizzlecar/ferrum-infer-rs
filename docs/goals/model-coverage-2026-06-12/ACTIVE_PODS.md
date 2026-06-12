# ACTIVE Vast instances — MUST be destroyed before goal completion

| instance | config | offer $/hr | created | purpose | status |
|---|---|---|---|---|---|
| 40700470 | 1x RTX 4090, 220GB | 0.351 | 2026-06-12 ~20:00 CST | W1 CUDA gate batch | ACTIVE |
| 40700477 | 2x RTX 4090, 160GB | 0.589 | 2026-06-12 ~20:00 CST | 70B dual-GPU lane | ACTIVE |

Destroy: `curl -X DELETE "https://console.vast.ai/api/v0/instances/{id}/?api_key=$VAST_API_KEY"`
Verify zero: instances list must return 0 before declaring the goal done.
