# ACTIVE Vast instances — MUST be destroyed before goal completion

| instance | config | offer $/hr | created | purpose | status |
|---|---|---|---|---|---|
| ~~40700470~~ | 1x RTX 4090, 220GB | 0.351 | 2026-06-12 ~20:00 | W1 CUDA gate batch (~9h) | DESTROYED 2026-06-13 ~04:40 |
| ~~40700477~~ | 2x RTX 4090 | 0.589 | | broken host (raw cuInit=999) | DESTROYED 2026-06-12 ~20:30 |
| ~~40703915~~ | 2x RTX 4090 | 0.336 | | host never booted (18min None) | DESTROYED |
| ~~40704262~~ | 2x RTX 4090, 160GB | 0.802 | 2026-06-12 ~21:50 | 70B dual lane (~5h) | DESTROYED 2026-06-13 ~02:50 |

Destroy: `curl -X DELETE "https://console.vast.ai/api/v0/instances/{id}/?api_key=$VAST_API_KEY"`
Verify zero: instances list must return 0 before declaring the goal done.

**2026-06-13 04:40 — API verified: 0 instances remaining.** Session GPU spend ~= $7.
