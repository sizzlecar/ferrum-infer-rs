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

| ~~40742315~~ | 1x RTX 4090 | 0.372 | | Quebec host killed cargo/nvcc twice | DESTROYED ~06:35 |
| ~~40744863~~ | 1x RTX 4090 | 0.136 | | Argentina host never booted (cluster issue) | DESTROYED ~06:50 |
| ~~40745272~~ | 1x RTX 4090 | 0.376 | | Korea host: session-scope process reaping killed builds 3x | DESTROYED ~08:20 |
| ~~40748763~~ | 1x RTX 4090 | 0.336 | | UK host never booted | DESTROYED ~08:40 |
| ~~40749191~~ | 1x RTX 4090 | 0.349 | | Iceland host stuck loading 18min | DESTROYED ~09:15 |
| ~~40749797~~ | 1x RTX 4090, 150GB | 0.376 | 2026-06-13 ~09:15 | Taiwan: host `docker_build() error writing dockerfile` | DESTROYED ~11:35 |
| ~~40750195~~ | 1x RTX 4090, 150GB | 0.136 | 2026-06-13 ~09:45 | Argentina racer: never booted (None ~25min) | DESTROYED ~11:50 |
| 40751023 | 1x RTX 4090, 120GB | 0.376 | 2026-06-13 ~11:40 | FINAL micro-session winner (Iceland host 1647, pytorch image cache-hit, booted <2min). cuInit=804 fixed by removing container compat libcuda 550 (GeForce unsupported by compat libs); host driver 535/CUDA 12.4 OK | ACTIVE |
