# P-H3 image editing on RTX 5090 — 2026-11-11

These are candidate measurements, not a benchmark of the final release archive.
The workload is OpenAI-compatible, non-streaming image editing at 2720×1536.
Text TTFT, TPOT and SSE ITL do not apply; the declared image SLO is request
end-to-end P99 ≤120 seconds. Server capacity stays at eight throughout C1/C8.

| Measurement | 48 GB run 1 | 48 GB run 2 |
|---|---:|---:|
| C1 request latency | 26.9351 s | 23.6458 s |
| C8 wall time, first dispatch to last completion | 119.1339 s | 118.3978 s |
| C8 throughput | 4.0291 images/min | 4.0541 images/min |
| C8 request P50 | 87.2356 s | 86.3801 s |
| C8 request P99 | 118.7580 s | 118.0798 s |
| C8 successes / errors | 8 / 0 | 8 / 0 |
| C8 requests within 120 s | 8 / 8 | 8 / 8 |
| C8 sampled process-tree peak RSS | 41,563,996,160 bytes | 42,225,098,752 bytes |
| C8 valid RSS samples / expected | 597 / 597 | 593 / 593 |
| C8 model-file read bytes during request window | 4,484,452,352 | 0 |

Peak VRAM was not sampled. The configured 29 GiB device ceiling is a capacity
setting, not a measured peak. RSS sampling used 200 ms intervals and includes
file-backed model pages; it does not mean the process needs that much anonymous
heap. Run 1 C1 RSS has one missing sample (135/136); no complete C1 peak is claimed.
Two C8 runs with only eight requests each do not establish a stable production
P99, particularly with this small margin to 120 seconds.

## Model, execution and workload

- RTX 5090, 32 GB VRAM; Linux under WSL, 48 GB configured guest-memory limit
  (47.04 GiB visible to Linux). Service cgroup swap limit/current were zero.
- P-H3 metadata: PrunaAI/P-H3-Image revision
  `7f98acef503cf2bcfba821a78cb1cf1a9928b2df`; Comfy quantized archives:
  PrunaAI/P-H3-Image-Comfy revision `44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543`.
- Comfy W4A8 text encoder/DiT, INT8 ConvRot VAE, native prompt expansion enabled
  with F16 execution from BF16 source weights. Eight denoising steps; seed 42;
  CPU MT19937 image/audio initialization. MTP and reference-result reuse off.
- CUDA 12.8 build with external CUDA 13/cuDNN 9.20 dependencies for cuDNN;
  fused F32 vision/VAE attention, fused INT8-to-BF16 ConvRot output, decoder
  spatial tile batch four. Device budget 29 GiB; managed Host budget 12 GiB.
- Same immutable server/client binary in both runs: SHA256
  `5de9f936ccc79963ace28ccb714bb10ba4ef8caca06b82e8dd7f84a1c64edc9d`,
  source `fcaa65389499c94ac3d07ca574b80d0729660cf1`, with tile four selected.
  This binary contains an unused tile-eight option; that option and a later
  unsuccessful Hadamard-shuffle experiment were reverted. Current source
  `373911540` retains the measured tile-four path.
- Fresh server for each run; one seed-44 edit warmup, then C1 and C8. Loopback
  HTTP excludes external network transport. Run 2 follows run 1 without a WSL
  restart, so host file-cache state differs. No GPU polling or compilation ran
  concurrently with the benchmark. Output/latent capture was disabled.
- C1 uses the original 1024×1024 teapot PNG. C8 submits eight fixed, distinct
  reference PNGs, one per request; these are synthetic diagnostic inputs, not
  a representative user dataset. The prompt changes the background to light
  blue while preserving the subject, pose, details and composition.

The Rust `bench-images` client used `--concurrency 1` or `8`,
`--arrival-mode closed-loop --repetitions 1 --warmup-repetitions 0`,
`--timeout-seconds 600 --latency-slo-p99-ms 120000`, and
`--resource-process-scope process-tree --resource-sample-interval-ms 200`.
The service used `serve --image-config server.json --host 127.0.0.1 --port 18188`.
The [example configuration](../crates/ferrum-cli/examples/config/p-h3-image-cuda.json)
retains the measured execution settings with illustrative model/library paths;
it omits the diagnostic trace and prompt capture.

All 18 measured C1/C8 PNG and decoded RGBA hashes matched the paired 31 GB
baseline. Expanded prompts, token counts, reference latents, DiT rows and sigma
schedules also matched. This establishes consistency for these samples, not a
general image-quality score. The trace confirms real multi-request DiT execution;
run 2 contains seven B1, twenty-four B2 and three B3 waves, covering all 64 member
steps. Client concurrency eight does not mean all eight fit into one GPU forward.

## What memory changed

The adjacent 31 GB run took 121.9952 s at C8, with 26.13 GB of model-file reads
and P99 121.5567 s (seven of eight requests within 120 s). Raising the WSL memory
limit reduced observed C8 wall time by 2.35–2.95%. The required restart also reset
OS/driver/cache state, so the experiment does not isolate every saved second.
Run 2 has zero file rereads/refaults yet still spends 58.7126 s in DiT and
14.3739 s in decoding. These are host-wall stage intervals, not GPU-active timing.
System-memory relief does not establish a faster GPU kernel or a portable
speedup on another machine.

## Evidence identity

Raw images, logs, traces and benchmark reports are kept outside Git. The retained
bundles are `memory31-baseline-r1`, `memory48-candidate-r1` and
`memory48-candidate-r2` under the 20261109 image-throughput evidence collection.

| Artifact | SHA256 |
|---|---|
| 31 GB baseline archive | `b5f41a230732e73fcf41fb3d6156b18060a043d7f73afab48f6c2cb678a39321` |
| 48 GB run 1 archive | `b0d96e19992d8e9055e65154e8166f0c976db3393f69b578ac12f765468a6cac` |
| 48 GB run 2 archive | `f7ec36e9cc5dafc003f365160277dc239d75b20760b9050fef9af3e7eb92cb0d` |
| Run 1 paired audit | `e7107a313f91483a4817f61b9103fafcf3ce09fd0af56cf52fd052c77a2887c2` |
| Run 2 paired audit | `afc11a588fa42ad29d297abb6393bd3937086f1c589164f8acbb6c0545a5b0a9` |

No new same-session ComfyUI comparison was run for these two cells. Earlier
ComfyUI timings are deliberately not used to claim a current speedup ratio.
