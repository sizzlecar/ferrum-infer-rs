# Ferrum v0.8.0 performance report

## Evidence boundary

The table below is the final R2 development checkpoint used to freeze the
v0.8.0 performance floors. It is **not** relabeled as a measurement of the
staged or published v0.8.0 binaries. After the release source is merged, the
production workflows build the exact Metal/CUDA release binaries once and a
bounded R3 exact staged-binary sample checks all release models and both product
entrypoints.
The final sample is a regression check, not a second full six-lane sweep.
Published performance is accepted only when staged and published tarball and
inner binary SHA256 values are identical.

R2 canonical evidence:

- Gate: `FERRUM GATE vnext-r2 PASS`
- Artifact: `/Users/chejinxuan/ferrum-artifacts/runtime-vnext-r2-unified-d324d509-20260814-r3`
- Gate manifest SHA256: `5f571d4af001d5ae758821235e5a40b5539db971641ac619443cdecd0174d799`
- Aggregate source: `d324d50911d8b2eb12a8889b889570d69b3f69c0`, clean
- Frozen 132-row floor file SHA256: `2a31bdc0845799b30e1b9d0f209bab10d01c4fd17bd87e1815f7c7911a53ac5b`

## R2 development measurements

### `ferrum serve` concurrency

`output tok/s` is aggregate usage-counted output throughput from 100 measured
requests per repeat, three repeats, with zero request errors. The table uses each
lane's highest release active-concurrency cell; Metal and CUDA use different
input lengths and are not presented as a same-workload hardware comparison.

| Model | Backend / hardware | Random input / output | Active concurrency | Output tok/s mean ± 95% CI half-width |
|---|---|---:|---:|---:|
| M1 Qwen3.5-4B | Metal / M1 Max 32 GiB, 24 GPU cores | 64 / 128 | 16 | 61.870 ± 0.107 |
| M1 Qwen3.5-4B | CUDA / 1x RTX 4090 24,564 MiB | 256 / 128 | 32 | 241.264 ± 0.615 |
| M2 Qwen3.5-35B-A3B | Metal / M1 Max 32 GiB, 24 GPU cores | 64 / 128 | 4 | 26.051 ± 0.234 |
| M2 Qwen3.5-35B-A3B | CUDA / 1x RTX 4090 24,564 MiB | 256 / 128 | 16 | 174.149 ± 0.963 |
| M3 Qwen3-30B-A3B | Metal / M1 Max 32 GiB, 24 GPU cores | 64 / 128 | 16 | 39.633 ± 1.150 |
| M3 Qwen3-30B-A3B | CUDA / 1x RTX 4090 24,564 MiB | 256 / 128 | 32 | 214.854 ± 2.723 |

### `ferrum run` parity

`run steady decode` is the median of three independent `ferrum run` processes
using the same fixed prompt, tokenizer, 128-token cap, thinking disabled, and
explicit EOS policy as each lane's serve-c1 parity probe.

| Model | Backend / hardware | Source SHA | Binary SHA256 | run steady decode median | Minimum physical headroom | Max HTTP-cell CV |
|---|---|---|---|---:|---:|---:|
| M1 Qwen3.5-4B | CUDA / 1x RTX 4090 24,564 MiB | `2d23deb302f2f50b7cce4174cea0cddc9bc4bc15` | `cd60d2a93f5f0d53980fceff89948f8e77163a6245ffceb901fec33cc96e81ae` | 83.611 tok/s | 11.71 GiB | 0.386% |
| M1 Qwen3.5-4B | Metal / M1 Max 32 GiB, 24 GPU cores | `2d23deb302f2f50b7cce4174cea0cddc9bc4bc15` | `0c80d38bd53909178d048ef2b72ace5e367ed23ad0dbde931c319ac9fbbf4d04` | 32.661 tok/s | 10.94 GiB | 0.417% |
| M2 Qwen3.5-35B-A3B | CUDA / 1x RTX 4090 24,564 MiB | `2d23deb302f2f50b7cce4174cea0cddc9bc4bc15` | `cd60d2a93f5f0d53980fceff89948f8e77163a6245ffceb901fec33cc96e81ae` | 94.230 tok/s | 1.58 GiB | 0.868% |
| M2 Qwen3.5-35B-A3B | Metal / M1 Max 32 GiB, 24 GPU cores | `00d6e34eb0daa4c801c89fe0bfaa794f03e05169` | `55cfc2df88b978e1af6bdc17c15c7943bcb38c66fd356f4e6634caed31958b90` | 20.534 tok/s | 2.74 GiB | 0.855% |
| M3 Qwen3-30B-A3B | CUDA / 1x RTX 4090 24,564 MiB | `2d23deb302f2f50b7cce4174cea0cddc9bc4bc15` | `cd60d2a93f5f0d53980fceff89948f8e77163a6245ffceb901fec33cc96e81ae` | 105.113 tok/s | 5.30 GiB | 0.417% |
| M3 Qwen3-30B-A3B | Metal / M1 Max 32 GiB, 24 GPU cores | `00d6e34eb0daa4c801c89fe0bfaa794f03e05169` | `55cfc2df88b978e1af6bdc17c15c7943bcb38c66fd356f4e6634caed31958b90` | 17.608 tok/s | 2.08 GiB | 2.425% |

The R2 aggregate validated 33 required HTTP cells, 99 repeat reports, 7,380
measured requests, 18 independent run samples, six build scenarios, zero
request errors, zero waivers, and zero external-comparator rows. Maximum basic
profile overhead was 0.0261%. These counts are derived from the raw lane
artifacts rather than copied summary claims.

## Canonical measurement commands

Each development lane was collected by the same product-only collector:

```text
python3 scripts/release/runtime_vnext_r2_ferrum_collector.py \
  --artifact-root <lane-root> --config <typed-lane-config> --resume
```

The collector launches `ferrum serve` with the locked model, backend, typed
admission/memory policy, scheduler trace, and effective-config output, then uses:

```text
ferrum bench-serve --base-url <url> --model <served-model> \
  --tokenizer <locked-tokenizer> --target-backend <metal|cuda> \
  --http-connection-mode fresh --dataset <random|sharegpt> \
  --concurrency <cell> --num-prompts <100|30> --warmup-requests 10 \
  --n-repeats 3 --seed 9271 --output json --fail-on-error --require-ci \
  --enable-thinking false
```

The exact expanded argv, model/file hashes, dataset hashes, effective config,
resource samples, hardware fingerprint, driver/toolchain observations, raw
repeat reports, and process receipts remain in each manifest-bound artifact.

## Final staged regression and publication rule

No number in this document may be relabeled as staged or published evidence.
The final report is an external immutable artifact generated after the merged
release source is tagged and the production workflow stages CPU/Metal/CUDA
tarballs. It records the unique RC SHA/tree/tag, staged tarball and inner binary
SHA256, selected sample cells, raw commands and repeats, both `ferrum run` and
`ferrum serve`, and the final exact PASS lines.

The approved final sample covers M1/M2/M3 on Metal and CUDA, plus the required
Llama 8B-class dense control. It must include correctness for both product
entrypoints and representative performance on each accelerator backend. It
sets `full_matrix_claim=false`; cells not selected by the frozen sample plan are
reported as not evaluated, never inferred from the R2 rows. If product source
changes after the R2 checkpoint, or published bytes differ from staged bytes,
the source closure/sample evidence is stale and publication stops.

The sample selection is frozen before the RC tag:

| Model | Metal performance sample | CUDA performance sample | Correctness sample |
|---|---|---|---|
| M1 Qwen3.5-4B | `random:c16` | `random:c32` | focused C17 with real `run` and `serve` |
| M2 Qwen3.5-35B-A3B | `random:c4` | `random:c16` | focused C17 with real `run` and `serve` |
| M3 Qwen3-30B-A3B | `random:c16` | `random:c32` | focused C17 with real `run` and `serve` |
| Llama 3.1 8B dense | `random:c1` | `random:c1` | dense supplemental `run` and `serve` |

Every HTTP performance sample uses three measured repeats, usage-derived token
counts, `--fail-on-error`, and its backend/model run-parity probe. Selecting a
different cell after seeing a result is forbidden; a failed selected sample
stops publication and remains a saved REJECT artifact.
