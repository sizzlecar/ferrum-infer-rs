# Release regression

This is a validation procedure, not a record of completed checks. Use the current
[English](../README.md) and [Chinese](../README_zh.md) README as the product contract.
Every Quick Start model must run on its advertised backend before release.
Sample performance-snapshot models by architecture, quantization and affected
execution path; an exhaustive model-by-backend matrix is not required.

## Prepare the candidate

Complete the workspace and relevant backend checks in [AGENTS.md](../AGENTS.md).
Build the release assets, then run model regressions against the extracted,
staged `ferrum` binaries that will be published. Building the regression runner
does not replace or rebuild the binary selected with `--ferrum-bin`.

Record the binary version and checksum, model repository/revision and weight
checksum, tokenizer/config sources, precision, backend, hardware and commands.
Keep reports, raw responses, logs and weights outside the repository. Reuse Cargo
and model caches; verify cached content against the selected upstream revision.
A cached weight must not hide stale tokenizer or chat-template sidecars. Record
which files were downloaded and which were reused; do not call reuse a fresh
download test.

Check binary architecture, accelerator features, driver/runtime dependencies,
available memory and storage before downloading large weights. CUDA assets target
sm89 and require compatible NVIDIA, CUDA and NCCL runtimes. Start paid hardware
when the candidate and test inputs are ready. Export evidence before releasing
temporary instances; retain paid storage only when its reuse justifies the cost.

## Mandatory Quick Start

The currently advertised paths are:

| Platform | MODEL | Source/format |
|---|---|---|
| Apple Silicon Metal | `qwen3.5:4b-q4_k_m` | `unsloth/Qwen3.5-4B-GGUF`, Q4_K_M; metadata from `Qwen/Qwen3.5-4B` |
| Linux CUDA | `qwen3.5:4b` | `Qwen/Qwen3.5-4B`, official safetensors |

Run the README's `--version`, `--help`, `doctor`, interactive `run`, and `serve`
commands, including `--disable-thinking` and `--served-model-name ferrum` where
shown. Submit a real prompt and the documented HTTP request; check meaningful,
non-empty output. Exercise both non-streaming and streaming responses.

Preserve the default context, memory and concurrency settings for these first-use
checks. A smaller-model substitution or capacity override is diagnostic evidence,
not completion of the advertised Quick Start. Capture startup/download failures,
timeouts, output errors and resource limits instead of silently changing flags.

## Rust model runner

Use the [model_regression example](../crates/ferrum-cli/examples/model_regression.rs)
for repeatable checks against an explicit binary. For example, on Metal:

```bash
cargo build --release --locked -p ferrum-cli --example model_regression
./target/release/examples/model_regression \
  --ferrum-bin /path/to/staged/ferrum \
  --model qwen3.5:4b-q4_k_m --backend metal \
  --report-dir /path/outside/repository/metal-quickstart \
  --checks basic,stop,structured,tools --disable-thinking
```

Use `--model qwen3.5:4b --backend cuda` for the CUDA Quick Start and a separate
report directory. Reports must use a new or empty directory. `--checks` defaults
to `basic`; explicitly select the checks relevant to the release:

- `basic`: two `run` turns; server non-streaming, streaming and history replay.
- `stop`: termination behavior through both entrypoints, including stream text.
- `structured`: server JSON/schema behavior and valid structured responses.
- `tools`: server tool calls and a tool-result continuation.

For reasoning-history compatibility, add `--reasoning-alias-replay` to a run
including `--checks tools`. Preserve the model's reasoning defaults for that run
by omitting `--disable-thinking`. It must replay actual, non-empty reasoning;
an empty reasoning field or a fabricated trace does not cover the behavior.
Choose a capable model and report a failure to produce that trace explicitly.

The runner complements separate literal README commands and installation checks.
It records an explicit backend, temperature 0, seed 7 and an output budget
(`--max-tokens`, default 512), without changing context, memory or concurrency.
It does not prove download behavior, general answer quality or throughput.

## Representative coverage and cost

Use the Quick Start dense hybrid model as one representative. Add cached MoE
models when scheduling, batching or resource ownership changed; choose an
attention-only or recurrent MoE path according to the affected code. Include
concurrent generation, late prefill, cancellation and a subsequent clean request
when those behaviors changed, using focused Rust tests or bounded model runs.

GPT-OSS is a useful separate representative for Harmony, reasoning channels and
tool handoff. Gemma is useful for its distinct template and compressed-tensors
path. Neither must be added merely to increase a model count. Sample AWQ, GPTQ,
MXFP4 or block-FP8 when the relevant loader/kernel changed; several related FP8
checkpoints do not all need to be downloaded for an unrelated protocol fix.

Prefer existing verified weights and one sufficiently sized GPU used sequentially.
For sharded models, fetch the files referenced by the selected index and required
sidecars; avoid duplicate original-format copies. Record the representatives,
omitted groups and reasoning so the limits of coverage remain visible.

## Interpret results and publish

Compilation proves code builds. Fixture tests prove the exercised protocol and
boundary behavior. Real-model runs additionally test loading, backend execution
and the specific semantic assertions used; HTTP 200 alone is not answer validity.
Keep failures and unexecuted checks visible. Do not substitute repetition totals,
PASS ratios or machine/commit identifiers for assertions about actual behavior.

Performance claims require same-hardware comparisons with recorded workloads,
precision, input/output lengths, concurrency, repetitions and uncertainty. Count
valid outputs and errors alongside timing; SSE text events are not usage tokens.
Correctness on a different GPU does not reproduce a README throughput number.

Resolve failures in mandatory Quick Start paths and selected required regressions
before publishing. Publish the validated staged bytes and verify the public
tarballs and checksums.
Check the advertised Homebrew formulas and crates.io installation command as well
as direct downloads: installed `--version`, `--help` and `doctor` must work with
their stated dependencies. Preserve the mandatory model-runtime evidence for
each backend, and disclose any installation or runtime check still incomplete.
