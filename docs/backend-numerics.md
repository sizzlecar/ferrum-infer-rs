# Required GPU numerical checks

The `GPU runtime` matrix in [CI](../.github/workflows/ci.yml) dispatches a Rust
checker to an Apple Silicon Metal worker and an NVIDIA CUDA worker. For code
changes, `CI required` requires both matrix legs to succeed along with the
existing hosted CPU, Metal and CUDA compilation/test jobs. Documentation-only
changes retain documentation checks. A GPU worker being offline, an execution
being skipped, or a timeout cannot become a passing code check.

## Current coverage

The checker exercises three existing `Backend` trait adapters against CPU F32
references. Metal uses F32 input/output buffers; CUDA uses F16 input/output
buffers and converts host readback to F32. The report identifies the selected
operation, shape and concrete entrypoint; these storage types do not describe
all model execution paths.

| Operation | Aligned sample | Tail sample | Metal / CUDA entrypoint |
| --- | --- | --- | --- |
| RMSNorm | tokens=4, dim=128, seed=42 | tokens=3, dim=33, seed=7 | `rms_norm_f32` / `rms_norm_f16` |
| GEMM, C=A×Bᵀ | m=64, n=32, k=32, seed=42 | m=64, n=33, k=35, seed=7 | `gemm_f32_v2` / `cublasGemmEx`, F32 accumulation |
| SiLU(gate)×up | tokens=4, intermediate=256, seed=42 | tokens=3, intermediate=257, seed=7 | `silu_mul_split_f32` / `fused_silu_mul_interleaved_f16` |

RMSNorm uses epsilon=1e-6. GEMM covers a full Metal 64×32 output tile with a
32-wide K tile, then a partial N tile and K reduction tile. CUDA uses cuBLAS
`CUBLAS_COMPUTE_32F_FAST_16F`; its internal tile choice is library-controlled.
A separately requested m=1 Metal sample selects `gemv_f32`, accurately recorded
instead of being labelled tiled GEMM. SiLU Mul covers complete and partial
256-thread launches with each token's `[gate | up]` row split. These samples
target concrete boundaries, not a model count or coverage-ratio requirement.
Zero dimensions, overflowing element/byte counts, and unsupported signed kernel
indices are rejected before allocation; valid arithmetic does not guarantee
that an arbitrary large requested shape fits device memory.

The checker requires nonempty outputs of the configured shape, finite values,
and NMSE strictly below the tolerance fixed before execution. Defaults reuse the
op-diff F32 threshold 1e-7 and F16 threshold 1e-6. For a near-zero reference mean
square (<1e-30), the existing comparator uses absolute MSE instead and records
that choice. Maximum absolute error is reported separately. Tolerances can be
set explicitly for a separately justified experiment, not raised after observing
a failure to make CI pass.

Metal checks command-buffer completion and driver errors before host readback;
CUDA propagates launch, synchronization and readback errors. Missing backend
features or devices produce `not_run` and a failing command. Unwind panics become
failed reports; process aborts, timeouts and missing reports remain job failures.

This lane validates only the three selected legacy backend operations and the
checker. It does not cover quantized Marlin, paged attention, production-plan
dispatch, full models, installation, performance or all shapes/precisions. The production `run` and `serve` interfaces
are unchanged by the validation CLI. Both still require their applicable protocol
and real-model checks. Unbound catalog obligations remain visible as gaps.

## Run locally

On a Metal host, using a new report path outside the repository:

```bash
cargo run --locked -p ferrum-testkit --example backend_numerics --features metal -- \
  --require-backend metal --op rms-norm --tokens 3 --dim 33 --seed 7 \
  --report /path/outside/repository/rms-norm-metal.json
```

For CUDA, use `--features cuda --require-backend cuda` on a configured CUDA host.
Building CUDA requires the pinned native operator-set lock, toolkit and runtime
libraries documented in [CI](../.github/workflows/ci.yml); compiling a CPU binary
and requesting CUDA deliberately fails. The existing RMSNorm command remains compatible. Select GEMM with
`--op gemm --m 64 --n 33 --k 35`, or SiLU Mul with
`--op silu-mul --tokens 3 --intermediate 257`, using a separate report path for
each execution. `--eps` belongs only to RMSNorm; irrelevant operator options
fail rather than silently changing coverage. `--seed` and `--max-nmse` are
shared; use `--help` for defaults.

Reports retain the fixture configuration, actual precision, elapsed execution
time, reference and GPU output as exact IEEE-754 bit arrays, numerical metrics
and failure reasons. CI uploads these even after a numerical failure. Existing
reports cannot be overwritten, and report creation/writing errors fail the
command. A serialized `passed` field is not release authorization: future release
validation must verify execution context and recompute assertions from raw data.

## Worker operation and cost

CPU and full-workspace Metal jobs use the existing GitHub-hosted runners. The
small numerical jobs use repository-scoped self-hosted labels `ferrum-metal` and
`ferrum-cuda` (currently CUDA sm89). Each worker accepts one job at a time; matrix
legs run independently and have a 30-minute job timeout. Reuse local Cargo
caches. This lane downloads no model weights and starts no paid cloud instance.
The additional samples contain only thousands of elements and reuse the same
compiled checker; they do not add another build or model download. Their actual
execution time is recorded in each report, not estimated as a fixed speedup.
Cold compilation, queueing and network transfers still count toward total CI
latency; a warm-cache target is not a measured percentile.

Repository-controlled PR branches and manual runs execute automatically.
External fork code is not dispatched to these personal workers by this workflow;
its missing GPU result blocks the aggregate check. Maintainers can review and
bring the change onto a repository branch for the normal complete CI. Because a
fork can also propose workflow changes, repository Actions settings require
approval for all external contributors before running their workflows. This
external-contribution boundary does not add approval to repository-owned PRs.
Use read-only job permissions and avoid persisting checkout credentials. These
controls do not turn a persistent self-hosted worker into an isolated sandbox;
see GitHub's [self-hosted runner guidance](https://docs.github.com/en/actions/concepts/runners/self-hosted-runners).

Keep network proxy addresses and credentials in each worker's local service
configuration, outside this repository. The CUDA runner uses the LAN Mac's Panda
Proxy for its Actions connections and job network traffic. Restarting a service
is necessary after changing its proxy environment. A Mac user LaunchAgent starts
on login; a WSL systemd service starts when that distribution starts. Registration
and an online status alone do not prove GPU execution or Windows-boot startup.
