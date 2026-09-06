# Required GPU numerical checks

The `GPU runtime` matrix in [CI](../.github/workflows/ci.yml) dispatches a Rust
checker to an Apple Silicon Metal worker and an NVIDIA CUDA worker. For code
changes, `CI required` requires both matrix legs to succeed along with the
existing hosted CPU, Metal and CUDA compilation/test jobs. Documentation-only
changes retain documentation checks. A GPU worker being offline, an execution
being skipped, or a timeout cannot become a passing code check.

## Current coverage

The first adapter is `RmsNormOp`: CPU F32 reference, Metal F32 buffers, and CUDA
F16 buffers with host readback as F32. CI exercises an aligned dimension
(tokens=4, dim=128, seed=42) and a partial-warp/SIMD tail
(tokens=3, dim=33, seed=7), with epsilon=1e-6. These shapes target different
kernel boundaries; they are not a model count or coverage-ratio requirement.

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

This lane validates the legacy backend RMSNorm operation and the checker. It
makes no claim for other operators, vNext execution, full models, installation,
performance or all shapes/precisions. The production `run` and `serve` interfaces
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
and requesting CUDA deliberately fails. CLI options include `--eps` and
`--max-nmse`; use `--help` for defaults.

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
