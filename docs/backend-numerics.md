# Required GPU numerical checks

The `GPU runtime` jobs in [CI](../.github/workflows/ci.yml) execute on Metal and
CUDA workers. Code PRs require both to succeed. Missing devices, skipped checks,
timeouts and numerical failures cannot pass the required CI check.

## Current coverage

CPU F32 is the reference. Metal uses F32 input/output buffers; CUDA uses F16
buffers with host readback as F32. Each report records the actual entrypoint.

| Operation | Aligned sample | Tail sample |
| --- | --- | --- |
| RMSNorm | tokens=4, dim=128, seed=42 | tokens=3, dim=33, seed=7 |
| GEMM, C=A×Bᵀ | m=64, n=32, k=32, seed=42 | m=64, n=33, k=35, seed=7 |
| SiLU(gate)×up | tokens=4, intermediate=256, seed=42 | tokens=3, intermediate=257, seed=7 |

RMSNorm uses epsilon=1e-6. These samples check the three legacy backend
operations; they do not establish coverage of quantized Marlin, paged attention,
production-plan dispatch, full models, performance or every shape/precision.

Outputs must have the configured shape, contain only finite values, and have
NMSE below 1e-7 for Metal or 1e-6 for CUDA. Near-zero references use absolute
MSE, recorded in the report. Set any alternative tolerance before execution
with a stated reason; do not raise it merely to pass a failed check.

## Run locally

Use a new report path in an existing directory outside the repository:

```bash
cargo run --locked -p ferrum-testkit --example backend_numerics --features metal -- \
  --require-backend metal --op rms-norm --tokens 3 --dim 33 --seed 7 \
  --report /path/outside/repository/rms-norm-metal.json
```

For CUDA, change both `--features metal` and `--require-backend metal` to `cuda`.
Use the toolkit, runtime libraries and pinned native operator-set lock specified
in [CI](../.github/workflows/ci.yml). A binary compiled without the requested
backend exits unsuccessfully with `not_run`.

Select the other operations with these options and distinct report paths:

- GEMM: `--op gemm --m 64 --n 33 --k 35`.
- SiLU Mul: `--op silu-mul --tokens 3 --intermediate 257`.

`--eps` applies only to RMSNorm; `--seed` and `--max-nmse` apply to all operations.
Use `--help` for defaults. Invalid dimensions and overflowing sizes are rejected
before allocation; an arbitrary large shape can still exceed device memory.

Reports retain configuration, precision, reference/GPU outputs, error metrics,
elapsed execution time and failure reasons. CI uploads them even after failure.
Existing reports cannot be overwritten. A successful numerical sample does not
replace the required protocol, model or installation checks.

## Workers and cost

The repository's self-hosted `ferrum-metal` and `ferrum-cuda` workers run these
small checks with a 30-minute job timeout. Reuse Cargo caches; no model weights
or paid cloud instances are needed. Offline workers leave code CI incomplete.

Repository-owned PRs run automatically. External fork code is not dispatched to
these personal workers; reviewed changes must reach a repository branch before
complete CI can run. Keep checkout credentials and publication secrets off these
workers, following GitHub's [self-hosted runner guidance](https://docs.github.com/en/actions/concepts/runners/self-hosted-runners).

Keep proxy settings and credentials in local runner service configuration.
The CUDA worker uses the LAN Mac's Panda Proxy. A Mac LaunchAgent starts on user
login; the WSL systemd service starts with its distribution. Windows-boot startup
is a separate operational prerequisite.
