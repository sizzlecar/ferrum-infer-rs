# Automated release delivery

The [delivery workflow](../.github/workflows/release-delivery.yml) connects the
[regression plan](release-regression-policy.zh.md), ordinary required CI,
[staged distributions](release-candidate-automation.md), real model tasks, and
publication. A workflow definition or a green planning step is not evidence that
a formal release has completed. The final job requires publication and actual
public Cargo/Homebrew installation checks to succeed.

## Development and release flow

1. Each PR runs workspace checks and registered correctness contracts. The
   contract executor reads Cargo's actual test artifacts, requires each named
   assertion to be registered, and executes it with `--exact --include-ignored`.
   A renamed/deleted test, missing harness, assertion failure or timeout fails.
   The [GPU lane](backend-numerics.md) separately requires real backend outputs;
   compilation and a CPU substitute cannot satisfy it.
2. Start **Prepare formal release** with a formal version greater than the
   current workspace version. Equal or older versions are rejected. The Rust
   preparer updates workspace/member/internal dependency versions and the lock
   file together. It opens a version PR and enables automatic merge after
   required CI. An existing branch is reused only when its proposed tree agrees;
   conflicting edits are not replaced.
3. A coordinated version merged into `main` starts **Deliver formal release**.
   It compares against the latest reachable formal tag, saves the plan and notes,
   and refuses missing coverage/checker bindings before building release assets
   or allocating paid hardware. Manual dispatch supports an explicit rerun.
4. Required CI and CPU/Metal/CUDA candidate builds run in parallel. The staging
   workflows produce immutable artifact IDs and separate model-runner archives.
   The controller checks the archive's actual embedded binary before execution.
5. Selected Metal tasks run sequentially on the local Mac. Selected CUDA tasks
   run sequentially on one [Vast lease](release-cloud-lifecycle.md). Checks for a
   profile share its existing server; Quick Start retains its actual advertised
   model, default backend selection and public options. Unsupported checks fail
   before renting; no cheaper replacement silently satisfies a required model.
6. The release gate verifies the original plan assignments, actual model reports,
   executed contract report, archive/startup results and the effective successful
   `Quality / CI required` job for that candidate. A numerical or performance
   obligation needs its own bound evidence; a model response cannot substitute.
7. The publisher uses Cargo's native multi-package publication order, reconciles
   every registry version and checksum, resumes missing release assets and
   updates both Homebrew formulas in one commit. Existing conflicting versions,
   tags, assets or formulas stop publication.
8. Separate jobs actually install the public Cargo package and install/upgrade
   the public Metal and CUDA Homebrew formulas. Their installed programs execute
   `--version`, `--help`, `run --help`, and `serve --help`; version, exit status,
   output and binary stability are checked. Only then does **complete** succeed.
   These startup checks do not claim another full-model regression.

The controller is Rust, in
[`release_delivery`](../crates/ferrum-bench-core/examples/release_delivery.rs).
`inspect --extract-only` explicitly records `not_run`. For a CUDA archive checked
on a CPU host, the final gate additionally requires successful `run` and `serve`
from verified real-model reports using that exact embedded binary. Extracting a
file or reading its metadata alone cannot authorize its distribution.

## Resources and credentials

Trusted controller jobs use repository Actions secrets:

- `RELEASE_GITHUB_TOKEN`: repository/workflow access and write access to the
  Homebrew tap. PR events must start ordinary CI; the default workflow token's
  event suppression is unsuitable for preparing the version PR.
- `CARGO_TOKEN`: crates.io publisher credential, exposed only to publication.
- `VAST_API_KEY`: exposed only to the cloud controller and expiry reaper.

Build jobs and model workers receive no publication credentials. Cloud SSH uses
a per-run key. Private key directories are excluded from artifacts, including
failure paths. Local workers retain their normal network/proxy configuration;
the 4050 runner uses the LAN proxy hosted on the Mac.

The current explicit cloud policy is one native-compatible 48 GB sm89 device,
300 GiB disk, at least 180 GiB measured free disk and 64,000 MB CPU RAM, at most
$0.75/hour and $0.004 per provider transfer GB. The controller allows one create
request, 15 minutes to bootstrap, a three-hour lease deadline and a one-hour
per-model task deadline. These are operator limits, not a provider invoice cap;
network/storage charges and delayed scheduled cleanup are described in the
[lease lifecycle](release-cloud-lifecycle.md). A future resource change must
preserve the selected model/precision/backend requirements.

## Failure and retry

Use **Re-run failed jobs** when successful model work remains valid. Evidence
artifact names are stable within the workflow run, so a failed publication or
public installation does not require rerunning successful model jobs. The gate
accepts the latest effective Quality result; an earlier success cannot hide a
newer failure or a still-running replacement.

Package checkpoints save verified crate archives and candidate input metadata.
They are cached and uploaded outside the source repository. Checkpoints are a
work-saving mechanism: the publisher still reads the actual registry and verifies
all immutable version/checksum pairs. Unknown network responses are not treated
as missing packages. Cargo can finish with the final index visibility wait still
incomplete, so the controller checks every published version independently.

Public channels cannot be committed atomically. If installation fails after
publication, the overall workflow stays failed and retains the published state;
it does not pretend to undo a public crate version. Fix the installation fault
and rerun the failed jobs. Missing/mutated evidence requires rebuilding or
re-executing the affected checks; it cannot be replaced by a saved `passed` flag.

## Coverage boundaries

The regression catalog is a product inventory, not a promise to execute every
model/backend combination for every edit. Scope refinement can prove test-only
AST changes and controlled development/release-tool dependency changes; unknown
runtime dependencies, production code, and product promises remain visible.
Scheduling/cancellation/capacity contracts use deterministic fixtures, while
actual model execution checks integration on selected architecture/protocol and
backend representatives.

Required GPU coverage includes RMSNorm, GEMM and SiLU Mul, each with aligned and
tail cases on Metal and CUDA. Current storage and operation limits are recorded
in [backend numerics](backend-numerics.md). Quantized matmul, attention and KV
operations remain explicit expansion work, selected by usage and changed code.
Do not report the entire GPU backend as validated merely because one operator
passed. Performance and device state require separate assertions and evidence.
