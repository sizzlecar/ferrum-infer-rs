# Release candidate preparation

The Rust `release_candidate` example prepares coordinated workspace versions and
produces build-time metadata for staged assets. The staging workflows consume a
formal target version instead of embedding one release number in their build
logic. They upload candidate artifacts with read-only repository permissions.

This is one part of the [release regression policy](release-regression-policy.zh.md).
It does **not** automatically run the complete release regression plan, authorize
promotion, publish crates, create GitHub releases, or update Homebrew.

## Prepare a version PR

Use a dedicated checkout with no concurrent editor of its manifests or lockfile.
The example below uses `0.8.8` as an illustrative next version; choose the actual
release version before running it.

```sh
cargo run --locked -p ferrum-bench-core --example release_candidate -- \
  prepare --workspace . --version 0.8.8 --dry-run

cargo run --locked -p ferrum-bench-core --example release_candidate -- \
  prepare --workspace . --version 0.8.8
```

`prepare` discovers actual workspace members through Cargo metadata and builds a
text edit plan for:

- `workspace.package.version` and coordinated explicit member versions;
- internal path dependency versions, including renamed, inherited, build, dev,
  and target-specific dependencies;
- source-less workspace package records and local version-qualified references
  in `Cargo.lock`.

Inherited member versions remain inherited. TOML comments, unrelated dependencies
and historical text remain intact. A registry package with the same name is not
an internal path dependency. The target must be an increasing formal semantic
version without prerelease or build metadata. Supported internal constraints are
complete versions with optional `^`, `~`, or `=`; complex constraints, inconsistent
current versions and ambiguous lock references fail with an explicit error.

`--dry-run` prints the proposed changed file list without applying the edits.
It does not perform the post-edit dependency resolution. An actual application
runs full `cargo metadata --locked` afterward and checks every workspace member's
version; Cargo may need registry access for that resolution.

Review the resulting diff, run the applicable repository checks, and submit and
merge a separate version PR. The tool does not commit or create a PR. Once the
candidate source is fixed, create its annotated RC tag and dispatch staging with
the matching inputs. The binary and crate version is the formal version, such as
`0.8.8`; `v0.8.8-rc.1` identifies the candidate, not a prerelease Cargo version.

### Failure and rollback

Each file is replaced atomically. On an ordinary write or post-edit validation
error, `prepare` attempts to restore files written by that invocation. If a file
has changed again, it preserves that concurrent edit and reports the rollback
failure instead of overwriting it.

This is not a transaction across the filesystem. Process termination, a machine
crash, disk failure, or rollback failure can leave a partially updated checkout.
Inspect the reported paths and Git diff before continuing; preserve unrelated
work. There is no persistent recovery journal or automatic recovery after a
killed process. Use an isolated preparation checkout to keep this boundary small.

## Verify and stage a candidate

[CPU/Metal staging](../.github/workflows/release.yml),
[CUDA staging](../.github/workflows/release-cuda.yml), and
[Windows staging](../.github/workflows/release-windows.yml) require:

| Input | Meaning |
| --- | --- |
| `version` | Formal workspace and binary version |
| `release_candidate_sha` | Full lowercase commit object ID |
| `release_candidate_tag` | Annotated `vVERSION-rc.N` tag, with positive `N` |
| `staging_label` | Artifact label using letters, digits, `.`, `_` or `-` |
| `publish_release` | Must be `false`; staging cannot turn into publication |

The workflows check inputs before setup and check that the annotated tag peels
to the checked-out commit. The Rust verification step checks the candidate input
relationship and the versions of **all** Cargo workspace members:

```sh
cargo run --locked -p ferrum-bench-core --example release_candidate -- \
  verify --workspace . --version 0.8.8 \
  --candidate-sha <full-commit-id> --candidate-tag v0.8.8-rc.1 \
  --staging-label v0.8.8-rc.1
```

Replace `<full-commit-id>` with the real object ID. The standalone `verify`
command does not inspect Git tag provenance itself; the workflows perform that
separate check. Neither version consistency nor source identity establishes
model correctness.

## Generate adjacent asset metadata

`manifest` reads an existing archive, an existing binary, and raw dependency
inspection output. For example, after packaging a Linux CPU binary:

```sh
cargo run --locked -p ferrum-bench-core --example release_candidate -- \
  manifest --version 0.8.8 \
  --candidate-sha <full-commit-id> --candidate-tag v0.8.8-rc.1 \
  --staging-label v0.8.8-rc.1 \
  --backend cpu --target-triple x86_64-unknown-linux-gnu \
  --asset ferrum-linux-x86_64.tar.gz --binary target/release/ferrum \
  --dependencies ferrum-linux-x86_64.dependencies.txt --output-dir .
```

For Metal, supply `--backend metal`, a Darwin target and `--cargo-features metal`.
For CUDA, supply `--backend cuda`, its Linux target, actual cargo features,
`--cuda-compute-capability` and `--cuda-toolkit-image`. The workflow passes its
actual build settings. `--workflow-run-id` and `--workflow-run-attempt` record the
GitHub run when supplied; local invocations default to `local` and `1`.

The tool creates `.sha256`, `.binary.sha256`, `.version.json`, `.dependency.json`
and `.abi.json` records inside the chosen output directory. That directory must
already exist. Existing records are not overwritten. An ordinary output error
removes only newly reserved records from that invocation; a killed process can
leave partial records, so inspect them or use a fresh output directory.

The dependency audit parses raw Linux `ldd` or Darwin `otool -L` output, with an
optional `file` description. Empty or malformed audits, forbidden Python/Torch/
vLLM dynamic linkage, and unresolved build-host dependencies fail. CUDA's missing
`libcuda.so.1` is explicitly recorded as deferred to an actual runtime host;
this is not a successful CUDA execution check.

The supplied archive and binary are hashed independently. This command does not
verify that the archive contains that same binary, run the binary, or establish
its model behavior. Final asset acceptance must verify the extracted bytes and
execute the required checks using those bytes.

## What the workflow tests establish

Rust tests parse the real staging workflows and check effective workflow and job
permissions, including attempts to elevate package, repository or OIDC access.
They reject forwarded or inherited secrets and publication secret expressions;
the automatic read-only GitHub token remains usable. Negative YAML fixtures
exercise these failures rather than counting build commands.

The tests also execute each job's actual first input guard, using the workflow's
real environment bindings. They cover valid inputs, mismatched versions/tags,
publication requests and injection strings. These checks establish the tested
input and declared credential boundaries. They are not a general proof that any
future shell script or third-party action can never publish.

## Publish a formal release

Start [Prepare formal release](../.github/workflows/prepare-release.yml) on `main`
with a higher formal version. It opens a version PR and enables merge after CI.
Merging starts [release delivery](../.github/workflows/release-delivery.yml):
candidate builds, selected model checks, Cargo/GitHub/Homebrew publication and
actual public installation checks. Require the final `complete` job to succeed.

CPU, Metal, Linux CUDA and Windows staging run independently. CPU and Metal
model checks use their existing native hosts; CUDA model checks use the bounded
lease below. Each backend prepares its selected tasks after its own artifacts
and quality checks are ready, without waiting for Windows packaging. Linux
delivery tools are built once and shared by artifact ID; Metal consumes its
staged native tools. Publication still requires every staging job, the complete
unfiltered task plan, and all selected model reports.

Model jobs print profile counts and case start/completion events. A 30-second
heartbeat indicates the controller is still waiting; it does not imply a passed
check. Full diagnostics remain in evidence artifacts, and deadlines and process
cleanup still apply. CPU requests have a 900-second allowance within the existing
one-hour task deadline.

Windows native object caching is separate from Rust dependency/build caching.
Successful native builds save their cache before application compilation, and
the job summary records actual object hits, compiled units and build time per
operator. Compiler and dependency validation still decide whether an object can
be reused. Windows CPU and CUDA packages are built independently by
[Windows staging](../.github/workflows/release-windows.yml). CUDA uses the dedicated
Windows runner with preinstalled MSVC, CUDA and Inno Setup; Cargo outputs and
native objects persist there. CPU uses a separate hosted Windows runner, so its
package/startup checks do not wait for CUDA compilation or require a GPU driver.
Both packages reuse the same released launcher. The publisher checks each
backend's actual successful staging attempt independently, including retries.
Packaging helpers use debug builds without debug information; distributed Ferrum
executables retain the release profile.

The native runner label is `ferrum-windows-native`. Configure its service with
`FERRUM_WINDOWS_VS_ROOT`, `FERRUM_WINDOWS_CUDA_ROOT`,
`FERRUM_WINDOWS_INNO_ROOT`, and `FERRUM_WINDOWS_OBJECT_CACHE`, plus persistent
Cargo/Rustup homes and PowerShell 7 on PATH. CUDA 12.4.1 and Inno 6.7.3 remain
required; runtime/license hashes and compiler provenance are still verified.
Trusted Windows CI jobs use this runner; fork PRs use hosted runners. These
changes remove repeated setup and serialization; no measured cold-build speedup
is claimed yet.

Configure these repository Actions secrets before starting:

- `RELEASE_GITHUB_TOKEN`: repository/workflow and tap write access; must trigger PR CI.
- `CARGO_TOKEN`: crates.io publishing credential.
- `VAST_API_KEY`: Vast account credential for GPU rental and cleanup.

Use **Re-run failed jobs** to keep successful platform builds and model results.
A failed Windows CUDA staging job does not invalidate Windows CPU or Metal
artifacts; each package is bound to its own successful execution. Avoid
**Re-run all jobs** for an isolated failure.

After a publication or installation failure, keep the original run and its
evidence artifacts;
missing or failed evidence blocks publication. Do not restart the whole workflow
solely to retry an upload.

During development, use the existing CPU, Metal and CUDA hosts for affected
compilation, unit tests and model checks. Freeze the release source, version,
binaries and model tasks before renting a GPU for release acceptance. Publish
the accepted binaries without rebuilding them. Development results remain useful
regression evidence, but do not certify a different release binary. A manual
delivery run can use `reuse_cuda_run_id` to reverify a previous delivery run's
CUDA results against the current binary, runner and tasks; a verified match
skips the lease. Investigate any mismatch before restarting delivery instead of
repeatedly renting machines while implementation is still changing.

Current rental limits: one 48 GB sm89 GPU, 300 GiB disk, $0.75/hour maximum,
$0.004 per transfer GB, three hours per lease, 15 minutes to bootstrap and one
hour per model task. Storage and transfer still contribute to the bill.
[Expired-lease cleanup](../.github/workflows/release-cloud-reaper.yml) is scheduled,
but delays or provider failures prevent a hard billing cap. Check cleanup results
and the Vast account after a failed run.
