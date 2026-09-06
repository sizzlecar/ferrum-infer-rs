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

Both [CPU/Metal staging](../.github/workflows/release.yml) and
[CUDA staging](../.github/workflows/release-cuda.yml) require:

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

Rust tests parse both real staging workflows and check effective workflow and job
permissions, including attempts to elevate package, repository or OIDC access.
They reject forwarded or inherited secrets and publication secret expressions;
the automatic read-only GitHub token remains usable. Negative YAML fixtures
exercise these failures rather than counting build commands.

The tests also execute each job's actual first input guard, using the workflow's
real environment bindings. They cover valid inputs, mismatched versions/tags,
publication requests and injection strings. These checks establish the tested
input and declared credential boundaries. They are not a general proof that any
future shell script or third-party action can never publish.

## Remaining execution and promotion work

The [model runner](release-regression.md#rust-model-runner),
[regression catalog](release-regression-catalog.md) and
[model-task gate](release-regression.md#verify-selected-model-tasks) now connect
a limited ModelRuntime scope. `model_gate prepare` reads a generated plan plus
adjacent `.abi.json`/`.version.json` staging records to fix expected tasks before
execution. The runner checks the actual staged binary digest/version, task
options and observed backend, executes its semantic assertions, and produces a
schema-2 terminal report. `model_gate verify` rejects missing, failed, duplicate,
unfinished or mismatched model reports and retains remaining plan gaps.

Quick Start uses the normal alias, automatic backend and disabled thinking with
controlled prompts, without capacity/template overrides or another model load.
Other checks for the same profile share its server instance. This validates the
implemented model-task obligations; it does not verify archive contents,
installation, numerical references, every declared execution path, or complete
release correctness. The gate reports `release_approved: false` even on success.

Automatic local/cloud capacity selection, full-plan execution, raw report replay,
and the complete evidence gate immediately before publishing remain pending. Cloud
provider/runner selection and its integration also remain pending; the catalog's
resource discussion is background, not a configured paid execution service.

Crate dependency-order publication, resumable GitHub asset upload, formal release
creation and Homebrew updates still require a later delivery implementation.
That implementation must reject missing, failed, not-run or inconclusive
required evidence and promote the same accepted asset bytes. This preparation
command's success cannot substitute for those checks.
