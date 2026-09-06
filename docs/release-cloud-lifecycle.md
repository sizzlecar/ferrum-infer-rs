# Bounded release GPU execution and cleanup

The Rust `release_delivery cloud` command runs the CUDA entries from existing
`model_gate` tasks on one Vast instance. It preserves their model, execution
target, checks and public options; other backend tasks and plan gaps remain
outstanding. It uses the existing model runner and report verifier. A successful
cloud batch does not authorize publication or replace the final release gate.

The caller supplies the archive, extracted Ferrum binary, Linux model runner,
trusted expected digests and a fresh report directory. The cloud command hashes
these local inputs, binds Ferrum to the prepared task digest, and checks uploaded
binary/runner hashes before loading models. `release_candidate verify` checks
candidate/version inputs; it does **not** check archive hashes or archive contents.
`release_candidate manifest` hashes an archive and a binary independently. The
caller must additionally verify that the accepted archive contains the executed
binary. These identities bind evidence to bytes; the actual runner owns behavior
assertions, observed-backend checks and terminal report validation.

Resource policy requires explicit storage, minimum free disk/host RAM, hourly and
network price ceilings, bootstrap/task/total execution deadlines and a create
attempt limit of one. The first implementation uses a pinned official Vast
CUDA12.4.1/Ubuntu22 amd64 image and a verified, on-demand, native 48GB sm89 GPU.
Remote device, free capacity, runtime-library and version checks precede inference.
A quote or declared architecture does not prove a model fits; actual load and
semantic failures remain failures, with no silent model or GPU-class substitution.

The controller records its unique create intent before allocation. An ambiguous
create response is reconciled by the exact ownership label; allocation is never
blindly retried. A temporary SSH public key is attached to that instance; the
private key and Vast API key stay on the controller. Each task's raw report is
collected, including after runner failure. Missing, incomplete or failed reports
and transfer failures cannot satisfy the task. Execution stops after a failure.

Normal success, error, timeout, SIGTERM and ctrl-C paths attempt destruction and
verify the instance is absent from the paginated account list. A lost controller
can interrupt these steps, so the independent
[reaper workflow](../.github/workflows/release-cloud-reaper.yml) runs from trusted
`main` on a standard CPU runner, using Rust 1.91 and only the small controller.
It cannot allocate GPU instances. The Vast secret is scoped to its cleanup step;
repository permissions are read-only. Manual dispatch optionally restricts one
release run; a different ref or fork does not run the job.

The reaper accepts only the exact `ferrum-rel:1` label format, matching repository
ID, optional run ID and expired Unix deadline, and rechecks unchanged ownership
before deletion. Malformed, foreign, unexpired or relabeled instances remain
untouched. DELETE acknowledgement alone is insufficient: absence must be observed.
Report artifacts exclude `private/**`; normal execution removes the private key,
but hard termination can happen before that removal. Uploads must keep this
exclusion when the parent release workflow collects cloud execution reports too.

The schedule is every 15 minutes away from the top of the hour. GitHub documents
[delayed/dropped schedules and inactivity-based disabling](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).
Cleanup can also fail during provider/network outages;
this is **not** a hard TTL or an absolute invoice cap. Polling/build/cleanup time,
network bytes and provider billing still matter. Keep the independent schedule
enabled, inspect failed cleanup reports, and retain the label/instance IDs for
recovery. Setup or cleanup errors fail the workflow; it does not rerun paid work.

See [candidate preparation](release-candidate-automation.md) and the
[release regression policy](release-regression-policy.zh.md) for the separate
source, artifact and product validation obligations.
