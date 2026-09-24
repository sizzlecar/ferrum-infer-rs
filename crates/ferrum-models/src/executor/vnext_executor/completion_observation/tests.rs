//! Typed no-submit branches and the accumulator used by the actual native
//! branch. These do not claim to exercise a submitted checkpoint transfer.
use super::*;
#[path = "../../../../../ferrum-interfaces/tests/vnext_resource_contract/support.rs"]
mod resource_support;
use resource_support::TestRuntime;

#[test]
fn completion_probe_retention_skip_remains_composite_without_inventing_submission() {
    let probe = CompletionProbe::default();
    assert_eq!(probe.work(), ExecutorCompletionWork::NoAdditionalWork);
    probe.retention();
    probe.recovery();
    probe.checkpoint(&NativeCheckpointStart::<TestRuntime>::Skipped(
        CheckpointAccessSkipReason::Disabled,
    ));
    probe.checkpoint(&NativeCheckpointStart::<TestRuntime>::NotSubmitted(
        VNextError::InvalidExecutionPlan {
            reason: "actual pre-submission refusal".into(),
        },
    ));
    probe.maintenance();
    let ExecutorCompletionWork::AdditionalOrUnproven(work) = probe.work() else {
        panic!("entered retention cannot become no-work")
    };
    assert_eq!(work.checkpoint_submitted, 0);
    assert!(!work.submission_indeterminate);
    assert!(work.recovery_entered && work.maintenance_entered);
    assert!(!work.evidence_lost);
}

#[test]
fn completion_probe_accumulates_attempts_and_overflow_does_not_wrap_to_no_work() {
    let probe = CompletionProbe::default();
    probe.retention();
    probe.record_submission(false);
    probe.record_submission(true);
    // A later skip cannot replace the earlier possibly submitted evidence.
    probe.checkpoint(&NativeCheckpointStart::<TestRuntime>::Skipped(
        CheckpointAccessSkipReason::Busy,
    ));
    let ExecutorCompletionWork::AdditionalOrUnproven(work) = probe.work() else {
        panic!("retention")
    };
    assert_eq!(work.checkpoint_submitted, 2);
    assert!(work.submission_indeterminate);
    probe.submitted.store(u32::MAX, Ordering::Relaxed);
    probe.record_submission(false);
    let ExecutorCompletionWork::AdditionalOrUnproven(work) = probe.work() else {
        panic!("retention")
    };
    assert_eq!(work.checkpoint_submitted, u32::MAX);
    assert!(work.evidence_lost && work.submission_indeterminate);
}
