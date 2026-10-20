//! Protocol checks only: these reports do not execute a model or reserve KV.

use super::*;
use crate::model_executor::{ExecutorExecutionCapacityStage, KvSlotCapacitySnapshot};

fn capacity() -> ExecutorExecutionDeferral {
    TokenAdmissionCapacityDeferral::new(
        AllocatorCapacityDomain::new(),
        2,
        KvSlotCapacitySnapshot {
            block_size: 16,
            total_blocks: 4,
            free_blocks: 1,
        },
    )
    .unwrap()
    .into()
}

fn frontier(issuer: &ExecutorSequenceIssuer, position: usize) -> ExecutionFrontier {
    issuer
        .frontier(&issuer.issue_sequence().unwrap(), position)
        .unwrap()
}

fn decode<'a>(
    frontier: &'a ExecutionFrontier,
    policy: &'a LogitsReturnPolicy,
) -> ExecutionWork<'a> {
    ExecutionWork::new(
        frontier,
        TokenWork::Decode {
            token: TokenId(1),
            position: frontier.consumed_tokens(),
        },
        policy,
    )
    .unwrap()
}

fn before_submit() -> ExecutionProgress {
    ExecutionProgress::NotSubmitted {
        stage: NoSubmissionStage::Preparation,
    }
}

fn product() -> ExecutionProduct {
    ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(vec![0.25, -0.5]))
}

#[test]
fn allocator_deferral_preserves_observation_without_native_epochs() {
    let deferral = capacity();
    assert_eq!(
        deferral.stage(),
        ExecutorExecutionCapacityStage::SequenceExtension
    );
    assert!(deferral.as_capacity().is_none());
    assert!(deferral.as_request_state().is_none());
    let observed = deferral.as_allocator_capacity().unwrap();
    assert_eq!(observed.required_blocks(), 2);
    assert_eq!(observed.snapshot().free_blocks, 1);
    assert_eq!(
        serde_json::to_value(&deferral).unwrap(),
        serde_json::json!({
            "reason": "allocator_capacity",
            "evidence": {
                "domain": serde_json::to_value(observed.domain()).unwrap(),
                "required_blocks": 2,
                "snapshot": {"block_size": 16, "total_blocks": 4, "free_blocks": 1}
            }
        })
    );
}

#[test]
fn capacity_deferral_keeps_committed_peer_out_of_retryable_subset() {
    let issuer = ExecutorSequenceIssuer::new();
    let first = frontier(&issuer, 2);
    let second = frontier(&issuer, 5);
    let policy = LogitsReturnPolicy::FullLogits;
    let inputs = [decode(&first, &policy), decode(&second, &policy)];
    let batch = issuer.batch(&inputs, 2).unwrap();
    let completed = batch
        .completion(0, ExecutedTokenWork::Decode { position: 2 }, product())
        .unwrap();
    let deferred = batch
        .participant(
            1,
            before_submit(),
            ContinuationDisposition::Retryable(second.clone()),
            0,
        )
        .unwrap();
    let ExecutionBatchOutcome::Interrupted(outcome) = batch
        .interrupted(vec![completed], vec![deferred], vec![capacity().into()])
        .unwrap()
    else {
        panic!("partial progress cannot become whole-batch zero submission");
    };
    assert_eq!(outcome.completed()[0].next_frontier().consumed_tokens(), 3);
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
        outcome.completed()[0].product()
    else {
        panic!("original full product was lost");
    };
    assert_eq!(logits, &[0.25, -0.5]);
    assert_eq!(outcome.remaining()[0].input_index(), 1);
    let ContinuationDisposition::Retryable(retained) = outcome.remaining()[0].continuation() else {
        panic!("capacity must preserve the original live frontier");
    };
    assert_eq!(retained, &second);
    assert!(outcome.causes()[0].as_error().is_none());
    assert!(outcome.causes()[0]
        .as_deferral()
        .unwrap()
        .as_allocator_capacity()
        .is_some());
}

#[test]
fn capacity_deferral_rejects_advanced_foreign_and_ended_frontiers() {
    let issuer = ExecutorSequenceIssuer::new();
    let original = frontier(&issuer, 2);
    let policy = LogitsReturnPolicy::FullLogits;
    let inputs = [decode(&original, &policy)];
    let batch = issuer.batch(&inputs, 2).unwrap();
    for wrong in [
        issuer.frontier(original.sequence(), 3).unwrap(),
        frontier(&issuer, 2),
    ] {
        assert!(batch
            .participant(
                0,
                before_submit(),
                ContinuationDisposition::Retryable(wrong),
                0
            )
            .is_err());
    }
    let retained = batch
        .participant(
            0,
            before_submit(),
            ContinuationDisposition::Retryable(original.clone()),
            0,
        )
        .unwrap();
    let ended = issuer.ended(original.sequence()).unwrap();
    let rejected = batch
        .interrupted(Vec::new(), vec![retained], vec![capacity().into()])
        .unwrap_err();
    let (_, (_, remaining, causes)) = rejected.into_parts();
    assert_eq!(remaining[0].sequence(), ended.sequence());
    assert!(causes[0].as_deferral().is_some());

    for progress in [
        before_submit(),
        ExecutionProgress::Terminal {
            disposition: TerminalModelDisposition::Successful,
            executed_range: Some(ExecutionRange::new(2, 1).unwrap()),
        },
    ] {
        let participant = batch
            .participant(
                0,
                progress,
                ContinuationDisposition::Ended(ended.clone()),
                0,
            )
            .unwrap();
        assert!(batch
            .interrupted(Vec::new(), vec![participant], vec![capacity().into()])
            .is_err());
    }
}

#[test]
fn rejected_capacity_deferral_retains_success_and_original_held_observer() {
    let issuer = ExecutorSequenceIssuer::new();
    let first = frontier(&issuer, 2);
    let second = frontier(&issuer, 5);
    let policy = LogitsReturnPolicy::FullLogits;
    let inputs = [decode(&first, &policy), decode(&second, &policy)];
    let batch = issuer.batch(&inputs, 2).unwrap();
    let completed = batch
        .completion(0, ExecutedTokenWork::Decode { position: 2 }, product())
        .unwrap();
    let (sender, receiver) = futures::channel::oneshot::channel();
    let held = issuer
        .held(
            issuer.recovery_key(second.sequence()).unwrap(),
            async move {
                receiver
                    .await
                    .map_err(|_| invalid("terminal publisher disappeared"))
            },
        )
        .unwrap();
    let unknown = batch
        .participant(
            1,
            ExecutionProgress::Unknown {
                last_known: LastKnownExecutionStage::Submitted,
                known_executed_range: None,
            },
            ContinuationDisposition::Held(held),
            0,
        )
        .unwrap();
    let rejected = batch
        .interrupted(vec![completed], vec![unknown], vec![capacity().into()])
        .unwrap_err();
    let ExecutionBatchOutcome::ContractViolation(mut violation) = rejected.into_outcome() else {
        panic!("invalid deferral cannot grant retry");
    };
    assert_eq!(violation.completed[0].next_frontier().consumed_tokens(), 3);
    assert!(violation.causes[0].as_deferral().is_some());
    let (_, _, _, continuation, _) = violation.remaining.pop().unwrap().into_parts();
    let ContinuationDisposition::Held(held) = continuation else {
        panic!("original recovery observer was lost");
    };
    sender
        .send(ExecutionTerminalOutcome::ended(
            issuer.ended(second.sequence()).unwrap(),
        ))
        .unwrap();
    assert_eq!(
        futures::executor::block_on(held.wait())
            .unwrap()
            .ended
            .sequence(),
        second.sequence()
    );
}

#[test]
fn capacity_deferral_rejects_incomplete_conflicting_or_foreign_coverage() {
    let issuer = ExecutorSequenceIssuer::new();
    let first = frontier(&issuer, 2);
    let second = frontier(&issuer, 5);
    let policy = LogitsReturnPolicy::FullLogits;
    let inputs = [decode(&first, &policy), decode(&second, &policy)];
    for (index, cause_index, foreign_batch, omit) in [
        (0, 0, false, false), // Success and deferral claim the same member.
        (1, 1, false, false), // Cause index does not exist.
        (1, 0, true, false),  // Same inputs, different original batch.
        (1, 0, false, true),  // A participant has no report.
    ] {
        let batch = issuer.batch(&inputs, 2).unwrap();
        let other = issuer.batch(&inputs, 2).unwrap();
        let completed = batch
            .completion(0, ExecutedTokenWork::Decode { position: 2 }, product())
            .unwrap();
        let remaining = if omit {
            Vec::new()
        } else {
            let scope = if foreign_batch { &other } else { &batch };
            vec![scope
                .participant(
                    index,
                    before_submit(),
                    ContinuationDisposition::Retryable(inputs[index].expected_frontier().clone()),
                    cause_index,
                )
                .unwrap()]
        };
        let rejected = batch
            .interrupted(vec![completed], remaining, vec![capacity().into()])
            .unwrap_err();
        let (_, (completed, remaining, causes)) = rejected.into_parts();
        assert_eq!(completed[0].next_frontier().consumed_tokens(), 3);
        assert_eq!(remaining.is_empty(), omit);
        assert!(causes[0].as_deferral().is_some());
    }
}
