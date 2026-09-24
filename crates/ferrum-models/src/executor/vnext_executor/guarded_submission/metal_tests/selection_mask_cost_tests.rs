//! The real product query and guarded native submission must agree on mask
//! contents, physical row order and cold/hot uploads with timing still Off.
use super::*;
use ferrum_interfaces::model_executor::TokenSelectionMask;

fn greedy(mask: TokenSelectionMask) -> LogitsReturnPolicy {
    LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        repetition_penalty: None,
    }
}

fn mask_counts(fixture: &Fixture) -> (u64, u64) {
    (
        fixture
            .executor
            .metrics
            .token_mask_upload_participants
            .load(Ordering::Relaxed),
        fixture
            .executor
            .metrics
            .token_mask_cache_hit_participants
            .load(Ordering::Relaxed),
    )
}

fn assert_outputs(actual: &[PlanRuntimeDecodeOutput], expected: &[PlanRuntimeDecodeOutput]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.kv_cache.num_tokens(), b.kv_cache.num_tokens());
        match (&a.sampling_output, &b.sampling_output) {
            (ExecutorSamplingOutput::GreedyToken(a), ExecutorSamplingOutput::GreedyToken(b)) => {
                assert_eq!(a, b)
            }
            (ExecutorSamplingOutput::FullLogits(a), ExecutorSamplingOutput::FullLogits(b)) => {
                fixture::assert_logits_same(a, b)
            }
            other => panic!("real guarded/ordinary output mode differs: {other:?}"),
        }
    }
}

async fn step(
    fixture: &Fixture,
    baseline: &Fixture,
    inputs: &mut [PlanRuntimeDecodeInput],
    others: &mut [PlanRuntimeDecodeInput],
    expected_uploads: u64,
) {
    assert_eq!(fixture.executor.device_timing_mode(), DeviceTimingMode::Off);
    fixture.warm(&[], inputs);
    let submissions = fixture.submissions();
    let before = mask_counts(fixture);
    let expected = fixture.expected(&[], inputs);
    assert_eq!(
        fixture.submissions(),
        submissions,
        "future query cannot encode or submit"
    );
    assert_eq!(
        mask_counts(fixture),
        before,
        "pure query cannot publish mask residency"
    );
    let gate = Gate::new(false);
    let outputs = submitted(
        fixture
            .executor
            .plan_runtime_batch_decode_guarded_observed(inputs, &expected, &gate, None)
            .await,
    );
    let ordinary = match baseline
        .executor
        .plan_runtime_batch_decode_with_capacity(others)
        .await
        .unwrap()
    {
        PlanRuntimeBatchDecodeOutcome::Completed(outputs) => outputs,
        PlanRuntimeBatchDecodeOutcome::Deferred(reason) => {
            panic!("ordinary baseline deferred: {reason:?}")
        }
    };
    assert_outputs(&outputs, &ordinary);
    assert_eq!(fixture.submissions() - submissions, 1);
    assert_eq!(
        gate.calls.load(Ordering::Relaxed),
        1,
        "the actual full canonical route passed its final guard once"
    );
    let after = mask_counts(fixture);
    assert_eq!(after.0 - before.0, expected_uploads);
    assert_eq!(after.1 - before.1, inputs.len() as u64 - expected_uploads);
    for (input, output) in inputs.iter_mut().zip(outputs) {
        input.kv_cache = output.kv_cache;
    }
    for (input, output) in others.iter_mut().zip(ordinary) {
        input.kv_cache = output.kv_cache;
    }
}

#[tokio::test]
async fn selection_mask_future_metal_route_matches_cold_hot_and_both_mode_transitions() {
    let fixture = Fixture::new(8, false).await;
    let baseline = Fixture::new(8, false).await;
    let original = TokenSelectionMask::new(vec![0, 1, 0]);
    let mut inputs = [fixture
        .seed_decode(&[0, 1])
        .await
        .with_logits_policy(greedy(original.clone()))];
    let mut others = [baseline
        .seed_decode(&[0, 1])
        .await
        .with_logits_policy(greedy(original.clone()))];
    step(&fixture, &baseline, &mut inputs, &mut others, 1).await;
    // Distinct immutable storage with equal complete contents is a real hit.
    // Keep original alive exactly as an admitted request's retained mask is.
    inputs[0].logits_policy = greedy(TokenSelectionMask::new(vec![0, 1, 0]));
    others[0].logits_policy = inputs[0].logits_policy.clone();
    step(&fixture, &baseline, &mut inputs, &mut others, 0).await;
    inputs[0].logits_policy = LogitsReturnPolicy::FullLogits;
    others[0].logits_policy = LogitsReturnPolicy::FullLogits;
    step(&fixture, &baseline, &mut inputs, &mut others, 1).await;
    step(&fixture, &baseline, &mut inputs, &mut others, 0).await;
    inputs[0].logits_policy = greedy(original.clone());
    others[0].logits_policy = greedy(original);
    step(&fixture, &baseline, &mut inputs, &mut others, 1).await;
}

#[tokio::test]
async fn selection_mask_future_metal_route_matches_packed_full_logits_peers_and_row_contents() {
    let fixture = Fixture::new(8, false).await;
    let baseline = Fixture::new(8, false).await;
    let a = TokenSelectionMask::new(vec![0, 1, 0]);
    let b = TokenSelectionMask::new(vec![0, 0, 1]);
    let mut inputs = [
        fixture
            .seed_decode(&[0, 1])
            .await
            .with_logits_policy(greedy(a.clone())),
        fixture
            .seed_decode(&[1, 0])
            .await
            .with_logits_policy(LogitsReturnPolicy::FullLogits),
    ];
    let mut others = [
        baseline
            .seed_decode(&[0, 1])
            .await
            .with_logits_policy(greedy(a)),
        baseline
            .seed_decode(&[1, 0])
            .await
            .with_logits_policy(LogitsReturnPolicy::FullLogits),
    ];
    // One FullLogits participant means AllValid uploads for the whole wave,
    // including its greedy peer; every actual output is FullLogits.
    step(&fixture, &baseline, &mut inputs, &mut others, 2).await;
    inputs[1].logits_policy = greedy(b.clone());
    others[1].logits_policy = greedy(b);
    step(&fixture, &baseline, &mut inputs, &mut others, 2).await;
    // A selected guarded wave seals physical row order. Unlike the ordinary
    // batch API, its caller cannot reorder that seal after cost projection.
    inputs.reverse();
    others.reverse();
    fixture.warm(&[], &inputs);
    let expected = fixture.expected(&[], &inputs);
    let before = mask_counts(&fixture);
    let submissions = fixture.submissions();
    let gate = Gate::new(false);
    assert!(matches!(
        fixture
            .executor
            .plan_runtime_batch_decode_guarded_observed(&inputs, &expected, &gate, None,)
            .await,
        GuardedDispatchOutcome::Unsupported
    ));
    assert_eq!(fixture.submissions(), submissions);
    assert_eq!(mask_counts(&fixture), before);
    assert_eq!(gate.calls.load(Ordering::Relaxed), 0);
    // Re-capture the same actual row order: both masks are still resident.
    inputs.reverse();
    others.reverse();
    step(&fixture, &baseline, &mut inputs, &mut others, 0).await;
}
