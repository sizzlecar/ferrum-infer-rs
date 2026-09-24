//! Product Metal execution with exact work and no predicted cost commitment.
//! These tests do not build a canonical route or supply a host policy hash.
use super::*;
use ferrum_interfaces::model_executor::TokenSelectionMask;
use std::num::NonZeroU64;

pub(super) fn complete(
    fixture: &Fixture,
    prefills: &[PlanRuntimePrefillInput],
    decodes: &[PlanRuntimeDecodeInput],
) -> ExpectedExecutionWave {
    let rows = fixture.rows(prefills, decodes); // actual SequenceAuthority order
    let requests: Vec<_> = rows
        .iter()
        .map(|(sequence, _, _)| ExecutorResourcePlanningRequest {
            request_id: sequence.request_id(),
            cache_id: decodes
                .iter()
                .any(|input| &input.request_id == sequence.request_id())
                .then_some(sequence.cache_id.as_str()),
        })
        .collect();
    let deadline = Instant::now() + Duration::from_secs(5);
    let view =
        loop {
            match fixture.executor.execution_resource_planning_view(
                &requests,
                ResourcePlanningLimits::default(),
                &mut || true,
            ) {
                ResourcePlanningAvailability::Known(view) => break view,
                ResourcePlanningAvailability::Unknown(
                    ResourcePlanningUnknown::ReadUnavailable(_),
                ) if Instant::now() < deadline => std::thread::yield_now(),
                other => panic!("complete-work resource snapshot: {other:?}"),
            }
        };
    let selections = rows
        .iter()
        .enumerate()
        .map(|(participant_index, (sequence, _, range))| {
            let (input, work, decode_policy) = match prefills
                .iter()
                .find(|input| &input.request_id == sequence.request_id())
            {
                Some(input) => (
                    ExpectedWaveInput::Prefill { chunk: input.chunk },
                    ActualRowWork::Prefill {
                        offset: range.start as u32,
                        count: range.len() as u32,
                        total_prompt_tokens: input.input_tokens.len() as u32,
                    },
                    None,
                ),
                None => {
                    let input = decodes
                        .iter()
                        .find(|input| &input.request_id == sequence.request_id())
                        .unwrap();
                    (
                        ExpectedWaveInput::Decode {
                            cache_id: input.kv_cache.cache_id(),
                        },
                        ActualRowWork::Decode {
                            kv_tokens: range.start as u32,
                        },
                        Some(input.logits_policy.clone()),
                    )
                }
            };
            ExpectedWorkSelection {
                participant_index,
                request_id: sequence.request_id().clone(),
                // This model-level fixture owns one host work generation. Its
                // physical owner/session/epoch comes only from the real view above.
                owner_incarnation: NonZeroU64::new(1).unwrap(),
                work_generation: NonZeroU64::new(1).unwrap(),
                input,
                work,
                decode_policy,
            }
        })
        .collect();
    let kind = if prefills.is_empty() {
        ActualWaveKind::Decode
    } else if decodes.is_empty() {
        ActualWaveKind::Prefill
    } else {
        ActualWaveKind::Mixed
    };
    ExpectedExecutionWave::complete_requests(
        ExpectedWaveWork::new(&view, kind, selections).unwrap(),
        CompletionOnlyReason::CostUnavailable,
    )
}

fn tail(input: &PlanRuntimePrefillInput, offset: usize) -> PlanRuntimePrefillInput {
    PlanRuntimePrefillInput::new(
        input.request_id.clone(),
        Arc::clone(&input.input_tokens),
        input.maximum_sequence_tokens,
        PrefillChunk::new(
            offset,
            input.input_tokens.len() - offset,
            input.input_tokens.len(),
        )
        .unwrap(),
    )
    .unwrap()
}

async fn ordinary_decode(
    fixture: &Fixture,
    input: PlanRuntimeDecodeInput,
) -> PlanRuntimeDecodeOutput {
    match fixture
        .executor
        .plan_runtime_batch_decode_with_capacity(&[input])
        .await
        .unwrap()
    {
        PlanRuntimeBatchDecodeOutcome::Completed(mut outputs) => {
            assert_eq!(outputs.len(), 1);
            outputs.remove(0)
        }
        PlanRuntimeBatchDecodeOutcome::Deferred(reason) => panic!("ordinary decode: {reason:?}"),
    }
}

#[tokio::test]
async fn guarded_metal_complete_partial_final_decode_matches_ordinary_without_cost() {
    let fixture = Fixture::new(8, false).await;
    let baseline = Fixture::new(8, false).await;
    let input = prompt(&[0, 1, 2, 1], 2);
    let other = prompt(&[0, 1, 2, 1], 2);
    fixture.admit(&input);
    baseline.admit(&other);
    fixture.warm(std::slice::from_ref(&input), &[]);
    let gate = Gate::new(false);
    let before = fixture.submissions();
    let work = complete(&fixture, std::slice::from_ref(&input), &[]);
    let partial = submitted(
        fixture
            .executor
            .plan_runtime_batch_prefill_guarded_work_observed(
                std::slice::from_ref(&input),
                &work,
                &gate,
                None,
            )
            .await,
    );
    assert_eq!(partial.len(), 1);
    assert_eq!(partial[0].capacity_probe_count(), 0);
    assert!(matches!(
        partial[0].output().product(),
        PlanRuntimePrefillProduct::Intermediate
    ));
    assert_prefill_same(&partial[0], &baseline.prefill(&other).await);
    fixture.assert_ready(&input, 2);
    let final_input = tail(&input, 2);
    fixture.warm(std::slice::from_ref(&final_input), &[]);
    let work = complete(&fixture, std::slice::from_ref(&final_input), &[]);
    let final_outputs = submitted(
        fixture
            .executor
            .plan_runtime_batch_prefill_guarded_work_observed(&[final_input], &work, &gate, None)
            .await,
    );
    let ordinary = baseline.prefill(&tail(&other, 2)).await;
    assert_prefill_same(&final_outputs[0], &ordinary);
    assert_eq!(final_outputs[0].output().committed_tokens(), 4);
    let decode = PlanRuntimeDecodeInput::new(
        input.request_id,
        TokenId::new(1),
        Arc::clone(final_outputs[0].output().kv_cache()),
    );
    let baseline_decode = PlanRuntimeDecodeInput::new(
        other.request_id,
        TokenId::new(1),
        Arc::clone(ordinary.output().kv_cache()),
    );
    fixture.warm(&[], std::slice::from_ref(&decode));
    let work = complete(&fixture, &[], std::slice::from_ref(&decode));
    let decoded = submitted(
        fixture
            .executor
            .plan_runtime_batch_decode_guarded_work_observed(&[decode], &work, &gate, None)
            .await,
    );
    let expected = ordinary_decode(&baseline, baseline_decode).await;
    fixture::assert_logits_same(
        fixture::logits(&decoded[0].sampling_output),
        fixture::logits(&expected.sampling_output),
    );
    assert_eq!(decoded[0].kv_cache.num_tokens(), 5);
    assert_eq!(fixture.submissions() - before, 3);
    assert_eq!(gate.calls.load(Ordering::Relaxed), 3);
}

#[tokio::test]
async fn guarded_metal_complete_mixed_is_one_exact_wave_with_partial_and_final_products() {
    let fixture = Fixture::new(8, false).await;
    let baseline = Fixture::new(8, false).await;
    let decode = fixture.seed_decode(&[0, 1]).await;
    let ordinary_decode_input = baseline.seed_decode(&[0, 1]).await;
    let inputs = [prompt(&[1, 2, 0, 1], 2), prompt(&[2, 1, 0], 3)];
    let others = [prompt(&[1, 2, 0, 1], 2), prompt(&[2, 1, 0], 3)];
    for input in &inputs {
        fixture.admit(input);
    }
    for input in &others {
        baseline.admit(input);
    }
    fixture.warm(&inputs, std::slice::from_ref(&decode));
    let work = complete(&fixture, &inputs, std::slice::from_ref(&decode));
    let before = fixture.submissions();
    let gate = Gate::new(false);
    let outputs = submitted(
        fixture
            .executor
            .plan_runtime_mixed_batch_guarded_work_observed(&inputs, &[decode], &work, &gate, None)
            .await,
    );
    assert_eq!(outputs.prefills.len(), 2);
    assert_eq!(outputs.decodes.len(), 1);
    for (input, other) in inputs.iter().zip(&others) {
        let actual = outputs
            .prefills
            .iter()
            .find(|output| output.output().request_id() == &input.request_id)
            .unwrap();
        assert_eq!(actual.capacity_probe_count(), 0);
        assert_prefill_same(actual, &baseline.prefill(other).await);
    }
    fixture.assert_ready(&inputs[0], 2);
    let ordinary = ordinary_decode(&baseline, ordinary_decode_input).await;
    fixture::assert_logits_same(
        fixture::logits(&outputs.decodes[0].sampling_output),
        fixture::logits(&ordinary.sampling_output),
    );
    assert_eq!(outputs.decodes[0].kv_cache.num_tokens(), 3);
    assert_eq!(fixture.submissions() - before, 1);
    assert_eq!(gate.calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn guarded_metal_complete_cold_defers_without_maintenance_then_fresh_work_advances() {
    let fixture = Fixture::new(8, false).await;
    let input = prompt(&[0, 1, 2, 0], 4);
    fixture.admit(&input);
    let before = fixture.submissions();
    let gate = Gate::new(false);
    let work = complete(&fixture, std::slice::from_ref(&input), &[]);
    let pool_bytes = || {
        fixture
            .executor
            .plan_resources
            .dynamic_pool_status()
            .unwrap()
            .pools()
            .iter()
            .map(|pool| {
                (
                    pool.domain_id(),
                    pool.resident_bytes(),
                    pool.free_bytes(),
                    pool.pending_growth_bytes(),
                )
            })
            .collect::<Vec<_>>()
    };
    let cold = pool_bytes();
    assert!(matches!(
        fixture
            .executor
            .plan_runtime_batch_prefill_guarded_work_observed(
                std::slice::from_ref(&input),
                &work,
                &gate,
                None,
            )
            .await,
        GuardedDispatchOutcome::MaintenanceDeferred { .. }
    ));
    assert_eq!(
        pool_bytes(),
        cold,
        "guarded work cannot perform hidden pool maintenance"
    );
    fixture.assert_ready(&input, 0);
    assert_eq!(fixture.submissions(), before);
    assert_eq!(gate.calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        fixture
            .executor
            .metrics
            .prefill_frontier_narrowings
            .load(Ordering::Relaxed),
        0
    );
    fixture.warm(std::slice::from_ref(&input), &[]);
    let refreshed = complete(&fixture, std::slice::from_ref(&input), &[]);
    let outputs = submitted(
        fixture
            .executor
            .plan_runtime_batch_prefill_guarded_work_observed(&[input], &refreshed, &gate, None)
            .await,
    );
    assert_eq!(outputs[0].output().committed_tokens(), 4);
    assert_eq!(outputs[0].capacity_probe_count(), 0);
    assert_eq!(fixture.submissions() - before, 1);
    assert_eq!(gate.calls.load(Ordering::Relaxed), 1);
}

struct OutputRevoked(AtomicUsize);
impl NonblockingHostSubmissionGuard for OutputRevoked {
    fn check(&self) -> std::result::Result<(), HostSubmissionRejection> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Err(HostSubmissionRejection::OutputRevoked)
    }
}

#[tokio::test]
async fn guarded_metal_complete_host_rejection_and_wrong_chunk_preserve_same_request() {
    let fixture = Fixture::new(8, false).await;
    let baseline = Fixture::new(8, false).await;
    let input = prompt(&[0, 1, 2, 1], 2);
    let other = prompt(&[0, 1, 2, 1], 2);
    fixture.admit(&input);
    baseline.admit(&other);
    fixture.warm(std::slice::from_ref(&input), &[]);
    let work = complete(&fixture, std::slice::from_ref(&input), &[]);
    let before = fixture.submissions();
    let gate = Gate::new(false);
    let wrong = PlanRuntimePrefillInput::new(
        input.request_id.clone(),
        Arc::clone(&input.input_tokens),
        input.maximum_sequence_tokens,
        PrefillChunk::new(0, 1, 4).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        fixture
            .executor
            .plan_runtime_batch_prefill_guarded_work_observed(&[wrong], &work, &gate, None)
            .await,
        GuardedDispatchOutcome::Unsupported
    ));
    assert_eq!(gate.calls.load(Ordering::Relaxed), 0);
    let revoked = OutputRevoked(AtomicUsize::new(0));
    match fixture
        .executor
        .plan_runtime_batch_prefill_guarded_work_observed(
            std::slice::from_ref(&input),
            &work,
            &revoked,
            None,
        )
        .await
    {
        GuardedDispatchOutcome::NotSubmittedAfterPreparation(receipt) => assert_eq!(
            receipt.reason(),
            GuardedNotSubmittedReason::HostRejected(HostSubmissionRejection::OutputRevoked)
        ),
        _ => panic!("output revocation must reconcile the actual encoded Step"),
    }
    assert_eq!(revoked.0.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.submissions(), before);
    fixture.assert_ready(&input, 0);
    let refreshed = complete(&fixture, std::slice::from_ref(&input), &[]);
    let outputs = submitted(
        fixture
            .executor
            .plan_runtime_batch_prefill_guarded_work_observed(&[input], &refreshed, &gate, None)
            .await,
    );
    assert_prefill_same(&outputs[0], &baseline.prefill(&other).await);
    assert_eq!(fixture.submissions() - before, 1);
}

#[tokio::test]
async fn guarded_metal_complete_cancelled_owner_cannot_submit_into_reused_request_id() {
    let fixture = Fixture::new(8, false).await;
    let input = prompt(&[0, 1, 2, 1], 4);
    fixture.admit(&input);
    fixture.warm(std::slice::from_ref(&input), &[]);
    let stale = complete(&fixture, std::slice::from_ref(&input), &[]);
    assert!(fixture.executor.cancel_prefill_admission(&input.request_id));
    fixture.admit(&input); // same product ID, new physical request/session owner
    fixture.warm(std::slice::from_ref(&input), &[]);
    let refreshed = complete(&fixture, std::slice::from_ref(&input), &[]);
    assert_ne!(
        stale.work().participants()[0].resource().authority(),
        refreshed.work().participants()[0].resource().authority()
    );
    let before = fixture.submissions();
    let gate = Gate::new(false);
    match fixture
        .executor
        .plan_runtime_batch_prefill_guarded_work_observed(
            std::slice::from_ref(&input),
            &stale,
            &gate,
            None,
        )
        .await
    {
        GuardedDispatchOutcome::NotSubmittedAfterPreparation(receipt) => assert_eq!(
            receipt.reason(),
            GuardedNotSubmittedReason::ResourceClaimMismatch
        ),
        _ => panic!("old owner must fail before native submission"),
    }
    assert_eq!(fixture.submissions(), before);
    assert_eq!(gate.calls.load(Ordering::Relaxed), 0);
    fixture.assert_ready(&input, 0);
    let refreshed = complete(&fixture, std::slice::from_ref(&input), &[]);
    let outputs = submitted(
        fixture
            .executor
            .plan_runtime_batch_prefill_guarded_work_observed(&[input], &refreshed, &gate, None)
            .await,
    );
    assert_eq!(outputs[0].output().committed_tokens(), 4);
    assert_eq!(fixture.submissions() - before, 1);
}

#[tokio::test]
async fn guarded_metal_complete_selection_mask_executes_but_replaced_policy_needs_new_capture() {
    let fixture = Fixture::new(8, false).await;
    let baseline = Fixture::new(8, false).await;
    let policy = || LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(TokenSelectionMask::new(vec![0, 1, 0])),
        repetition_penalty: None,
    };
    let mut decode = fixture
        .seed_decode(&[0, 1])
        .await
        .with_logits_policy(policy());
    let other = baseline
        .seed_decode(&[0, 1])
        .await
        .with_logits_policy(policy());
    fixture.warm(&[], std::slice::from_ref(&decode));
    fixture
        .try_expected(&[], std::slice::from_ref(&decode))
        .expect("selection-mask cost projection now has actual content evidence");
    let captured = complete(&fixture, &[], std::slice::from_ref(&decode));
    decode.logits_policy = policy(); // identical contents, distinct immutable Arc
    let before = fixture.submissions();
    let gate = Gate::new(false);
    assert!(matches!(
        fixture
            .executor
            .plan_runtime_batch_decode_guarded_work_observed(
                std::slice::from_ref(&decode),
                &captured,
                &gate,
                None
            )
            .await,
        GuardedDispatchOutcome::Unsupported
    ));
    assert_eq!(fixture.submissions(), before);
    assert_eq!(gate.calls.load(Ordering::Relaxed), 0);
    let refreshed = complete(&fixture, &[], std::slice::from_ref(&decode));
    let outputs = submitted(
        fixture
            .executor
            .plan_runtime_batch_decode_guarded_work_observed(&[decode], &refreshed, &gate, None)
            .await,
    );
    let ordinary = ordinary_decode(&baseline, other).await;
    assert!(
        matches!(outputs[0].sampling_output, ExecutorSamplingOutput::GreedyToken(token) if token == TokenId::new(1))
    );
    assert!(
        matches!(ordinary.sampling_output, ExecutorSamplingOutput::GreedyToken(token) if token == TokenId::new(1))
    );
    assert_eq!(outputs[0].kv_cache.num_tokens(), 3);
    assert_eq!(fixture.submissions() - before, 1);
    assert_eq!(gate.calls.load(Ordering::Relaxed), 1);
}
