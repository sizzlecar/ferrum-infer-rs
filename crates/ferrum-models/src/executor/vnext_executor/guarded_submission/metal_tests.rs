//! Actual product executor, selected Metal providers and the final native gate.
//! No mocked receipt, capability override or provider outcome is used.
use super::*;
use ferrum_interfaces::execution_cost::*;
use ferrum_interfaces::model_executor::ExecutorResourcePlanningRequest;
use ferrum_kernels::backend::metal::{
    vnext_ops::MetalVNextComposition, vnext_runtime::MetalDeviceRuntime,
};
use ferrum_types::ModelId;
mod completion_work_tests;
mod execution_maintenance_tests;
mod fixture;
mod resolver;
mod selection_mask_cost_tests;
mod weights;
use fixture::{assert_prefill_same, prompt, Fixture};

struct Gate {
    reject: bool,
    calls: AtomicUsize,
}
impl Gate {
    fn new(reject: bool) -> Self {
        Self {
            reject,
            calls: AtomicUsize::new(0),
        }
    }
}
impl NonblockingHostSubmissionGuard for Gate {
    fn check(&self) -> std::result::Result<(), HostSubmissionRejection> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.reject {
            Err(HostSubmissionRejection::WitnessExpired)
        } else {
            Ok(())
        }
    }
}

#[track_caller]
fn submitted<T>(outcome: GuardedDispatchOutcome<T>) -> T {
    match outcome {
        GuardedDispatchOutcome::Submitted(result) => result.unwrap(),
        GuardedDispatchOutcome::Unsupported => panic!("actual product guard is unsupported"),
        GuardedDispatchOutcome::Deferred(reason) => {
            panic!("prepared capacity deferred: {reason:?}")
        }
        GuardedDispatchOutcome::MaintenanceDeferred { deferral, .. } => {
            panic!("prepared capacity requires maintenance: {deferral:?}")
        }
        GuardedDispatchOutcome::ReplanBeforeEncode => panic!("prepared resource evidence changed"),
        GuardedDispatchOutcome::NotSubmittedAfterPreparation(receipt) => {
            panic!("actual route rejected: {:?}", receipt.reason())
        }
    }
}

#[tokio::test]
async fn guarded_metal_product_prefill_exact_partial_final_rejection_then_same_request() {
    let fixture = Fixture::new(8, false).await;
    assert_eq!(
        fixture.executor.slo_execution_capability(),
        ferrum_interfaces::model_executor::ExecutorSloCapability::GuardedEagerWaves
    );
    let baseline = Fixture::new(8, false).await;
    let input = prompt(&[0, 1, 2, 1], 2);
    let other = prompt(&[0, 1, 2, 1], 2);
    fixture.admit(&input);
    baseline.admit(&other);
    fixture.warm(std::slice::from_ref(&input), &[]);
    let before = fixture.submissions();
    let expected = fixture.expected(std::slice::from_ref(&input), &[]);
    let reject = Gate::new(true);
    match fixture
        .executor
        .plan_runtime_batch_prefill_guarded_observed(
            std::slice::from_ref(&input),
            &expected,
            &reject,
            None,
        )
        .await
    {
        GuardedDispatchOutcome::NotSubmittedAfterPreparation(receipt) => assert_eq!(
            receipt.reason(),
            GuardedNotSubmittedReason::HostRejected(HostSubmissionRejection::WitnessExpired)
        ),
        _ => panic!("final native guard did not return a reconciled rejection"),
    }
    assert_eq!(reject.calls.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.submissions(), before);
    fixture.assert_ready(&input, 0);
    let expected = fixture.expected(std::slice::from_ref(&input), &[]);
    let allow = Gate::new(false);
    let partial = submitted(
        fixture
            .executor
            .plan_runtime_batch_prefill_guarded_observed(
                std::slice::from_ref(&input),
                &expected,
                &allow,
                None,
            )
            .await,
    );
    assert_eq!(fixture.submissions() - before, 1);
    assert_eq!(allow.calls.load(Ordering::Relaxed), 1);
    assert_eq!(partial[0].capacity_probe_count(), 0);
    assert_eq!(partial[0].completed_chunk(), input.chunk);
    assert!(matches!(
        partial[0].output().product(),
        PlanRuntimePrefillProduct::Intermediate
    ));
    fixture.assert_ready(&input, 2);
    let uploads_after_partial = fixture
        .executor
        .metrics
        .token_mask_upload_participants
        .load(Ordering::Relaxed);
    let hits_after_partial = fixture
        .executor
        .metrics
        .token_mask_cache_hit_participants
        .load(Ordering::Relaxed);
    assert_eq!(uploads_after_partial, 1);
    assert_prefill_same(&partial[0], &baseline.prefill(&other).await);
    let tail = PlanRuntimePrefillInput::new(
        input.request_id.clone(),
        input.input_tokens.clone(),
        input.maximum_sequence_tokens,
        PrefillChunk::new(2, 2, 4).unwrap(),
    )
    .unwrap();
    let other_tail = PlanRuntimePrefillInput::new(
        other.request_id.clone(),
        other.input_tokens.clone(),
        other.maximum_sequence_tokens,
        tail.chunk,
    )
    .unwrap();
    fixture.warm(std::slice::from_ref(&tail), &[]);
    let expected = fixture.expected(std::slice::from_ref(&tail), &[]);
    let final_outputs = submitted(
        fixture
            .executor
            .plan_runtime_batch_prefill_guarded_observed(&[tail], &expected, &allow, None)
            .await,
    );
    assert_prefill_same(&final_outputs[0], &baseline.prefill(&other_tail).await);
    assert_eq!(fixture.submissions() - before, 2);
    assert!(matches!(
        final_outputs[0].output().product(),
        PlanRuntimePrefillProduct::FinalLogits(_)
    ));
    assert_eq!(final_outputs[0].output().kv_cache().num_tokens(), 4);
    assert_eq!(
        fixture
            .executor
            .metrics
            .token_mask_upload_participants
            .load(Ordering::Relaxed),
        uploads_after_partial
    );
    assert_eq!(
        fixture
            .executor
            .metrics
            .token_mask_cache_hit_participants
            .load(Ordering::Relaxed),
        hits_after_partial + 1,
        "final must exercise the actual retained-slot hit, not a forced reupload"
    );
}

#[tokio::test]
async fn guarded_metal_product_mixed_rejection_preserves_decode_and_partial_final_prefills() {
    let fixture = Fixture::new(8, false).await;
    let baseline = Fixture::new(8, false).await;
    let decode = fixture.seed_decode(&[0, 1]).await;
    let baseline_decode = baseline.seed_decode(&[0, 1]).await;
    let inputs = [prompt(&[1, 2, 0, 1], 2), prompt(&[2, 1, 0], 3)];
    let others = [prompt(&[1, 2, 0, 1], 2), prompt(&[2, 1, 0], 3)];
    for input in &inputs {
        fixture.admit(input);
    }
    for input in &others {
        baseline.admit(input);
    }
    fixture.warm(&inputs, std::slice::from_ref(&decode));
    let expected = fixture.expected(&inputs, std::slice::from_ref(&decode));
    let before = fixture.submissions();
    let reject = Gate::new(true);
    assert!(matches!(
        fixture
            .executor
            .plan_runtime_mixed_batch_guarded_observed(
                &inputs,
                std::slice::from_ref(&decode),
                &expected,
                &reject,
                None
            )
            .await,
        GuardedDispatchOutcome::NotSubmittedAfterPreparation(_)
    ));
    assert_eq!(reject.calls.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.submissions(), before);
    for input in &inputs {
        fixture.assert_ready(input, 0);
    }
    let active = fixture
        .executor
        .sequence_for_cache(&decode.kv_cache.cache_id())
        .unwrap();
    assert!(active.active.load(Ordering::Acquire));
    assert_eq!(*active.tokens.lock(), vec![0, 1]);
    let expected = fixture.expected(&inputs, std::slice::from_ref(&decode));
    let allow = Gate::new(false);
    let outputs = submitted(
        fixture
            .executor
            .plan_runtime_mixed_batch_guarded_observed(&inputs, &[decode], &expected, &allow, None)
            .await,
    );
    assert_eq!(fixture.submissions() - before, 1);
    assert_eq!(outputs.prefills.len(), 2);
    assert_eq!(outputs.decodes.len(), 1);
    for (index, other) in others.iter().enumerate() {
        assert_eq!(
            outputs.prefills[index].output().request_id(),
            &inputs[index].request_id
        );
        assert_prefill_same(&outputs.prefills[index], &baseline.prefill(other).await);
    }
    let expected_decode = match baseline
        .executor
        .plan_runtime_batch_decode_with_capacity(&[baseline_decode])
        .await
        .unwrap()
    {
        PlanRuntimeBatchDecodeOutcome::Completed(mut outputs) => outputs.remove(0),
        _ => panic!("baseline decode deferred"),
    };
    fixture::assert_logits_same(
        fixture::logits(&outputs.decodes[0].sampling_output),
        fixture::logits(&expected_decode.sampling_output),
    );
    assert_eq!(outputs.decodes[0].kv_cache.num_tokens(), 3);
    assert_eq!(*active.tokens.lock(), vec![0, 1, 1]);
}

#[tokio::test]
async fn guarded_metal_product_cold_capacity_cannot_invent_witness_and_cancel_is_terminal() {
    let fixture = Fixture::new(8, false).await;
    let input = prompt(&[0, 1, 2, 0], 4);
    fixture.admit(&input);
    let gate = Gate::new(false);
    let before = fixture.submissions();
    // These cold Step/Invocation pools have positive minimum demand and no
    // resident backing. Their logical capacity is residency-derived, so the
    // projection fails its logical charge before testing physical extents.
    // Pure planning must not maintain the pools or manufacture a witness.
    let pools = fixture
        .executor
        .plan_resources
        .dynamic_pool_status()
        .unwrap();
    let cold_pools: Vec<_> = pools
        .pools()
        .iter()
        .filter(|pool| {
            pool.resident_bytes() == 0
                && (pool.contract().minimum_step_bytes() > 0
                    || pool.contract().minimum_invocation_peak_bytes() > 0)
        })
        .map(|pool| {
            (
                pool.domain_id(),
                pool.contract().minimum_step_bytes(),
                pool.contract().minimum_invocation_peak_bytes(),
                pool.resident_bytes(),
            )
        })
        .collect();
    assert!(!cold_pools.is_empty(), "fixture must exercise cold demand");
    let cold = fixture.try_expected(std::slice::from_ref(&input), &[]);
    assert!(
        matches!(
            &cold,
            Err(ExecutionCostRouteUnknown::Resource(
                ResourcePlanningUnknown::LogicalCapacity
            ))
        ),
        "cold route: {cold:?}; cold domain/step/invocation/resident (first 8 of {}): {:?}",
        cold_pools.len(),
        &cold_pools[..cold_pools.len().min(8)],
    );
    assert_eq!(
        fixture
            .executor
            .plan_resources
            .dynamic_pool_status()
            .unwrap(),
        pools,
        "cost projection must not maintain cold pools"
    );
    assert_eq!(gate.calls.load(Ordering::Relaxed), 0);
    assert_eq!(fixture.submissions(), before);
    assert_eq!(
        fixture
            .executor
            .metrics
            .prefill_frontier_narrowings
            .load(Ordering::Relaxed),
        0
    );
    fixture.assert_ready(&input, 0);
    fixture.warm(std::slice::from_ref(&input), &[]);
    let expected = fixture.expected(std::slice::from_ref(&input), &[]);
    assert!(fixture.executor.cancel_prefill_admission(&input.request_id));
    assert!(matches!(
        fixture
            .executor
            .plan_runtime_batch_prefill_guarded_observed(&[input], &expected, &gate, None)
            .await,
        GuardedDispatchOutcome::Submitted(Err(_))
    ));
    assert_eq!(gate.calls.load(Ordering::Relaxed), 0);
    assert_eq!(fixture.submissions(), before);
}

#[tokio::test]
async fn guarded_metal_product_one_token_prefill_keeps_its_prefill_workspace_class() {
    let fixture = Fixture::new(8, false).await;
    let baseline = Fixture::new(8, false).await;
    let prefill_bucket = fixture
        .executor
        .reusable_bucket_for_shape(VNextExecutionWaveKind::Prefill, 1, 1, 0)
        .expect("actual product prefill workspace bucket");
    let decode_bucket = fixture
        .executor
        .reusable_bucket_for_shape(VNextExecutionWaveKind::Decode, 1, 1, 0)
        .expect("actual product decode workspace bucket");
    assert_ne!(prefill_bucket, decode_bucket);
    let input = prompt(&[0, 1, 2], 1);
    let other = prompt(&[0, 1, 2], 1);
    fixture.admit(&input);
    baseline.admit(&other);
    fixture.warm(std::slice::from_ref(&input), &[]);
    let expected = fixture.expected(std::slice::from_ref(&input), &[]);
    let gate = Gate::new(false);
    let before = fixture.submissions();
    let outputs = submitted(
        fixture
            .executor
            .plan_runtime_batch_prefill_guarded_observed(
                std::slice::from_ref(&input),
                &expected,
                &gate,
                None,
            )
            .await,
    );
    assert_eq!(gate.calls.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.submissions() - before, 1);
    fixture.assert_ready(&input, 1);
    assert_prefill_same(&outputs[0], &baseline.prefill(&other).await);
}

#[tokio::test]
async fn guarded_metal_product_prefix_capture_is_explicitly_unsupported_before_claims() {
    let fixture = Fixture::new(8, true).await;
    assert_eq!(
        fixture.executor.slo_execution_capability(),
        ferrum_interfaces::model_executor::ExecutorSloCapability::Unavailable
    );
    let input = prompt(&[0, 1], 2);
    fixture.admit(&input);
    fixture.warm(std::slice::from_ref(&input), &[]);
    let expected = fixture.expected(std::slice::from_ref(&input), &[]);
    let before = fixture.submissions();
    let gate = Gate::new(false);
    assert!(matches!(
        fixture
            .executor
            .plan_runtime_batch_prefill_guarded_observed(
                std::slice::from_ref(&input),
                &expected,
                &gate,
                None
            )
            .await,
        GuardedDispatchOutcome::Unsupported
    ));
    assert_eq!(gate.calls.load(Ordering::Relaxed), 0);
    assert_eq!(fixture.submissions(), before);
    fixture.assert_ready(&input, 0);
}
