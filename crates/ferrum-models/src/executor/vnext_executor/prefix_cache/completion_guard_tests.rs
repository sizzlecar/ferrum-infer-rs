//! Completion ownership uses actual admitted CPU sessions; these tests do not
//! claim native checkpoint-copy or numerical coverage.
use super::*;
use std::future::{pending, Future};
use std::task::{Context, Waker};

#[path = "../../../../../ferrum-interfaces/tests/vnext_resource_contract/support.rs"]
mod resource_support;
use resource_support::{admit_logical_sequence, close_plan_runtime, TestRuntime};

fn completion_sequence(
    suffix: &str,
) -> (
    checkpoint_fixture::Fixture,
    Arc<PlanRuntimeResources<TestRuntime>>,
    Arc<VNextSequence<TestRuntime>>,
) {
    let fixture = checkpoint_fixture::Fixture::build(checkpoint_fixture::Spec::default()).unwrap();
    let (driver, _) = resource_support::configured_driver(&fixture.plan, &[], &[]);
    let root = resource_support::plan_runtime(&fixture.plan, driver, suffix);
    let session = admit_logical_sequence(&root, suffix, suffix)
        .open_session()
        .unwrap();
    let request = VNextRequestRoot::bind_initial(
        RequestId::new(),
        session.resources().request_id(),
        &session,
    )
    .unwrap();
    let active_binding = Arc::new(TrustedActiveSequenceBinding::from_session(&session).unwrap());
    let sequence = Arc::new(VNextSequence {
        prefix_capture_interests: Mutex::new(Vec::new()),
        cache_id: suffix.to_owned(),
        request,
        session,
        active_binding,
        request_origin: ExecutorRequestOrigin::Product,
        tokens: Mutex::new(vec![1]),
        pending_decode: Mutex::new(None),
        maximum_tokens: 1,
        active: AtomicBool::new(true),
        operation: AsyncMutex::new(()),
        events: None,
        product_prompt_tokens: 1,
        replayed_output_tokens: 0,
        prefill_tokens_processed: AtomicUsize::new(0),
    });
    (fixture, root, sequence)
}

fn assert_aborted(sequence: &VNextSequence<TestRuntime>) {
    assert!(!sequence.active.load(Ordering::Acquire));
    assert!(sequence
        .session
        .write_release_capacity_sources(&mut Vec::new())
        .is_err());
    assert!(sequence.session.try_complete().is_err());
}

#[test]
fn active_prefix_snapshot_uses_original_inputs_and_filters_terminal_incarnations() {
    let (_current_fixture, current_root, current) = completion_sequence("coverage-current");
    let (_peer_fixture, peer_root, mut peer) = completion_sequence("coverage-peer");
    // Same public request id is insufficient to exclude another incarnation.
    let request = VNextRequestRoot::bind_initial(
        current.request_id().clone(),
        peer.session.resources().request_id(),
        &peer.session,
    )
    .unwrap();
    Arc::get_mut(&mut peer).unwrap().request = request;
    *peer.tokens.lock() = vec![2, 90, 91];
    let registry = Mutex::new(VNextSequenceRegistry::default());
    registry
        .lock()
        .active
        .insert(current.cache_id.clone(), Arc::clone(&current));
    let slot = VNextPrefillSlot::new(
        peer.request_id().clone(),
        ResourceWorkShape::single(resource_support::one_token_span()).unwrap(),
    );
    registry
        .lock()
        .prefills
        .insert(peer.request_id().clone(), Arc::clone(&slot));
    assert!(active_prefix_inputs(&registry, &current).is_empty()); // Probing
    *slot.state.lock() = VNextPrefillSlotState::Ready(Arc::clone(&peer));
    let peer_owners = Arc::strong_count(&peer);
    let input = active_prefix_inputs(&registry, &current);
    assert_eq!(input.len(), 1);
    assert_eq!(input[0].tokens, vec![2]); // Executed generation is not the prompt.
    assert_eq!(Arc::strong_count(&peer), peer_owners);
    *slot.state.lock() = VNextPrefillSlotState::Executing(Arc::clone(&peer));
    assert_eq!(active_prefix_inputs(&registry, &current).len(), 1);
    slot.cancelled.store(true, Ordering::Release);
    assert!(active_prefix_inputs(&registry, &current).is_empty());
    slot.cancelled.store(false, Ordering::Release);
    *slot.state.lock() = VNextPrefillSlotState::Terminal;
    assert!(active_prefix_inputs(&registry, &current).is_empty());
    registry
        .lock()
        .active
        .insert(peer.cache_id.clone(), Arc::clone(&peer));
    assert_eq!(active_prefix_inputs(&registry, &current).len(), 1);
    peer.active.store(false, Ordering::Release);
    assert!(active_prefix_inputs(&registry, &current).is_empty());
    registry.lock().active.remove(&peer.cache_id);
    Arc::get_mut(&mut peer).unwrap().request_origin = ExecutorRequestOrigin::Diagnostic;
    peer.active.store(true, Ordering::Release);
    registry
        .lock()
        .active
        .insert(peer.cache_id.clone(), Arc::clone(&peer));
    assert!(active_prefix_inputs(&registry, &current).is_empty());
    drop(registry);
    drop(slot);
    current.abort();
    peer.abort();
    drop(current);
    drop(peer);
    close_plan_runtime(current_root);
    close_plan_runtime(peer_root);
    // The retained result contains no sequence/slot/checkpoint owner.
    assert_eq!(input[0].tokens, vec![2]);
}

#[tokio::test]
async fn dropping_completion_waiting_for_operation_cancels_a_still_retained_sequence() {
    let (_fixture, root, sequence) = completion_sequence("completion-waiting");
    let retained = Arc::clone(&sequence);
    let operation = sequence.operation.lock().await;
    let mut completion = Box::pin(async {
        let mut guard = PendingSequenceCompletion {
            sequence: &sequence,
            operation: None,
            completed: false,
        };
        guard.operation = Some(sequence.operation.lock().await);
        pending::<()>().await;
    });
    assert!(completion
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
        .is_pending());
    assert!(sequence.active.load(Ordering::Acquire));
    drop(completion);
    assert_aborted(&retained);
    drop(operation);
    assert!(retained.operation.try_lock().is_ok());
    drop(retained);
    drop(sequence);
    close_plan_runtime(root);
}

#[tokio::test]
async fn dropping_completion_after_operation_acquisition_cancels_and_unlocks() {
    let (_fixture, root, sequence) = completion_sequence("completion-capturing");
    let retained = Arc::clone(&sequence);
    let mut completion = Box::pin(async {
        let mut guard = PendingSequenceCompletion {
            sequence: &sequence,
            operation: None,
            completed: false,
        };
        guard.operation = Some(sequence.operation.lock().await);
        pending::<()>().await;
    });
    assert!(completion
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
        .is_pending());
    assert!(retained.operation.try_lock().is_err());
    drop(completion);
    assert_aborted(&retained);
    assert!(retained.operation.try_lock().is_ok());
    drop(retained);
    drop(sequence);
    close_plan_runtime(root);
}

#[tokio::test]
async fn completed_native_session_disarms_completion_ownership() {
    let (_fixture, root, sequence) = completion_sequence("completion-success");
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&sequence.session)]).unwrap();
    let lane = root.create_execution_lane().unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![resource_support::one_token_span()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let StepResourceAdmissionDecision::Admitted(step) =
        batch.try_begin_step(request, &lane).unwrap()
    else {
        panic!("initialized CPU pools must admit this one-token step");
    };
    // A normally retired frame is necessary for real session completion. This
    // exercises lifecycle authority, without manufacturing a FullPlan seal.
    step.try_retire_normal().unwrap();
    let completion = ExecutorSequenceCompletion::new(
        sequence.request_id().clone(),
        sequence.cache_id.clone(),
        1,
        1,
    )
    .unwrap();
    let mut guard = PendingSequenceCompletion {
        sequence: &sequence,
        operation: Some(sequence.operation.lock().await),
        completed: false,
    };
    sequence.complete(&completion).await.unwrap();
    guard.completed = true;
    drop(guard);
    assert!(!sequence.active.load(Ordering::Acquire));
    assert!(sequence.operation.try_lock().is_ok());
    assert!(sequence.session.request_cancel().is_err());
    drop(batch);
    drop(lane);
    drop(sequence);
    close_plan_runtime(root);
}
