//! Exercises the production registry reader over real core admission/session
//! authorities. This fixture does not claim language-model numeric execution.
use super::*;

#[path = "../../../../../ferrum-interfaces/tests/vnext_device_operation_contract/mod.rs"]
mod contract;
use contract::{fixture, logical_resources, one_token_work, Fixture, TestRuntime};

struct RegistryHarness {
    fixture: Fixture,
    registry: Mutex<VNextSequenceRegistry<TestRuntime>>,
    request: RequestId,
}

impl RegistryHarness {
    fn new() -> Self {
        let fixture = fixture();
        assert!(fixture.reusable_execution_bucket.is_none());
        let resources = logical_resources(
            &fixture.plan_resources,
            "run.resource-planning",
            "request.resource-planning",
        );
        let session = resources.open_session().unwrap();
        let request = RequestId::new();
        let root =
            VNextRequestRoot::bind_initial(request.clone(), resources.request_id(), &session)
                .unwrap();
        let active_binding =
            Arc::new(TrustedActiveSequenceBinding::from_session(&session).unwrap());
        let sequence = Arc::new(VNextSequence {
            prefix_capture_interests: Mutex::new(Vec::new()),
            cache_id: "resource-planning-cache".into(),
            request: root,
            session,
            active_binding,
            request_origin: ExecutorRequestOrigin::Product,
            tokens: Mutex::new(vec![1]),
            maximum_tokens: 1,
            active: AtomicBool::new(true),
            operation: AsyncMutex::new(()),
            events: None,
            product_prompt_tokens: 1,
            replayed_output_tokens: 0,
            prefill_tokens_processed: AtomicUsize::new(0),
        });
        let mut registry = VNextSequenceRegistry::default();
        let slot = registry
            .begin_prefill_probe(&request, &one_token_work())
            .unwrap();
        // Same publication state as publish_prefill_probe(Ready), backed by the
        // actual admitted request and sequence above, without fake executor I/O.
        *slot.state.lock() = VNextPrefillSlotState::Ready(sequence);
        Self {
            fixture,
            registry: Mutex::new(registry),
            request,
        }
    }

    fn capture(
        &self,
        request: &RequestId,
        cache_id: Option<&str>,
    ) -> ResourcePlanningAvailability<ResourcePlanningView> {
        match capture_registry_view(
            &self.registry,
            &self.fixture.plan_resources,
            None,
            &[ExecutorResourcePlanningRequest {
                request_id: request,
                cache_id,
            }],
            ResourcePlanningLimits::default(),
            &mut || true,
        ) {
            Ok(value) => value,
            Err(reason) => ResourcePlanningAvailability::Unknown(reason),
        }
    }

    fn ready_sequence(&self) -> Arc<VNextSequence<TestRuntime>> {
        let registry = self.registry.lock();
        let slot = registry.prefills.get(&self.request).unwrap();
        let state = slot.state.lock();
        match &*state {
            VNextPrefillSlotState::Ready(sequence) => Arc::clone(sequence),
            _ => panic!("expected the actual Ready registry entry"),
        }
    }

    fn close(self) {
        drop(self.registry);
        drop(self.fixture.registry);
        drop(self.fixture.impostor_registry);
        drop(self.fixture.runtime);
        assert!(matches!(
            PlanRuntimeResources::close(self.fixture.plan_resources),
            Ok(PlanRuntimeCloseOutcome::Closed(_))
        ));
    }
}

fn known(
    mut capture: impl FnMut() -> ResourcePlanningAvailability<ResourcePlanningView>,
) -> ResourcePlanningView {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut seen = BTreeSet::new();
    loop {
        match capture() {
            ResourcePlanningAvailability::Known(view) => return view,
            ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::ReadUnavailable(
                stage,
            )) => {
                if seen.insert(stage) {
                    eprintln!(
                        "{}: resource-planning positive read contention: {stage:?}",
                        std::thread::current().name().unwrap_or("unnamed test")
                    );
                }
                assert!(
                    Instant::now() < deadline,
                    "resource read did not become available: {seen:?}"
                );
                std::thread::yield_now();
            }
            ResourcePlanningAvailability::Unknown(reason) => {
                panic!("expected admitted numeric view: {reason:?}")
            }
        }
    }
}

fn unknown(
    mut capture: impl FnMut() -> ResourcePlanningAvailability<ResourcePlanningView>,
    expected: ResourcePlanningUnknown,
) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match capture() {
            ResourcePlanningAvailability::Unknown(reason) if reason == expected => return,
            ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::ReadUnavailable(
                stage,
            )) => {
                assert!(
                    Instant::now() < deadline,
                    "resource read remained contended at {stage:?}, expected {expected:?}"
                );
                std::thread::yield_now();
            }
            other => panic!("expected {expected:?}, observed {other:?}"),
        }
    }
}

#[test]
fn resource_planning_adapter_maps_ready_prefill_then_exact_activated_cache() {
    let harness = RegistryHarness::new();
    let before = {
        let trace = harness.fixture.runtime_trace.lock().unwrap();
        (trace.allocation_calls, trace.submit_calls)
    };
    let prefill = known(|| harness.capture(&harness.request, None));
    let (slot, sequence) = harness
        .registry
        .lock()
        .begin_prefill_execution(&harness.request)
        .unwrap();
    assert_eq!(
        prefill.participants()[0].authority(),
        sequence.session.sequence_authority()
    );
    unknown(
        || harness.capture(&harness.request, None),
        ResourcePlanningUnknown::BusyOrUnavailable,
    );
    harness
        .registry
        .lock()
        .restore_prefill_ready(&slot, &sequence)
        .unwrap();
    assert_eq!(
        prefill.participants(),
        known(|| harness.capture(&harness.request, None)).participants()
    );
    drop(slot);
    drop(sequence);
    let (slot, sequence) = harness
        .registry
        .lock()
        .begin_prefill_execution(&harness.request)
        .unwrap();
    harness.registry.lock().activate(&slot, &sequence).unwrap();
    let decode = known(|| harness.capture(&harness.request, Some(&sequence.cache_id)));
    assert_eq!(
        decode.participants()[0].authority(),
        prefill.participants()[0].authority()
    );
    unknown(
        || harness.capture(&harness.request, None),
        ResourcePlanningUnknown::BusyOrUnavailable,
    );
    let after = {
        let trace = harness.fixture.runtime_trace.lock().unwrap();
        (trace.allocation_calls, trace.submit_calls)
    };
    assert_eq!(before, after);
    drop(slot);
    drop(sequence);
    harness.close();
    assert_eq!(decode.participants().len(), 1);
}

#[test]
fn resource_planning_adapter_rejects_wrong_cache_owner_and_request_replay() {
    let harness = RegistryHarness::new();
    let (slot, sequence) = harness
        .registry
        .lock()
        .begin_prefill_execution(&harness.request)
        .unwrap();
    harness.registry.lock().activate(&slot, &sequence).unwrap();
    unknown(
        || harness.capture(&harness.request, Some("foreign-cache")),
        ResourcePlanningUnknown::StaleIdentity,
    );
    unknown(
        || harness.capture(&RequestId::new(), Some(&sequence.cache_id)),
        ResourcePlanningUnknown::StaleIdentity,
    );
    let request = ExecutorResourcePlanningRequest {
        request_id: &harness.request,
        cache_id: Some(&sequence.cache_id),
    };
    assert!(matches!(
        capture_registry_view(
            &harness.registry,
            &harness.fixture.plan_resources,
            None,
            &[request, request],
            ResourcePlanningLimits::default(),
            &mut || true
        ),
        Err(ResourcePlanningUnknown::InvalidInput)
    ));
    drop(slot);
    drop(sequence);
    harness.close();
}

#[test]
fn resource_planning_adapter_busy_operation_and_real_frame_are_unknown() {
    let harness = RegistryHarness::new();
    let sequence = harness.ready_sequence();
    let operation = sequence.operation.try_lock().unwrap();
    unknown(
        || harness.capture(&harness.request, None),
        ResourcePlanningUnknown::ReadUnavailable(ResourcePlanningReadStage::ModelSequenceOperation),
    );
    drop(operation);
    let before = known(|| harness.capture(&harness.request, None));
    let registry_guard = harness.registry.lock();
    unknown(
        || harness.capture(&harness.request, None),
        ResourcePlanningUnknown::ReadUnavailable(ResourcePlanningReadStage::ModelRegistry),
    );
    drop(registry_guard);
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&sequence.session)]).unwrap();
    let lane = harness
        .fixture
        .plan_resources
        .create_execution_lane()
        .unwrap();
    let step = contract::begin_single_participant_step_on_lane_with_bucket(&batch, &lane, None);
    unknown(
        || harness.capture(&harness.request, None),
        ResourcePlanningUnknown::BusyOrUnavailable,
    );
    step.try_retire_normal().unwrap();
    let after = known(|| harness.capture(&harness.request, None));
    assert!(!before.same_live_evidence(&after));
    assert_eq!(
        harness.fixture.runtime_trace.lock().unwrap().submit_calls,
        0
    );
    drop(batch);
    drop(lane);
    drop(sequence);
    harness.close();
}

#[test]
fn resource_planning_adapter_cancelled_slot_and_retained_view_do_not_pin_resources() {
    let harness = RegistryHarness::new();
    let sequence = harness.ready_sequence();
    let weak = Arc::downgrade(&sequence);
    let view = known(|| harness.capture(&harness.request, None));
    let state = view.initial_state();
    harness.registry.lock().cancel_prefill(&harness.request);
    unknown(
        || harness.capture(&harness.request, None),
        ResourcePlanningUnknown::BusyOrUnavailable,
    );
    assert!(!sequence.active.load(Ordering::Acquire));
    drop(sequence);
    assert!(weak.upgrade().is_none());
    harness.close();
    assert_eq!(state.projected_waves(), 0);
    assert_eq!(view.participants().len(), 1);
}
