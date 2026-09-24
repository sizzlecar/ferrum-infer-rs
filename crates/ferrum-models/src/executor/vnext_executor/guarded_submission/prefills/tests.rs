//! Real admitted sessions and the production Ready/Executing/RAII transitions.
//! These CPU tests establish lifetime safety, not language-model numeric parity.
use super::*;
#[path = "../../../../../../ferrum-interfaces/tests/vnext_device_operation_contract/mod.rs"]
mod contract;
use contract::{fixture, logical_resources, one_token_work, Fixture, TestRuntime};

struct Harness {
    fixture: Fixture,
    registry: Mutex<VNextSequenceRegistry<TestRuntime>>,
    requests: Vec<RequestId>,
}
impl Harness {
    fn new() -> Self {
        let fixture = fixture();
        let mut registry = VNextSequenceRegistry::default();
        let mut requests = Vec::new();
        for index in 0..2 {
            let resources = logical_resources(
                &fixture.plan_resources,
                "run.guarded-prefill",
                &format!("request.guarded-prefill.{index}"),
            );
            let session = resources.open_session().unwrap();
            let request = RequestId::new();
            let root =
                VNextRequestRoot::bind_initial(request.clone(), resources.request_id(), &session)
                    .unwrap();
            let binding = Arc::new(TrustedActiveSequenceBinding::from_session(&session).unwrap());
            let sequence = Arc::new(VNextSequence {
                prefix_capture_interests: Mutex::new(Vec::new()),
                cache_id: format!("guarded.{index}"),
                request: root,
                session,
                active_binding: binding,
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
            let slot = registry
                .begin_prefill_probe(&request, &one_token_work())
                .unwrap();
            *slot.state.lock() = VNextPrefillSlotState::Ready(sequence);
            requests.push(request);
        }
        Self {
            fixture,
            registry: Mutex::new(registry),
            requests,
        }
    }
    fn begin(&self) -> Vec<VNextPrefillCandidate<TestRuntime>> {
        self.registry
            .lock()
            .begin_prefill_batch_execution(&self.requests)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(original_index, (slot, sequence))| VNextPrefillCandidate {
                original_index,
                slot,
                sequence,
                tokens: vec![1],
                maximum_tokens: 1,
                planned_chunk: PrefillChunk::new(0, 1, 1).unwrap(),
            })
            .collect()
    }
    fn guards<'a>(
        &'a self,
        rows: &[VNextPrefillCandidate<TestRuntime>],
    ) -> Vec<VNextPrefillExecutionGuard<'a, TestRuntime>> {
        rows.iter()
            .map(|row| {
                VNextPrefillExecutionGuard::new(
                    &self.registry,
                    Arc::clone(&row.slot),
                    Arc::clone(&row.sequence),
                )
            })
            .collect()
    }
    fn close(self) {
        assert_eq!(self.fixture.runtime_trace.lock().unwrap().submit_calls, 0);
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

#[test]
fn guarded_prefill_reconciled_batch_restores_same_authorities_for_retry() {
    let harness = Harness::new();
    let first = harness.begin();
    let mut guards = harness.guards(&first);
    restore_prefill_execution_batch(&harness.registry, &first, &mut guards).unwrap();
    drop(guards);
    let second = harness.begin();
    for (before, after) in first.iter().zip(&second) {
        assert!(Arc::ptr_eq(&before.slot, &after.slot));
        assert!(Arc::ptr_eq(&before.sequence, &after.sequence));
        assert!(after.sequence.active.load(Ordering::Acquire));
        assert_eq!(
            after
                .sequence
                .prefill_tokens_processed
                .load(Ordering::Acquire),
            0
        );
    }
    drop(harness.guards(&second));
    assert!(harness.registry.lock().prefills.is_empty());
    drop(second);
    drop(first);
    harness.close();
}

#[test]
fn guarded_prefill_cancelled_row_never_partially_reopens_its_peers() {
    let harness = Harness::new();
    let rows = harness.begin();
    let mut guards = harness.guards(&rows);
    assert!(harness.registry.lock().cancel_prefill(&harness.requests[1]));
    assert!(restore_prefill_execution_batch(&harness.registry, &rows, &mut guards).is_err());
    assert!(matches!(
        &*rows[0].slot.state.lock(),
        VNextPrefillSlotState::Executing(_)
    ));
    drop(guards);
    assert!(harness.registry.lock().prefills.is_empty());
    assert!(rows
        .iter()
        .all(|row| !row.sequence.active.load(Ordering::Acquire)));
    drop(rows);
    harness.close();
}

#[test]
fn guarded_prefill_partial_and_final_products_keep_distinct_registry_ownership() {
    let harness = Harness::new();
    let rows = harness.begin();
    let mut guards = harness.guards(&rows);
    harness
        .registry
        .lock()
        .commit_prefill_batch_execution(&[
            (&rows[0].slot, &rows[0].sequence, false),
            (&rows[1].slot, &rows[1].sequence, true),
        ])
        .unwrap();
    for guard in &mut guards {
        guard.disarm();
    }
    drop(guards);
    let registry = harness.registry.lock();
    assert!(matches!(
        &*rows[0].slot.state.lock(),
        VNextPrefillSlotState::Ready(_)
    ));
    assert!(registry.prefills.contains_key(&harness.requests[0]));
    assert!(!registry.prefills.contains_key(&harness.requests[1]));
    assert!(Arc::ptr_eq(
        registry.active.get(&rows[1].sequence.cache_id).unwrap(),
        &rows[1].sequence
    ));
    drop(registry);
    drop(rows);
    harness.close();
}

#[tokio::test]
async fn guarded_prefill_dropped_waiting_future_cleans_exact_executing_slots() {
    let harness = Harness::new();
    let entered = tokio::sync::Notify::new();
    let future = async {
        let rows = harness.begin();
        let _guards = harness.guards(&rows);
        entered.notify_one();
        std::future::pending::<()>().await;
    };
    let mut future = Box::pin(future);
    tokio::select! { _ = entered.notified() => {}, _ = &mut future => panic!("parked work completed") }
    drop(future);
    assert!(harness.registry.lock().prefills.is_empty());
    harness.close();
}
