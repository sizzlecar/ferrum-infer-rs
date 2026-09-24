use super::*;

#[path = "../../../../../ferrum-interfaces/tests/vnext_device_operation_contract/mod.rs"]
mod contract;

struct Harness {
    fixture: Option<contract::Fixture>,
    lane: Option<Arc<ExecutionLane<contract::TestRuntime>>>,
    registry: Mutex<VNextSequenceRegistry<contract::TestRuntime>>,
    worker: VNextCompletionWorker,
    residency: Mutex<VNextProductTokenMaskResidency>,
}
impl Harness {
    fn new() -> Self {
        let fixture = contract::fixture();
        let lane = fixture.plan_resources.create_execution_lane().unwrap();
        Self {
            fixture: Some(fixture),
            lane: Some(lane),
            registry: Mutex::new(VNextSequenceRegistry::default()),
            worker: VNextCompletionWorker::new().unwrap(),
            residency: Mutex::new(VNextProductTokenMaskResidency::default()),
        }
    }
    fn invalidate(&self) -> Outcome {
        invalidate(
            &self.registry,
            &self.worker,
            self.lane.as_ref().unwrap(),
            &self.residency,
        )
    }
}
impl Drop for Harness {
    fn drop(&mut self) {
        drop(std::mem::take(self.registry.get_mut()));
        self.lane.take();
        let fixture = self.fixture.take().unwrap();
        drop(fixture.registry);
        drop(fixture.impostor_registry);
        drop(fixture.runtime);
        assert!(matches!(
            PlanRuntimeResources::close(fixture.plan_resources),
            Ok(PlanRuntimeCloseOutcome::Closed(_))
        ));
    }
}

#[test]
fn calibration_token_policy_invalidation_clears_real_ledger_and_next_prepare_requires_upload() {
    let harness = Harness::new();
    let selection = VNextProductTokenMaskContent::from_policy(
        Some(&LogitsReturnPolicy::GreedyArgmax {
            token_mask: Some(ferrum_interfaces::model_executor::TokenSelectionMask::new(
                vec![1, 0, 1],
            )),
            repetition_penalty: None,
        }),
        VNextProductOutputMode::GreedyToken,
        3,
    );
    let contents = [
        selection,
        VNextProductTokenMaskContent::AllValid { vocabulary_size: 3 },
    ];
    let targets = (0..2)
        .map(|participant_index| VNextProductTokenMaskSlotTarget {
            identity: VNextProductTokenMaskSlotIdentity::Test(17),
            participant_index,
        })
        .collect::<Vec<_>>();
    for (target, content) in targets.iter().zip(&contents) {
        let mut cache = harness.residency.lock();
        let prepared = cache.prepare(Some(target.clone()), content.clone());
        assert!(prepared.upload_required);
        cache.publish(std::slice::from_ref(&prepared));
        assert!(
            !cache
                .prepare(Some(target.clone()), content.clone())
                .upload_required
        );
    }
    let metrics = harness.worker.metrics_snapshot();
    assert_eq!(
        harness.invalidate(),
        Outcome::Cleared { cleared_entries: 2 }
    );
    assert_eq!(harness.worker.metrics_snapshot(), metrics);
    assert_eq!(
        harness.invalidate(),
        Outcome::Cleared { cleared_entries: 0 }
    );
    for (target, content) in targets.iter().zip(&contents) {
        let mut cache = harness.residency.lock();
        let prepared = cache.prepare(Some(target.clone()), content.clone());
        assert!(prepared.upload_required);
        // Preparing alone cannot restore residency; only successful publication can.
        assert!(
            cache
                .prepare(Some(target.clone()), content.clone())
                .upload_required
        );
        cache.publish(std::slice::from_ref(&prepared));
        assert!(
            !cache
                .prepare(Some(target.clone()), content.clone())
                .upload_required
        );
    }
}

#[tokio::test]
async fn calibration_token_policy_invalidation_rejects_reserved_completion_and_registry_work() {
    let harness = Harness::new();
    let reservation = harness.worker.reserve().await.unwrap();
    assert_eq!(harness.invalidate(), unavailable(Busy::CompletionWork));
    drop(reservation);
    let residency = harness.residency.lock();
    assert_eq!(harness.invalidate(), unavailable(Busy::ResidencyBusy));
    drop(residency);
    let lock = harness.registry.lock();
    assert_eq!(harness.invalidate(), unavailable(Busy::ExecutorStateBusy));
    drop(lock);
    let request = RequestId::new();
    harness
        .registry
        .lock()
        .begin_prefill_probe(&request, &contract::one_token_work())
        .unwrap();
    assert_eq!(harness.invalidate(), unavailable(Busy::ActiveRequests));
    // This is an actual registry probe, not an allocated or submitted wave.
    assert_eq!(harness.worker.metrics_snapshot()["scheduled_tasks"], 0);
}
