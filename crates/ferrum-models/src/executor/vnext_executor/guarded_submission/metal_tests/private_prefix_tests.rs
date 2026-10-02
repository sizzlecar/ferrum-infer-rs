//! Actual private checkpoints with the product prefix cache disabled. The
//! fixture resolves real F32-master providers; no checkpoint owner is invented.
use super::*;
use ferrum_interfaces::model_executor::{
    PlanRuntimePrefixRestoreInput, PlanRuntimePrefixRestoreOutcome, PrefixCaptureBoundary,
    PrefixCaptureLease, PrefixCapturePurpose, PrefixCaptureRequest, PrefixCaptureStatus,
};

struct TransferGate {
    kind: NativeCheckpointTransferKind,
    source: Option<NativeCheckpointTransferIdentity>,
    allowed: bool,
    calls: AtomicUsize,
}

impl TransferGate {
    fn capture() -> Arc<Self> {
        Arc::new(Self {
            kind: NativeCheckpointTransferKind::Capture,
            source: None,
            allowed: true,
            calls: AtomicUsize::new(0),
        })
    }

    fn restore(source: NativeCheckpointTransferIdentity, allowed: bool) -> Arc<Self> {
        Arc::new(Self {
            kind: NativeCheckpointTransferKind::Restore,
            source: Some(source),
            allowed,
            calls: AtomicUsize::new(0),
        })
    }
}

impl CheckpointTransferSubmissionGuard for TransferGate {
    fn check(
        &self,
        prepared: &PreparedCheckpointTransfer<'_>,
    ) -> std::result::Result<(), GuardedNotSubmittedReason> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(prepared.identity().kind(), self.kind);
        assert!(prepared.cost_domain().geometry().copy_bytes() > 0);
        match (&self.source, prepared.source_capture_identity()) {
            (Some(expected), Some(actual)) => assert!(expected.same_transfer(actual)),
            (None, None) => {}
            _ => panic!("native guard lost its exact capture identity"),
        }
        if self.allowed {
            Ok(())
        } else {
            Err(GuardedNotSubmittedReason::HostRejected(
                HostSubmissionRejection::WitnessExpired,
            ))
        }
    }
}

struct Transfers(std::sync::Mutex<Vec<NativeCheckpointTransferObservation>>);

impl NativeCheckpointObservationSink for Transfers {
    fn try_record(&self, observation: NativeCheckpointTransferObservation) -> bool {
        let Ok(mut entries) = self.0.try_lock() else {
            return false;
        };
        if entries.len() == 3 {
            return false;
        }
        entries.push(observation);
        true
    }
}

fn observe(fixture: &Fixture) -> Arc<Transfers> {
    let sink = Arc::new(Transfers(std::sync::Mutex::new(Vec::with_capacity(3))));
    let erased: Arc<dyn NativeCheckpointObservationSink> = sink.clone();
    fixture
        .executor
        .install_checkpoint_observation_sink(Arc::downgrade(&erased))
        .unwrap();
    sink
}

fn input(maximum: usize, offset: usize, count: usize) -> PlanRuntimePrefillInput {
    PlanRuntimePrefillInput::new(
        RequestId::new(),
        vec![0, 1, 2, 1]
            .into_iter()
            .map(TokenId::new)
            .collect::<Vec<_>>(),
        maximum,
        PrefillChunk::new(offset, count, 4).unwrap(),
    )
    .unwrap()
}

fn capture_request(input: &PlanRuntimePrefillInput) -> PrefixCaptureRequest<'_> {
    PrefixCaptureRequest {
        purpose: PrefixCapturePurpose::PrivateCalibration,
        source_request_id: &input.request_id,
        source_tokens: &input.input_tokens,
        maximum_sequence_tokens: input.maximum_sequence_tokens,
        boundary: 2,
        expires_at: Instant::now() + Duration::from_secs(30),
    }
}

fn restore_input<'a>(
    target: &'a PlanRuntimePrefillInput,
    lease: Option<&'a dyn PrefixCaptureLease>,
) -> PlanRuntimePrefixRestoreInput<'a> {
    PlanRuntimePrefixRestoreInput {
        request_id: &target.request_id,
        input_tokens: &target.input_tokens,
        maximum_sequence_tokens: target.maximum_sequence_tokens,
        checkpoint: lease,
        retry: None,
    }
}

fn checkpoint_occupancy(fixture: &Fixture) -> (u64, u64) {
    fixture
        .executor
        .plan_resources
        .dynamic_pool_status()
        .unwrap()
        .pools()
        .iter()
        .map(|pool| {
            let live = pool.live_occupancy();
            let transient = live.transient().checkpoint();
            let stable = live.lane_stable().checkpoint();
            (
                transient.claim_count() + stable.claim_count(),
                transient.physical_bytes() + stable.physical_bytes(),
            )
        })
        .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
}

fn assert_shared_cache_empty(fixture: &Fixture) {
    assert!(!fixture.executor.supports_plan_runtime_prefix_restore());
    let metrics = fixture.executor.prefix_cache_metrics_snapshot();
    assert_eq!(metrics["requested"], false);
    assert_eq!(metrics["enabled"], false);
    for field in ["entries", "bytes", "hits", "misses", "saved_prefill_tokens"] {
        assert_eq!(
            metrics[field], 0,
            "private operation changed shared {field}"
        );
    }
}

async fn capture_and_retire_seed(
    fixture: &Fixture,
    seed: &PlanRuntimePrefillInput,
) -> Arc<dyn PrefixCaptureLease> {
    assert!(fixture
        .executor
        .supports_guarded_prefix_maintenance_for(PrefixCapturePurpose::PrivateCalibration));
    let boundary = fixture
        .executor
        .plan_prefix_capture_boundary_for(
            PrefixCapturePurpose::PrivateCalibration,
            PrefixCaptureBoundary {
                processed_tokens: 0,
                source_prompt_tokens: 4,
                common_prefix_tokens: 2,
                follower_prompt_tokens: &[4],
            },
        )
        .expect("resolved real providers must declare this proper boundary");
    assert_eq!(boundary.boundary, 2);
    let copies_before = fixture
        .executor
        .reaper
        .checkpoint_timing_snapshot()
        .capture
        .submitted_copies;
    fixture.admit(seed);
    drop(fixture.prefill(seed).await);
    fixture.assert_ready(seed, 2);
    assert_eq!(checkpoint_occupancy(fixture), (0, 0));
    assert_eq!(
        fixture
            .executor
            .reaper
            .checkpoint_timing_snapshot()
            .capture
            .submitted_copies,
        copies_before,
        "ordinary prefill must not capture while the product cache is disabled"
    );
    let sequence = {
        let rows = fixture.rows(std::slice::from_ref(seed), &[]);
        Arc::downgrade(&rows[0].0)
    };
    let lease = fixture
        .executor
        .retain_prefix_capture_interest(capture_request(seed))
        .unwrap()
        .expect("private interest must not require the shared cache");
    let gate = TransferGate::capture();
    assert!(fixture
        .executor
        .try_capture_plan_runtime_prefix_guarded(capture_request(seed), gate.clone())
        .await
        .unwrap());
    assert_eq!(gate.calls.load(Ordering::Relaxed), 1);
    assert_eq!(lease.purpose(), PrefixCapturePurpose::PrivateCalibration);
    assert_eq!(lease.status(), PrefixCaptureStatus::Ready);
    let retained = checkpoint_occupancy(fixture);
    assert!(retained.0 > 0 && retained.1 > 0);
    assert!(fixture.executor.cancel_prefill_admission(&seed.request_id));
    assert!(
        sequence.upgrade().is_none(),
        "lease must not retain the seed owner"
    );
    assert_eq!(fixture.executor.reaper.retained_count(), 0);
    assert_eq!(lease.status(), PrefixCaptureStatus::Ready);
    assert_eq!(checkpoint_occupancy(fixture), retained);
    assert_shared_cache_empty(fixture);
    lease
}

#[tokio::test]
async fn private_checkpoint_survives_seed_retirement_and_restores_without_shared_cache() {
    let fixture = Fixture::private_calibration(8).await;
    let cold = Fixture::private_calibration(8).await;
    let sink = observe(&fixture);
    let seed = input(8, 0, 2);
    let lease = capture_and_retire_seed(&fixture, &seed).await;
    let source = {
        let observations = sink.0.lock().unwrap();
        assert_eq!(observations.len(), 1, "capture must publish its native ACK");
        assert_eq!(
            observations[0].kind(),
            NativeCheckpointTransferKind::Capture
        );
        observations[0].identity().clone()
    };
    let target = input(12, 0, 2);
    let baseline = input(12, 0, 2);
    assert_ne!(seed.maximum_sequence_tokens, target.maximum_sequence_tokens);
    fixture.admit(&target);
    cold.admit(&baseline);
    drop(cold.prefill(&baseline).await);
    let gate = TransferGate::restore(source, true);
    let restored = fixture
        .executor
        .try_restore_plan_runtime_prefix_guarded(
            restore_input(&target, Some(lease.as_ref())),
            gate.clone(),
        )
        .await
        .unwrap();
    let PlanRuntimePrefixRestoreOutcome::Restored(restored) = restored else {
        panic!("real private checkpoint must restore into its fresh admitted target");
    };
    assert_eq!(gate.calls.load(Ordering::Relaxed), 1);
    assert_eq!(restored.restored_tokens(), 2);
    fixture.assert_ready(&target, 0);
    assert_eq!(
        sink.0.lock().unwrap().len(),
        1,
        "copy completion is not restore ACK"
    );
    drop(restored.acknowledge().unwrap());
    fixture.assert_ready(&target, 2);
    {
        let observations = sink.0.lock().unwrap();
        assert_eq!(observations.len(), 2);
        assert_eq!(
            observations[1].kind(),
            NativeCheckpointTransferKind::Restore
        );
        assert!(observations[0]
            .identity()
            .same_transfer(observations[1].source_capture_identity().unwrap()));
    }
    let suffix = PlanRuntimePrefillInput {
        chunk: PrefillChunk::new(2, 2, 4).unwrap(),
        ..target
    };
    let cold_suffix = PlanRuntimePrefillInput {
        chunk: suffix.chunk,
        ..baseline
    };
    let actual = fixture.prefill(&suffix).await;
    let expected = cold.prefill(&cold_suffix).await;
    assert_prefill_same(&actual, &expected);
    assert_shared_cache_empty(&fixture);
    fixture
        .executor
        .release_cache(&actual.output().kv_cache().cache_id());
    cold.executor
        .release_cache(&expected.output().kv_cache().cache_id());
    drop(actual);
    drop(expected);
    assert!(checkpoint_occupancy(&fixture).1 > 0);
    drop(lease);
    assert_eq!(checkpoint_occupancy(&fixture), (0, 0));
    assert_eq!(fixture.executor.reaper.retained_count(), 0);
    assert_shared_cache_empty(&fixture);
    // Pool backing may remain resident. A new actual capture proves the
    // released checkpoint charge/extents can be used by the same executor.
    let reused = capture_and_retire_seed(&fixture, &input(8, 0, 2)).await;
    assert_eq!(sink.0.lock().unwrap().len(), 3);
    drop(reused);
    assert_eq!(checkpoint_occupancy(&fixture), (0, 0));
}

#[tokio::test]
async fn private_checkpoint_requires_its_purpose_exact_lease_and_native_guard() {
    let fixture = Fixture::private_calibration(8).await;
    let sink = observe(&fixture);
    let seed = input(8, 0, 2);
    let lease = capture_and_retire_seed(&fixture, &seed).await;
    let source = sink.0.lock().unwrap()[0].identity().clone();
    let target = input(12, 0, 2);
    fixture.admit(&target);
    let before = fixture.executor.reaper.checkpoint_timing_snapshot();
    let retained = checkpoint_occupancy(&fixture);
    let allowed = TransferGate::restore(source.clone(), true);
    assert!(matches!(
        fixture
            .executor
            .try_restore_plan_runtime_prefix_guarded(restore_input(&target, None), allowed.clone())
            .await
            .unwrap(),
        PlanRuntimePrefixRestoreOutcome::Unavailable
    ));
    assert_eq!(allowed.calls.load(Ordering::Relaxed), 0);
    assert!(matches!(
        fixture
            .executor
            .try_restore_plan_runtime_prefix(restore_input(&target, Some(lease.as_ref())))
            .await
            .unwrap(),
        PlanRuntimePrefixRestoreOutcome::Unavailable
    ));
    let rejected = TransferGate::restore(source, false);
    assert!(matches!(
        fixture
            .executor
            .try_restore_plan_runtime_prefix_guarded(
                restore_input(&target, Some(lease.as_ref())),
                rejected.clone()
            )
            .await
            .unwrap(),
        PlanRuntimePrefixRestoreOutcome::Unavailable
    ));
    assert_eq!(rejected.calls.load(Ordering::Relaxed), 1);
    fixture.assert_ready(&target, 0);
    drop(fixture.prefill(&target).await);
    let mut wrong_purpose = capture_request(&target);
    wrong_purpose.purpose = PrefixCapturePurpose::SharedCache;
    assert!(fixture
        .executor
        .retain_prefix_capture_interest(wrong_purpose)
        .unwrap()
        .is_none());
    let capture_gate = TransferGate::capture();
    assert!(!fixture
        .executor
        .try_capture_plan_runtime_prefix_guarded(wrong_purpose, capture_gate.clone())
        .await
        .unwrap());
    assert_eq!(capture_gate.calls.load(Ordering::Relaxed), 0);
    let after = fixture.executor.reaper.checkpoint_timing_snapshot();
    assert_eq!(
        after.capture.submitted_copies,
        before.capture.submitted_copies
    );
    assert_eq!(
        after.restore.submitted_copies,
        before.restore.submitted_copies
    );
    assert_eq!(
        after.capture.indeterminate_copies,
        before.capture.indeterminate_copies
    );
    assert_eq!(
        after.restore.indeterminate_copies,
        before.restore.indeterminate_copies
    );
    assert_eq!(sink.0.lock().unwrap().len(), 1);
    assert_eq!(checkpoint_occupancy(&fixture), retained);
    assert!(fixture
        .executor
        .cancel_prefill_admission(&target.request_id));
    drop(lease);
    assert_eq!(checkpoint_occupancy(&fixture), (0, 0));
    assert_shared_cache_empty(&fixture);
}

#[tokio::test]
async fn unsupported_private_checkpoint_keeps_automatic_cold_inference_available() {
    let fixture = Fixture::automatic_without_native_checkpoint(8).await;
    assert!(!fixture
        .executor
        .supports_guarded_prefix_maintenance_for(PrefixCapturePurpose::PrivateCalibration));
    assert!(fixture
        .executor
        .plan_prefix_capture_boundary_for(
            PrefixCapturePurpose::PrivateCalibration,
            PrefixCaptureBoundary {
                processed_tokens: 0,
                source_prompt_tokens: 4,
                common_prefix_tokens: 2,
                follower_prompt_tokens: &[4],
            },
        )
        .is_none());
    assert_eq!(
        fixture.executor.slo_execution_capability(),
        ferrum_interfaces::model_executor::ExecutorSloCapability::GuardedEagerWaves
    );
    let request = input(8, 0, 2);
    fixture.admit(&request);
    fixture.warm(std::slice::from_ref(&request), &[]);
    let expected = completion_work_tests::complete(&fixture, std::slice::from_ref(&request), &[]);
    let gate = Gate::new(false);
    let before = fixture.submissions();
    let output = submitted(
        fixture
            .executor
            .plan_runtime_batch_prefill_guarded_work_observed(
                std::slice::from_ref(&request),
                &expected,
                &gate,
                None,
            )
            .await,
    );
    assert_eq!(gate.calls.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.submissions() - before, 1);
    fixture.assert_ready(&request, 2);
    drop(output);
    assert!(fixture
        .executor
        .cancel_prefill_admission(&request.request_id));
    assert_eq!(checkpoint_occupancy(&fixture), (0, 0));
    assert_shared_cache_empty(&fixture);
}
