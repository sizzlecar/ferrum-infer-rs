//! Exercise Core's real guarded copy batches on the selected native backend.
use super::*;
use ferrum_interfaces::execution_cost::{GuardedNotSubmittedReason, HostSubmissionRejection};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Gate {
    kind: NativeCheckpointTransferKind,
    allowed: AtomicBool,
    calls: AtomicUsize,
}

impl Gate {
    fn new(kind: NativeCheckpointTransferKind) -> Self {
        Self {
            kind,
            allowed: AtomicBool::new(false),
            calls: AtomicUsize::new(0),
        }
    }
}

impl CheckpointTransferSubmissionGuard for Gate {
    fn check(
        &self,
        actual: &PreparedCheckpointTransfer<'_>,
    ) -> Result<(), GuardedNotSubmittedReason> {
        assert_eq!(actual.identity().kind(), self.kind);
        assert_eq!(actual.cost_domain().kind(), self.kind);
        assert!(actual.cost_domain().geometry().copy_bytes() > 0);
        assert_eq!(
            actual.source_capture_identity().is_some(),
            self.kind == NativeCheckpointTransferKind::Restore
        );
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.allowed.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(GuardedNotSubmittedReason::HostRejected(
                HostSubmissionRejection::Cancelled,
            ))
        }
    }
}

fn assert_rejected(start: NativeCheckpointStart<Runtime>, gate: &Gate, fixture: &Fixture) {
    assert!(matches!(
        start,
        NativeCheckpointStart::GuardRejected(GuardedNotSubmittedReason::HostRejected(
            HostSubmissionRejection::Cancelled
        ))
    ));
    assert_eq!(gate.calls.load(Ordering::Acquire), 1);
    assert_eq!(fixture.lane.in_flight_count(), 0);
    assert_eq!(fixture.reaper.retained_count(), 0);
}

pub(crate) fn verify(timing: DeviceTimingMode, configured: bool) {
    let fixture = Fixture::new(AttentionKind::CausalInt8).with_checkpoint_timing(timing);
    let tokens: Arc<[u32]> = Arc::from([3, 4, 5, 6, 7]);
    let source = fixture.admit("guarded-checkpoint-source", Arc::clone(&tokens));
    fixture.execute(&source, Arc::clone(&tokens), 0..2);
    fixture
        .execute(&source, Arc::clone(&tokens), 2..3)
        .assert_state_nonzero();
    if configured {
        fixture
            .lane
            .configure_reusable_executables(DeviceReusableExecutionPlan::on_demand(1).unwrap())
            .unwrap();
    }
    let graph_snapshot = || {
        let state = fixture
            .lane
            .cost_graph_stream_state()
            .unwrap()
            .expect("CUDA exposes the actual graph configuration");
        assert_eq!(
            state.configuration(),
            if configured {
                DeviceCostGraphConfiguration::OnDemand
            } else {
                DeviceCostGraphConfiguration::Unconfigured
            }
        );
        if !configured {
            assert!(state.is_unconfigured_empty());
        }
        // Catalog and preparation inspection require prior configuration;
        // the typed stream state proves the unconfigured empty case.
        let prepared = configured.then(|| {
            (
                fixture
                    .lane
                    .reusable_execution_catalog()
                    .unwrap()
                    .into_parts(),
                fixture.lane.reusable_executable_preparation().unwrap(),
            )
        });
        (state, prepared)
    };
    let before_graph = graph_snapshot();
    let binding = fixture.resources.trusted_runtime_binding().unwrap();
    let capture_gate = Gate::new(NativeCheckpointTransferKind::Capture);
    let capture = || loop {
        let start = fixture
            .reaper
            .try_capture_sequence_checkpoint_guarded(
                fixture.compilation.executable().execution_plan(),
                &binding,
                Arc::clone(&source),
                Arc::clone(&fixture.lane),
                timing,
                CheckpointTransferObservationStart::now(),
                &capture_gate,
            )
            .unwrap();
        match start {
            NativeCheckpointStart::CapacityMaintenance { maintenance, .. } => {
                assert!(matches!(
                    maintenance.try_maintain().unwrap(),
                    CheckpointCapacityMaintenanceOutcome::Ready(_)
                ));
            }
            other => break other,
        }
    };
    assert_rejected(capture(), &capture_gate, &fixture);
    assert_eq!(graph_snapshot(), before_graph);
    assert_eq!(
        fixture
            .reaper
            .checkpoint_timing_snapshot()
            .capture
            .submitted_copies
            .samples,
        0
    );
    assert_eq!(
        CapacitySnapshot::observe(&fixture.resources).checkpoint_claims,
        0
    );
    capture_gate.allowed.store(true, Ordering::Release);
    let NativeCheckpointResult::Captured(checkpoint) = finish(capture()) else {
        panic!("guarded capture did not publish a checkpoint");
    };
    assert_eq!(capture_gate.calls.load(Ordering::Acquire), 2);
    fixture.assert_checkpoint_timing(
        fixture.reaper.checkpoint_timing_snapshot().capture,
        checkpoint.logical_bytes(),
    );
    assert_eq!(graph_snapshot(), before_graph);
    let expected = [3..4, 4..5].map(|range| fixture.execute(&source, Arc::clone(&tokens), range));
    source.try_complete().unwrap();
    drop(source);

    let target = fixture.admit("guarded-checkpoint-target", Arc::clone(&tokens));
    let restore_gate = Gate::new(NativeCheckpointTransferKind::Restore);
    let before_graph = graph_snapshot();
    let restore = || {
        fixture
            .reaper
            .try_restore_sequence_checkpoint_guarded(
                fixture.compilation.executable().execution_plan(),
                Arc::clone(&target),
                &checkpoint,
                Arc::clone(&tokens),
                Arc::clone(&fixture.lane),
                timing,
                CheckpointTransferObservationStart::now(),
                &restore_gate,
            )
            .unwrap()
    };
    assert_rejected(restore(), &restore_gate, &fixture);
    assert_eq!(graph_snapshot(), before_graph);
    assert_eq!(
        fixture
            .reaper
            .checkpoint_timing_snapshot()
            .restore
            .submitted_copies
            .samples,
        0
    );
    restore_gate.allowed.store(true, Ordering::Release);
    let NativeCheckpointResult::Restored(publication) = finish(restore()) else {
        panic!("guarded restore did not publish its target");
    };
    assert_eq!(restore_gate.calls.load(Ordering::Acquire), 2);
    assert!(publication.matches_target(&target));
    publication.acknowledge().unwrap();
    fixture.assert_checkpoint_timing(
        fixture.reaper.checkpoint_timing_snapshot().restore,
        checkpoint.logical_bytes(),
    );
    assert_eq!(graph_snapshot(), before_graph);
    for (range, expected) in [3..4, 4..5].into_iter().zip(expected) {
        expected.assert_same(
            &fixture.execute(&target, Arc::clone(&tokens), range),
            "guarded checkpoint retry suffix",
        );
    }
    target.try_complete().unwrap();
    drop((target, checkpoint));
    assert_eq!(fixture.lane.in_flight_count(), 0);
    assert_eq!(fixture.reaper.retained_count(), 0);
    assert_eq!(
        CapacitySnapshot::observe(&fixture.resources).checkpoint_claims,
        0
    );
}
