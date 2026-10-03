use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Projection {
    shape: ActualWaveShape,
    calls: Arc<AtomicUsize>,
    mismatch: bool,
    expanded_limit: usize,
}
impl PendingActualWaveProjection for Projection {
    fn rows(&self) -> &[ActualWaveRow] {
        &self.shape.rows
    }
    fn bounds(&self) -> PendingWaveBounds {
        PendingWaveBounds {
            retained_bytes: std::mem::size_of::<Self>() + std::mem::size_of::<ActualWaveRow>(),
            maximum_resolved_bytes: self.expanded_limit,
            retained_rows: 1,
        }
    }
    fn project(&self) -> Result<ActualWaveShape, ActualWaveEvidenceUnknown> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let mut shape = self.shape.clone();
        if self.mismatch {
            shape.rows[0].work_generation += 1;
        }
        Ok(shape)
    }
}
fn pending(
    shape: ActualWaveShape,
    calls: Arc<AtomicUsize>,
    mismatch: bool,
    expanded_limit: usize,
) -> PendingActualWave {
    PendingActualWave::new(Arc::new(Projection {
        shape,
        calls,
        mismatch,
        expanded_limit,
    }))
    .unwrap()
}
fn space() -> BoundedWaveRecorder {
    BoundedWaveRecorder::new(
        NonZeroU64::new(7).unwrap(),
        CostRecorderLimits {
            max_waves: 2,
            max_rows_per_wave: 8,
            max_retained_rows: 8192,
        },
    )
    .unwrap()
}

#[test]
fn pending_wave_preserves_original_lifecycle_without_implicit_projection() {
    let shape = decode();
    let calls = Arc::new(AtomicUsize::new(0));
    let input = pending(shape.clone(), calls.clone(), false, 4096);
    let _diagnostic = format!("{input:?}");
    let _bounds = input.bounds();
    let mut recorder = space();
    let handle = recorder
        .begin_pending(
            input,
            WaveObservationBoundary::IsolatedPreparationToCommit,
            10,
        )
        .unwrap();
    recorder.submission_started(&handle, 12).unwrap();
    recorder
        .terminal(&handle, ActualWaveOutcome::Completed, 18, None)
        .unwrap();
    recorder.host_committed(&handle, 20).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(matches!(
        recorder.coverage(),
        CostObservationCoverage::Unknown {
            reason: CostObservationUnknownReason::IncompleteObservation,
            ..
        }
    ));
    recorder.resolve_pending().unwrap();
    recorder.resolve_pending().unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(recorder.observations()[0].shape.as_ref(), Some(&shape));
    assert_eq!(recorder.trainable_wall_ns(&handle).unwrap().get(), 10);
}

#[test]
fn pending_wave_identity_or_expansion_failure_remains_unknown() {
    for (mismatch, bound, expected) in [
        (
            true,
            4096,
            ActualWaveEvidenceUnknown::ParticipantCorrelation,
        ),
        (false, 1, ActualWaveEvidenceUnknown::Capacity),
        // Expansion remains the first numeric guard even with invalid rows.
        (true, 1, ActualWaveEvidenceUnknown::Capacity),
    ] {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut recorder = space();
        recorder
            .begin_pending(
                pending(decode(), calls.clone(), mismatch, bound),
                WaveObservationBoundary::IsolatedPreparationToCommit,
                10,
            )
            .unwrap();
        assert_eq!(recorder.resolve_pending(), Err(expected));
        assert!(recorder.observations()[0].shape.is_none());
        assert_eq!(recorder.observations()[0].shape_unknown, Some(expected));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}

struct OwnedMetadataProjection {
    shape: ActualWaveShape,
    metadata: Box<[u64]>,
    calls: Arc<AtomicUsize>,
}
impl PendingActualWaveProjection for OwnedMetadataProjection {
    fn rows(&self) -> &[ActualWaveRow] {
        &self.shape.rows
    }
    fn bounds(&self) -> PendingWaveBounds {
        let raw = std::mem::size_of_val(self)
            + self.metadata.len() * std::mem::size_of::<u64>()
            + self.shape.rows.capacity() * std::mem::size_of::<ActualWaveRow>();
        PendingWaveBounds {
            retained_bytes: raw,
            maximum_resolved_bytes: raw + 4096,
            retained_rows: self.shape.rows.capacity(),
        }
    }
    fn project(&self) -> Result<ActualWaveShape, ActualWaveEvidenceUnknown> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert!(self.metadata.iter().all(|v| *v == 7));
        Ok(self.shape.clone())
    }
}

#[test]
fn pending_wave_owned_metadata_bytes_do_not_consume_physical_rows() {
    // Vary real owned metadata independently of physical participants. Large
    // immutable plans are legal for a one-row wave; rows are not byte units.
    let rows_limit = 8;
    for metadata_words in [rows_limit, rows_limit * 1024, rows_limit * 8192] {
        let calls = Arc::new(AtomicUsize::new(0));
        let pending = PendingActualWave::new(Arc::new(OwnedMetadataProjection {
            shape: decode(),
            metadata: vec![7; metadata_words].into_boxed_slice(),
            calls: calls.clone(),
        }))
        .unwrap();
        let bounds = pending.bounds();
        let mut recorder = BoundedWaveRecorder::new_with_byte_limits(
            NonZeroU64::new(17).unwrap(),
            CostRecorderLimits {
                max_waves: 1,
                max_rows_per_wave: rows_limit,
                max_retained_rows: rows_limit,
            },
            CostRecorderByteLimits {
                maximum_retained_bytes: bounds.retained_bytes,
                maximum_working_bytes: bounds.maximum_resolved_bytes,
            },
        )
        .unwrap();
        recorder
            .begin_pending(pending, WaveObservationBoundary::ExecutorOnly, 10)
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(recorder.memory_audit().requested_rows_peak, 1);
        assert_eq!(
            recorder.memory_audit().requested_raw_bytes_peak,
            bounds.retained_bytes
        );
        assert!(recorder.memory_audit().first_rejection.is_none());
        recorder.resolve_pending().unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(recorder.observations().len(), 1);
    }
}

#[test]
fn pending_wave_byte_and_row_limits_reject_their_actual_resources() {
    for resource in [
        CostRecorderCapacityResource::RawBytes,
        CostRecorderCapacityResource::WorkingBytes,
    ] {
        let calls = Arc::new(AtomicUsize::new(0));
        let pending = PendingActualWave::new(Arc::new(OwnedMetadataProjection {
            shape: decode(),
            metadata: vec![7; 8192].into_boxed_slice(),
            calls: calls.clone(),
        }))
        .unwrap();
        let bounds = pending.bounds();
        let mut recorder = BoundedWaveRecorder::new_with_byte_limits(
            NonZeroU64::new(18).unwrap(),
            CostRecorderLimits {
                max_waves: 1,
                max_rows_per_wave: 1,
                max_retained_rows: 1,
            },
            CostRecorderByteLimits {
                maximum_retained_bytes: bounds.retained_bytes
                    - usize::from(resource == CostRecorderCapacityResource::RawBytes),
                maximum_working_bytes: bounds.maximum_resolved_bytes
                    - usize::from(resource == CostRecorderCapacityResource::WorkingBytes),
            },
        )
        .unwrap();
        assert_eq!(
            recorder
                .begin_pending(pending, WaveObservationBoundary::ExecutorOnly, 10)
                .unwrap_err(),
            CostRecorderError::RowCapacity
        );
        assert_eq!(
            recorder.memory_audit().first_rejection.unwrap().resource,
            resource
        );
        assert!(recorder.observations().is_empty());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert!(matches!(
            recorder.coverage(),
            CostObservationCoverage::Unknown {
                lost_observations: 1,
                ..
            }
        ));
    }
}
struct PhysicalProjection {
    shape: ActualWaveShape,
    physical: ActualWavePhysicalEvidenceV1,
    unknown: ActualWaveEvidenceUnknown,
    bytes: usize,
}
impl PendingActualWaveProjection for PhysicalProjection {
    fn rows(&self) -> &[ActualWaveRow] {
        &self.shape.rows
    }
    fn bounds(&self) -> PendingWaveBounds {
        PendingWaveBounds {
            retained_bytes: std::mem::size_of::<Self>(),
            maximum_resolved_bytes: self.bytes,
            retained_rows: 1,
        }
    }
    fn project(&self) -> Result<ActualWaveShape, ActualWaveEvidenceUnknown> {
        Err(self.unknown)
    }
    fn project_with_physical_evidence(&self) -> ActualWaveProjection {
        ActualWaveProjection {
            shape: self.project(),
            physical_evidence: Some(self.physical),
        }
    }
}
fn physical_pending(
    shape: ActualWaveShape,
    physical: ActualWavePhysicalEvidenceV1,
    unknown: ActualWaveEvidenceUnknown,
    bytes: usize,
) -> PendingActualWave {
    PendingActualWave::new(Arc::new(PhysicalProjection {
        shape,
        physical,
        unknown,
        bytes,
    }))
    .unwrap()
}

#[test]
fn pending_physical_evidence_cannot_rescue_unknown_or_exceed_its_byte_bound() {
    let shape = decode();
    let physical = ActualWavePhysicalEvidenceV1::from_shape(&shape);
    let bytes = std::mem::size_of_val(&physical);
    let exact = physical_pending(
        shape.clone(),
        physical,
        ActualWaveEvidenceUnknown::GraphPath,
        bytes,
    )
    .resolve_with_physical_evidence();
    assert_eq!(exact.shape, Err(ActualWaveEvidenceUnknown::GraphPath));
    assert_eq!(exact.physical_evidence, Some(physical));
    let short = physical_pending(
        shape.clone(),
        physical,
        ActualWaveEvidenceUnknown::GraphPath,
        bytes - 1,
    )
    .resolve_with_physical_evidence();
    assert_eq!(short.shape, Err(ActualWaveEvidenceUnknown::Capacity));
    assert!(short.physical_evidence.is_none());
    let unsupported = physical_pending(
        shape.clone(),
        physical,
        ActualWaveEvidenceUnknown::ProviderPath,
        bytes,
    )
    .resolve_with_physical_evidence();
    assert_eq!(
        unsupported.shape,
        Err(ActualWaveEvidenceUnknown::ProviderPath)
    );
    assert!(unsupported.physical_evidence.is_none());
    for physical in [
        ActualWavePhysicalEvidenceV1 {
            kind: ActualWaveKind::Prefill,
            ..physical
        },
        ActualWavePhysicalEvidenceV1 {
            restore_bytes: 1,
            ..physical
        },
        ActualWavePhysicalEvidenceV1 {
            maintenance_units: 1,
            ..physical
        },
    ] {
        let invalid = physical_pending(
            shape.clone(),
            physical,
            ActualWaveEvidenceUnknown::GraphPath,
            bytes,
        )
        .resolve_with_physical_evidence();
        assert_eq!(invalid.shape, Err(ActualWaveEvidenceUnknown::GraphPath));
        assert!(invalid.physical_evidence.is_none());
    }
}

// Recorder-local protocol fixture. Actual Core selection/submission authority
// is independently exercised by the engine prefix preparation regression.
fn prepared_physical_rows(rows: Vec<ActualWaveRow>, submitted: bool) -> PreparedCallRouteV1 {
    PreparedCallRouteV1 {
        route: PreparedCostRouteV1 {
            class: PreparedCostRouteClassV1::GraphDisabled,
            reason: PreparedCostRouteReasonV1::DeclaredGraphUnsupported,
            program_id: None,
            non_reusable_wave: None,
            lane_id: 3,
            lane_epoch: 7,
            catalog_epoch: None,
            graph_state: None,
            batch_step: Some(11),
            batch_invocation: Some(12),
        },
        selected_at_ns: 11,
        rows,
        submitted: submitted.then(|| SubmittedRouteEvidenceV1 {
            batch_step: 11,
            batch_invocation: 12,
            plan_hash: "plan".into(),
            runtime_implementation_fingerprint: "runtime".into(),
            lane_id: 3,
            submission_started_at_ns: 12,
            graph: None,
        }),
    }
}

#[test]
fn pending_physical_evidence_requires_exact_completed_single_submission() {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Binding {
        Exact,
        MissingSubmission,
        WrongOwner,
        WrongGeneration,
        Partial,
        ExecutorOnly,
        ExtraWave,
    }
    for binding in [
        Binding::Exact,
        Binding::MissingSubmission,
        Binding::WrongOwner,
        Binding::WrongGeneration,
        Binding::Partial,
        Binding::ExecutorOnly,
        Binding::ExtraWave,
    ] {
        let shape = decode();
        let physical = ActualWavePhysicalEvidenceV1::from_shape(&shape);
        let mut rows = shape.rows.clone();
        if binding == Binding::WrongOwner {
            rows[0].owner_incarnation += 1;
        }
        if binding == Binding::WrongGeneration {
            rows[0].work_generation += 1;
        }
        let mut recorder = space();
        recorder.record_prepared_route(Some(prepared_physical_rows(
            rows,
            binding != Binding::MissingSubmission,
        )));
        let handle = recorder
            .begin_pending(
                physical_pending(
                    shape.clone(),
                    physical,
                    ActualWaveEvidenceUnknown::GraphPath,
                    std::mem::size_of_val(&physical),
                ),
                if binding == Binding::ExecutorOnly {
                    WaveObservationBoundary::ExecutorOnly
                } else {
                    WaveObservationBoundary::IsolatedPreparationToCommit
                },
                10,
            )
            .unwrap();
        recorder.submission_started(&handle, 12).unwrap();
        recorder
            .terminal(
                &handle,
                if binding == Binding::Partial {
                    ActualWaveOutcome::PartiallyCompleted
                } else {
                    ActualWaveOutcome::Completed
                },
                18,
                None,
            )
            .unwrap();
        if binding == Binding::ExtraWave {
            recorder
                .begin(shape, WaveObservationBoundary::ExecutorOnly, 19)
                .unwrap();
        }
        assert_eq!(
            recorder.resolve_pending(),
            Err(ActualWaveEvidenceUnknown::GraphPath)
        );
        assert_eq!(
            recorder.observations()[0].physical_evidence,
            (binding == Binding::Exact).then_some(physical)
        );
        assert!(recorder.observations()[0].shape.is_none());
        assert_eq!(
            recorder.observations()[0].shape_unknown,
            Some(ActualWaveEvidenceUnknown::GraphPath)
        );
    }
}
