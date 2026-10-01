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
