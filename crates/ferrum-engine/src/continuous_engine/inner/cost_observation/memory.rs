//! CPU observation ownership only. These permits cannot authorize execution.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
pub(super) struct ObservationBytePool {
    maximum: usize,
    retained: AtomicUsize,
    peak: AtomicUsize,
}
impl ObservationBytePool {
    pub(super) fn new(maximum: usize) -> Arc<Self> {
        Arc::new(Self {
            maximum,
            retained: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        })
    }
    pub(super) fn reserve(self: &Arc<Self>, bytes: usize) -> Option<ObservationBytePermit> {
        let old = self
            .retained
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                n.checked_add(bytes).filter(|n| *n <= self.maximum)
            })
            .ok()?;
        self.peak.fetch_max(old + bytes, Ordering::Relaxed);
        Some(ObservationBytePermit {
            pool: Arc::clone(self),
            bytes: AtomicUsize::new(bytes),
        })
    }
    pub(super) fn retained(&self) -> usize {
        self.retained.load(Ordering::Acquire)
    }
    pub(super) fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }
}
/// Moved into the queue without a per-call Arc allocation. Only the CPU worker
/// shares the working permit with completed immutable results.
#[derive(Debug)]
pub(super) struct ObservationBytePermit {
    pool: Arc<ObservationBytePool>,
    bytes: AtomicUsize,
}
impl ObservationBytePermit {
    /// Sole owner expansion. Charge the shared ledger before the caller grows
    /// retained training state; another snapshot's lease is never reused.
    pub(super) fn grow_to(&mut self, retained_bytes: usize) -> bool {
        let old = *self.bytes.get_mut();
        if retained_bytes <= old {
            return true;
        }
        let Some(mut extra) = self.pool.reserve(retained_bytes - old) else {
            return false;
        };
        *self.bytes.get_mut() = retained_bytes;
        // Move this charge into self; dropping extra releases zero bytes.
        *extra.bytes.get_mut() = 0;
        true
    }
    pub(super) fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Acquire)
    }
    /// Release only temporary expansion after all immutable outputs have been
    /// measured. Remaining shared owners keep the retained payload charged.
    pub(super) fn shrink_to(&self, retained_bytes: usize) -> bool {
        let Ok(old) = self
            .bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                (retained_bytes <= old).then_some(retained_bytes)
            })
        else {
            return false;
        };
        self.pool
            .retained
            .fetch_sub(old - retained_bytes, Ordering::AcqRel);
        true
    }
}
impl Drop for ObservationBytePermit {
    fn drop(&mut self) {
        let old = self
            .pool
            .retained
            .fetch_sub(*self.bytes.get_mut(), Ordering::AcqRel);
        debug_assert!(old >= *self.bytes.get_mut());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub(super) struct CaptureCapacityFailure {
    pub call_id: u64,
    pub gate: &'static str,
    pub diagnostic: CostRecorderCapacityDiagnostic,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub(super) struct ObservationMemoryStats {
    pub requested_rows_peak: usize,
    pub requested_raw_bytes_peak: usize,
    pub requested_working_bytes_peak: usize,
    pub accepted_raw_bytes_peak: usize,
    pub accepted_working_bytes_peak: usize,
    pub queued_raw_bytes: usize,
    pub queued_raw_bytes_peak: usize,
    pub worker_and_result_bytes: usize,
    pub worker_and_result_bytes_peak: usize,
    pub first_capture_rejection: Option<CaptureCapacityFailure>,
}

impl ObservationMemoryStats {
    pub(super) fn maximum() -> Self {
        Self {
            requested_rows_peak: usize::MAX,
            requested_raw_bytes_peak: usize::MAX,
            requested_working_bytes_peak: usize::MAX,
            accepted_raw_bytes_peak: usize::MAX,
            accepted_working_bytes_peak: usize::MAX,
            queued_raw_bytes: usize::MAX,
            queued_raw_bytes_peak: usize::MAX,
            worker_and_result_bytes: usize::MAX,
            worker_and_result_bytes_peak: usize::MAX,
            first_capture_rejection: Some(CaptureCapacityFailure {
                call_id: u64::MAX,
                gate: "worker_and_result_bytes",
                diagnostic: CostRecorderCapacityDiagnostic {
                    resource: CostRecorderCapacityResource::WorkingBytes,
                    requested: Some(usize::MAX),
                    limit: usize::MAX,
                },
            }),
        }
    }
}

/// Construction bound before the numerical input's final axes check. The
/// current builder has <=4A+26 basis and <=11A+34 support coordinates; two
/// doubling Vecs and at most four original/scoped/settled inputs may coexist.
/// Include all row vectors and actual causes, even for a rejected projection.
pub(super) fn maximum_resolution_overhead(rows: usize) -> Option<usize> {
    let axes = MAX_COST_COMMANDS
        .checked_mul(15)?
        .checked_add(60)?
        .checked_mul(std::mem::size_of::<u64>())?
        .checked_mul(8)?;
    let host = rows.checked_mul(
        8 * (std::mem::size_of::<StructuredHostRowV1>()
            + 3 * std::mem::size_of::<u32>()
            + std::mem::size_of::<(u32, ferrum_types::FinishReason)>()),
    )?;
    axes.checked_add(host)?.checked_add(4096)
}

pub(super) fn shape_bytes(
    shape: &ferrum_scheduler::implementations::continuous::cost_model::WaveExecutionShape,
) -> Option<usize> {
    std::mem::size_of_val(shape)
        .checked_add(
            shape
                .decode_kv_tokens
                .capacity()
                .checked_mul(std::mem::size_of::<u32>())?,
        )?
        .checked_add(
            shape
                .prefill_chunks
                .capacity()
                .checked_mul(std::mem::size_of::<
                    ferrum_scheduler::implementations::continuous::cost_model::PrefillShape,
                >())?,
        )?
        .checked_add(shape.numeric_features.as_ref().map_or(Some(0), |v| {
            v.rows
                .capacity()
                .checked_mul(std::mem::size_of::<CostRowNumericFeatures>())
        })?)?
        .checked_add(shape.row_multiset_features.as_ref().map_or(Some(0), |v| {
            v.rows
                .capacity()
                .checked_mul(std::mem::size_of::<HostRowStaticCostFeaturesV2>())
        })?)
}

pub(super) fn statistics_bytes(statistics: &StatisticalWaveEvidenceV1) -> Option<usize> {
    std::mem::size_of_val(statistics).checked_add(match statistics.structured_capture() {
        Some(Ok(recipe)) => recipe.retained_bytes().ok()?,
        _ => 0,
    })
}

impl HostStageEvidenceV1 {
    pub(super) fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of_val(self)
            .checked_add(
                self.rows
                    .capacity()
                    .checked_mul(std::mem::size_of::<HostRowStageV1>())?,
            )?
            .checked_add(self.actual_shape.as_ref().map_or(Some(0), shape_bytes)?)?
            .checked_add(
                self.statistical_evidence
                    .as_ref()
                    .map_or(Some(0), statistics_bytes)?,
            )?
            .checked_add(match &self.structured_evidence {
                Some(Ok(v)) => v.recipe().retained_bytes().ok()?,
                _ => 0,
            })?
            .checked_add(
                self.route_evidence
                    .as_ref()
                    .map_or(Some(0), |v| v.retained_bytes())?,
            )
    }
}

impl CostCalibrationResult {
    pub(super) fn retained_payload_bytes(&self) -> Option<usize> {
        let mut bytes = std::mem::size_of_val(self);
        if let Self::Observed {
            sample,
            actual_rows,
            commits,
            host_features,
            ..
        } = self
        {
            bytes = bytes
                .checked_add(std::mem::size_of_val(sample.as_ref()))?
                .checked_add(shape_bytes(&sample.actual_shape)?)?
                .checked_add(
                    actual_rows
                        .capacity()
                        .checked_mul(std::mem::size_of::<ActualWaveRow>())?,
                )?
                .checked_add(
                    commits
                        .capacity()
                        .checked_mul(std::mem::size_of::<HostCommitEvidence>())?,
                )?
                .checked_add(
                    host_features
                        .capacity()
                        .checked_mul(std::mem::size_of::<Option<HostCostFeaturesV1>>())?,
                )?;
        }
        bytes.checked_add(2 * std::mem::size_of::<usize>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observation_bytes_remain_charged_until_last_result_owner_drops() {
        let pool = ObservationBytePool::new(1024);
        let raw = pool.reserve(700).unwrap();
        assert!(pool.reserve(325).is_none());
        let resolved = Arc::new(raw);
        let calibration = Arc::clone(&resolved);
        let stages = Arc::clone(&resolved);
        drop(resolved);
        drop(calibration);
        assert_eq!(pool.retained(), 700);
        assert!(pool.reserve(325).is_none());
        drop(stages);
        assert_eq!(pool.retained(), 0);
        assert!(pool.reserve(1024).is_some());
        assert_eq!(pool.retained(), 0);
        assert!(pool.reserve(usize::MAX).is_none());
    }
}
