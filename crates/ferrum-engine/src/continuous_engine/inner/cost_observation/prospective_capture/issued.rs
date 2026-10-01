//! Fixed-size issued-bound diagnostics. These counters never feed training or feedback.
use super::*;

#[derive(Default)]
pub(in crate::continuous_engine::inner::cost_observation) struct IssuedPredictionAudit {
    pub(super) population: Arc<CaptureAudit>,
    paired: AtomicU64,
    underestimates: AtomicU64,
    overestimates: AtomicU64,
    total_planning_ns: AtomicU64,
    total_actual_ns: AtomicU64,
    total_absolute_error_ns: AtomicU64,
    total_underestimate_ns: AtomicU64,
    total_overestimate_ns: AtomicU64,
    maximum_underestimate_ns: AtomicU64,
    maximum_overestimate_ns: AtomicU64,
    invalid_measurement: AtomicU64,
    matched_but_not_retained: AtomicU64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct IssuedPredictionAuditSnapshot {
    pub protocol: &'static str,
    pub boundary: &'static str,
    pub population: CaptureAuditSnapshot,
    pub paired_count: u64,
    pub underestimates: u64,
    pub overestimates: u64,
    pub total_planning_ns: u64,
    pub total_actual_ns: u64,
    pub total_absolute_error_ns: u64,
    pub total_underestimate_ns: u64,
    pub total_overestimate_ns: u64,
    pub maximum_underestimate_ns: u64,
    pub maximum_overestimate_ns: u64,
    pub invalid_measurement: u64,
    pub matched_but_not_retained: u64,
}
impl IssuedPredictionAudit {
    fn add(&self, target: &AtomicU64, amount: u64) {
        if target
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                v.checked_add(amount)
            })
            .is_err()
        {
            self.population
                .counter_exhausted
                .store(true, Ordering::Relaxed);
        }
    }
    pub(super) fn not_retained(&self) {
        self.add(&self.matched_but_not_retained, 1);
    }
    pub(super) fn record(&self, ordinal: u64, receipt: &ProspectiveCaptureReceiptV1) {
        let (Some(planning), Some(actual)) =
            (receipt.issued_planning_ns, receipt.actual_full_wall_ns)
        else {
            self.add(&self.invalid_measurement, 1);
            return;
        };
        if ordinal == 0 || receipt.call_id == 0 || planning == 0 || actual == 0 {
            self.add(&self.invalid_measurement, 1);
            return;
        }
        let underestimate = actual.saturating_sub(planning);
        let overestimate = planning.saturating_sub(actual);
        self.add(&self.paired, 1);
        self.add(&self.underestimates, u64::from(underestimate > 0));
        self.add(&self.overestimates, u64::from(overestimate > 0));
        self.add(&self.total_planning_ns, planning);
        self.add(&self.total_actual_ns, actual);
        self.add(&self.total_absolute_error_ns, actual.abs_diff(planning));
        self.add(&self.total_underestimate_ns, underestimate);
        self.add(&self.total_overestimate_ns, overestimate);
        self.maximum_underestimate_ns
            .fetch_max(underestimate, Ordering::Relaxed);
        self.maximum_overestimate_ns
            .fetch_max(overestimate, Ordering::Relaxed);
        tracing::trace!(target: "ferrum::structured_presubmit_audit",
            call_id = receipt.call_id, accepted_ordinal = ordinal,
            model_epoch = receipt.model_epoch, profile_sha256 = ?receipt.source.profile_sha256,
            owner_domain = ?receipt.source.domain,
            planning_ns = planning, actual_ns = actual,
            underestimate_ns = underestimate, overestimate_ns = overestimate,
            "original issued structured witness matched retained complete same-call settlement");
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn snapshot(
        &self,
    ) -> IssuedPredictionAuditSnapshot {
        let read = |v: &AtomicU64| v.load(Ordering::Acquire);
        IssuedPredictionAuditSnapshot {
            protocol: "ferrum.issued-structured-wave-error.v1",
            boundary: "original preparation to complete host settlement; issued planning bound; no retrospective lookup or client latency; cumulative since runtime construction; inspect only quiescent counters for exact denominator",
            population: self.population.snapshot(),
            paired_count: read(&self.paired),
            underestimates: read(&self.underestimates),
            overestimates: read(&self.overestimates),
            total_planning_ns: read(&self.total_planning_ns),
            total_actual_ns: read(&self.total_actual_ns),
            total_absolute_error_ns: read(&self.total_absolute_error_ns),
            total_underestimate_ns: read(&self.total_underestimate_ns),
            total_overestimate_ns: read(&self.total_overestimate_ns),
            maximum_underestimate_ns: read(&self.maximum_underestimate_ns),
            maximum_overestimate_ns: read(&self.maximum_overestimate_ns),
            invalid_measurement: read(&self.invalid_measurement),
            matched_but_not_retained: read(&self.matched_but_not_retained),
        }
    }
}
