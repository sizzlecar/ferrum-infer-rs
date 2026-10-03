//! Prospective, witness-only capture. Frozen before input preparation and
//! reconciled with the original private host settlement. No source3 session,
//! cohort/phase assignment, fit input, TTL refresh or execution authority.
use super::*;
use ferrum_interfaces::execution_cost::{ExpectedExecutionWave, HostTerminalExpectationV1};
#[cfg(test)]
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredQueryV2;
use ferrum_scheduler::implementations::continuous::slo_planner::{SchedulerSnapshot, SelectedWave};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
    OnceLock,
};
use std::time::Instant;
mod issued;
pub(super) use issued::{IssuedPredictionAudit, IssuedPredictionAuditSnapshot};

struct IssuedPrediction {
    planning_ns: u64,
    audit: Arc<IssuedPredictionAudit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProspectiveCaptureOutcomeV1 {
    Matched = 1,
    ReplayEvidenceMissing,
    CatalogUnavailable,
    ClockUnavailable,
    Expired,
    EpochChanged,
    AttachmentConflict,
    ParticipantMismatch,
    ActualMismatch,
    SettlementRejected,
    UnsupportedTerminal,
    NotSubmitted,
    Abandoned,
}
impl ProspectiveCaptureOutcomeV1 {
    const COUNT: usize = Self::Abandoned as usize;
    const ALL: [Self; Self::COUNT] = [
        Self::Matched,
        Self::ReplayEvidenceMissing,
        Self::CatalogUnavailable,
        Self::ClockUnavailable,
        Self::Expired,
        Self::EpochChanged,
        Self::AttachmentConflict,
        Self::ParticipantMismatch,
        Self::ActualMismatch,
        Self::SettlementRejected,
        Self::UnsupportedTerminal,
        Self::NotSubmitted,
        Self::Abandoned,
    ];
}

#[derive(Debug, Clone, serde::Serialize)]
pub(super) struct SourceIdentity {
    pub domain: [u8; 32],
    pub profile_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
    pub parameters_sha256: [u8; 32],
    pub protocol_sha256: [u8; 32],
    pub capture_identity: [u8; 32],
}

/// Serialize-only diagnostic receipt. `Matched` proves one actual wave matched
/// a prospective declaration and private settlement; it is NOT model/source
/// qualification. FIFO acceptance is separately owned by the original sink.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProspectiveCaptureReceiptV1 {
    protocol: &'static str,
    population: &'static str,
    source: SourceIdentity,
    model_epoch: u64,
    declared_at_ns: u64,
    call_id: u64,
    outcome: ProspectiveCaptureOutcomeV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    issued_planning_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    actual_full_wall_ns: Option<u64>,
    #[serde(skip)]
    settlement_binding: Option<[u8; 32]>,
}
impl ProspectiveCaptureReceiptV1 {
    pub fn outcome(&self) -> ProspectiveCaptureOutcomeV1 {
        self.outcome
    }
    /// Detect a diagnostic receipt copied to another call or mutated stages.
    /// Success still grants no source3/cohort/phase membership.
    pub fn validates_host_stages(&self, stages: &HostStageEvidenceV1) -> bool {
        self.outcome == ProspectiveCaptureOutcomeV1::Matched
            && self.call_id == stages.call_id
            && stages
                .structured_evidence
                .as_ref()
                .and_then(|s| s.as_ref().ok())
                .is_some_and(|s| {
                    Some(s.stage_binding()) == self.settlement_binding
                        && s.validate_host_stages(stages).is_ok()
                })
    }
}

#[derive(Debug, Default)]
pub(super) struct CaptureAudit {
    declared: AtomicU64,
    outcomes: [AtomicU64; ProspectiveCaptureOutcomeV1::COUNT],
    queued: AtomicU64,
    queue_dropped: AtomicU64,
    counter_exhausted: AtomicBool,
}
#[derive(Debug, Clone, serde::Serialize)]
pub(super) struct CaptureAuditSnapshot {
    protocol: &'static str,
    population: &'static str,
    declared: u64,
    outcomes: Vec<(ProspectiveCaptureOutcomeV1, u64)>,
    queued: u64,
    queue_dropped: u64,
    counter_exhausted: bool,
}
impl CaptureAudit {
    fn add(&self, value: &AtomicU64) {
        if value
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .is_err()
        {
            self.counter_exhausted.store(true, Ordering::Release);
        }
    }
    fn outcome(&self, outcome: ProspectiveCaptureOutcomeV1) {
        self.add(&self.outcomes[outcome as usize - 1]);
    }
    pub fn snapshot(&self) -> CaptureAuditSnapshot {
        CaptureAuditSnapshot {
            protocol: "ferrum.prospective-first-wave-capture.v1",
            population: "published structured V2 cost witnesses; completion-only and legacy waves excluded; matched is not source3/model qualification",
            declared: self.declared.load(Ordering::Acquire),
            outcomes: ProspectiveCaptureOutcomeV1::ALL.iter().map(|v| (*v, self.outcomes[*v as usize - 1].load(Ordering::Acquire))).collect(),
            queued: self.queued.load(Ordering::Acquire),
            queue_dropped: self.queue_dropped.load(Ordering::Acquire),
            counter_exhausted: self.counter_exhausted.load(Ordering::Acquire),
        }
    }
}

pub(in crate::continuous_engine::inner) struct ProspectiveCapture {
    exact: Arc<CanonicalWaveCostShape>,
    selected: Arc<StatisticalWaveEvidenceV1>,
    participants: Vec<CostObservationParticipant>,
    baseline: super::profile::ProspectiveIdentity,
    source: SourceIdentity,
    epoch: u64,
    declared_at_ns: u64,
    valid_until: Instant,
    audit: Arc<CaptureAudit>,
    issued_prediction: Option<IssuedPrediction>,
    issued_recorded: AtomicBool,
    claimed: AtomicBool,
    outcome: AtomicU8,
    receipt: OnceLock<ProspectiveCaptureReceiptV1>,
    queue_recorded: AtomicBool,
    #[cfg(test)]
    before_finish: parking_lot::Mutex<Option<Box<dyn FnOnce(ProspectiveCaptureOutcomeV1) + Send>>>,
}
impl ProspectiveCapture {
    /// Read an already produced normal-execution receipt. This never reconciles
    /// a call, creates a receipt or changes the original once-only outcome.
    #[cfg(test)]
    pub(in crate::continuous_engine::inner) fn settled_receipt_for_test(
        &self,
    ) -> Option<&ProspectiveCaptureReceiptV1> {
        self.receipt.get()
    }

    pub(super) fn retained_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(std::mem::size_of::<CaptureAudit>())?
            .checked_add(if self.issued_prediction.is_some() {
                // Include both audit Arcs conservatively even when their population is shared.
                std::mem::size_of::<IssuedPredictionAudit>()
                    + std::mem::size_of::<CaptureAudit>()
                    + 4 * std::mem::size_of::<usize>()
            } else {
                0
            })?
            .checked_add(super::sealed::canonical_retained_bytes(&self.exact)?)?
            .checked_add(std::mem::size_of::<StatisticalWaveEvidenceV1>())?
            .checked_add(
                self.participants
                    .capacity()
                    .checked_mul(std::mem::size_of::<CostObservationParticipant>())?,
            )?
            .checked_add(
                match self.selected.structured_capture().and_then(Result::ok) {
                    Some(recipe) => recipe.retained_bytes().ok()?,
                    None => 0,
                },
            )?
            .checked_add(8 * std::mem::size_of::<usize>() + 4 * std::mem::size_of::<AtomicU64>())
    }
    #[cfg(test)]
    pub(super) fn before_receipt_finish_for_test(
        &self,
        hook: impl FnOnce(ProspectiveCaptureOutcomeV1) + Send + 'static,
    ) {
        *self.before_finish.lock() = Some(Box::new(hook));
    }

    #[cfg(test)]
    pub(super) fn from_fixture(
        runtime: &EngineCostRuntime,
        exact: CanonicalWaveCostShape,
        selected: StatisticalWaveEvidenceV1,
        query: &StructuredQueryV2,
        participants: Vec<CostObservationParticipant>,
        valid_until: Instant,
    ) -> Arc<Self> {
        let baseline = runtime.snapshot().unwrap();
        // The fixture uses the real source3/profile10 importer. This helper
        // supplies only the independent-replay edge, not a synthetic model.
        baseline
            .audit_structured_query_v2(query, runtime.clock.now_ns().unwrap())
            .unwrap();
        let audit = runtime
            .prospective_capture
            .clone()
            .or_else(|| {
                runtime
                    .issued_structured_prediction
                    .as_ref()
                    .map(|a| a.population.clone())
            })
            .unwrap();
        audit.add(&audit.declared);
        Arc::new(Self {
            source: baseline.prospective_source(query).unwrap(),
            epoch: baseline.model_version(),
            baseline: baseline
                .prospective_identity()
                .expect("validated structured snapshot"),
            exact: Arc::new(exact),
            selected: Arc::new(selected),
            participants,
            declared_at_ns: runtime.clock.now_ns().unwrap(),
            valid_until,
            audit,
            issued_prediction: None,
            issued_recorded: AtomicBool::new(false),
            claimed: AtomicBool::new(false),
            outcome: AtomicU8::new(0),
            receipt: OnceLock::new(),
            queue_recorded: AtomicBool::new(false),
            #[cfg(test)]
            before_finish: parking_lot::Mutex::new(None),
        })
    }
    #[cfg(test)]
    pub(super) fn with_issued_bound_for_test(
        mut self: Arc<Self>,
        runtime: &EngineCostRuntime,
        planning_ns: u64,
    ) -> Arc<Self> {
        let this = Arc::get_mut(&mut self).expect("unshared fixture declaration");
        let audit = runtime
            .issued_structured_prediction
            .as_ref()
            .unwrap()
            .clone();
        if !Arc::ptr_eq(&this.audit, &audit.population) {
            audit.population.add(&audit.population.declared);
        }
        this.issued_prediction = Some(IssuedPrediction { planning_ns, audit });
        self
    }
    fn each_audit(&self, visit: impl Fn(&CaptureAudit)) {
        visit(&self.audit);
        if let Some(issued) = &self.issued_prediction {
            if !Arc::ptr_eq(&self.audit, &issued.audit.population) {
                visit(&issued.audit.population);
            }
        }
    }
    /// Sole original resolver calls this only after its final retention check.
    /// Do not re-query the current model: an update cannot replace the issued bound.
    pub(super) fn record_issued_settlement(&self, ordinal: u64) {
        let Some(issued) = &self.issued_prediction else {
            return;
        };
        let Some(receipt) = self
            .receipt
            .get()
            .filter(|r| r.outcome == ProspectiveCaptureOutcomeV1::Matched)
        else {
            return;
        };
        if !self.issued_recorded.swap(true, Ordering::AcqRel) {
            issued.audit.record(ordinal, receipt);
        }
    }
    pub(in crate::continuous_engine::inner) fn not_submitted(&self, now: Instant) {
        self.finish(if now > self.valid_until {
            ProspectiveCaptureOutcomeV1::Expired
        } else if !self.baseline.current() {
            ProspectiveCaptureOutcomeV1::EpochChanged
        } else {
            ProspectiveCaptureOutcomeV1::NotSubmitted
        });
    }
    fn finish(&self, outcome: ProspectiveCaptureOutcomeV1) -> ProspectiveCaptureOutcomeV1 {
        match self
            .outcome
            .compare_exchange(0, outcome as u8, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                self.each_audit(|audit| audit.outcome(outcome));
                outcome
            }
            // The declaration has one winning terminal result, even when a
            // concurrent duplicate attach races settlement publication.
            Err(value) => ProspectiveCaptureOutcomeV1::ALL[value as usize - 1],
        }
    }
    fn attachable(
        &self,
        call: &EngineCostCall,
        now: Instant,
    ) -> Result<(), ProspectiveCaptureOutcomeV1> {
        use ProspectiveCaptureOutcomeV1 as O;
        if self.claimed.swap(true, Ordering::AcqRel)
            || call.context_created
            || call.prospective_capture.is_some()
        {
            return Err(O::AttachmentConflict);
        }
        if now > self.valid_until {
            return Err(O::Expired);
        }
        if !self.baseline.current() || self.baseline.model_version() != self.epoch {
            return Err(O::EpochChanged);
        }
        if call
            .prepare_started_at_ns
            .is_none_or(|at| at < self.declared_at_ns)
        {
            return Err(O::ClockUnavailable);
        }
        if self.participants != call.participants {
            return Err(O::ParticipantMismatch);
        }
        Ok(())
    }
    pub(super) fn receipt(
        &self,
        actual: &ActualWaveShape,
        original: host_stages::OriginalHostStages<'_>,
    ) -> ProspectiveCaptureReceiptV1 {
        let stages = original.get();
        self.receipt
            .get_or_init(|| {
                let proposed = self.reconcile(actual, stages);
                #[cfg(test)]
                if let Some(hook) = self.before_finish.lock().take() {
                    hook(proposed);
                }
                let outcome = self.finish(proposed);
                ProspectiveCaptureReceiptV1 {
                    protocol: "ferrum.prospective-first-wave-capture.v1",
                    population: "published_structured_v2_cost_witness",
                    source: self.source.clone(),
                    model_epoch: self.epoch,
                    declared_at_ns: self.declared_at_ns,
                    call_id: stages.call_id,
                    outcome,
                    issued_planning_ns: self
                        .issued_prediction
                        .as_ref()
                        .map(|issued| issued.planning_ns),
                    actual_full_wall_ns: self.issued_prediction.as_ref().and(stages.full_wall_ns),
                    settlement_binding: stages
                        .structured_evidence
                        .as_ref()
                        .and_then(|v| v.as_ref().ok())
                        .map(|v| v.stage_binding()),
                }
            })
            .clone()
    }
    fn reconcile(
        &self,
        actual: &ActualWaveShape,
        stages: &HostStageEvidenceV1,
    ) -> ProspectiveCaptureOutcomeV1 {
        use ProspectiveCaptureOutcomeV1 as O;
        if self.outcome.load(Ordering::Acquire) == O::AttachmentConflict as u8 {
            return O::AttachmentConflict;
        }
        let Some(Ok(settled)) = &stages.structured_evidence else {
            return O::SettlementRejected;
        };
        // receipt only accepts the original producer's private borrow, immediately
        // after qualification. Do not serialize/hash the same host stages again.
        if stages.completeness != HostStageCompleteness::CompleteSingleWave
            || stages.full_wall_ns != Some(settled.full_wall_ns())
            || stages.fingerprint.as_ref() != Some(self.baseline.fingerprint())
        {
            return O::SettlementRejected;
        }
        // Statistical equality deliberately excludes sidecars; compare both
        // exact bindings AND the full immutable structured recipe explicitly.
        let Some(statistics) = &actual.statistical_evidence else {
            return O::ActualMismatch;
        };
        let Some(Ok(expected_recipe)) = self.selected.structured_capture() else {
            return O::ActualMismatch;
        };
        let Some(Ok(actual_recipe)) = statistics.structured_capture() else {
            return O::ActualMismatch;
        };
        // The private settlement already validated actual statistics/recipe
        // against the actual wave. Their equalities include exact_binding;
        // comparing all sidecars preserves that proof for the independently
        // validated replay value without hashing the actual shape twice again.
        if self.selected.as_ref() != statistics
            || self.selected.independent_attention_v2() != statistics.independent_attention_v2()
            || expected_recipe.as_ref() != actual_recipe.as_ref()
            || self.exact.host_content_features != actual.host_content_features
            || !std::ptr::eq(settled.recipe(), actual_recipe.as_ref())
            || self.participants.len() != actual.rows.len()
            || self.participants.iter().zip(&actual.rows).any(|(p, r)| {
                p.request_id != r.request_id
                    || p.owner_incarnation != r.owner_incarnation
                    || p.work_generation != r.work_generation
                    || p.input_index != r.input_index
            })
        {
            return O::ActualMismatch;
        }
        for (declared, row) in actual_recipe.physical_host_rows().iter().zip(&stages.rows) {
            match (declared.terminal_expectation, row.terminal.as_ref()) {
                (HostTerminalExpectationV1::NoTokenProduced, None)
                | (HostTerminalExpectationV1::TokenMayTerminate, None) => {}
                (HostTerminalExpectationV1::LengthBoundary, Some(terminal))
                    if terminal.finish_reason == ferrum_types::FinishReason::Length => {}
                _ => return O::UnsupportedTerminal,
            }
        }
        O::Matched
    }
    pub(super) fn abandon_unresolved(&self) {
        self.finish(ProspectiveCaptureOutcomeV1::Abandoned);
    }
    pub(super) fn queue_result(&self, result: &Result<u64, CostSampleDrop>) {
        // FIFO acceptance/loss is independent of later actual matching. A raw
        // drop cannot project a receipt or be silently omitted from this count.
        if !self.queue_recorded.swap(true, Ordering::AcqRel) {
            self.each_audit(|audit| {
                audit.add(if result.is_ok() {
                    &audit.queued
                } else {
                    &audit.queue_dropped
                })
            });
        }
    }
}
impl Drop for ProspectiveCapture {
    fn drop(&mut self) {
        self.finish(ProspectiveCaptureOutcomeV1::Abandoned);
        if self.outcome.load(Ordering::Acquire) == ProspectiveCaptureOutcomeV1::Matched as u8
            && !self.issued_recorded.load(Ordering::Acquire)
        {
            if let Some(issued) = &self.issued_prediction {
                issued.audit.not_retained();
            }
        }
    }
}

impl EngineCostRuntime {
    /// Called only after common controller publication checks. The exact
    /// candidate/query/recipe all originate in the same independent replay.
    pub(in crate::continuous_engine::inner) fn declare_prospective_capture(
        &self,
        selected: &SelectedWave,
        snapshot: &SchedulerSnapshot,
        expected: &ExpectedExecutionWave,
        valid_until: Instant,
    ) -> Option<Arc<ProspectiveCapture>> {
        let audit = self.prospective_capture.clone().or_else(|| {
            self.issued_structured_prediction
                .as_ref()
                .map(|a| a.population.clone())
        })?;
        audit.add(&audit.declared);
        let issued_audit = self.issued_structured_prediction.as_ref();
        if let Some(issued) = issued_audit.filter(|a| !Arc::ptr_eq(&audit, &a.population)) {
            issued.population.add(&issued.population.declared);
        }
        let attempt = (|| {
            let (exact, statistics, query) =
                selected
                    .replayed_first_wave_structured_v2(snapshot)
                    .ok_or(ProspectiveCaptureOutcomeV1::ReplayEvidenceMissing)?;
            let baseline = self
                .try_snapshot()
                .flatten()
                .filter(|m| m.model_version() == selected.cost_model_version)
                .ok_or(ProspectiveCaptureOutcomeV1::CatalogUnavailable)?;
            let source = baseline
                .prospective_source(query)
                .ok_or(ProspectiveCaptureOutcomeV1::CatalogUnavailable)?;
            let declared_at_ns = self
                .clock
                .now_ns()
                .ok_or(ProspectiveCaptureOutcomeV1::ClockUnavailable)?;
            let ferrum_interfaces::execution_cost::WaveCommitment::CostWitness(witness) =
                expected.commitment()
            else {
                return Err(ProspectiveCaptureOutcomeV1::ReplayEvidenceMissing);
            };
            if witness.canonical() != exact.as_ref() {
                return Err(ProspectiveCaptureOutcomeV1::ReplayEvidenceMissing);
            }
            let participants = witness
                .participants()
                .iter()
                .map(|p| p.host.clone())
                .collect();
            Ok(Arc::new(ProspectiveCapture {
                exact: exact.clone(),
                selected: statistics.clone(),
                participants,
                epoch: selected.cost_model_version,
                baseline: baseline
                    .prospective_identity()
                    .expect("validated structured snapshot"),
                source,
                declared_at_ns,
                valid_until,
                audit: audit.clone(),
                issued_prediction: issued_audit.map(|audit| IssuedPrediction {
                    planning_ns: selected.predicted_wall_ns,
                    audit: audit.clone(),
                }),
                issued_recorded: AtomicBool::new(false),
                claimed: AtomicBool::new(false),
                outcome: AtomicU8::new(0),
                receipt: OnceLock::new(),
                queue_recorded: AtomicBool::new(false),
                #[cfg(test)]
                before_finish: parking_lot::Mutex::new(None),
            }))
        })();
        match attempt {
            Ok(value) => Some(value),
            Err(reason) => {
                audit.outcome(reason);
                if let Some(issued) = issued_audit.filter(|a| !Arc::ptr_eq(&audit, &a.population)) {
                    issued.population.outcome(reason);
                }
                None
            }
        }
    }
}
impl EngineCostCall {
    pub(in crate::continuous_engine::inner) fn attach_prospective_capture(
        &mut self,
        capture: Arc<ProspectiveCapture>,
        now: Instant,
    ) {
        match capture.attachable(self, now) {
            Ok(()) => self.prospective_capture = Some(capture),
            Err(reason) => {
                capture.finish(reason);
            }
        }
    }
}
