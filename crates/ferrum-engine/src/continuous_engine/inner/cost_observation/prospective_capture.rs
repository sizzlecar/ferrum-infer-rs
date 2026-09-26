//! Prospective, witness-only capture. Frozen before input preparation and
//! reconciled with the original private host settlement. No source3 session,
//! cohort/phase assignment, fit input, TTL refresh or execution authority.
use super::profile::EngineCostSnapshot;
use super::*;
use ferrum_interfaces::execution_cost::{ExpectedExecutionWave, HostTerminalExpectationV1};
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::StructuredQueryV2,
    slo_planner::{SchedulerSnapshot, SelectedWave},
};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
    OnceLock,
};
use std::time::Instant;

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
    baseline: Arc<EngineCostSnapshot>,
    source: SourceIdentity,
    epoch: u64,
    declared_at_ns: u64,
    valid_until: Instant,
    audit: Arc<CaptureAudit>,
    claimed: AtomicBool,
    outcome: AtomicU8,
    receipt: OnceLock<ProspectiveCaptureReceiptV1>,
    queue_recorded: AtomicBool,
    #[cfg(test)]
    before_finish: parking_lot::Mutex<Option<Box<dyn FnOnce(ProspectiveCaptureOutcomeV1) + Send>>>,
}
impl ProspectiveCapture {
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
        let audit = runtime.prospective_capture.as_ref().unwrap().clone();
        audit.add(&audit.declared);
        Arc::new(Self {
            source: baseline.prospective_source(query).unwrap(),
            epoch: baseline.model_version(),
            baseline,
            exact: Arc::new(exact),
            selected: Arc::new(selected),
            participants,
            declared_at_ns: runtime.clock.now_ns().unwrap(),
            valid_until,
            audit,
            claimed: AtomicBool::new(false),
            outcome: AtomicU8::new(0),
            receipt: OnceLock::new(),
            queue_recorded: AtomicBool::new(false),
            #[cfg(test)]
            before_finish: parking_lot::Mutex::new(None),
        })
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
                self.audit.outcome(outcome);
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
        stages: &HostStageEvidenceV1,
    ) -> ProspectiveCaptureReceiptV1 {
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
        if settled.validate_host_stages(stages).is_err()
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
    pub(super) fn queue_result(&self, result: &Result<u64, CostSampleDrop>) {
        if self.receipt.get().is_some() && !self.queue_recorded.swap(true, Ordering::AcqRel) {
            self.audit.add(if result.is_ok() {
                &self.audit.queued
            } else {
                &self.audit.queue_dropped
            });
        }
    }
}
impl Drop for ProspectiveCapture {
    fn drop(&mut self) {
        self.finish(ProspectiveCaptureOutcomeV1::Abandoned);
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
        let audit = self.prospective_capture.as_ref()?;
        audit.add(&audit.declared);
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
                baseline,
                source,
                declared_at_ns,
                valid_until,
                audit: audit.clone(),
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
