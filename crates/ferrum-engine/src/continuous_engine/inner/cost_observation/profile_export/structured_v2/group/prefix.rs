//! A single physical preparation stream. Only the manual engine's private
//! captured values enter this writer; ordinary numerical receipts are unchanged.
use super::*;
use crate::continuous_engine::inner::calibration::token_preparation::{
    CapturedPrefixReleaseV5, CapturedPrefixWaveV5, PrefixPreparedRowV5,
};
use crate::continuous_engine::inner::cost_observation::host_stages::ExportEvidence;
use prefixes::{StructuredPrefixCohortV5, StructuredPrefixPlanV5, StructuredPrefixSlotV5};
use std::collections::HashSet;

pub(super) struct PrefixSourceState {
    plan: StructuredPrefixPlanV5,
    // Bind the original declaration independently of the mutable numerical
    // phase. A failed collector must still let an already-released request
    // drain its ordinary suffix to the original Length boundary.
    active: Option<(usize, usize)>,
    released: HashSet<RequestId>,
    pending: bool,
}
impl PrefixSourceState {
    pub fn new(plan: StructuredPrefixPlanV5) -> Self {
        Self {
            plan,
            active: None,
            released: HashSet::new(),
            pending: false,
        }
    }
    pub fn begin(&mut self, phase: usize, ordinal: usize) -> Result<(), ExportError> {
        if self.active.is_some() || self.pending {
            return Err(ExportError::Source("prefix previous cohort remains active"));
        }
        if self
            .plan
            .phases
            .get(phase)
            .and_then(|p| p.get(ordinal))
            .is_none()
        {
            return Err(ExportError::Source("prefix cohort not declared"));
        }
        self.active = Some((phase, ordinal));
        self.released.clear();
        Ok(())
    }
    pub fn end(&mut self) {
        self.active = None;
        self.released.clear();
    }
}
impl StructuredCalibrationGroupV2 {
    fn prefix_cohort(&self) -> Result<Option<&StructuredPrefixCohortV5>, ExportError> {
        let Some(prefix) = &self.prefix else {
            return Ok(None);
        };
        let (phase, ordinal) = prefix
            .active
            .ok_or(ExportError::Source("prefix cohort is not active"))?;
        prefix.plan.phases[phase]
            .get(ordinal)
            .map(Option::as_ref)
            .ok_or(ExportError::Source("prefix cohort not declared"))
    }
    pub fn prefix_declaration_for_next_request(
        &self,
    ) -> Result<Option<(u64, StructuredPrefixSlotV5)>, ExportError> {
        let Some(cohort) = self.prefix_cohort()? else {
            return Ok(None);
        };
        let slot = self.children[0].cohorts.next_admission_slot()?;
        Ok(Some((
            cohort.release_generated,
            cohort
                .slots
                .get(slot)
                .ok_or(ExportError::Source(
                    "prefix request exceeds declared cohort",
                ))?
                .clone(),
        )))
    }
    pub fn preparing_prefix(&self) -> Result<bool, ExportError> {
        Ok(self.prefix_cohort()?.is_some_and(|cohort| {
            self.prefix.as_ref().unwrap().released.len() < cohort.slots.len()
        }))
    }
    pub fn prefix_pending(&self) -> bool {
        self.prefix.as_ref().is_some_and(|p| p.pending)
    }
    pub fn prefix_release_generated(&self) -> Result<Option<u64>, ExportError> {
        Ok(self.prefix_cohort()?.map(|c| c.release_generated))
    }
    pub fn prefix_release_order(&self) -> Result<Vec<RequestId>, ExportError> {
        self.children[0].cohorts.admitted_ids()
    }
    pub(super) fn require_prefix_released(&self) -> Result<(), ExportError> {
        if self.preparing_prefix()? {
            return Err(ExportError::Source(
                "every cohort slot must release before ordinary measurement",
            ));
        }
        Ok(())
    }
    pub fn offer_preparation(&mut self, rows: &[PrefixPreparedRowV5]) -> Result<(), ExportError> {
        if !self.collecting() || !self.preparing_prefix()? || self.prefix_pending() {
            return Err(ExportError::Source(
                "source5 preparation offer outside declared lifecycle",
            ));
        }
        let release = self.prefix_cohort()?.unwrap().release_generated;
        if rows
            .iter()
            .any(|r| r.before().generated_tokens as u64 >= release)
        {
            return Err(ExportError::Source(
                "preparation cannot pass its common release frontier",
            ));
        }
        // All ledgers advance the same original offer/FIFO, with zero numerical
        // members. No forged Prepared facts or child physical records exist.
        for child in &mut self.children {
            child.cohorts.preparation_offered(rows)?;
            let cohort = child.cohorts.active_ordinal()?;
            child.ledger.offer(
                child.phase,
                cohort,
                child.options.maximum_offered_waves.get(),
            )?;
        }
        let first = self.children[0].ledger.attempt.as_ref().unwrap();
        #[derive(Serialize)]
        struct Offered<'a> {
            kind: &'static str,
            offered: u64,
            phase: StructuredCapturePhase,
            cohort: usize,
            rows: &'a [PrefixPreparedRowV5],
        }
        self.shared.as_mut().unwrap().record_borrowed(&Offered {
            kind: "preparation_offered",
            offered: first.offered,
            phase: first.phase,
            cohort: first.cohort,
            rows,
        })?;
        self.prefix.as_mut().unwrap().pending = true;
        Ok(())
    }
    pub fn complete_preparation(
        &mut self,
        captured: &CapturedPrefixWaveV5,
    ) -> Result<(), ExportError> {
        if !self.collecting() || !self.prefix_pending() {
            return Err(ExportError::Source(
                "source5 preparation has no original offer",
            ));
        }
        let first = self.children[0]
            .ledger
            .attempt
            .as_ref()
            .ok_or(ExportError::Source("source5 preparation attempt missing"))?;
        let e = captured.evidence();
        #[derive(Serialize)]
        struct Completed<'a> {
            kind: &'static str,
            offered: u64,
            phase: StructuredCapturePhase,
            cohort: usize,
            reconciled: bool,
            queue: Option<HostStageQueueReceipt>,
            host_stages: Option<ExportEvidence<'a>>,
            rows: &'a [crate::continuous_engine::inner::calibration::PrefixRowEvidenceV1],
            failure: Option<&'a str>,
        }
        // Even a failed actual attempt is written once before fail-closed.
        self.shared.as_mut().unwrap().record_borrowed(&Completed {
            kind: "preparation_completed", offered: first.offered, phase: first.phase, cohort: first.cohort,
            reconciled: e.submission == crate::continuous_engine::inner::calibration::CalibrationSubmissionState::HostReconciled && e.error.is_none(),
            queue: e.host_stage_queue, host_stages: e.host_stages.as_deref().map(|s| ExportEvidence::new(s, true)),
            rows: &e.rows, failure: e.error.as_deref().or(e.chain_error.as_deref()),
        })?;
        if e.chain_error.is_some() || e.error.is_some() {
            return Err(ExportError::Source(
                "source5 preparation actual receipt failed",
            ));
        }
        for child in &mut self.children {
            child.cohorts.preparation_completed(captured)?;
            child.ledger.accept_fifo(e.host_stage_queue)?;
            child
                .ledger
                .attempt
                .take()
                .ok_or(ExportError::Source("source5 child offer missing"))?;
        }
        self.prefix.as_mut().unwrap().pending = false;
        Ok(())
    }
    pub fn release_preparation(
        &mut self,
        captured: &CapturedPrefixReleaseV5,
    ) -> Result<(), ExportError> {
        if !self.collecting() || self.prefix_pending() {
            return Err(ExportError::Source(
                "source5 release before complete physical wave",
            ));
        }
        let first = &self.children[0];
        let receipt = captured.receipt();
        let slot = first.cohorts.request_slot(&receipt.frontier.request_id)?;
        let declaration = self
            .prefix_cohort()?
            .ok_or(ExportError::Source("ordinary cohort has no prefix release"))?;
        let declared = declaration
            .slots
            .get(slot)
            .ok_or(ExportError::Source("undeclared prefix release slot"))?;
        if receipt.frontier.generated_tokens as u64 != declaration.release_generated
            || receipt.frontier.pending_utf8 != declared.expected_pending().map_err(numeric_error)?
            // The sealed receipt binds this slot's actual final preparation
            // call; other slots may have settled later in the same cohort.
            || receipt.through_fifo_ordinal == 0 || receipt.through_fifo_ordinal > first.ledger.last_fifo
            || receipt.actor_applied_output_ordinal != receipt.frontier.output_accepted_ordinal
            || !self.prefix.as_mut().unwrap().released.insert(receipt.frontier.request_id.clone())
        {
            return Err(ExportError::Source(
                "source5 release differs from declared actual frontier",
            ));
        }
        #[derive(Serialize)]
        struct Released<'a> {
            kind: &'static str,
            phase: StructuredCapturePhase,
            cohort: usize,
            slot: usize,
            receipt: &'a crate::continuous_engine::inner::calibration::PrefixReleasedV1,
        }
        self.shared.as_mut().unwrap().record_borrowed(&Released {
            kind: "preparation_released",
            phase: first.phase,
            cohort: first.cohorts.active_ordinal()?,
            slot,
            receipt,
        })
    }
}
