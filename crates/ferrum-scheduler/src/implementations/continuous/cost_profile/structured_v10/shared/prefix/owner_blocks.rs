//! The source8 driver ledger reuses the original preparation/release validator.
//! These serializable values are untrusted evidence, never live capabilities.
use super::*;
use crate::implementations::continuous::cost_profile::structured_v10::lifecycle::{
    Lifecycle, LifecycleMode,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StructuredPreparationEventV8(PreparationRecord);
impl StructuredPreparationEventV8 {
    pub fn from_diagnostic(value: serde_json::Value) -> Result<Self, CostProfileError> {
        Ok(Self(serde_json::from_value(value)?))
    }
    pub(in super::super::super) fn position(&self) -> Option<(u64, u64, u64, u64, usize, u64)> {
        match &self.0 {
            PreparationRecord::PreparationCompleted {
                offered,
                queue: Some(q),
                host_stages: Some(s),
                ..
            } => Some((
                *offered,
                q.accepted_ordinal?,
                s.prepare_started_at_ns?,
                s.call_id,
                s.rows.len(),
                s.finalized_at_ns?,
            )),
            _ => None,
        }
    }
    pub(in super::super::super) fn offered(&self) -> Option<u64> {
        match self.0 {
            PreparationRecord::PreparationOffered { offered, .. } => Some(offered),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StructuredCohortEventV8(Record);
impl StructuredCohortEventV8 {
    pub fn from_diagnostic(value: serde_json::Value) -> Result<Self, CostProfileError> {
        let out = Self(serde_json::from_value(value)?);
        out.validate()?;
        Ok(out)
    }
    fn validate(&self) -> Result<(), CostProfileError> {
        if !matches!(
            self.0,
            Record::CohortBegin { .. }
                | Record::RequestAdmitted { .. }
                | Record::RequestCompleted { .. }
                | Record::CohortEnd { .. }
        ) {
            return Err(invalid("source8 non-lifecycle event in cohort ledger"));
        }
        Ok(())
    }
}

pub(in super::super::super) struct PreparedCohortLedgerV8 {
    plan: CohortPlanV2,
    preparation: Preparation,
    lifecycle: Lifecycle,
    active: Option<(usize, usize)>,
    preparation_offered: u64,
    last_finalized: u64,
}
impl PreparedCohortLedgerV8 {
    #[allow(clippy::too_many_arguments)]
    pub(in super::super::super) fn no_submission(
        &mut self,
        preparation: Option<(usize, usize)>,
        offered: u64,
        call: u64,
        finalized: u64,
        participants: &[ferrum_interfaces::execution_cost::CallNoSubmissionParticipantV1],
    ) -> Result<(), CostProfileError> {
        let (phase, cohort) = self.cohort()?;
        self.lifecycle.not_submitted(phase, cohort, participants)?;
        if let Some(expected) = preparation {
            if expected != (phase, cohort) || offered != self.preparation_offered {
                return Err(invalid(
                    "source8 preparation no-submission phase/offer differs",
                ));
            }
            self.preparation
                .not_submitted(offered, phase, cohort, participants)?;
        } else if !self.preparation.idle() {
            return Err(invalid(
                "source8 ordinary no-submission hides pending preparation",
            ));
        }
        if finalized < self.last_finalized {
            return Err(invalid("source8 no-submission final clock regressed"));
        }
        self.preparation.observed_call(call)?;
        self.last_finalized = finalized;
        Ok(())
    }
    pub(in super::super::super) fn outside(
        &mut self,
        rows: &[crate::implementations::continuous::cost_profile::structured_v10::lifecycle::OriginalCohortRow<'_>],
        s: &Stages,
        fifo: u64,
    ) -> Result<(), CostProfileError> {
        let (phase, cohort) = self.cohort()?;
        self.preparation.outside_completed(rows)?;
        self.preparation.observed_call(s.call_id)?;
        let phase = match phase {
            0 => StructuredProfilePhaseV10::Fit,
            1 => StructuredProfilePhaseV10::Residual,
            _ => StructuredProfilePhaseV10::Qualification,
        };
        self.lifecycle
            .outside_completed(phase, cohort, rows, s, fifo)?;
        self.last_finalized = s
            .finalized_at_ns
            .ok_or_else(|| invalid("source8 outside settlement lacks final clock"))?;
        Ok(())
    }
    pub(in super::super::super) fn new(
        plan: CohortPlanV2,
        prefix: StructuredPrefixPlanV5,
        native: Option<crate::implementations::continuous::cost_profile::structured_v10::service::StructuredNativePrefixAcquisitionPlanV1>,
    ) -> Self {
        Self {
            lifecycle: Lifecycle::with_mode(
                plan.clone(),
                LifecycleMode::OriginalInstalledPlainText,
            ),
            preparation: Preparation::with_native_acquisition(
                prefix,
                LifecycleMode::OriginalInstalledPlainText,
                native,
            ),
            plan,
            active: None,
            preparation_offered: 0,
            last_finalized: 0,
        }
    }
    pub(in super::super::super) fn idle(&self) -> bool {
        self.preparation.idle()
    }
    pub(in super::super::super) fn complete(&self) -> Result<(), CostProfileError> {
        self.preparation.ready()?;
        for phase in 0..3 {
            self.lifecycle.freeze(phase)?;
        }
        Ok(())
    }
    pub(in super::super::super) fn cohort(&self) -> Result<(usize, usize), CostProfileError> {
        self.active
            .ok_or_else(|| invalid("source8 physical work outside original cohort"))
    }
    pub(in super::super::super) fn event(
        &mut self,
        e: &StructuredCohortEventV8,
        limits: &CostProfileLoadLimits,
    ) -> Result<(), CostProfileError> {
        e.validate()?;
        match &e.0 {
            Record::CohortBegin {
                phase,
                cohort,
                manifest_case,
                repetition,
            } => {
                self.lifecycle
                    .begin(phase.index(), *cohort, *manifest_case, *repetition)?;
                self.preparation.begin(phase.index(), *cohort, &self.plan)?;
                self.active = Some((phase.index(), *cohort));
            }
            Record::RequestAdmitted {
                phase,
                cohort,
                slot,
                request_id,
                maximum_output,
            } => {
                self.lifecycle.admit(
                    phase.index(),
                    *cohort,
                    *slot,
                    request_id.clone(),
                    *maximum_output,
                    limits.max_source_field_bytes.get(),
                )?;
                self.preparation.admit(*slot, request_id)?;
            }
            Record::RequestCompleted { request } => {
                self.lifecycle.request_completed(request.clone())?
            }
            Record::CohortEnd {
                phase,
                cohort,
                admitted_count,
                completed_count,
            } => {
                self.lifecycle
                    .end(phase.index(), *cohort, *admitted_count, *completed_count)?;
                self.preparation.end()?;
                self.active = None;
            }
            _ => unreachable!(),
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(in super::super::super) fn prepare(
        &mut self,
        e: &StructuredPreparationEventV8,
        offered: u64,
        maximum_offered: u64,
        last_fifo: u64,
        fingerprint: &ProfileFingerprint,
        source_opened_at_ns: u64,
        earliest: u64,
        limits: &CostProfileLoadLimits,
    ) -> Result<(bool, u64), CostProfileError> {
        let (phase, _) = self.cohort()?;
        if e.offered().is_some() {
            self.preparation_offered = offered;
        }
        let mut last_fifo = last_fifo;
        // The population core checks its complete unique-call ledger before
        // this local validation and commits the same call only after success.
        let mut calls = HashSet::new();
        let mut rows = 0;
        let settled = self.preparation.handle(
            serde_json::to_value(&e.0)?,
            &mut Progress {
                phase,
                offered: &mut self.preparation_offered,
                maximum_offered,
                last_fifo: &mut last_fifo,
                last_finalized: &mut self.last_finalized,
                earliest,
                calls: &mut calls,
                total_rows: &mut rows,
                fingerprint,
                source_opened_at_ns,
                limits,
                lifecycle: &mut self.lifecycle,
            },
        )?;
        Ok((settled, last_fifo))
    }
    pub(in super::super::super) fn ordinary(
        &mut self,
        p: &Prepared,
        s: &Stages,
        fifo: u64,
    ) -> Result<(), CostProfileError> {
        let (phase, cohort) = self.cohort()?;
        self.preparation.prepared(p)?;
        self.preparation.observed_call(s.call_id)?;
        self.lifecycle.prepared(phase, cohort, p)?;
        let phase = match phase {
            0 => StructuredProfilePhaseV10::Fit,
            1 => StructuredProfilePhaseV10::Residual,
            _ => StructuredProfilePhaseV10::Qualification,
        };
        self.lifecycle.completed(phase, cohort, p, s, fifo)?;
        self.preparation.ordinary_completed(p)?;
        self.last_finalized = s
            .finalized_at_ns
            .ok_or_else(|| invalid("source8 incomplete ordinary clock"))?;
        Ok(())
    }
    pub(in super::super::super) fn retained_payload_bytes(&self) -> Option<usize> {
        let mut n = std::mem::size_of::<Self>()
            .checked_add(self.lifecycle.retained_heap_bytes()?)?
            .checked_add(self.preparation.retained_heap_bytes()?)?;
        for phase in &self.plan.phases {
            n = n.checked_add(phase.capacity().checked_mul(std::mem::size_of::<
                crate::implementations::continuous::cost_model::structured_v2::windows::CohortV2,
            >())?)?;
            for c in phase {
                n = n.checked_add(c.requests.capacity().checked_mul(std::mem::size_of::<crate::implementations::continuous::cost_model::structured_v2::windows::CohortRequestV2>())?)?;
            }
        }
        Some(n)
    }
}
