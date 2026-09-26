//! Finite pre-wave diagnostics using the existing complete-request driver.
//! The caller declares trigger attempts or physical frontier stages and paths
//! before any request executes. Readiness is independent of cost availability.
use super::*;
use ferrum_engine::continuous_engine::{
    CalibrationFrontier, CalibrationSession, RequiredFutureAuditCostV2, RequiredFutureAuditPlanV2,
    RequiredFutureAuditReportV2,
};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU64;
mod trigger;
use trigger::{AuditFrontierTriggerV2, StageMatch, TriggerProgress, UntriggeredReason};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AuditConfigV2 {
    pub triggers: Vec<AuditTriggerV2>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AuditTriggerV2 {
    pub case_index: usize,
    pub repetition: usize,
    /// One-based nonempty Wave attempt in this complete cohort. A rejected
    /// physical attempt still consumes its ordinal; no retry until Known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_wave_attempt: Option<NonZeroU64>,
    /// Physical readiness is read only after the declared stage matches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<AuditFrontierTriggerV2>,
    pub plan: RequiredFutureAuditPlanV2,
}
impl AuditConfigV2 {
    #[cfg(test)]
    fn trigger(
        &self,
        case: usize,
        repetition: usize,
        attempt: u64,
    ) -> Option<(usize, &AuditTriggerV2)> {
        self.triggers.iter().enumerate().find(|(_, trigger)| {
            (
                trigger.case_index,
                trigger.repetition,
                trigger.before_wave_attempt.map(NonZeroU64::get),
            ) == (case, repetition, Some(attempt))
        })
    }
    pub(super) fn validate(&self, manifest: &manifest::Manifest) -> Result<()> {
        let invalid = || FerrumError::config("invalid finite required-future audit declaration");
        if self.triggers.is_empty()
            || self.triggers.len() > 64
            || manifest.reference.is_some()
            || !manifest.validation.is_empty()
        {
            return Err(invalid());
        }
        let mut keys = std::collections::BTreeSet::new();
        let mut queries = 0usize;
        let mut coordinates = 0usize;
        for trigger in &self.triggers {
            let case = manifest
                .training
                .get(trigger.case_index)
                .ok_or_else(invalid)?;
            if trigger.repetition >= case.repetitions.get()
                || trigger.before_wave_attempt.is_some() == trigger.when.is_some()
                || trigger
                    .before_wave_attempt
                    .is_some_and(|n| n.get() > manifest.protocol.maximum_wave_attempts.get())
                || (trigger.when.is_some() && case.rolling_window.is_some())
                || !keys.insert((
                    trigger.case_index,
                    trigger.repetition,
                    trigger.before_wave_attempt,
                    trigger.when,
                ))
            {
                return Err(invalid());
            }
            trigger.plan.validate(case.maximum_in_flight())?;
            queries = queries
                .checked_add(trigger.plan.limits.maximum_queries.get())
                .filter(|n| *n <= 8192)
                .ok_or_else(invalid)?;
            coordinates = coordinates
                .checked_add(trigger.plan.limits.maximum_coordinates.get())
                .filter(|n| *n <= 1_048_576)
                .ok_or_else(invalid)?;
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub(super) struct AuditSummaryV2 {
    pub scope: &'static str,
    pub declared_triggers: usize,
    pub attempted_triggers: usize,
    pub structurally_complete_triggers: usize,
    pub unavailable_triggers: usize,
    pub truncated_triggers: usize,
    pub known_cost_queries: usize,
    pub cost_unknown_reasons: std::collections::BTreeMap<String, usize>,
    pub input_unknown_reasons: std::collections::BTreeMap<String, usize>,
    pub remaining_trigger_indices: Vec<usize>,
    pub trigger_progress: Vec<TriggerProgress>,
    pub collection_completed: bool,
    pub all_declared_requirements_recorded: bool,
}
impl AuditSummaryV2 {
    fn new(count: usize) -> Self {
        Self {
            scope: "only_predeclared_finite_paths_not_all_planner_candidates_or_feasibility",
            declared_triggers: count,
            attempted_triggers: 0,
            structurally_complete_triggers: 0,
            unavailable_triggers: 0,
            truncated_triggers: 0,
            known_cost_queries: 0,
            cost_unknown_reasons: Default::default(),
            input_unknown_reasons: Default::default(),
            remaining_trigger_indices: (0..count).collect(),
            trigger_progress: (0..count).map(|_| TriggerProgress::default()).collect(),
            collection_completed: false,
            all_declared_requirements_recorded: false,
        }
    }
    fn record(&mut self, index: usize, report: Option<&RequiredFutureAuditReportV2>) -> Result<()> {
        let position = self
            .remaining_trigger_indices
            .iter()
            .position(|i| *i == index)
            .ok_or_else(|| FerrumError::internal("required-future trigger repeated"))?;
        self.remaining_trigger_indices.remove(position);
        self.attempted_triggers += 1;
        let Some(report) = report else {
            self.unavailable_triggers += 1;
            return Ok(());
        };
        self.structurally_complete_triggers += usize::from(report.completed_all_declared_paths);
        self.unavailable_triggers += usize::from(
            report.unavailable.is_some() || report.paths.iter().any(|p| p.stopped.is_some()),
        );
        self.truncated_triggers += usize::from(report.truncated);
        for query in &report.queries {
            if let Some(reason) = &query.input_unknown {
                *self
                    .input_unknown_reasons
                    .entry(reason.clone())
                    .or_default() += 1;
            }
            match &query.cost {
                RequiredFutureAuditCostV2::KnownAtRead { .. } => self.known_cost_queries += 1,
                RequiredFutureAuditCostV2::Unknown { reason, .. } => {
                    *self.cost_unknown_reasons.entry(reason.clone()).or_default() += 1;
                }
                RequiredFutureAuditCostV2::NotQueried { .. } => {}
            }
        }
        Ok(())
    }
    fn finish(&mut self) {
        self.collection_completed = true;
        self.all_declared_requirements_recorded = self.remaining_trigger_indices.is_empty()
            && self.structurally_complete_triggers == self.declared_triggers;
    }
}

pub(super) async fn collect(
    session: &mut CalibrationSession,
    manifest: &manifest::Manifest,
    inputs: &inputs::PreparedInputs,
    artifacts: &mut report::Artifacts,
    summary: &mut report::Summary,
) -> Result<()> {
    let config = manifest
        .validation_model
        .required_audit()
        .ok_or_else(|| FerrumError::internal("missing required-future audit declaration"))?;
    config.validate(manifest)?;
    summary.required_future_audit_v2 = Some(AuditSummaryV2::new(config.triggers.len()));
    for (phase, cases) in [
        (report::Phase::Warmup, manifest.validation_model.warmup_v2()),
        (report::Phase::Discovery, manifest.training.as_slice()),
    ] {
        for (index, case) in cases.iter().enumerate() {
            for repetition in 0..case.repetitions.get() {
                driver::cohort(
                    session, manifest, inputs, case, phase, index, repetition, None, None,
                    artifacts, summary,
                )
                .await?;
            }
        }
    }
    let result = summary
        .required_future_audit_v2
        .as_mut()
        .expect("initialized above");
    result.finish();
    artifacts.record(&serde_json::json!({
        "schema_version": 1, "event": "required_future_audit_completed", "summary": result,
    }))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn before_wave(
    session: &CalibrationSession,
    manifest: &manifest::Manifest,
    case_index: usize,
    repetition: usize,
    attempt: u64,
    frontiers: &[CalibrationFrontier],
    full_population: usize,
    artifacts: &mut report::Artifacts,
    totals: &mut report::Summary,
) -> Result<()> {
    let config = manifest
        .validation_model
        .required_audit()
        .ok_or_else(|| FerrumError::internal("missing required-future audit declaration"))?;
    for (index, trigger) in config.triggers.iter().enumerate() {
        if trigger.case_index != case_index || trigger.repetition != repetition {
            continue;
        }
        let summary = totals
            .required_future_audit_v2
            .as_mut()
            .ok_or_else(|| FerrumError::internal("required-future audit summary missing"))?;
        if !summary.remaining_trigger_indices.contains(&index)
            || summary.trigger_progress[index].untriggered_reason.is_some()
        {
            continue;
        }
        if let Some(when) = trigger.when {
            let progress = &mut summary.trigger_progress[index];
            match when.match_stage(
                manifest.training[case_index].prompts.len(),
                frontiers
                    .iter()
                    .map(|f| (f.generated_tokens(), f.prefill_progress())),
            ) {
                StageMatch::Before => continue,
                StageMatch::Passed => {
                    progress.untriggered_reason =
                        Some(UntriggeredReason::StagePassedBeforeReadiness);
                    continue;
                }
                StageMatch::PopulationChanged => {
                    progress.untriggered_reason =
                        Some(UntriggeredReason::OriginalCohortNoLongerComplete);
                    continue;
                }
                StageMatch::At => progress.stage_seen = true,
            }
            progress.readiness_checks = progress
                .readiness_checks
                .checked_add(1)
                .ok_or_else(|| FerrumError::resource_exhausted("audit readiness count overflow"))?;
            let ready = match session.audit_frontier_readiness_v2(frontiers) {
                Ok(readiness) => {
                    let ready = readiness.all_ready();
                    progress.last_readiness = Some(readiness);
                    progress.last_readiness_error = None;
                    ready
                }
                Err(error) => {
                    progress.last_readiness = None;
                    progress.last_readiness_error = Some(error.to_string());
                    false
                }
            };
            artifacts.record(&serde_json::json!({
                "schema_version": 1, "event": "required_future_trigger_readiness",
                "trigger_index": index, "case": case_index, "repetition": repetition,
                "before_wave_attempt": attempt, "declaration": trigger,
                "progress": progress, "ready": ready,
            }))?;
            if !ready {
                continue;
            }
        } else if trigger.before_wave_attempt.map(NonZeroU64::get) != Some(attempt) {
            continue;
        }
        let result = if frontiers.len() != full_population {
            Err(FerrumError::invalid_request(
                "audit cohort ordering omitted a live owner",
            ))
        } else {
            session
                .audit_required_owners_v2(frontiers, &trigger.plan)
                .await
        };
        let (audit, error) = match result {
            Ok(report) => (Some(report), None),
            Err(error) => (None, Some(error.to_string())),
        };
        totals
            .required_future_audit_v2
            .as_mut()
            .expect("checked above")
            .record(index, audit.as_ref())?;
        // Once fired, either mode consumes its declaration even on Unknown.
        // Readiness never suppresses ordinary admission, Wave or request drain.
        artifacts.record(&serde_json::json!({
            "schema_version": 1, "event": "required_future_owner_audit", "phase": "discovery",
            "trigger_index": index, "case": case_index, "repetition": repetition,
            "before_wave_attempt": attempt, "declaration": trigger,
            "report": audit, "error": error,
        }))?;
    }
    Ok(())
}

pub(super) fn cohort_completed(
    manifest: &manifest::Manifest,
    case: usize,
    repetition: usize,
    artifacts: &mut report::Artifacts,
    totals: &mut report::Summary,
) -> Result<()> {
    let config = manifest
        .validation_model
        .required_audit()
        .ok_or_else(|| FerrumError::internal("missing required-future audit declaration"))?;
    let summary = totals
        .required_future_audit_v2
        .as_mut()
        .ok_or_else(|| FerrumError::internal("required-future audit summary missing"))?;
    for (index, trigger) in config.triggers.iter().enumerate() {
        if trigger.case_index == case
            && trigger.repetition == repetition
            && summary.remaining_trigger_indices.contains(&index)
        {
            let progress = &mut summary.trigger_progress[index];
            progress
                .untriggered_reason
                .get_or_insert(UntriggeredReason::CohortCompletedBeforeTrigger);
            artifacts.record(&serde_json::json!({
                "schema_version": 1, "event": "required_future_trigger_untriggered",
                "trigger_index": index, "case": case, "repetition": repetition,
                "declaration": trigger, "progress": progress,
            }))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
