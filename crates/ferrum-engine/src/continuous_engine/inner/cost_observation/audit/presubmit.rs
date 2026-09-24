//! Separate issued-bound audit; never feeds the retrospective margin policy.
use super::super::CostEvidenceEntry;
use super::{add_one, SelectedServingEvaluation};

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct PresubmitAudit {
    pub drained: u64,
    pub incomplete_or_invalid_actual: u64,
    pub no_issued_bound: u64,
    pub exact_mismatch: u64,
    pub invalid_bound: u64,
    pub compared: u64,
    pub terminal_compared: u64,
    pub underestimates: u64,
    pub total_underestimate_ns: u64,
    pub maximum_underestimate_ns: u64,
}
impl PresubmitAudit {
    pub fn record(
        &mut self,
        entry: &CostEvidenceEntry,
        evaluation: &SelectedServingEvaluation,
        exhausted: &mut bool,
    ) {
        add_one(&mut self.drained, exhausted);
        let SelectedServingEvaluation::Complete { terminal, .. } = evaluation else {
            add_one(&mut self.incomplete_or_invalid_actual, exhausted);
            return;
        };
        let stages = match entry {
            CostEvidenceEntry::Training { stages, .. } => stages.as_deref(),
            CostEvidenceEntry::StagesOnly { stages, .. } => Some(stages.as_ref()),
        };
        let Some(stages) = stages else {
            add_one(&mut self.incomplete_or_invalid_actual, exhausted);
            return;
        };
        let Some(receipt) = &stages.presubmit_prediction else {
            add_one(&mut self.no_issued_bound, exhausted);
            return;
        };
        if !receipt.exact_actual_matches {
            add_one(&mut self.exact_mismatch, exhausted);
            return;
        }
        let Some(actual) = stages.full_wall_ns.filter(|n| *n > 0) else {
            add_one(&mut self.invalid_bound, exhausted);
            return;
        };
        if receipt.schema_version != 1 || receipt.model_epoch == 0 || receipt.planning_ns == 0 {
            add_one(&mut self.invalid_bound, exhausted);
            return;
        }
        add_one(&mut self.compared, exhausted);
        if *terminal {
            add_one(&mut self.terminal_compared, exhausted);
        }
        let error = actual.saturating_sub(receipt.planning_ns);
        if error > 0 {
            add_one(&mut self.underestimates, exhausted);
            self.maximum_underestimate_ns = self.maximum_underestimate_ns.max(error);
            match self.total_underestimate_ns.checked_add(error) {
                Some(n) => self.total_underestimate_ns = n,
                None => *exhausted = true,
            }
        }
        tracing::trace!(target: "ferrum::selected_presubmit_audit", call_id = stages.call_id,
            model_epoch = receipt.model_epoch, profile_sha256 = ?receipt.profile_sha256,
            planning_ns = receipt.planning_ns, actual_ns = actual, underestimate_ns = error,
            "same-call complete actual wave compared to the issued witness; separate from retrospective lookup");
    }
}
