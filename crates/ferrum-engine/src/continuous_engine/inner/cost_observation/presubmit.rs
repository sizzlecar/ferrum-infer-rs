//! Correlates the actual selected witness to its call's complete host receipt.
//! No authority/lease is retained, and no retrospective lookup is called an
//! issued prediction. Legacy commit-boundary costs are deliberately excluded.
use super::*;
use ferrum_interfaces::execution_cost::CanonicalWaveCostShape;

pub(super) struct PendingPrediction {
    profile_sha256: [u8; 32],
    epoch: u64,
    planning_ns: u64,
    exact: CanonicalWaveCostShape,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct PresubmitPredictionReceiptV1 {
    pub schema_version: u32,
    pub profile_sha256: [u8; 32],
    pub model_epoch: u64,
    pub planning_ns: u64,
    pub exact_actual_matches: bool,
}

impl PendingPrediction {
    pub(super) fn receipt(&self, actual: &ActualWaveShape) -> PresubmitPredictionReceiptV1 {
        let a = &self.exact;
        PresubmitPredictionReceiptV1 {
            schema_version: 1,
            profile_sha256: self.profile_sha256,
            model_epoch: self.epoch,
            planning_ns: self.planning_ns,
            exact_actual_matches: a.kind == actual.kind
                && a.path == actual.path
                && a.graph == actual.graph
                && a.row_order == actual.row_order
                && a.provider_signature == actual.provider_signature
                && a.output_policy_signature == actual.output_policy_signature
                && a.numeric_features == actual.numeric_features
                && a.host_content_features == actual.host_content_features
                && a.row_multiset_features == actual.row_multiset_features
                && a.recurrent_state_bytes == actual.recurrent_state_bytes
                && a.rows.len() == actual.rows.len()
                && a.rows
                    .iter()
                    .zip(&actual.rows)
                    .all(|(work, row)| work == &row.work),
        }
    }
}

impl EngineCostRuntime {
    pub(in crate::continuous_engine) fn attach_selected_witness_prediction(
        &self,
        call: &mut EngineCostCall,
        epoch: u64,
        planning_ns: u64,
        exact: &CanonicalWaveCostShape,
    ) {
        // Same immutable model and dynamic validity as the issuing controller;
        // the mandatory host guard still makes the final admission decision.
        let Some(snapshot) = self
            .try_snapshot()
            .flatten()
            .filter(|s| s.feedback_enabled() && s.model_version() == epoch)
        else {
            return;
        };
        let Some(model) = snapshot.selected_import() else {
            return;
        };
        if planning_ns == 0 || call.context_created || call.presubmit_prediction.is_some() {
            return;
        }
        call.presubmit_prediction = Some(PendingPrediction {
            profile_sha256: model.file_sha256,
            epoch,
            planning_ns,
            exact: exact.clone(),
        });
    }
}
