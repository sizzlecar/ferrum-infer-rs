//! Bounded cold-failure facts. No request IDs, prompt/token data, or shapes are
//! retained. This never changes the original receipt validator's decision.
use super::*;
use crate::continuous_engine::inner::cost_observation::host_stages::StructuredSettlementUnknown;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::continuous_engine::inner::cost_observation) enum PrivateSettlementDiagnostic {
    Missing,
    ProducerRejected(StructuredSettlementUnknown),
    BindingRejected(StructuredSettlementUnknown),
    Qualified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::continuous_engine::inner::cost_observation) enum RecipeDiagnostic {
    Missing,
    Rejected(StatisticalEvidenceUnknown),
    Present,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct SettlementFailureDiagnostic {
    pub accepted_ordinal: u64,
    pub call_id: Option<u64>,
    #[serde(serialize_with = "serialize_model_unknown")]
    pub model_error: ModelUnknown,
    pub host_content_rejection: Option<HostContentRejection>,
    pub legacy_rejection: Option<CostCallRejection>,
    pub host_stage_completeness: Option<HostStageCompleteness>,
    pub private_settlement: PrivateSettlementDiagnostic,
    pub selected_evidence_present: bool,
    pub recipe: RecipeDiagnostic,
    pub numeric_features_present: bool,
    pub row_multiset_features_present: bool,
}

// ModelUnknown and its nested evidence error are closed, payload-free enums.
// Preserve the typed value in memory without adding serialization to the model.
fn serialize_model_unknown<S: serde::Serializer>(
    value: &ModelUnknown,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_str(&format_args!("{value:?}"))
}

impl SettlementFailureDiagnostic {
    /// Called only after the unchanged validator failed and only for the first
    /// failed entry in this generation. Rechecking the cold sample exposes the
    /// original HostContentRejection hidden behind ModelUnknown::InvalidSample.
    pub fn capture(
        entry: &CostEvidenceEntry,
        accepted_ordinal: u64,
        model_error: ModelUnknown,
    ) -> Self {
        let (stages, legacy_rejection) = match entry {
            CostEvidenceEntry::Training { stages, .. } => (stages.as_deref(), None),
            CostEvidenceEntry::StagesOnly {
                stages,
                legacy_rejection,
            } => (Some(stages.as_ref()), Some(*legacy_rejection)),
        };
        let selected = stages.and_then(|value| value.statistical_evidence.as_ref());
        let shape = stages.and_then(|value| value.actual_shape.as_ref());
        let private_settlement = match stages.and_then(|value| {
            value
                .structured_evidence
                .as_ref()
                .map(|private| (value, private))
        }) {
            None => PrivateSettlementDiagnostic::Missing,
            Some((_, Err(reason))) => PrivateSettlementDiagnostic::ProducerRejected(*reason),
            Some((stages, Ok(private))) => match private.validate_host_stages(stages) {
                Ok(()) => PrivateSettlementDiagnostic::Qualified,
                Err(reason) => PrivateSettlementDiagnostic::BindingRejected(reason),
            },
        };
        Self {
            accepted_ordinal,
            call_id: stages.map(|value| value.call_id),
            model_error,
            host_content_rejection: sample(stages, legacy_rejection, true).err(),
            legacy_rejection,
            host_stage_completeness: stages.map(|value| value.completeness),
            private_settlement,
            selected_evidence_present: selected.is_some(),
            recipe: match selected.and_then(|value| value.structured_capture()) {
                None => RecipeDiagnostic::Missing,
                Some(Err(reason)) => RecipeDiagnostic::Rejected(reason),
                Some(Ok(_)) => RecipeDiagnostic::Present,
            },
            numeric_features_present: shape.is_some_and(|value| value.numeric_features.is_some()),
            row_multiset_features_present: shape
                .is_some_and(|value| value.row_multiset_features.is_some()),
        }
    }
}
