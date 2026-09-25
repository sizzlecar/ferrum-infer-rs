//! Engine-private pre-execution facts minted from the real published route.
//! No deserializer, public receipt factory, or post-execution reconstruction.
use ferrum_interfaces::execution_cost::{
    ActualRowWork, CanonicalWaveCostShape, StatisticalWaveEvidenceV1,
    UnsettledStructuredWaveEvidenceV1,
};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    windows::{PreparedRowFactsV2, PreparedWorkV2},
    StructuredInputV2, StructuredOwnerFactsV2, StructuredOwnerKeyV2, StructuredUnknownV2,
};
use ferrum_types::RequestId;
use serde::Serialize;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize)]
pub(in crate::continuous_engine::inner) struct PreparedRowBindingV2 {
    pub request_id: RequestId,
    pub owner_incarnation: u64,
    pub work_generation: u64,
    pub frontier: PreparedRowFactsV2,
}
pub(in crate::continuous_engine::inner) struct PreparedStructuredFactsV2 {
    pub exact: CanonicalWaveCostShape,
    pub selected: StatisticalWaveEvidenceV1,
    pub recipe: Arc<UnsettledStructuredWaveEvidenceV1>,
    pub owner: StructuredOwnerFactsV2,
    pub rows: Vec<PreparedRowBindingV2>,
}
impl PreparedStructuredFactsV2 {
    pub fn validate(&self) -> Result<StructuredOwnerKeyV2, StructuredUnknownV2> {
        let owner = StructuredInputV2::owner_for(&self.exact, &self.selected, &self.recipe)?;
        if owner != self.owner.owner_key()? || self.rows.len() != owner.rows as usize {
            return Err(StructuredUnknownV2::InvalidInput);
        }
        let numeric = self
            .exact
            .numeric_features
            .as_ref()
            .ok_or(StructuredUnknownV2::MissingEvidence)?;
        if numeric.rows.len() != self.rows.len() || self.exact.rows.len() != self.rows.len() {
            return Err(StructuredUnknownV2::InvalidInput);
        }
        for (index, row) in self.rows.iter().enumerate() {
            row.frontier.validate()?;
            if row.owner_incarnation == 0
                || row.work_generation == 0
                || row.frontier.physical_position as usize != index
                || self.rows[..index]
                    .iter()
                    .any(|old| old.request_id == row.request_id)
                || row.frontier.generated_before != numeric.rows[index].generated_tokens_before
                || row.frontier.maximum_output != numeric.rows[index].maximum_output_tokens
            {
                return Err(StructuredUnknownV2::InvalidInput);
            }
            let expected = match row.frontier.work {
                PreparedWorkV2::Decode { kv_tokens } => ActualRowWork::Decode { kv_tokens },
                PreparedWorkV2::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                } => ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                },
            };
            if self.exact.rows[index] != expected {
                return Err(StructuredUnknownV2::InvalidInput);
            }
        }
        Ok(owner)
    }
}
