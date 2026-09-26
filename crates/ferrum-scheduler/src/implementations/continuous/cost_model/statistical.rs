//! Shared actual/future work aggregation for a separately versioned
//! whole-wave model. This does not change any existing key, predictor, profile,
//! training eligibility, min-sample rule, TTL or runtime cost-model setting.
pub mod model;
use ferrum_interfaces::execution_cost::{
    ActualRowWork, ActualWaveShape, CanonicalWaveCostFeatures, CanonicalWaveCostShape,
    DeviceNumericWorkV1, StatisticalEvidenceUnknown as Unknown, StatisticalWaveEvidenceV1,
    MAX_COST_ROWS,
};

/// Units are counts or bytes, never additive latency estimates. Dimensions are
/// interpreted jointly within a complete selected-algorithm family. The opt-in `model` child implements the independent fit/residual protocol.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostAndSequenceNumericWorkV1 {
    pub rows: u64,
    pub prefill_tokens: u64,
    pub attention_pairs: u64,
    pub kv_tokens_sum: u64,
    pub kv_tokens_max: u64,
    pub prompt_tokens_sum: u64,
    pub prompt_tokens_max: u64,
    pub generated_tokens_sum: u64,
    pub output_budget_sum: u64,
    pub sampling_history_sum: u64,
    pub sampling_history_max: u64,
    pub repetition_tokens_sum: u64,
    pub decoded_prefix_sum: u64,
    pub decoded_text_bytes_sum: u64,
    pub decode_scratch_bytes_sum: u64,
    pub recurrent_bytes: u64,
}
/// Explicit empirical family interpretation, independent of execution authority.
/// OrderedV1 remains the default entry point and profile-6 protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedStatisticalFamily {
    OrderedV1,
    IndependentAttentionV2,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatisticalModelInputV1 {
    family_signature: [u8; 32],
    independent_attention_family_v2: Option<[u8; 32]>,
    device: DeviceNumericWorkV1,
    host_and_sequence: HostAndSequenceNumericWorkV1,
}
impl StatisticalModelInputV1 {
    pub fn family_signature(&self) -> &[u8; 32] {
        &self.family_signature
    }
    pub fn family_signature_for(
        &self,
        family: SelectedStatisticalFamily,
    ) -> Result<&[u8; 32], Unknown> {
        match family {
            SelectedStatisticalFamily::OrderedV1 => Ok(&self.family_signature),
            SelectedStatisticalFamily::IndependentAttentionV2 => self
                .independent_attention_family_v2
                .as_ref()
                .ok_or(Unknown::MissingProducer),
        }
    }
    pub fn device(&self) -> DeviceNumericWorkV1 {
        self.device
    }
    pub fn host_and_sequence(&self) -> HostAndSequenceNumericWorkV1 {
        self.host_and_sequence
    }

    pub fn from_future(
        shape: &CanonicalWaveCostShape,
        evidence: &StatisticalWaveEvidenceV1,
    ) -> Result<Self, Unknown> {
        if shape.graph != ferrum_interfaces::execution_cost::ActualWaveGraphState::Disabled {
            return Err(Unknown::UnsupportedReplay);
        }
        evidence.validate_exact(shape)?;
        Self::aggregate(
            evidence,
            shape.numeric_features.as_ref(),
            shape.rows.iter().copied(),
            shape.recurrent_state_bytes,
        )
    }
    /// Numerical projection shared only by the V2 live recipe validator and
    /// profile10 original-source replay. Does not construct live evidence or
    /// broaden legacy selected/V1 entrypoints.
    pub(super) fn from_future_structured_v2(
        shape: &CanonicalWaveCostShape,
        evidence: &StatisticalWaveEvidenceV1,
    ) -> Result<Self, Unknown> {
        if !matches!(
            shape.graph,
            ferrum_interfaces::execution_cost::ActualWaveGraphState::Disabled
                | ferrum_interfaces::execution_cost::ActualWaveGraphState::Warm
                | ferrum_interfaces::execution_cost::ActualWaveGraphState::ConfiguredEager
        ) {
            return Err(Unknown::UnsupportedReplay);
        }
        evidence.validate_exact(shape)?;
        Self::aggregate(
            evidence,
            shape.numeric_features.as_ref(),
            shape.rows.iter().copied(),
            shape.recurrent_state_bytes,
        )
    }
    pub fn from_actual(shape: &ActualWaveShape) -> Result<Self, Unknown> {
        if shape.graph != ferrum_interfaces::execution_cost::ActualWaveGraphState::Disabled {
            return Err(Unknown::UnsupportedReplay);
        }
        let evidence = shape
            .statistical_evidence
            .as_ref()
            .ok_or(Unknown::MissingProducer)?;
        evidence.validate_actual(shape)?;
        Self::aggregate(
            evidence,
            shape.numeric_features.as_ref(),
            shape.rows.iter().map(|r| r.work),
            shape.recurrent_state_bytes,
        )
    }
    fn aggregate(
        evidence: &StatisticalWaveEvidenceV1,
        numeric: Option<&CanonicalWaveCostFeatures>,
        rows: impl ExactSizeIterator<Item = ActualRowWork>,
        recurrent_bytes: u64,
    ) -> Result<Self, Unknown> {
        let count = rows.len();
        if count == 0 || count > MAX_COST_ROWS {
            return Err(Unknown::Capacity);
        }
        let numeric = numeric.ok_or(Unknown::MissingHostDomain)?;
        numeric.validate(count).map_err(|_| Unknown::InvalidWork)?;
        let mut out = HostAndSequenceNumericWorkV1 {
            rows: count as u64,
            recurrent_bytes,
            ..Default::default()
        };
        for (work, row) in rows.zip(&numeric.rows) {
            let (kv, pairs) = match work {
                ActualRowWork::Decode { kv_tokens } => {
                    let kv = u64::from(kv_tokens)
                        .checked_add(1)
                        .ok_or(Unknown::Overflow)?;
                    (kv, kv)
                }
                ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                } => {
                    let offset = u64::from(offset);
                    let count = u64::from(count);
                    let total = u64::from(total_prompt_tokens);
                    let end = offset.checked_add(count).ok_or(Unknown::Overflow)?;
                    if count == 0 || end > total {
                        return Err(Unknown::InvalidWork);
                    }
                    // Exact causal query/key pair count: count*offset + count*(count+1)/2.
                    let pairs = count
                        .checked_mul(offset)
                        .and_then(|a| {
                            count
                                .checked_mul(count + 1)
                                .and_then(|b| a.checked_add(b / 2))
                        })
                        .ok_or(Unknown::Overflow)?;
                    add(&mut out.prefill_tokens, count)?;
                    add(&mut out.prompt_tokens_sum, total)?;
                    out.prompt_tokens_max = out.prompt_tokens_max.max(total);
                    (end, pairs)
                }
                _ => return Err(Unknown::UnsupportedWave),
            };
            add(&mut out.attention_pairs, pairs)?;
            add(&mut out.kv_tokens_sum, kv)?;
            out.kv_tokens_max = out.kv_tokens_max.max(kv);
            add(&mut out.generated_tokens_sum, row.generated_tokens_before)?;
            add(&mut out.output_budget_sum, row.maximum_output_tokens)?;
            add(&mut out.sampling_history_sum, row.sampling_history_tokens)?;
            out.sampling_history_max = out.sampling_history_max.max(row.sampling_history_tokens);
            add(&mut out.repetition_tokens_sum, row.repetition_tokens)?;
            add(&mut out.decoded_prefix_sum, row.decoded_prefix_tokens)?;
            add(
                &mut out.decoded_text_bytes_sum,
                row.decoded_text_bytes_bound,
            )?;
            add(
                &mut out.decode_scratch_bytes_sum,
                row.decode_scratch_bytes_bound,
            )?;
        }
        Ok(Self {
            family_signature: *evidence.family_signature(),
            // The enclosing private V1 evidence binds both digests to one exact
            // receipt. This copies only a fixed digest: do not rehash/reaggregate
            // the same edge for the new predictor in the planner hot path.
            independent_attention_family_v2: evidence
                .independent_attention_v2()
                .map(|v| *v.family_signature()),
            device: evidence.work(),
            host_and_sequence: out,
        })
    }
}
fn add(target: &mut u64, value: u64) -> Result<(), Unknown> {
    *target = target.checked_add(value).ok_or(Unknown::Overflow)?;
    Ok(())
}
#[cfg(test)]
mod tests;
