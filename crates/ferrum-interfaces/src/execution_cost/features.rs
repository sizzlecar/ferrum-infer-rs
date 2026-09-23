//! Versioned numeric work evidence. These are observations or declared bounds,
//! never a claim that device latency is monotonic in any particular coordinate.
use super::{ActualRowWork, CostRowOutput, MAX_COST_ROWS};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod row_multiset;
pub use row_multiset::*;

pub const COST_NUMERIC_FEATURE_SCHEMA_V1: u32 = 1;
pub const HOST_CONTENT_FEATURE_SCHEMA_V1: u32 = 1;

/// An installed algorithm domain for an explicitly empirical content model.
/// This bounds which policies may share calibration, not their latency or the
/// number of resampling attempts. Unseen token content remains a disturbance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostContentDomainV1 {
    PlainTextGreedyV1,
}

/// Complete-wave identity with content-dependent host branches marginalized.
/// The global device output mode and actual provider/readback path stay exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostContentCostFeaturesV1 {
    pub schema_version: u32,
    pub output_policy_signature: [u8; 32],
}

impl HostContentCostFeaturesV1 {
    pub fn validate(&self) -> Result<(), CostFeatureError> {
        if self.schema_version != HOST_CONTENT_FEATURE_SCHEMA_V1 {
            return Err(CostFeatureError::UnsupportedSchema);
        }
        Ok(())
    }
}

/// A zero-command readback may still synchronize the host. The numeric identity
/// must distinguish that path from submission-owned staging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreReadbackRoute {
    SubmissionStaged,
    /// Staging was attempted and rolled back before synchronized fallback.
    SubmissionFallbackSynchronized,
    HostSynchronized,
    NoReadback,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostSamplingHistoryScope {
    FullGeneration,
    HiddenStructuredOutput,
    VisibleStructuredOutput,
}

/// Cached from the actually installed bounded decoder, sampler and codec.
/// No request id, history length, maximum output length or random seed belongs
/// in categorical_signature. The per-token bounds are also part of that hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostCostPolicyV2 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub empirical_content_domain: Option<HostContentDomainV1>,
    pub categorical_signature: [u8; 32],
    pub decoder_text_bytes_per_token: u64,
    pub decoder_scratch_bytes_per_token: u64,
    pub raw_token_bytes_bound: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostCostStateV1 {
    pub generated_tokens_before: u64,
    pub maximum_output_tokens: u64,
    pub sampling_history_tokens: u64,
    pub sampling_history_scope: CostSamplingHistoryScope,
    pub pending_decoded_utf8: bool,
    /// Actual completion phase/matcher progress. Unknown future token content
    /// cannot justify reusing a pending state's signature after a simulated step.
    pub completion_state_signature: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostCostFeaturesV1 {
    pub policy: HostCostPolicyV2,
    pub state: HostCostStateV1,
}

impl HostCostFeaturesV1 {
    pub fn supports_empirical_plain_text_content(&self) -> bool {
        self.policy.empirical_content_domain == Some(HostContentDomainV1::PlainTextGreedyV1)
            && self.state.sampling_history_scope == CostSamplingHistoryScope::FullGeneration
            && self.state.sampling_history_tokens == self.state.generated_tokens_before
            && self.state.completion_state_signature == satisfied_completion_cost_signature()
    }
}

pub fn satisfied_completion_cost_signature() -> [u8; 32] {
    Sha256::digest(b"ferrum.host-completion-state.v1\0satisfied").into()
}

/// The largest full prefix which this row may decode, with capacities for ONE
/// such decode. These fields do not sum all calls or bound total resident memory.
/// maximum_output_tokens is capacity/termination metadata, not a work axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostRowNumericFeatures {
    pub generated_tokens_before: u64,
    pub maximum_output_tokens: u64,
    pub sampling_history_tokens: u64,
    pub repetition_tokens: u64,
    pub decoded_prefix_tokens: u64,
    pub decoded_text_bytes_bound: u64,
    pub decode_scratch_bytes_bound: u64,
}

impl CostRowNumericFeatures {
    pub fn validate(&self) -> Result<(), CostFeatureError> {
        if self.maximum_output_tokens == 0
            || self.generated_tokens_before > self.maximum_output_tokens
            || self.sampling_history_tokens > self.generated_tokens_before
            || self.repetition_tokens > self.sampling_history_tokens
            || self.decoded_prefix_tokens < self.generated_tokens_before
            || self.decoded_prefix_tokens > self.maximum_output_tokens
            || self.decoded_prefix_tokens - self.generated_tokens_before > 1
            || self.decoded_text_bytes_bound < self.decoded_prefix_tokens
            || (self.decoded_prefix_tokens == 0
                && (self.decoded_text_bytes_bound != 0 || self.decode_scratch_bytes_bound != 0))
        {
            return Err(CostFeatureError::InvalidState);
        }
        self.decoded_text_bytes_bound
            .checked_add(self.decode_scratch_bytes_bound)
            .ok_or(CostFeatureError::Overflow)?;
        Ok(())
    }
}

/// Rows remain in physical order. The new output hash identifies only actual
/// categorical policies/branches; the old exact output hash remains separate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalWaveCostFeatures {
    pub schema_version: u32,
    pub output_policy_signature: [u8; 32],
    #[serde(deserialize_with = "bounded_rows")]
    pub rows: Vec<CostRowNumericFeatures>,
}

impl CanonicalWaveCostFeatures {
    pub fn validate(&self, expected_rows: usize) -> Result<(), CostFeatureError> {
        if self.schema_version != COST_NUMERIC_FEATURE_SCHEMA_V1 {
            return Err(CostFeatureError::UnsupportedSchema);
        }
        if expected_rows == 0
            || expected_rows > MAX_COST_ROWS
            || self.rows.len() != expected_rows
            || self.rows.capacity() > MAX_COST_ROWS
        {
            return Err(CostFeatureError::RowCapacity);
        }
        self.rows
            .iter()
            .try_for_each(CostRowNumericFeatures::validate)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CostFeatureError {
    #[error("unsupported cost numeric feature schema")]
    UnsupportedSchema,
    #[error("cost numeric feature row capacity or correspondence is invalid")]
    RowCapacity,
    #[error("cost numeric feature state is invalid or unavailable")]
    InvalidState,
    #[error("cost numeric feature arithmetic overflow")]
    Overflow,
}

/// Both actual observation and future shape declaration use this projection.
/// The caller must supply the real/justified host state and actual output mode;
/// the helper cannot infer state transitions from an ungenerated token.
pub fn project_host_cost_features(
    host: HostCostFeaturesV1,
    work: ActualRowWork,
    output: CostRowOutput,
) -> Result<CostRowNumericFeatures, CostFeatureError> {
    let state = host.state;
    let history_valid = match state.sampling_history_scope {
        CostSamplingHistoryScope::FullGeneration => {
            state.sampling_history_tokens == state.generated_tokens_before
        }
        CostSamplingHistoryScope::HiddenStructuredOutput => state.sampling_history_tokens == 0,
        CostSamplingHistoryScope::VisibleStructuredOutput => {
            state.sampling_history_tokens <= state.generated_tokens_before
        }
    };
    if !history_valid
        || host.policy.decoder_text_bytes_per_token == 0
        || host.policy.raw_token_bytes_bound == 0
    {
        return Err(CostFeatureError::InvalidState);
    }
    let (commits_token, repetition_tokens) = match (work, output) {
        (
            ActualRowWork::Decode { kv_tokens },
            CostRowOutput::Decode {
                repetition_tokens, ..
            },
        ) if kv_tokens > 0 => (true, repetition_tokens),
        (
            ActualRowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            },
            CostRowOutput::Prefill { final_logits },
        ) if count > 0
            && offset.checked_add(count).is_some_and(|end| {
                end <= total_prompt_tokens && (end == total_prompt_tokens) == final_logits
            }) =>
        {
            (final_logits, 0)
        }
        _ => return Err(CostFeatureError::InvalidState),
    };
    let prefix = state
        .generated_tokens_before
        .checked_add(u64::from(commits_token))
        .ok_or(CostFeatureError::Overflow)?;
    let numbers = CostRowNumericFeatures {
        generated_tokens_before: state.generated_tokens_before,
        maximum_output_tokens: state.maximum_output_tokens,
        sampling_history_tokens: state.sampling_history_tokens,
        repetition_tokens,
        decoded_prefix_tokens: prefix,
        decoded_text_bytes_bound: host
            .policy
            .decoder_text_bytes_per_token
            .checked_mul(prefix)
            .ok_or(CostFeatureError::Overflow)?,
        decode_scratch_bytes_bound: host
            .policy
            .decoder_scratch_bytes_per_token
            .checked_mul(prefix)
            .ok_or(CostFeatureError::Overflow)?,
    };
    numbers.validate()?;
    Ok(numbers)
}

fn bounded_rows<'de, D>(deserializer: D) -> Result<Vec<CostRowNumericFeatures>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Rows;
    impl<'de> serde::de::Visitor<'de> for Rows {
        type Value = Vec<CostRowNumericFeatures>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "at most {MAX_COST_ROWS} ordered cost rows")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> Result<Self::Value, A::Error> {
            // Never trust an input's size hint as allocation authority.
            let mut rows = Vec::new();
            while let Some(row) = seq.next_element()? {
                if rows.len() == MAX_COST_ROWS {
                    return Err(serde::de::Error::custom("cost numeric row limit exceeded"));
                }
                rows.push(row);
            }
            Ok(rows)
        }
    }
    deserializer.deserialize_seq(Rows)
}

#[cfg(test)]
mod tests;
