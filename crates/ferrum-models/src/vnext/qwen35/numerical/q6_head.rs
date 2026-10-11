//! Explicit head-only replacement on the unchanged AA body profile.
use super::*;
use ferrum_interfaces::vnext::{Q6MmqF32Policy, LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID};

#[cfg(test)]
mod tests;

pub(in crate::vnext::qwen35) fn eligible(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
) -> bool {
    if !upstream_extra_marker_v2_eligible(config, text) {
        return false;
    }
    // Match semantic_program's actual head source (explicit lm_head wins;
    // otherwise the embedding is tied), rather than inferring from a label.
    config
        .weights
        .iter()
        .find(|w| w.layer_index.is_none() && w.role == "lm_head")
        .or_else(|| {
            config
                .weights
                .iter()
                .find(|w| w.layer_index.is_none() && w.role == "embed_tokens")
        })
        .is_some_and(|w| {
            let FamilyWeightSourceEncoding::BlockQuantized(block) = &w.source_encoding else {
                return false;
            };
            let [n, k] = w.dimensions.as_slice() else {
                return false;
            };
            Q6MmqF32Policy::eligible_weight(Some(block), *k, *n)
        })
}

pub(super) fn profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut result = upstream_marker_v2::extra_all_rows_profile(master)?;
    result.id = NumericalProfileId::new(F32_MASTER_GGUF_Q8_HEAD_V1_NUMERICAL_PROFILE_ID)
        .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    let head = result
        .operations
        .iter_mut()
        .find(|op| op.operation_id.as_str() == LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID)
        .ok_or_else(|| {
            invalid_config("numerical_profile", "F32 head is absent from body profile")
        })?;
    head.operation_id = operation_id(LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID)?;
    head.version = ContractVersion::new(1, 0);
    head.multiplication_type = None;
    head.accumulation_type = None;
    head.staged_arithmetic = Some(Q6MmqF32Policy::new().staged());
    head.composite_arithmetic = None;
    result
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    result.validate()?;
    Ok(result)
}
