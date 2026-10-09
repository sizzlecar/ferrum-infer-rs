//! Explicit upstream projection policy shared by run/serve family lowering.
//! Source eligibility is a declaration, not backend or output-quality approval.
use super::*;
use ferrum_interfaces::vnext::UpstreamMarkerV2Profile;

#[cfg(test)]
mod tests;

pub(in crate::vnext::qwen35) fn eligible(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
) -> bool {
    eligible_for(
        config,
        text,
        [
            UpstreamMarkerV2Profile::SwiGlu,
            UpstreamMarkerV2Profile::GatedDelta,
            UpstreamMarkerV2Profile::Causal,
        ],
    )
}

pub(in crate::vnext::qwen35) fn extra_eligible(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
) -> bool {
    eligible_for(
        config,
        text,
        [
            UpstreamMarkerV2Profile::SwiGluExtraPrefill,
            UpstreamMarkerV2Profile::GatedDeltaExtraPrefill,
            UpstreamMarkerV2Profile::CausalExtraPrefill,
        ],
    )
}

fn eligible_for(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
    kinds: [UpstreamMarkerV2Profile; 3],
) -> bool {
    if config.weight_format != FamilyWeightFormat::GgufNative
        || text.moe.is_some()
        || config.recurrent_weight_abi != RecurrentWeightAbi::NegativeRateInterleaved
        || config.gguf_hadamard.is_some()
    {
        return false;
    }
    // Owned declarations are constructed in preparation, never per encode.
    let [ffn, gdn, causal] = kinds.map(UpstreamMarkerV2Profile::arithmetic);
    config.weights.iter().any(|weight| {
        let Some(layer) = weight
            .layer_index
            .map(|v| v as usize)
            .filter(|&i| i < text.num_hidden_layers)
        else {
            return false;
        };
        let (arithmetic, role, expected_layer) = match weight.role.as_str() {
            "mlp_gate" | "mlp_up" => (&ffn, ProjectionRole::SwiGluGateUp, None),
            "mlp_down" => (&ffn, ProjectionRole::SwiGluDown, None),
            "linear_attn_qkv" | "linear_attn_z" | "linear_attn_a" | "linear_attn_b" => (
                &gdn,
                ProjectionRole::GatedDeltaInput,
                Some(Qwen35LayerType::LinearAttention),
            ),
            "linear_attn_out" => (
                &gdn,
                ProjectionRole::GatedDeltaOutput,
                Some(Qwen35LayerType::LinearAttention),
            ),
            "self_attn_q" => (
                &causal,
                ProjectionRole::CausalQuery,
                Some(Qwen35LayerType::FullAttention),
            ),
            "self_attn_k" => (
                &causal,
                ProjectionRole::CausalKey,
                Some(Qwen35LayerType::FullAttention),
            ),
            "self_attn_v" => (
                &causal,
                ProjectionRole::CausalValue,
                Some(Qwen35LayerType::FullAttention),
            ),
            "self_attn_o" => (
                &causal,
                ProjectionRole::CausalOutput,
                Some(Qwen35LayerType::FullAttention),
            ),
            _ => return false,
        };
        if expected_layer.is_some_and(|kind| text.layer_types.get(layer) != Some(&kind)) {
            return false;
        }
        let FamilyWeightSourceEncoding::BlockQuantized(block) = &weight.source_encoding else {
            return false;
        };
        let [n, k] = weight.dimensions.as_slice() else {
            return false;
        };
        matches!(
            arithmetic.declared_projection_arithmetic(role, Some(block), *k, *n, false),
            Ok(DeclaredProjectionArithmetic::Staged(_))
        )
    })
}

pub(super) fn profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    profile_for(master, false)
}
pub(super) fn profile_for(
    master: &NumericalExecutionProfile,
    prefill: bool,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut profile = master.clone();
    profile.id = NumericalProfileId::new(if prefill {
        F32_MASTER_FFN_ATTENTION_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_NUMERICAL_PROFILE_ID
    } else {
        F32_MASTER_FFN_ATTENTION_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_NUMERICAL_PROFILE_ID
    })
    .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    replace_operations(
        &mut profile,
        if prefill {
            [
                UpstreamMarkerV2Profile::SwiGluPrefill,
                UpstreamMarkerV2Profile::GatedDeltaPrefill,
                UpstreamMarkerV2Profile::CausalPrefill,
            ]
        } else {
            [
                UpstreamMarkerV2Profile::SwiGlu,
                UpstreamMarkerV2Profile::GatedDelta,
                UpstreamMarkerV2Profile::Causal,
            ]
        },
    )?;
    Ok(profile)
}
pub(super) fn hybrid_profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut profile = master.clone();
    profile.id = NumericalProfileId::new(
        F32_MASTER_FFN_ATTENTION_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_NUMERICAL_PROFILE_ID,
    )
    .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    replace_operations(
        &mut profile,
        [
            UpstreamMarkerV2Profile::SwiGluG32MmqPrefill,
            UpstreamMarkerV2Profile::GatedDeltaG32MmqPrefill,
            UpstreamMarkerV2Profile::CausalG32MmqPrefill,
        ],
    )?;
    Ok(profile)
}
pub(super) fn extra_profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut profile = master.clone();
    profile.id = NumericalProfileId::new(F32_MASTER_FFN_ATTENTION_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_NUMERICAL_PROFILE_ID)
        .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    replace_operations(
        &mut profile,
        [
            UpstreamMarkerV2Profile::SwiGluExtraPrefill,
            UpstreamMarkerV2Profile::GatedDeltaExtraPrefill,
            UpstreamMarkerV2Profile::CausalExtraPrefill,
        ],
    )?;
    Ok(profile)
}

pub(super) fn extra_large_prefill_profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut profile = master.clone();
    profile.id = NumericalProfileId::new(F32_MASTER_FFN_ATTENTION_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_NUMERICAL_PROFILE_ID)
        .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    replace_operations(
        &mut profile,
        [
            UpstreamMarkerV2Profile::SwiGluExtraLargePrefill,
            UpstreamMarkerV2Profile::GatedDeltaExtraLargePrefill,
            UpstreamMarkerV2Profile::CausalExtraLargePrefill,
        ],
    )?;
    Ok(profile)
}

pub(super) fn extra_all_rows_profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut profile = master.clone();
    profile.id = NumericalProfileId::new(F32_MASTER_FFN_ATTENTION_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_NUMERICAL_PROFILE_ID)
        .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    replace_operations(
        &mut profile,
        [
            UpstreamMarkerV2Profile::SwiGluExtraAllRows,
            UpstreamMarkerV2Profile::GatedDeltaExtraAllRows,
            UpstreamMarkerV2Profile::CausalExtraAllRows,
        ],
    )?;
    Ok(profile)
}

#[cfg(test)]
mod extra_tests;

fn replace_operations(
    profile: &mut NumericalExecutionProfile,
    kinds: [UpstreamMarkerV2Profile; 3],
) -> Result<(), VNextError> {
    for kind in kinds {
        if let Some(operation) = profile
            .operations
            .iter_mut()
            .find(|op| op.operation_id.as_str() == kind.strict_operation_id())
        {
            operation.operation_id = operation_id(kind.operation_id())?;
            operation.version = ContractVersion::new(1, 0);
            operation.multiplication_type = None;
            operation.accumulation_type = None;
            operation.staged_arithmetic = None;
            operation.composite_arithmetic = Some(kind.arithmetic());
        }
    }
    profile
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    profile.validate()?;
    Ok(())
}

#[cfg(test)]
mod extra_prefill_tests;

#[cfg(test)]
mod all_rows_tests;
