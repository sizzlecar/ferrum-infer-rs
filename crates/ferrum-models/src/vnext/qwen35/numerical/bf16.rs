//! Native BF16 is a distinct physical and numerical ABI. The shared F16
//! topology defines dimensions; only explicitly declared half boundaries move
//! to BF16, leaving DeltaNet coefficients/state and arithmetic in F32.
use super::*;

pub(super) fn profiles(
    family_id: &ModelFamilyId,
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
) -> Result<FamilyNumericalProfiles, VNextError> {
    if config.weight_format != FamilyWeightFormat::SafetensorsDense || text.is_moe() {
        return Err(invalid_config(
            "dense_precision",
            "BF16 profiles require dense non-MoE safetensors",
        ));
    }
    let (states, kv_storage) =
        super::states(text, config.max_position_embeddings, KvStorageFormat::F16)?;
    let f16 = super::profile(family_id, text, &states, &kv_storage, false, false)?;
    let base = native_profile(f16.clone(), BF16_NUMERICAL_PROFILE_ID)?;
    let automatic = vec![base.id.clone()];
    let mut profiles = vec![base];
    if text.layer_types.contains(&Qwen35LayerType::FullAttention) {
        profiles.push(native_profile(
            super::visual_prompt_profile(&f16)?,
            BF16_VISUAL_PROMPT_NUMERICAL_PROFILE_ID,
        )?);
    }
    FamilyNumericalProfiles::new(family_id, ContractVersion::new(1, 3), profiles, automatic)
}

fn native_profile(
    mut profile: NumericalExecutionProfile,
    id: &str,
) -> Result<NumericalExecutionProfile, VNextError> {
    profile.id = NumericalProfileId::new(id)
        .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    for dtype in profile.boundaries.values_mut() {
        if *dtype == ElementType::F16 {
            *dtype = ElementType::Bf16;
        }
    }
    for state in &mut profile.states {
        if state.tensor.element_type == ElementType::F16 {
            state.tensor.element_type = ElementType::Bf16;
        }
    }
    for storage in &mut profile.kv_storage {
        let KvStateStorage::F16 { state } = storage else {
            return Err(invalid_config(
                "numerical_profile.kv_storage",
                "BF16 source profile must have plain half KV",
            ));
        };
        *storage = KvStateStorage::Bf16 {
            state: state.clone(),
        };
    }
    for operation in &mut profile.operations {
        operation.operation_id = operation_id(match operation.operation_id.as_str() {
            TOKEN_EMBEDDING_OPERATION_ID => ferrum_runtime::TOKEN_EMBEDDING_BF16_OPERATION_ID,
            RMS_NORM_OPERATION_ID => ferrum_runtime::RMS_NORM_BF16_OPERATION_ID,
            LAST_TOKEN_DENSE_LINEAR_OPERATION_ID => {
                ferrum_runtime::LAST_TOKEN_DENSE_LINEAR_BF16_OPERATION_ID
            }
            LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID => {
                ferrum_runtime::LAST_TOKEN_MASKED_ARGMAX_BF16_OPERATION_ID
            }
            DENSE_SWIGLU_OPERATION_ID => ferrum_runtime::DENSE_SWIGLU_BF16_OPERATION_ID,
            RESIDUAL_ADD_OPERATION_ID => ferrum_runtime::RESIDUAL_ADD_BF16_OPERATION_ID,
            GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID => {
                ferrum_runtime::GATED_DELTA_RECURRENT_ATTENTION_BF16_OPERATION_ID
            }
            CAUSAL_PAGED_ATTENTION_OPERATION_ID => {
                ferrum_runtime::CAUSAL_PAGED_ATTENTION_BF16_OPERATION_ID
            }
            ferrum_runtime::ROW_SELECT_F16_OPERATION_ID => {
                ferrum_runtime::ROW_SELECT_BF16_OPERATION_ID
            }
            ferrum_runtime::CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_OPERATION_ID => {
                ferrum_runtime::CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_BF16_OPERATION_ID
            }
            _ => {
                return Err(invalid_config(
                    "numerical_profile.operations",
                    "unsupported operation in BF16 dense family",
                ))
            }
        })?;
        operation.version = ContractVersion::new(1, 0);
    }
    profile
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    profile.validate()?;
    Ok(profile)
}
