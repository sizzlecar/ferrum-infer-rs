//! CUDA tensor configuration policy, checked before opening a CUDA context.
use ferrum_runtime::{CapabilityId, DenseLinearFp8Group, VNextError};
use std::collections::BTreeSet;

/// Shared by the grouped factory and CPU configuration tests.
pub(crate) fn add_grouped_fp8(capabilities: &mut BTreeSet<CapabilityId>) -> Result<(), VNextError> {
    for group in [DenseLinearFp8Group::Pair, DenseLinearFp8Group::Triple] {
        capabilities.insert(CapabilityId::new(group.capability_id())?);
    }
    Ok(())
}

/// The actual constructor allowlist; no driver or native operator is loaded.
pub(crate) fn validate(capabilities: &BTreeSet<CapabilityId>) -> Result<(), String> {
    if capabilities.is_empty() {
        return Err("CUDA tensor eager capabilities must not be empty".into());
    }
    for capability in capabilities {
        if !matches!(
            capability.as_str(),
            ferrum_runtime::SILU_BF16_CAPABILITY_ID
                | ferrum_runtime::PHILOX_NORMAL_BF16_CAPABILITY_ID
                | ferrum_runtime::PHILOX_NORMAL_F32_CAPABILITY_ID
                | ferrum_runtime::DENSE_LINEAR_BF16_CAPABILITY_ID
                | ferrum_runtime::DENSE_LINEAR_F32_CAPABILITY_ID
                | ferrum_runtime::AFFINE_LINEAR_F32_CAPABILITY_ID
                | ferrum_runtime::CAST_F32_TO_BF16_CAPABILITY_ID
                | ferrum_runtime::PLAIN_RESIDUAL_ADD_F32_CAPABILITY_ID
                | ferrum_runtime::SILU_MUL_F32_CAPABILITY_ID
                | ferrum_runtime::GELU_TANH_F32_CAPABILITY_ID
                | ferrum_runtime::GELU_ERF_F32_CAPABILITY_ID
                | ferrum_runtime::PATCH_PROJECTION_F32_CAPABILITY_ID
                | ferrum_runtime::RESHAPE_COPY_F32_CAPABILITY_ID
                | ferrum_runtime::ROW_SCATTER_REPLACE_F32_CAPABILITY_ID
                | ferrum_runtime::ROW_SCATTER_ADD_F32_CAPABILITY_ID
                | ferrum_runtime::COMFY_CONVROT_W4A8_BF16_CAPABILITY_ID
                | ferrum_runtime::COMFY_CONVROT_W8A8_BF16_CAPABILITY_ID
                | ferrum_runtime::COMFY_CONVROT_W4A8_F32_CAPABILITY_ID
                | ferrum_runtime::COMFY_CONVROT_W8A8_F32_CAPABILITY_ID
                | ferrum_runtime::COMFY_CONVROT_INT8_EMBEDDING_F32_CAPABILITY_ID
                | ferrum_runtime::HEAD_RMS_NORM_F32_CAPABILITY_ID
                | ferrum_runtime::HALF_SPLIT_ROPE_F32_CAPABILITY_ID
                | ferrum_runtime::UNIT_HEAD_RMS_PARTIAL_ROPE_F32_CAPABILITY_ID
                | ferrum_runtime::GROUPED_COLUMN_SLICE_F32_CAPABILITY_ID
                | ferrum_runtime::CHANNEL_BIAS_ADD_F32_CAPABILITY_ID
                | ferrum_runtime::CHANNEL_SCALED_RESIDUAL_F32_CAPABILITY_ID
                | ferrum_runtime::NAN_TO_NUM_F32_CAPABILITY_ID
                | ferrum_runtime::ROW_SELECTED_MODULATION_BF16_F32_CAPABILITY_ID
                | ferrum_runtime::ROW_SELECTED_MODULATION_BF16_CAPABILITY_ID
                | ferrum_runtime::ROW_SELECTED_ADDCMUL_BF16_CAPABILITY_ID
                | ferrum_runtime::BATCHED_SUFFIX_APPEND_F32_CAPABILITY_ID
                | ferrum_runtime::LAYER_NORM_F32_CAPABILITY_ID
                | ferrum_runtime::CONV2D_F32_CAPABILITY_ID
                | ferrum_runtime::GROUP_NORM_SILU_NCHW_F32_CAPABILITY_ID
                | ferrum_runtime::PLAIN_RESIDUAL_ADD_NCHW_F32_CAPABILITY_ID
                | ferrum_runtime::COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_CAPABILITY_ID
                | ferrum_runtime::COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_CAPABILITY_ID
                | ferrum_runtime::DENSE_LINEAR_BLOCK_BF16_CAPABILITY_ID
                | ferrum_runtime::DENSE_LINEAR_FP8_BF16_CAPABILITY_ID
                | ferrum_runtime::DENSE_LINEAR_FP8_PAIR_BF16_CAPABILITY_ID
                | ferrum_runtime::DENSE_LINEAR_FP8_TRIPLE_BF16_CAPABILITY_ID
                | ferrum_runtime::SILU_MUL_BF16_CAPABILITY_ID
                | ferrum_runtime::TANH_BF16_CAPABILITY_ID
                | ferrum_runtime::MODULATION_SCALE_BF16_CAPABILITY_ID
                | ferrum_runtime::GATED_RESIDUAL_BF16_CAPABILITY_ID
                | ferrum_runtime::LAYER_NORM_BF16_CAPABILITY_ID
                | ferrum_runtime::HEAD_RMS_NORM_BF16_CAPABILITY_ID
                | ferrum_runtime::ADJACENT_ROPE_BF16_CAPABILITY_ID
                | ferrum_runtime::STRUCTURED_ATTENTION_BF16_CAPABILITY_ID
                | ferrum_runtime::STRUCTURED_ATTENTION_FUSED_BF16_CAPABILITY_ID
                | ferrum_runtime::CROSS_ATTENTION_FUSED_BF16_CAPABILITY_ID
                | ferrum_runtime::ZERO_CENTERED_RMS_NORM_BF16_CAPABILITY_ID
                | ferrum_runtime::GELU_TANH_BF16_CAPABILITY_ID
                | ferrum_runtime::ROW_GATHER_BF16_CAPABILITY_ID
                | ferrum_runtime::BATCHED_ROW_CONCAT_BF16_CAPABILITY_ID
                | ferrum_runtime::PLAIN_RESIDUAL_ADD_BF16_CAPABILITY_ID
                | ferrum_runtime::HALF_SPLIT_ROPE_BF16_CAPABILITY_ID
                | ferrum_runtime::PARTIAL_HALF_SPLIT_ROPE_BF16_CAPABILITY_ID
                | ferrum_runtime::FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_CAPABILITY_ID
                | ferrum_runtime::ROW_RMS_NORM_SINGLE_ROUND_BF16_CAPABILITY_ID
                | ferrum_runtime::ADDCMUL_BF16_CAPABILITY_ID
                | ferrum_runtime::STRUCTURED_GQA_ATTENTION_BF16_CAPABILITY_ID
                | ferrum_runtime::STRUCTURED_GQA_ATTENTION_F32_CAPABILITY_ID
                | ferrum_runtime::CONV2D_BF16_CAPABILITY_ID
                | ferrum_runtime::CHANNEL_L2_NORM_BF16_CAPABILITY_ID
                | ferrum_runtime::CHANNEL_L2_NORM_IMAGE_BF16_CAPABILITY_ID
                | ferrum_runtime::SPATIAL_ATTENTION_BF16_CAPABILITY_ID
                | ferrum_runtime::SPATIAL_UPSAMPLE_GATHER_BF16_CAPABILITY_ID
                | ferrum_runtime::SPATIAL_GROUP_MEAN_BF16_CAPABILITY_ID
                | ferrum_runtime::AFFINE_LINEAR_BF16_CAPABILITY_ID
                | ferrum_runtime::PATCH_PROJECTION_BF16_CAPABILITY_ID
                | ferrum_runtime::AFFINE_LAYER_NORM_BF16_CAPABILITY_ID
                | ferrum_runtime::GELU_ERF_BF16_CAPABILITY_ID
                | ferrum_runtime::HALF_ROTARY_BF16_CAPABILITY_ID
                | ferrum_runtime::ROW_SCATTER_ADD_BF16_CAPABILITY_ID
                | ferrum_runtime::RESHAPE_COPY_BF16_CAPABILITY_ID
                | ferrum_runtime::SEGMENTED_ATTENTION_BF16_CAPABILITY_ID
                | ferrum_runtime::SILU_NCHW_BF16_CAPABILITY_ID
                | ferrum_runtime::PLAIN_RESIDUAL_ADD_NCHW_BF16_CAPABILITY_ID
        ) {
            return Err(format!(
                "unsupported CUDA tensor eager capability: {capability}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn ordinary_factory_configuration_passes_constructor_policy_without_a_device() {
        let config = crate::backend::cuda::vnext_ops::cuda_tensor_runtime_config(
            0,
            ferrum_runtime::DeviceId::new("device.cuda.0").unwrap(),
        )
        .unwrap();
        validate(&config.capabilities).unwrap();
    }

    #[test]
    fn grouped_fp8_factory_configuration_passes_constructor_policy() {
        let mut capabilities = BTreeSet::from([
            CapabilityId::new(ferrum_runtime::DENSE_LINEAR_BF16_CAPABILITY_ID).unwrap(),
            CapabilityId::new(ferrum_runtime::DENSE_LINEAR_FP8_BF16_CAPABILITY_ID).unwrap(),
        ]);
        validate(&capabilities).unwrap();
        add_grouped_fp8(&mut capabilities).unwrap();
        validate(&capabilities).unwrap();
        for group in [DenseLinearFp8Group::Pair, DenseLinearFp8Group::Triple] {
            assert!(capabilities.contains(&CapabilityId::new(group.capability_id()).unwrap()));
        }
    }

    #[test]
    fn constructor_policy_still_rejects_empty_and_unknown_capabilities() {
        assert!(validate(&BTreeSet::new()).is_err());
        let mut capabilities = BTreeSet::new();
        add_grouped_fp8(&mut capabilities).unwrap();
        let unsupported = CapabilityId::new("capability.operation.unsupported").unwrap();
        capabilities.insert(unsupported.clone());
        assert!(validate(&capabilities)
            .unwrap_err()
            .contains(unsupported.as_str()));
    }
}
