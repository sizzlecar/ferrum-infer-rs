//! The explicit native operation union of Comfy text, DiT, image codec and
//! initial noise. One paired inventory keeps contract/selection wiring local.
use ferrum_kernels::backend::cuda::vnext_ops::CudaTensorOperation as Op;
use ferrum_runtime::*;

pub(super) fn selected() -> Result<Vec<(Op, Box<dyn OperationContract>)>, VNextError> {
    Ok(vec![
        (
            Op::ComfyConvRotInt8EmbeddingF32,
            Box::new(comfy_convrot_int8_embedding_f32_contract()?),
        ),
        (
            Op::ComfyConvRotW4A8F32,
            Box::new(comfy_convrot_w4a8_f32_contract()?),
        ),
        (
            Op::ComfyConvRotW8A8F32,
            Box::new(comfy_convrot_w8a8_f32_contract()?),
        ),
        (Op::HeadRmsNormF32, Box::new(head_rms_norm_f32_contract()?)),
        (
            Op::HalfSplitRopeF32,
            Box::new(half_split_rope_f32_contract()?),
        ),
        (
            Op::StructuredGqaAttentionF32,
            Box::new(structured_gqa_attention_f32_contract()?),
        ),
        (
            Op::PlainResidualAddF32,
            Box::new(plain_residual_add_f32_contract()?),
        ),
        (Op::SiluMulF32, Box::new(silu_mul_f32_contract()?)),
        (Op::ReshapeCopyF32, Box::new(reshape_copy_f32_contract()?)),
        (Op::GeluTanhF32, Box::new(gelu_tanh_f32_contract()?)),
        (Op::GeluErfF32, Box::new(gelu_erf_f32_contract()?)),
        (
            Op::RowScatterReplaceF32,
            Box::new(row_scatter_replace_f32_contract()?),
        ),
        (
            Op::RowScatterAddF32,
            Box::new(row_scatter_add_f32_contract()?),
        ),
        (Op::AffineLinear, Box::new(affine_linear_bf16_contract()?)),
        (Op::AffineLinearF32, Box::new(affine_linear_f32_contract()?)),
        (
            Op::PatchProjectionF32,
            Box::new(patch_projection_f32_contract()?),
        ),
        (Op::CastF32ToBf16, Box::new(cast_f32_to_bf16_contract()?)),
        (Op::ReshapeCopy, Box::new(reshape_copy_bf16_contract()?)),
        (Op::RowGather, Box::new(row_gather_bf16_contract()?)),
        (
            Op::BatchedRowConcat,
            Box::new(batched_row_concat_bf16_contract()?),
        ),
        (
            Op::RowRmsNormSingleRound,
            Box::new(row_rms_norm_single_round_bf16_contract()?),
        ),
        (
            Op::ModulationScale,
            Box::new(modulation_scale_bf16_contract()?),
        ),
        (
            Op::RowSelectedModulationBf16,
            Box::new(row_selected_modulation_bf16_contract()?),
        ),
        (
            Op::RowSelectedAddcmulBf16,
            Box::new(row_selected_addcmul_bf16_contract()?),
        ),
        (
            Op::PlainResidualAdd,
            Box::new(plain_residual_add_bf16_contract()?),
        ),
        (
            Op::FusedHeadRmsPartialRope,
            Box::new(fused_head_rms_partial_rope_bf16_contract()?),
        ),
        (Op::Addcmul, Box::new(addcmul_bf16_contract()?)),
        (
            Op::SegmentedAttention,
            Box::new(segmented_attention_bf16_contract()?),
        ),
        (
            Op::RowSelectedModulationBf16F32,
            Box::new(row_selected_modulation_bf16_f32_contract()?),
        ),
        (
            Op::ComfyConvRotW4A8,
            Box::new(comfy_convrot_w4a8_bf16_contract()?),
        ),
        (
            Op::ComfyConvRotSwiGluF32W4A8,
            Box::new(comfy_convrot_swiglu_f32_w4a8_bf16_contract()?),
        ),
        (
            Op::ComfyConvRotW8A8,
            Box::new(comfy_convrot_w8a8_bf16_contract()?),
        ),
        (
            Op::ComfyConvRotSwiGluF32W8A8,
            Box::new(comfy_convrot_swiglu_f32_w8a8_bf16_contract()?),
        ),
        (
            Op::UnitHeadRmsPartialRopeF32,
            Box::new(unit_head_rms_partial_rope_f32_contract()?),
        ),
        (
            Op::GroupedColumnSliceF32,
            Box::new(grouped_column_slice_f32_contract()?),
        ),
        (
            Op::ChannelBiasAddF32,
            Box::new(channel_bias_add_f32_contract()?),
        ),
        (
            Op::ChannelScaledResidualF32,
            Box::new(channel_scaled_residual_f32_contract()?),
        ),
        (Op::NanToNumF32, Box::new(nan_to_num_f32_contract()?)),
        (
            Op::BatchedSuffixAppendF32,
            Box::new(batched_suffix_append_f32_contract()?),
        ),
        (Op::LayerNormF32, Box::new(layer_norm_f32_contract()?)),
        (
            Op::GroupNormSiluNchwF32,
            Box::new(group_norm_silu_nchw_f32_contract()?),
        ),
        (
            Op::PlainResidualAddNchwF32,
            Box::new(plain_residual_add_nchw_f32_contract()?),
        ),
        (Op::PhiloxNormalF32, Box::new(philox_normal_f32_contract()?)),
    ])
}

pub(super) fn independent_contracts() -> Result<Vec<Box<dyn OperationContract>>, VNextError> {
    Ok(vec![
        Box::new(dense_linear_f32_contract()?),
        // The CNN factory adds this provider under its one cuDNN owner; it is
        // intentionally not another CudaTensorOperation/native-module load.
        Box::new(conv2d_f32_contract()?),
    ])
}
