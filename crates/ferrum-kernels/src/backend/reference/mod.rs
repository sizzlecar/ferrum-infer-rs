//! Portable vNext reference backend.
//!
//! This backend is a production contract oracle, not a product fallback. It is
//! registered explicitly as [`DeviceClass::Reference`] and advertises only the
//! operations it really executes.

mod spatial_decoder;
mod spatial_group_mean;
mod tensor_primitives;
pub use composition::reference_vnext_spatial_group_mean_bf16_provider;
pub use composition::reference_vnext_tensor_primitive_bf16_provider;
pub use composition::{
    reference_vnext_channel_l2_norm_image_bf16_provider,
    reference_vnext_spatial_attention_bf16_provider,
    reference_vnext_spatial_upsample_gather_bf16_provider,
};
pub use spatial_decoder::{
    REFERENCE_CHANNEL_L2_NORM_IMAGE_BF16_NUMERICAL_PROFILE_ID,
    REFERENCE_SPATIAL_ATTENTION_BF16_NUMERICAL_PROFILE_ID,
    REFERENCE_SPATIAL_UPSAMPLE_GATHER_BF16_NUMERICAL_PROFILE_ID,
};
pub use spatial_group_mean::REFERENCE_SPATIAL_GROUP_MEAN_BF16_NUMERICAL_PROFILE_ID;
pub use tensor_primitives::reference_tensor_primitive_bf16_profile;
mod spatial_bf16;
pub use composition::{
    reference_vnext_channel_l2_norm_bf16_provider, reference_vnext_conv2d_bf16_provider,
    reference_vnext_plain_residual_add_nchw_bf16_provider, reference_vnext_silu_nchw_bf16_provider,
};
pub use spatial_bf16::{
    REFERENCE_CHANNEL_L2_NORM_BF16_NUMERICAL_PROFILE_ID,
    REFERENCE_CONV2D_BF16_NUMERICAL_PROFILE_ID,
    REFERENCE_PLAIN_RESIDUAL_ADD_NCHW_BF16_NUMERICAL_PROFILE_ID,
    REFERENCE_SILU_NCHW_BF16_NUMERICAL_PROFILE_ID,
};
mod plain_residual_add;
pub use plain_residual_add::REFERENCE_PLAIN_RESIDUAL_ADD_BF16_NUMERICAL_PROFILE_ID;
mod fused_head_rms_partial_rope;
mod head_rms_partial_rope_single_round_bf16;
pub use head_rms_partial_rope_single_round_bf16::{
    reference_vnext_head_rms_partial_rope_single_round_bf16_provider,
    REFERENCE_HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_PROFILE,
};
mod row_rms_addcmul;
pub use composition::{
    reference_vnext_addcmul_bf16_provider, reference_vnext_addcmul_fma_bf16_provider,
    reference_vnext_row_rms_norm_single_round_bf16_provider,
};
pub use row_rms_addcmul::{
    REFERENCE_ADDCMUL_BF16_NUMERICAL_PROFILE_ID, REFERENCE_ADDCMUL_FMA_BF16_NUMERICAL_PROFILE_ID,
    REFERENCE_ROW_RMS_NORM_SINGLE_ROUND_BF16_NUMERICAL_PROFILE_ID,
};
mod half_split_rope;
pub use composition::reference_vnext_fused_head_rms_partial_rope_bf16_provider;
pub use fused_head_rms_partial_rope::REFERENCE_FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_NUMERICAL_PROFILE_ID;
pub use half_split_rope::{
    REFERENCE_HALF_SPLIT_ROPE_BF16_NUMERICAL_PROFILE_ID,
    REFERENCE_PARTIAL_HALF_SPLIT_ROPE_BF16_NUMERICAL_PROFILE_ID,
};
mod structured_gqa_attention;
pub use structured_gqa_attention::REFERENCE_STRUCTURED_GQA_ATTENTION_BF16_NUMERICAL_PROFILE_ID;
mod structured_gqa_attention_f32;
pub use composition::reference_vnext_structured_gqa_attention_f32_provider;
pub use structured_gqa_attention_f32::REFERENCE_STRUCTURED_GQA_ATTENTION_F32_NUMERICAL_PROFILE_ID;
mod adjacent_rope;
mod affine_f32;
mod batched_f32;
mod batched_suffix_append_f32;
mod channelwise_f32;
mod head_axis_pack;
mod head_fp8_roundtrip_bf16;
pub use head_axis_pack::{
    reference_vnext_pack_head_axis_bf16_provider,
    reference_vnext_pack_head_axis_f32_to_bf16_provider, REFERENCE_PACK_HEAD_AXIS_BF16_PROFILE_ID,
    REFERENCE_PACK_HEAD_AXIS_F32_TO_BF16_PROFILE_ID,
};
pub use head_fp8_roundtrip_bf16::{
    reference_vnext_head_fp8_roundtrip_bf16_provider, REFERENCE_HEAD_FP8_ROUNDTRIP_BF16_PROFILE_ID,
};
mod float_cast;
pub use batched_f32::{reference_vnext_batched_f32_provider, REFERENCE_BATCHED_F32_PROFILE_ID};
mod channel_bias_add_bf16;
pub use channel_bias_add_bf16::{
    reference_vnext_channel_bias_add_bf16_provider, REFERENCE_CHANNEL_BIAS_ADD_BF16_PROFILE_ID,
};
mod row_center_bf16;
pub use row_center_bf16::{
    reference_vnext_row_center_bf16_provider, REFERENCE_ROW_CENTER_BF16_PROFILE_ID,
};
mod masked_tile_mean;
mod signal_f32;
pub use masked_tile_mean::{
    reference_vnext_masked_tile_mean_bf16_f32_provider, REFERENCE_MASKED_TILE_MEAN_PROFILE_ID,
};
pub use signal_f32::{reference_vnext_signal_f32_provider, REFERENCE_SIGNAL_F32_PROFILE_ID};
mod row_selected_bf16;
mod row_selected_modulation;
pub use batched_suffix_append_f32::{
    reference_vnext_batched_suffix_append_f32_provider,
    REFERENCE_BATCHED_SUFFIX_APPEND_F32_NUMERICAL_PROFILE_ID,
};
pub use row_selected_bf16::{
    reference_vnext_row_selected_bf16_provider,
    REFERENCE_ROW_SELECTED_ADDCMUL_BF16_NUMERICAL_PROFILE_ID,
    REFERENCE_ROW_SELECTED_ADDCMUL_FMA_BF16_NUMERICAL_PROFILE_ID,
    REFERENCE_ROW_SELECTED_MODULATION_BF16_NUMERICAL_PROFILE_ID,
};
mod cnn_f32;
mod conv_transpose2d_f32;
pub use conv_transpose2d_f32::{
    reference_vnext_conv_transpose2d_f32_provider, REFERENCE_CONV_TRANSPOSE2D_F32_PROFILE,
};
mod layer_norm_f32;
pub use cnn_f32::{reference_cnn_f32_numerical_profile, reference_vnext_cnn_f32_provider};
pub use layer_norm_f32::{
    reference_vnext_layer_norm_f32_provider, REFERENCE_LAYER_NORM_F32_NUMERICAL_PROFILE_ID,
};
pub use row_selected_modulation::{
    reference_vnext_row_selected_modulation_bf16_f32_provider,
    REFERENCE_ROW_SELECTED_MODULATION_BF16_F32_NUMERICAL_PROFILE_ID,
};
mod gelu_f32;
mod row_scatter_f32;
pub use row_scatter_f32::{
    reference_vnext_row_scatter_add_f32_provider, reference_vnext_row_scatter_replace_f32_provider,
    REFERENCE_ROW_SCATTER_ADD_F32_NUMERICAL_PROFILE_ID,
    REFERENCE_ROW_SCATTER_REPLACE_F32_NUMERICAL_PROFILE_ID,
};
mod vision_layout_f32;
pub use vision_layout_f32::{
    reference_vnext_patch_projection_f32_provider, reference_vnext_reshape_copy_f32_provider,
    REFERENCE_PATCH_PROJECTION_F32_NUMERICAL_PROFILE_ID,
    REFERENCE_RESHAPE_COPY_F32_NUMERICAL_PROFILE_ID,
};
mod packed_swiglu_bf16;
mod row_selected_rms_modulation_bf16;
pub use row_selected_rms_modulation_bf16::{
    reference_vnext_row_selected_rms_modulation_bf16_provider,
    REFERENCE_ROW_SELECTED_RMS_MODULATION_PROFILE,
};
mod pointwise_f32;
pub use channelwise_f32::{
    reference_channelwise_f32_numerical_profile, reference_vnext_channelwise_f32_provider,
};
pub use gelu_f32::{
    reference_vnext_gelu_erf_f32_provider, reference_vnext_gelu_tanh_f32_provider,
    REFERENCE_GELU_ERF_F32_NUMERICAL_PROFILE_ID, REFERENCE_GELU_TANH_F32_NUMERICAL_PROFILE_ID,
};
pub use packed_swiglu_bf16::{
    reference_vnext_packed_swiglu_bf16_provider, REFERENCE_PACKED_SWIGLU_BF16_NUMERICAL_PROFILE_ID,
};
pub use pointwise_f32::{
    reference_vnext_plain_residual_add_f32_provider, reference_vnext_silu_f32_provider,
    reference_vnext_silu_mul_f32_provider, REFERENCE_PLAIN_RESIDUAL_ADD_F32_NUMERICAL_PROFILE_ID,
    REFERENCE_SILU_F32_NUMERICAL_PROFILE_ID, REFERENCE_SILU_MUL_F32_NUMERICAL_PROFILE_ID,
};
mod norm_rope_f32;
pub use float_cast::{
    reference_vnext_cast_bf16_to_f32_provider, reference_vnext_cast_f32_to_bf16_provider,
    REFERENCE_CAST_BF16_TO_F32_NUMERICAL_PROFILE_ID,
    REFERENCE_CAST_F32_TO_BF16_NUMERICAL_PROFILE_ID,
};
pub use norm_rope_f32::{
    reference_vnext_half_split_rope_f32_provider, reference_vnext_head_rms_norm_f32_provider,
    reference_vnext_unit_head_rms_partial_rope_f32_provider,
    REFERENCE_HALF_SPLIT_ROPE_F32_NUMERICAL_PROFILE_ID,
    REFERENCE_HEAD_RMS_NORM_F32_NUMERICAL_PROFILE_ID,
    REFERENCE_UNIT_HEAD_RMS_PARTIAL_ROPE_F32_NUMERICAL_PROFILE_ID,
};
mod composition;
mod cross_attention;
mod dense_linear;
pub use affine_f32::{
    reference_vnext_affine_linear_f32_provider, REFERENCE_AFFINE_LINEAR_F32_NUMERICAL_PROFILE_ID,
};
mod gelu_tanh;
mod head_rms_norm;
mod layer_norm;
mod pointwise;
mod row_transport;
mod runtime;
mod structured_attention;
mod transcendental;
mod zero_centered_rms_norm;

pub use adjacent_rope::REFERENCE_ADJACENT_ROPE_BF16_NUMERICAL_PROFILE_ID;
pub use composition::{
    reference_vnext_adjacent_rope_bf16_provider, reference_vnext_batched_row_concat_bf16_provider,
    reference_vnext_capabilities, reference_vnext_cross_attention_bf16_provider,
    reference_vnext_dense_linear_bf16_provider, reference_vnext_dense_linear_f32_provider,
    reference_vnext_dense_linear_provider, reference_vnext_gated_residual_bf16_provider,
    reference_vnext_gelu_tanh_bf16_provider, reference_vnext_half_split_rope_bf16_provider,
    reference_vnext_head_rms_norm_bf16_provider, reference_vnext_layer_norm_bf16_provider,
    reference_vnext_modulation_scale_bf16_provider, reference_vnext_operation_registry,
    reference_vnext_partial_half_split_rope_bf16_provider,
    reference_vnext_plain_residual_add_bf16_provider, reference_vnext_row_gather_bf16_provider,
    reference_vnext_silu_bf16_provider, reference_vnext_silu_mul_bf16_provider,
    reference_vnext_structured_attention_bf16_provider,
    reference_vnext_structured_gqa_attention_bf16_provider, reference_vnext_tanh_bf16_provider,
    reference_vnext_zero_centered_rms_norm_bf16_provider, ReferenceVNextComposition,
    REFERENCE_DENSE_SAFETENSORS_FORMAT_ID,
};
pub use cross_attention::REFERENCE_CROSS_ATTENTION_BF16_NUMERICAL_PROFILE_ID;
pub use gelu_tanh::REFERENCE_GELU_TANH_BF16_NUMERICAL_PROFILE_ID;
pub use head_rms_norm::REFERENCE_HEAD_RMS_NORM_BF16_NUMERICAL_PROFILE_ID;
pub use row_transport::{
    REFERENCE_BATCHED_ROW_CONCAT_BF16_NUMERICAL_PROFILE_ID,
    REFERENCE_ROW_GATHER_BF16_NUMERICAL_PROFILE_ID,
};
pub use runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceFence, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceDeviceRuntimeSnapshot, ReferenceDeviceStream,
};
pub use structured_attention::REFERENCE_STRUCTURED_ATTENTION_BF16_NUMERICAL_PROFILE_ID;
pub use zero_centered_rms_norm::REFERENCE_ZERO_CENTERED_RMS_NORM_BF16_NUMERICAL_PROFILE_ID;
