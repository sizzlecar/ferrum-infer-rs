use super::*;
use ferrum_runtime::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum MetalTensorOperation {
    Silu,
    SiluMul,
    Tanh,
    ModulationScale,
    GatedResidual,
    GeluTanh,
    PlainResidualAdd,
    DenseLinear,
    BlockLinear,
    LayerNorm,
    HeadRmsNorm,
    ZeroCenteredRmsNorm,
    AdjacentRope,
    HalfSplitRope,
    StructuredAttention,
    StructuredGqaAttention,
    RowGather,
    BatchedRowConcat,
    RowScatterAdd,
    Conv2d,
    ChannelL2Norm,
    ChannelL2NormImage,
    SiluNchw,
    PlainResidualAddNchw,
    SpatialAttention,
    SpatialUpsampleGather,
    SpatialGroupMean,
    AffineLinear,
    PatchProjection,
    AffineLayerNorm,
    GeluErf,
    HalfRotary,
    ReshapeCopy,
    SegmentedAttention,
}

impl MetalTensorOperation {
    pub const POINTWISE: [Self; 7] = [
        Self::Silu,
        Self::SiluMul,
        Self::Tanh,
        Self::ModulationScale,
        Self::GatedResidual,
        Self::GeluTanh,
        Self::PlainResidualAdd,
    ];
    pub const ALL: [Self; 34] = [
        Self::Silu,
        Self::SiluMul,
        Self::Tanh,
        Self::ModulationScale,
        Self::GatedResidual,
        Self::GeluTanh,
        Self::PlainResidualAdd,
        Self::DenseLinear,
        Self::BlockLinear,
        Self::LayerNorm,
        Self::HeadRmsNorm,
        Self::ZeroCenteredRmsNorm,
        Self::AdjacentRope,
        Self::HalfSplitRope,
        Self::StructuredAttention,
        Self::StructuredGqaAttention,
        Self::RowGather,
        Self::BatchedRowConcat,
        Self::RowScatterAdd,
        Self::Conv2d,
        Self::ChannelL2Norm,
        Self::ChannelL2NormImage,
        Self::SiluNchw,
        Self::PlainResidualAddNchw,
        Self::SpatialAttention,
        Self::SpatialUpsampleGather,
        Self::SpatialGroupMean,
        Self::AffineLinear,
        Self::PatchProjection,
        Self::AffineLayerNorm,
        Self::GeluErf,
        Self::HalfRotary,
        Self::ReshapeCopy,
        Self::SegmentedAttention,
    ];
    pub(super) fn is_pointwise(self) -> bool {
        Self::POINTWISE.contains(&self)
            || matches!(self, Self::SiluNchw | Self::PlainResidualAddNchw)
    }
    pub(super) fn is_linear(self) -> bool {
        matches!(
            self,
            Self::DenseLinear | Self::BlockLinear | Self::AffineLinear | Self::PatchProjection
        )
    }
    pub(super) fn is_norm(self) -> bool {
        matches!(
            self,
            Self::LayerNorm | Self::HeadRmsNorm | Self::ZeroCenteredRmsNorm | Self::AffineLayerNorm
        )
    }
    pub(super) fn is_rope(self) -> bool {
        matches!(self, Self::AdjacentRope | Self::HalfSplitRope)
    }
    pub(super) fn is_attention(self) -> bool {
        matches!(
            self,
            Self::StructuredAttention | Self::StructuredGqaAttention | Self::SegmentedAttention
        )
    }
    pub(super) fn is_transport(self) -> bool {
        matches!(self, Self::RowGather | Self::BatchedRowConcat)
    }
    pub(super) fn primitive(self) -> Option<TensorPrimitiveBf16> {
        Some(match self {
            Self::AffineLinear => TensorPrimitiveBf16::AffineLinear,
            Self::PatchProjection => TensorPrimitiveBf16::PatchProjection,
            Self::AffineLayerNorm => TensorPrimitiveBf16::AffineLayerNorm,
            Self::GeluErf => TensorPrimitiveBf16::GeluErf,
            Self::HalfRotary => TensorPrimitiveBf16::HalfRotary,
            Self::ReshapeCopy => TensorPrimitiveBf16::ReshapeCopy,
            Self::SegmentedAttention => TensorPrimitiveBf16::SegmentedAttention,
            _ => return None,
        })
    }
    pub(super) fn is_element_primitive(self) -> bool {
        matches!(self, Self::GeluErf | Self::HalfRotary | Self::ReshapeCopy)
    }
    pub(super) fn is_channel_norm(self) -> bool {
        matches!(self, Self::ChannelL2Norm | Self::ChannelL2NormImage)
    }
    pub(super) fn is_spatial(self) -> bool {
        self.is_channel_norm()
            || matches!(
                self,
                Self::Conv2d
                    | Self::SpatialAttention
                    | Self::SpatialUpsampleGather
                    | Self::SpatialGroupMean
            )
    }
    pub(super) fn pointwise_shader_kind(self) -> u32 {
        match self {
            Self::SiluNchw => Self::Silu as u32,
            Self::PlainResidualAddNchw => Self::PlainResidualAdd as u32,
            _ => self as u32,
        }
    }
    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        if let Some(kind) = self.primitive() {
            return kind.contract();
        }
        match self {
            Self::Silu => silu_bf16_contract(),
            Self::SiluMul => silu_mul_bf16_contract(),
            Self::Tanh => tanh_bf16_contract(),
            Self::ModulationScale => modulation_scale_bf16_contract(),
            Self::GatedResidual => gated_residual_bf16_contract(),
            Self::GeluTanh => gelu_tanh_bf16_contract(),
            Self::PlainResidualAdd => plain_residual_add_bf16_contract(),
            Self::DenseLinear => dense_linear_bf16_contract(),
            Self::BlockLinear => dense_linear_block_bf16_contract(),
            Self::LayerNorm => layer_norm_bf16_contract(),
            Self::HeadRmsNorm => head_rms_norm_bf16_contract(),
            Self::ZeroCenteredRmsNorm => zero_centered_rms_norm_bf16_contract(),
            Self::AdjacentRope => adjacent_rope_bf16_contract(),
            Self::HalfSplitRope => half_split_rope_bf16_contract(),
            Self::StructuredAttention => structured_attention_bf16_contract(),
            Self::StructuredGqaAttention => structured_gqa_attention_bf16_contract(),
            Self::RowGather => row_gather_bf16_contract(),
            Self::BatchedRowConcat => batched_row_concat_bf16_contract(),
            Self::RowScatterAdd => row_scatter_add_bf16_contract(),
            Self::Conv2d => conv2d_bf16_contract(),
            Self::ChannelL2Norm => channel_l2_norm_bf16_contract(),
            Self::ChannelL2NormImage => channel_l2_norm_image_bf16_contract(),
            Self::SiluNchw => silu_nchw_bf16_contract(),
            Self::PlainResidualAddNchw => plain_residual_add_nchw_bf16_contract(),
            Self::SpatialAttention => spatial_attention_bf16_contract(),
            Self::SpatialUpsampleGather => spatial_upsample_gather_bf16_contract(),
            Self::SpatialGroupMean => spatial_group_mean_bf16_contract(),
            _ => unreachable!("primitive contracts handled above"),
        }
    }
    pub fn profile(self) -> &'static str {
        match self {
            Self::Silu => "metal.precise.silu.bf16.v1",
            Self::SiluMul => "metal.precise.silu_mul.bf16.v1",
            Self::Tanh => "metal.precise.tanh.bf16.v1",
            Self::ModulationScale => "metal.precise.modulation_scale.bf16.v1",
            Self::GatedResidual => "metal.precise.gated_residual.bf16.v1",
            Self::GeluTanh => "metal.precise.gelu_tanh.bf16.v1",
            Self::PlainResidualAdd => "metal.precise.plain_residual_add.bf16.v1",
            Self::DenseLinear => super::linear::DENSE_PROFILE,
            Self::BlockLinear => super::linear::BLOCK_PROFILE,
            Self::LayerNorm => "metal.precise.layer_norm.bf16.padded_pairwise_sqrt_div.v1",
            Self::HeadRmsNorm => "metal.precise.head_rms_norm.bf16.padded_pairwise_sqrt_div.v1",
            Self::ZeroCenteredRmsNorm => {
                "metal.precise.zero_centered_rms_norm.bf16.padded_pairwise_sqrt_div.v1"
            }
            Self::AdjacentRope => "metal.precise.adjacent_rope.bf16.fma.v1",
            Self::HalfSplitRope => "metal.precise.half_split_rope.bf16.staged.v1",
            Self::StructuredAttention => {
                "metal.simdgroup.f32.structured_attention.bf16.tile16x32_online.v1"
            }
            Self::StructuredGqaAttention => {
                "metal.simdgroup.f32.structured_gqa_attention.bf16.tile16x32_online.v1"
            }
            Self::RowGather => "metal.bitcopy.row_gather.bf16.v1",
            Self::BatchedRowConcat => "metal.bitcopy.batched_row_concat.bf16.v1",
            Self::RowScatterAdd => "metal.precise.row_scatter_add.bf16.v1",
            Self::Conv2d => "metal.simdgroup.f32.conv2d.bf16.pad_then_dot_then_bias.v1",
            Self::ChannelL2Norm => "metal.precise.channel_l2_norm.bf16.padded_pairwise_staged.v1",
            Self::ChannelL2NormImage => {
                "metal.precise.channel_l2_norm_image.bf16.padded_pairwise_staged.v1"
            }
            Self::SiluNchw => "metal.precise.silu.nchw.bf16.v1",
            Self::PlainResidualAddNchw => "metal.precise.plain_residual_add.nchw.bf16.v1",
            Self::SpatialAttention => {
                "metal.simdgroup.f32.spatial_attention.bf16.tile16x32_managed_accum.v1"
            }
            Self::SpatialUpsampleGather => "metal.bitcopy.spatial_upsample_gather.bf16.v1",
            Self::SpatialGroupMean => {
                "metal.precise.spatial_group_mean.bf16.serial_first_element.v1"
            }
            Self::AffineLinear => "metal.simdgroup.f32.affine_linear.bf16.v1",
            Self::PatchProjection => "metal.simdgroup.f32.patch_projection.bf16.dot_then_bias.v1",
            Self::AffineLayerNorm => "metal.precise.affine_layer_norm.bf16.padded_pairwise.v1",
            Self::GeluErf => "metal.precise.gelu_erf.bf16.sunpro_f32.v1",
            Self::HalfRotary => "metal.precise.half_rotary.bf16.f32_tables.v1",
            Self::ReshapeCopy => "metal.bitcopy.reshape.bf16.v1",
            Self::SegmentedAttention => {
                "metal.simdgroup.f32.segmented_attention.bf16.tile16x32_online.v1"
            }
        }
    }
    pub(super) fn capability(self) -> &'static str {
        if let Some(kind) = self.primitive() {
            return kind.capability_id();
        }
        match self {
            Self::Silu => SILU_BF16_CAPABILITY_ID,
            Self::SiluMul => SILU_MUL_BF16_CAPABILITY_ID,
            Self::Tanh => TANH_BF16_CAPABILITY_ID,
            Self::ModulationScale => MODULATION_SCALE_BF16_CAPABILITY_ID,
            Self::GatedResidual => GATED_RESIDUAL_BF16_CAPABILITY_ID,
            Self::GeluTanh => GELU_TANH_BF16_CAPABILITY_ID,
            Self::PlainResidualAdd => PLAIN_RESIDUAL_ADD_BF16_CAPABILITY_ID,
            Self::DenseLinear => DENSE_LINEAR_BF16_CAPABILITY_ID,
            Self::BlockLinear => DENSE_LINEAR_BLOCK_BF16_CAPABILITY_ID,
            Self::LayerNorm => LAYER_NORM_BF16_CAPABILITY_ID,
            Self::HeadRmsNorm => HEAD_RMS_NORM_BF16_CAPABILITY_ID,
            Self::ZeroCenteredRmsNorm => ZERO_CENTERED_RMS_NORM_BF16_CAPABILITY_ID,
            Self::AdjacentRope => ADJACENT_ROPE_BF16_CAPABILITY_ID,
            Self::HalfSplitRope => HALF_SPLIT_ROPE_BF16_CAPABILITY_ID,
            Self::StructuredAttention => STRUCTURED_ATTENTION_BF16_CAPABILITY_ID,
            Self::StructuredGqaAttention => STRUCTURED_GQA_ATTENTION_BF16_CAPABILITY_ID,
            Self::RowGather => ROW_GATHER_BF16_CAPABILITY_ID,
            Self::BatchedRowConcat => BATCHED_ROW_CONCAT_BF16_CAPABILITY_ID,
            Self::RowScatterAdd => ROW_SCATTER_ADD_BF16_CAPABILITY_ID,
            Self::Conv2d => CONV2D_BF16_CAPABILITY_ID,
            Self::ChannelL2Norm => CHANNEL_L2_NORM_BF16_CAPABILITY_ID,
            Self::ChannelL2NormImage => CHANNEL_L2_NORM_IMAGE_BF16_CAPABILITY_ID,
            Self::SiluNchw => SILU_NCHW_BF16_CAPABILITY_ID,
            Self::PlainResidualAddNchw => PLAIN_RESIDUAL_ADD_NCHW_BF16_CAPABILITY_ID,
            Self::SpatialAttention => SPATIAL_ATTENTION_BF16_CAPABILITY_ID,
            Self::SpatialUpsampleGather => SPATIAL_UPSAMPLE_GATHER_BF16_CAPABILITY_ID,
            Self::SpatialGroupMean => SPATIAL_GROUP_MEAN_BF16_CAPABILITY_ID,
            _ => unreachable!("primitive capabilities handled above"),
        }
    }
    pub(super) fn operation(self) -> &'static str {
        if let Some(kind) = self.primitive() {
            return kind.operation_id();
        }
        match self {
            Self::Silu => SILU_BF16_OPERATION_ID,
            Self::SiluMul => SILU_MUL_BF16_OPERATION_ID,
            Self::Tanh => TANH_BF16_OPERATION_ID,
            Self::ModulationScale => MODULATION_SCALE_BF16_OPERATION_ID,
            Self::GatedResidual => GATED_RESIDUAL_BF16_OPERATION_ID,
            Self::GeluTanh => GELU_TANH_BF16_OPERATION_ID,
            Self::PlainResidualAdd => PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
            Self::DenseLinear => DENSE_LINEAR_BF16_OPERATION_ID,
            Self::BlockLinear => DENSE_LINEAR_BLOCK_BF16_OPERATION_ID,
            Self::LayerNorm => LAYER_NORM_BF16_OPERATION_ID,
            Self::HeadRmsNorm => HEAD_RMS_NORM_BF16_OPERATION_ID,
            Self::ZeroCenteredRmsNorm => ZERO_CENTERED_RMS_NORM_BF16_OPERATION_ID,
            Self::AdjacentRope => ADJACENT_ROPE_BF16_OPERATION_ID,
            Self::HalfSplitRope => HALF_SPLIT_ROPE_BF16_OPERATION_ID,
            Self::StructuredAttention => STRUCTURED_ATTENTION_BF16_OPERATION_ID,
            Self::StructuredGqaAttention => STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID,
            Self::RowGather => ROW_GATHER_BF16_OPERATION_ID,
            Self::BatchedRowConcat => BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
            Self::RowScatterAdd => ROW_SCATTER_ADD_BF16_OPERATION_ID,
            Self::Conv2d => CONV2D_BF16_OPERATION_ID,
            Self::ChannelL2Norm => CHANNEL_L2_NORM_BF16_OPERATION_ID,
            Self::ChannelL2NormImage => CHANNEL_L2_NORM_IMAGE_BF16_OPERATION_ID,
            Self::SiluNchw => SILU_NCHW_BF16_OPERATION_ID,
            Self::PlainResidualAddNchw => PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID,
            Self::SpatialAttention => SPATIAL_ATTENTION_BF16_OPERATION_ID,
            Self::SpatialUpsampleGather => SPATIAL_UPSAMPLE_GATHER_BF16_OPERATION_ID,
            Self::SpatialGroupMean => SPATIAL_GROUP_MEAN_BF16_OPERATION_ID,
            _ => unreachable!("primitive operations handled above"),
        }
    }
    pub(super) fn name(self) -> &'static str {
        if let Some(kind) = self.primitive() {
            return kind.name();
        }
        match self {
            Self::Silu => "silu",
            Self::SiluMul => "silu_mul",
            Self::Tanh => "tanh",
            Self::ModulationScale => "modulation_scale",
            Self::GatedResidual => "gated_residual",
            Self::GeluTanh => "gelu_tanh",
            Self::PlainResidualAdd => "plain_residual_add",
            Self::DenseLinear => "dense_linear",
            Self::BlockLinear => "native_block_linear",
            Self::LayerNorm => "layer_norm",
            Self::HeadRmsNorm => "head_rms_norm",
            Self::ZeroCenteredRmsNorm => "zero_centered_rms_norm",
            Self::AdjacentRope => "adjacent_rope",
            Self::HalfSplitRope => "half_split_rope",
            Self::StructuredAttention => "structured_attention",
            Self::StructuredGqaAttention => "structured_gqa_attention",
            Self::RowGather => "row_gather",
            Self::BatchedRowConcat => "batched_row_concat",
            Self::RowScatterAdd => "row_scatter_add",
            Self::Conv2d => "conv2d",
            Self::ChannelL2Norm => "channel_l2_norm",
            Self::ChannelL2NormImage => "channel_l2_norm_image",
            Self::SiluNchw => "silu_nchw",
            Self::PlainResidualAddNchw => "plain_residual_add_nchw",
            Self::SpatialAttention => "spatial_attention",
            Self::SpatialUpsampleGather => "spatial_upsample_gather",
            Self::SpatialGroupMean => "spatial_group_mean",
            _ => unreachable!("primitive names handled above"),
        }
    }
    pub(super) fn inputs(self) -> u32 {
        if let Some(kind) = self.primitive() {
            return kind.input_count();
        }
        match self {
            Self::Silu | Self::Tanh | Self::GeluTanh => 1,
            Self::SiluMul | Self::ModulationScale | Self::PlainResidualAdd => 2,
            Self::GatedResidual => 3,
            Self::DenseLinear | Self::BlockLinear => 2,
            Self::LayerNorm => 1,
            Self::HeadRmsNorm | Self::ZeroCenteredRmsNorm | Self::AdjacentRope => 2,
            Self::HalfSplitRope => 3,
            Self::StructuredAttention | Self::StructuredGqaAttention => 3,
            Self::RowGather => 1,
            Self::BatchedRowConcat => 2,
            Self::RowScatterAdd => 2,
            Self::Conv2d => 3,
            Self::ChannelL2Norm | Self::ChannelL2NormImage | Self::PlainResidualAddNchw => 2,
            Self::SiluNchw | Self::SpatialAttention | Self::SpatialUpsampleGather => 1,
            Self::SpatialGroupMean => 1,
            _ => unreachable!("primitive ports handled above"),
        }
    }
    pub(super) fn broadcasts(self) -> bool {
        matches!(self, Self::ModulationScale | Self::GatedResidual)
    }
    pub(super) fn pointwise_operations(self) -> u64 {
        match self {
            Self::Silu | Self::SiluNchw => 4,
            Self::SiluMul => 5,
            Self::Tanh | Self::PlainResidualAdd | Self::PlainResidualAddNchw => 1,
            Self::ModulationScale | Self::GatedResidual => 2,
            Self::GeluTanh => 9,
            _ => unreachable!("non-pointwise work is computed by its own geometry"),
        }
    }
}
