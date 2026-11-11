//! Typed selection from one fixed native module. Selection never grants a
//! capability absent from the runtime descriptor and is checked before load.

use super::super::tensor_dense_bf16::{self as dense, RowBiasProjection};
use super::*;
use ferrum_runtime::TensorPrimitiveBf16;

/// BF16 fixed-tensor operations implemented by the shared CUDA native module.
/// Bias-free dense remains separate. F32 affine/patch combine the admitted
/// lane BLAS with one broadcast in this same admitted native module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum CudaTensorOperation {
    /// Native block weights, BF16 decode scratch, and the admitted lane BLAS.
    BlockLinear,
    Silu,
    SiluMul,
    Tanh,
    ModulationScale,
    GatedResidual,
    LayerNorm,
    HeadRmsNorm,
    AdjacentRope,
    StructuredAttention,
    ZeroCenteredRmsNorm,
    GeluTanh,
    RowGather,
    BatchedRowConcat,
    PlainResidualAdd,
    HalfSplitRope,
    PartialHalfSplitRope,
    StructuredGqaAttention,
    ChannelL2Norm,
    SiluNchw,
    PlainResidualAddNchw,
    ChannelL2NormImage,
    SpatialAttention,
    SpatialUpsampleGather,
    SpatialGroupMean,
    AffineLinear,
    PatchProjection,
    AffineLayerNorm,
    GeluErf,
    HalfRotary,
    RowScatterAdd,
    ReshapeCopy,
    SegmentedAttention,
    /// Explicit-state normal generation. Its launch geometry is queried from
    /// the existing runtime only when this operation is selected.
    PhiloxNormal,
    /// Canonical codebook W4 storage, regular ConvRot A8, integer GEMM.
    ComfyConvRotW4A8,
    /// Already-rotated signed INT8 weights and per-output [N,1] scales.
    ComfyConvRotW8A8,
    FusedHeadRmsPartialRope,
    RowRmsNormSingleRound,
    Addcmul,
    /// F32 SiLU(gate)*up retained through ConvRot; no M-dependent rounding.
    ComfyConvRotSwiGluF32W4A8,
    ComfyConvRotSwiGluF32W8A8,
    CastF32ToBf16,
    ComfyConvRotW4A8F32,
    ComfyConvRotW8A8F32,
    ComfyConvRotInt8EmbeddingF32,
    StructuredGqaAttentionF32,
    /// Explicit external CUTLASS module; requires its independently admitted factory.
    StructuredGqaAttentionFusedF32,
    HeadRmsNormF32,
    HalfSplitRopeF32,
    PlainResidualAddF32,
    SiluMulF32,
    GroupedColumnSliceF32,
    ChannelBiasAddF32,
    ChannelScaledResidualF32,
    NanToNumF32,
    UnitHeadRmsPartialRopeF32,
    RowSelectedModulationBf16F32,
    BatchedSuffixAppendF32,
    LayerNormF32,
    GroupNormSiluNchwF32,
    PlainResidualAddNchwF32,
    /// Direct native F32 draws; shares explicit seed/offset and launch mapping.
    PhiloxNormalF32,
    GeluTanhF32,
    GeluErfF32,
    ReshapeCopyF32,
    RowScatterReplaceF32,
    RowScatterAddF32,
    RowSelectedModulationBf16,
    RowSelectedAddcmulBf16,
    /// Pedantic SGEMM dot-store followed by one retained F32 row broadcast.
    AffineLinearF32,
    /// Original rank5 patch weights, SGEMM then the same F32 row broadcast.
    PatchProjectionF32,
}

impl CudaTensorOperation {
    /// Pure provider metadata for matching a selected operation to its actual
    /// registered contract before native module acquisition. Philox requires
    /// explicitly checked launch geometry; this never opens or queries a GPU.
    /// F32 affine/patch query the loaded cuBLAS software version with a null
    /// handle, without creating a context, module or BLAS handle.
    pub fn provider_descriptor(
        self,
        device: &DeviceDescriptor,
        philox_geometry: Option<CudaPhiloxDeviceGeometry>,
    ) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
        let mut selected = selected_descriptors(device, &[self], philox_geometry)?;
        Ok(
            match selected.pop().expect("one explicit operation descriptor") {
                SelectedDescriptor::Fixed(_, descriptor) => descriptor,
                SelectedDescriptor::Philox(prepared) => prepared.descriptor().clone(),
                SelectedDescriptor::PhiloxF32(prepared) => prepared.descriptor().clone(),
            },
        )
    }

    // Legacy factories select the original leading operation subsets.
    // Append new capabilities so those callers never acquire a new dependency.
    pub const ALL: [Self; 70] = [
        Self::Silu,
        Self::SiluMul,
        Self::Tanh,
        Self::ModulationScale,
        Self::GatedResidual,
        Self::LayerNorm,
        Self::HeadRmsNorm,
        Self::AdjacentRope,
        Self::StructuredAttention,
        Self::ZeroCenteredRmsNorm,
        Self::GeluTanh,
        Self::RowGather,
        Self::BatchedRowConcat,
        Self::PlainResidualAdd,
        Self::HalfSplitRope,
        Self::StructuredGqaAttention,
        Self::ChannelL2Norm,
        Self::SiluNchw,
        Self::PlainResidualAddNchw,
        Self::ChannelL2NormImage,
        Self::SpatialAttention,
        Self::SpatialUpsampleGather,
        Self::SpatialGroupMean,
        Self::AffineLinear,
        Self::PatchProjection,
        Self::AffineLayerNorm,
        Self::GeluErf,
        Self::HalfRotary,
        Self::RowScatterAdd,
        Self::ReshapeCopy,
        Self::SegmentedAttention,
        Self::PhiloxNormal,
        Self::BlockLinear,
        Self::ComfyConvRotW4A8,
        Self::PartialHalfSplitRope,
        Self::ComfyConvRotW8A8,
        Self::FusedHeadRmsPartialRope,
        Self::RowRmsNormSingleRound,
        Self::Addcmul,
        Self::ComfyConvRotSwiGluF32W4A8,
        Self::ComfyConvRotSwiGluF32W8A8,
        Self::CastF32ToBf16,
        Self::ComfyConvRotW4A8F32,
        Self::ComfyConvRotW8A8F32,
        Self::ComfyConvRotInt8EmbeddingF32,
        Self::StructuredGqaAttentionF32,
        Self::HeadRmsNormF32,
        Self::HalfSplitRopeF32,
        Self::PlainResidualAddF32,
        Self::SiluMulF32,
        Self::GroupedColumnSliceF32,
        Self::ChannelBiasAddF32,
        Self::ChannelScaledResidualF32,
        Self::NanToNumF32,
        Self::UnitHeadRmsPartialRopeF32,
        Self::RowSelectedModulationBf16F32,
        Self::BatchedSuffixAppendF32,
        Self::LayerNormF32,
        Self::GroupNormSiluNchwF32,
        Self::PlainResidualAddNchwF32,
        Self::PhiloxNormalF32,
        Self::GeluTanhF32,
        Self::GeluErfF32,
        Self::ReshapeCopyF32,
        Self::RowScatterReplaceF32,
        Self::RowScatterAddF32,
        Self::RowSelectedModulationBf16,
        Self::RowSelectedAddcmulBf16,
        Self::AffineLinearF32,
        Self::PatchProjectionF32,
    ];

    fn primitive(self) -> Option<TensorPrimitiveBf16> {
        match self {
            Self::AffineLinear => Some(TensorPrimitiveBf16::AffineLinear),
            Self::PatchProjection => Some(TensorPrimitiveBf16::PatchProjection),
            Self::AffineLayerNorm => Some(TensorPrimitiveBf16::AffineLayerNorm),
            Self::GeluErf => Some(TensorPrimitiveBf16::GeluErf),
            Self::HalfRotary => Some(TensorPrimitiveBf16::HalfRotary),
            Self::RowScatterAdd => Some(TensorPrimitiveBf16::RowScatterAdd),
            Self::ReshapeCopy => Some(TensorPrimitiveBf16::ReshapeCopy),
            Self::SegmentedAttention => Some(TensorPrimitiveBf16::SegmentedAttention),
            _ => None,
        }
    }

    fn pointwise(self) -> Option<Kind> {
        match self {
            Self::Silu => Some(Kind::Silu),
            Self::SiluMul => Some(Kind::SiluMul),
            Self::Tanh => Some(Kind::Tanh),
            Self::ModulationScale => Some(Kind::Modulation),
            Self::GatedResidual => Some(Kind::GatedResidual),
            Self::GeluTanh => Some(Kind::GeluTanh),
            Self::PlainResidualAdd => Some(Kind::PlainResidualAdd),
            _ => None,
        }
    }

    fn normalization(self) -> Option<norm::Kind> {
        match self {
            Self::LayerNorm => Some(norm::Kind::Layer),
            Self::HeadRmsNorm => Some(norm::Kind::HeadRms),
            Self::ZeroCenteredRmsNorm => Some(norm::Kind::ZeroCenteredRms),
            _ => None,
        }
    }

    fn descriptor(
        self,
        device: &DeviceDescriptor,
    ) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
        if let Some(kind) = self.primitive() {
            return primitives::descriptor(device, kind);
        }
        if let Some(kind) = self.pointwise() {
            return descriptor(device, kind);
        }
        if let Some(kind) = self.normalization() {
            return norm::descriptor(device, kind);
        }
        match self {
            Self::PlainResidualAddF32 => {
                pointwise_f32::descriptor(device, ferrum_runtime::F32PointwiseOp::ResidualAdd)
            }
            Self::SiluMulF32 => {
                pointwise_f32::descriptor(device, ferrum_runtime::F32PointwiseOp::SiluMul)
            }
            Self::GeluTanhF32 => gelu_f32::descriptor(device, ferrum_runtime::GeluF32Variant::Tanh),
            Self::GeluErfF32 => gelu_f32::descriptor(device, ferrum_runtime::GeluF32Variant::Erf),
            Self::ReshapeCopyF32 => reshape_copy_f32::descriptor(device),
            Self::AffineLinearF32 => dense::row_bias_descriptor(device, RowBiasProjection::Affine),
            Self::PatchProjectionF32 => {
                dense::row_bias_descriptor(device, RowBiasProjection::Patch)
            }
            Self::RowScatterReplaceF32 => {
                row_scatter_f32::descriptor(device, ferrum_runtime::RowScatterF32Op::Replace)
            }
            Self::RowScatterAddF32 => {
                row_scatter_f32::descriptor(device, ferrum_runtime::RowScatterF32Op::Add)
            }
            Self::BlockLinear => block_linear::descriptor(device),
            Self::ComfyConvRotW4A8 => comfy_convrot::descriptor(device),
            Self::ComfyConvRotW8A8 => comfy_convrot::int8_descriptor(device),
            Self::ComfyConvRotSwiGluF32W4A8 => comfy_convrot::swiglu_w4_descriptor(device),
            Self::ComfyConvRotSwiGluF32W8A8 => comfy_convrot::swiglu_w8_descriptor(device),
            Self::CastF32ToBf16 => cast_f32_to_bf16::descriptor(device),
            Self::ComfyConvRotW4A8F32 => comfy_convrot::f32_w4_descriptor(device),
            Self::ComfyConvRotW8A8F32 => comfy_convrot::f32_w8_descriptor(device),
            Self::ComfyConvRotInt8EmbeddingF32 => comfy_embedding::descriptor(device),
            Self::FusedHeadRmsPartialRope => fused_head_rms_partial_rope::descriptor(device),
            Self::RowRmsNormSingleRound => {
                row_rms_addcmul::descriptor(device, row_rms_addcmul::Kind::RowRms)
            }
            Self::Addcmul => row_rms_addcmul::descriptor(device, row_rms_addcmul::Kind::Addcmul),
            Self::ChannelL2NormImage => {
                spatial::descriptor(device, spatial::SpatialKind::ChannelL2Image)
            }
            Self::SpatialAttention => {
                spatial_extended::descriptor(device, spatial_extended::Kind::Attention)
            }
            Self::SpatialUpsampleGather => {
                spatial_extended::descriptor(device, spatial_extended::Kind::Gather)
            }
            Self::SpatialGroupMean => {
                spatial_extended::descriptor(device, spatial_extended::Kind::GroupMean)
            }
            Self::ChannelL2Norm => spatial::descriptor(device, spatial::SpatialKind::ChannelL2),
            Self::SiluNchw => spatial::descriptor(device, spatial::SpatialKind::Silu),
            Self::PlainResidualAddNchw => {
                spatial::descriptor(device, spatial::SpatialKind::ResidualAdd)
            }
            Self::AdjacentRope => rope::descriptor(device),
            Self::StructuredAttention => attention::descriptor(device),
            Self::HalfSplitRope => half_split_rope::descriptor(device),
            Self::PartialHalfSplitRope => half_split_rope::partial_descriptor(device),
            Self::StructuredGqaAttention => gqa::descriptor(device),
            Self::StructuredGqaAttentionF32 => gqa_f32::descriptor(device),
            Self::StructuredGqaAttentionFusedF32 => gqa_fused_f32::descriptor(device),
            Self::GroupedColumnSliceF32 => channelwise_f32::descriptor(
                device,
                ferrum_runtime::F32ChannelwiseOp::GroupedColumnSlice,
            ),
            Self::ChannelBiasAddF32 => {
                channelwise_f32::descriptor(device, ferrum_runtime::F32ChannelwiseOp::BiasAdd)
            }
            Self::ChannelScaledResidualF32 => channelwise_f32::descriptor(
                device,
                ferrum_runtime::F32ChannelwiseOp::ScaledResidual,
            ),
            Self::NanToNumF32 => {
                channelwise_f32::descriptor(device, ferrum_runtime::F32ChannelwiseOp::NanToNum)
            }
            Self::UnitHeadRmsPartialRopeF32 => {
                norm_rope_f32::descriptor(device, norm_rope_f32::Kind::UnitHeadRmsPartialRope)
            }
            Self::HeadRmsNormF32 => norm_rope_f32::descriptor(device, norm_rope_f32::Kind::HeadRms),
            Self::RowSelectedModulationBf16F32 => row_selected_modulation::descriptor(device),
            Self::RowSelectedModulationBf16 => row_selected_bf16::descriptor(
                device,
                ferrum_runtime::RowSelectedBf16Kind::Modulation,
            ),
            Self::RowSelectedAddcmulBf16 => {
                row_selected_bf16::descriptor(device, ferrum_runtime::RowSelectedBf16Kind::Addcmul)
            }
            Self::BatchedSuffixAppendF32 => batched_suffix_append_f32::descriptor(device),
            Self::LayerNormF32 => layer_norm_f32::descriptor(device),
            Self::GroupNormSiluNchwF32 => {
                cnn_f32::descriptor(device, ferrum_runtime::F32CnnOp::GroupNormSilu)
            }
            Self::PlainResidualAddNchwF32 => {
                cnn_f32::descriptor(device, ferrum_runtime::F32CnnOp::ResidualAdd)
            }
            Self::HalfSplitRopeF32 => {
                norm_rope_f32::descriptor(device, norm_rope_f32::Kind::HalfSplitRope)
            }
            Self::RowGather => transport::descriptor(device, transport::Kind::RowGather),
            Self::BatchedRowConcat => {
                transport::descriptor(device, transport::Kind::BatchedRowConcat)
            }
            _ => Err(CudaDeviceRuntimeError::contract(
                "unknown CUDA tensor operation",
            )),
        }
    }

    fn provider(
        self,
        descriptor: OperationProviderDescriptor,
        kernels: &Arc<RetainedFunctions>,
    ) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
        if let Some(kind) = self.primitive() {
            return primitives::provider(kind, descriptor, kernels);
        }
        if let Some(kind) = self.pointwise() {
            return Box::new(CudaTensorPointwiseProvider {
                descriptor,
                kind,
                kernels: Arc::clone(kernels),
            });
        }
        if let Some(kind) = self.normalization() {
            return norm::provider(kind, descriptor, kernels);
        }
        match self {
            Self::PlainResidualAddF32 => pointwise_f32::provider(
                descriptor,
                kernels,
                ferrum_runtime::F32PointwiseOp::ResidualAdd,
            ),
            Self::SiluMulF32 => pointwise_f32::provider(
                descriptor,
                kernels,
                ferrum_runtime::F32PointwiseOp::SiluMul,
            ),
            Self::GeluTanhF32 => {
                gelu_f32::provider(descriptor, kernels, ferrum_runtime::GeluF32Variant::Tanh)
            }
            Self::GeluErfF32 => {
                gelu_f32::provider(descriptor, kernels, ferrum_runtime::GeluF32Variant::Erf)
            }
            Self::ReshapeCopyF32 => reshape_copy_f32::provider(descriptor, kernels),
            Self::AffineLinearF32 => dense::row_bias_provider(
                descriptor,
                RetainedF32RowBias::new(kernels),
                RowBiasProjection::Affine,
            ),
            Self::PatchProjectionF32 => dense::row_bias_provider(
                descriptor,
                RetainedF32RowBias::new(kernels),
                RowBiasProjection::Patch,
            ),
            Self::RowScatterReplaceF32 => row_scatter_f32::provider(
                descriptor,
                kernels,
                ferrum_runtime::RowScatterF32Op::Replace,
            ),
            Self::RowScatterAddF32 => {
                row_scatter_f32::provider(descriptor, kernels, ferrum_runtime::RowScatterF32Op::Add)
            }
            Self::BlockLinear => block_linear::provider(descriptor, kernels),
            Self::ComfyConvRotW4A8 => comfy_convrot::provider(descriptor, kernels),
            Self::ComfyConvRotW8A8 => comfy_convrot::int8_provider(descriptor, kernels),
            Self::ComfyConvRotSwiGluF32W4A8 => {
                comfy_convrot::swiglu_w4_provider(descriptor, kernels)
            }
            Self::ComfyConvRotSwiGluF32W8A8 => {
                comfy_convrot::swiglu_w8_provider(descriptor, kernels)
            }
            Self::CastF32ToBf16 => cast_f32_to_bf16::provider(descriptor, kernels),
            Self::ComfyConvRotW4A8F32 => comfy_convrot::f32_w4_provider(descriptor, kernels),
            Self::ComfyConvRotW8A8F32 => comfy_convrot::f32_w8_provider(descriptor, kernels),
            Self::ComfyConvRotInt8EmbeddingF32 => comfy_embedding::provider(descriptor, kernels),
            Self::RowRmsNormSingleRound => {
                row_rms_addcmul::provider(row_rms_addcmul::Kind::RowRms, descriptor, kernels)
            }
            Self::Addcmul => {
                row_rms_addcmul::provider(row_rms_addcmul::Kind::Addcmul, descriptor, kernels)
            }
            Self::FusedHeadRmsPartialRope => {
                fused_head_rms_partial_rope::provider(descriptor, kernels)
            }
            Self::ChannelL2NormImage => {
                spatial::provider(spatial::SpatialKind::ChannelL2Image, descriptor, kernels)
            }
            Self::SpatialAttention => {
                spatial_extended::provider(spatial_extended::Kind::Attention, descriptor, kernels)
            }
            Self::SpatialUpsampleGather => {
                spatial_extended::provider(spatial_extended::Kind::Gather, descriptor, kernels)
            }
            Self::SpatialGroupMean => {
                spatial_extended::provider(spatial_extended::Kind::GroupMean, descriptor, kernels)
            }
            Self::ChannelL2Norm => {
                spatial::provider(spatial::SpatialKind::ChannelL2, descriptor, kernels)
            }
            Self::SiluNchw => spatial::provider(spatial::SpatialKind::Silu, descriptor, kernels),
            Self::PlainResidualAddNchw => {
                spatial::provider(spatial::SpatialKind::ResidualAdd, descriptor, kernels)
            }
            Self::AdjacentRope => rope::provider(descriptor, kernels),
            Self::StructuredAttention => attention::provider(descriptor, kernels),
            Self::HalfSplitRope => half_split_rope::provider(descriptor, kernels),
            Self::PartialHalfSplitRope => half_split_rope::provider(descriptor, kernels),
            Self::StructuredGqaAttention => gqa::provider(descriptor, kernels),
            Self::StructuredGqaAttentionF32 => gqa_f32::provider(descriptor, kernels),
            Self::GroupedColumnSliceF32 => channelwise_f32::provider(
                descriptor,
                kernels,
                ferrum_runtime::F32ChannelwiseOp::GroupedColumnSlice,
            ),
            Self::ChannelBiasAddF32 => channelwise_f32::provider(
                descriptor,
                kernels,
                ferrum_runtime::F32ChannelwiseOp::BiasAdd,
            ),
            Self::ChannelScaledResidualF32 => channelwise_f32::provider(
                descriptor,
                kernels,
                ferrum_runtime::F32ChannelwiseOp::ScaledResidual,
            ),
            Self::NanToNumF32 => channelwise_f32::provider(
                descriptor,
                kernels,
                ferrum_runtime::F32ChannelwiseOp::NanToNum,
            ),
            Self::UnitHeadRmsPartialRopeF32 => norm_rope_f32::provider(
                norm_rope_f32::Kind::UnitHeadRmsPartialRope,
                descriptor,
                kernels,
            ),
            Self::HeadRmsNormF32 => {
                norm_rope_f32::provider(norm_rope_f32::Kind::HeadRms, descriptor, kernels)
            }
            Self::RowSelectedModulationBf16F32 => {
                row_selected_modulation::provider(descriptor, kernels)
            }
            Self::RowSelectedModulationBf16 => row_selected_bf16::provider(
                descriptor,
                kernels,
                ferrum_runtime::RowSelectedBf16Kind::Modulation,
            ),
            Self::RowSelectedAddcmulBf16 => row_selected_bf16::provider(
                descriptor,
                kernels,
                ferrum_runtime::RowSelectedBf16Kind::Addcmul,
            ),
            Self::BatchedSuffixAppendF32 => {
                batched_suffix_append_f32::provider(descriptor, kernels)
            }
            Self::LayerNormF32 => layer_norm_f32::provider(descriptor, kernels),
            Self::GroupNormSiluNchwF32 => {
                cnn_f32::provider(descriptor, kernels, ferrum_runtime::F32CnnOp::GroupNormSilu)
            }
            Self::PlainResidualAddNchwF32 => {
                cnn_f32::provider(descriptor, kernels, ferrum_runtime::F32CnnOp::ResidualAdd)
            }
            Self::HalfSplitRopeF32 => {
                norm_rope_f32::provider(norm_rope_f32::Kind::HalfSplitRope, descriptor, kernels)
            }
            Self::RowGather => transport::provider(transport::Kind::RowGather, descriptor, kernels),
            Self::BatchedRowConcat => {
                transport::provider(transport::Kind::BatchedRowConcat, descriptor, kernels)
            }
            _ => unreachable!("operation kind was exhaustively classified"),
        }
    }
}

/// A Philox descriptor cannot lose or substitute its fingerprinted geometry
/// between metadata validation and sharing the admitted module owner.
pub(super) enum SelectedDescriptor {
    Fixed(CudaTensorOperation, OperationProviderDescriptor),
    Philox(philox::PreparedProvider),
    PhiloxF32(philox_f32::PreparedProvider),
}

impl SelectedDescriptor {
    #[cfg(test)]
    fn operation(&self) -> CudaTensorOperation {
        match self {
            Self::Fixed(operation, _) => *operation,
            Self::Philox(_) => CudaTensorOperation::PhiloxNormal,
            Self::PhiloxF32(_) => CudaTensorOperation::PhiloxNormalF32,
        }
    }

    #[cfg(test)]
    fn descriptor(&self) -> &OperationProviderDescriptor {
        match self {
            Self::Fixed(_, descriptor) => descriptor,
            Self::Philox(prepared) => prepared.descriptor(),
            Self::PhiloxF32(prepared) => prepared.descriptor(),
        }
    }

    fn into_provider(
        self,
        kernels: &Arc<RetainedFunctions>,
    ) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
        match self {
            Self::Fixed(operation, descriptor) => operation.provider(descriptor, kernels),
            Self::Philox(prepared) => prepared.into_provider(kernels),
            Self::PhiloxF32(prepared) => prepared.into_provider(kernels),
        }
    }
}

pub(super) fn validate_selection(
    selected: &[CudaTensorOperation],
) -> Result<(), CudaDeviceRuntimeError> {
    if selected.is_empty() || selected.len() > CudaTensorOperation::ALL.len() + 1 {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA tensor operation selection must be nonempty and bounded by the module",
        ));
    }
    for (index, operation) in selected.iter().enumerate() {
        if selected[..index].contains(operation) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA tensor operation selection contains a duplicate",
            ));
        }
    }
    Ok(())
}

/// Pure metadata preparation. Philox needs explicitly supplied, checked device
/// geometry; absence never falls back to creating a context or querying a GPU.
pub(super) fn selected_descriptors(
    device: &DeviceDescriptor,
    selected: &[CudaTensorOperation],
    philox_geometry: Option<CudaPhiloxDeviceGeometry>,
) -> Result<Vec<SelectedDescriptor>, CudaDeviceRuntimeError> {
    validate_selection(selected)?;
    selected
        .iter()
        .copied()
        .map(|operation| {
            if matches!(
                operation,
                CudaTensorOperation::PhiloxNormal | CudaTensorOperation::PhiloxNormalF32
            ) {
                let geometry = philox_geometry.ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "Philox selection requires explicit device launch geometry",
                    )
                })?;
                if operation == CudaTensorOperation::PhiloxNormal {
                    philox::PreparedProvider::new(device, geometry).map(SelectedDescriptor::Philox)
                } else {
                    philox_f32::PreparedProvider::new(device, geometry)
                        .map(SelectedDescriptor::PhiloxF32)
                }
            } else {
                operation
                    .descriptor(device)
                    .map(|d| SelectedDescriptor::Fixed(operation, d))
            }
        })
        .collect()
}

pub(super) fn runtime_descriptors(
    runtime: &CudaDeviceRuntime,
    selected: &[CudaTensorOperation],
) -> Result<Vec<SelectedDescriptor>, CudaDeviceRuntimeError> {
    runtime_descriptors_with_query(runtime.descriptor(), selected, || {
        CudaPhiloxDeviceGeometry::query(runtime)
    })
}

fn runtime_descriptors_with_query(
    device: &DeviceDescriptor,
    selected: &[CudaTensorOperation],
    query: impl FnOnce() -> Result<CudaPhiloxDeviceGeometry, CudaDeviceRuntimeError>,
) -> Result<Vec<SelectedDescriptor>, CudaDeviceRuntimeError> {
    validate_selection(selected)?;
    if selected.contains(&CudaTensorOperation::StructuredGqaAttentionFusedF32) {
        return Err(CudaDeviceRuntimeError::contract(
            "fused F32 attention requires the explicit independently admitted factory",
        ));
    }
    let geometry = if selected.iter().any(|op| {
        matches!(
            op,
            CudaTensorOperation::PhiloxNormal | CudaTensorOperation::PhiloxNormalF32
        )
    }) {
        Some(query()?)
    } else {
        None
    };
    selected_descriptors(device, selected, geometry)
}

pub(super) fn providers(
    runtime: &CudaDeviceRuntime,
    selected: &[CudaTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    let descriptors = runtime_descriptors(runtime, selected)?;
    let kernels = RetainedFunctions::load(runtime)?;
    Ok(providers_from_descriptors(descriptors, &kernels))
}

pub(super) fn providers_from_descriptors(
    descriptors: Vec<SelectedDescriptor>,
    kernels: &Arc<RetainedFunctions>,
) -> Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>> {
    descriptors
        .into_iter()
        .map(|descriptor| descriptor.into_provider(kernels))
        .collect()
}

#[cfg(test)]
mod tests;
