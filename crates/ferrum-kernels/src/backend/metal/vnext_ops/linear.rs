//! Native Metal linear providers over typed physical weight layouts.
mod cost_route;
mod head_cost_route;
mod head_selected;
#[cfg(test)]
mod m8_tests;
mod selected;
#[cfg(test)]
pub(crate) use selected::tests::runtime_fixture as selected_runtime_fixture;

use std::ffi::c_void;
use std::ops::Range;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    dense_linear_contract, dense_swiglu_contract, last_token_dense_linear_contract,
    last_token_dense_linear_f32_contract, BatchedOperationInvocation, CheckpointBoundaryConstraint,
    CheckpointCompletedInputCapture, CheckpointInputDependency, CheckpointPartitionNumerics,
    DeviceBatchingForm, DynamicStorageRequirement, ElementType, EncodedDeviceOperation,
    OperationFailure, OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, PhysicalWeightPadding,
    ProviderCheckpointCapability, ProviderCheckpointContract, ProviderWorkspaceRequirement,
    ProviderWorkspaceReusePolicy, ProviderWorkspaceScope, ProviderWorkspaceSizeFormula,
    ResolvedTensorLayout, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, VNextError, WeightEncoding, DENSE_LINEAR_F16_CAPABILITY_ID,
    DENSE_LINEAR_OPERATION_ID, DENSE_SWIGLU_F16_CAPABILITY_ID, DENSE_SWIGLU_OPERATION_ID,
    LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID, LAST_TOKEN_DENSE_LINEAR_F32_CAPABILITY_ID,
    LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID, LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
};
use metal::{CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize};

use crate::backend::metal::k_quant_gemm::MetalKQuantGemmPipelines;
use crate::gguf_blocks::GgufBlockFormat;

use super::hadamard::{self, HadamardTransform, MetalHadamardPipelines};
use super::native_blocks::{
    bind_native_block, dispatch_m64_grid, pq2_complete_outputs_supported, pq2_full_tiles_supported,
    pq2_full_tiles_vector_input_supported, MetalNativeBlockPipelines,
};

use super::super::vnext_runtime::{
    MetalBufferRegion, MetalDeviceBuffer, MetalDeviceCommand, MetalDeviceRuntime,
    MetalDeviceRuntimeError,
};
use super::weights::{
    resolve_weight, validate_hadamard_transform, MetalResolvedCompositePart, MetalResolvedWeight,
    MetalResolvedWeightComponent, MetalResolvedWeightLayout,
};
use super::{
    authorize_reusable_topology, binding, checked_u32, contiguous_bindings, contiguous_region,
    contiguous_token_region, ensure_invocation, f16_contiguous, implementation_fingerprint,
    invalid_plan, provider_descriptor, provider_failure, shared_scratch_region,
    shared_token_region, token_binding_is_packed, unsigned_attribute, DENSE_SAFETENSORS_FORMAT_ID,
    GGUF_NATIVE_BLOCK_FORMAT_ID, Q4_K_FORMAT_ID, Q5_K_FORMAT_ID, Q6_K_FORMAT_ID, Q8_0_FORMAT_ID,
    THREADS_PER_GROUP, VALUE_ALIGNMENT_BYTES,
};

const SHADER_SOURCE: &str = include_str!("linear.metal");
pub(super) const FINGERPRINT_SOURCE: &str = concat!(
    include_str!("linear.rs"),
    include_str!("linear.metal"),
    include_str!("linear/small_batch.rs"),
    include_str!("linear/small_batch.metal"),
    include_str!("linear/plain_prefill.rs"),
    include_str!("linear/cost_route.rs"),
    include_str!("linear/head_cost_route.rs"),
    include_str!("linear/head_selected.rs"),
    include_str!("linear/selected.rs"),
    include_str!("weights.rs"),
    include_str!("linear/staged_prefill.rs"),
    include_str!("linear/transformed_prefill.rs"),
    include_str!("linear/half_head.rs"),
    include_str!("linear/half_head.metal"),
    include_str!("linear/half_head_tiled.metal"),
    include_str!("hadamard.rs"),
    include_str!("hadamard.metal"),
    include_str!("../q4_k_gemv_v2.metal"),
    include_str!("../q5_k_gemv.metal"),
    include_str!("../q6_k_gemv.metal"),
    include_str!("../k_quant_gemm.metal"),
    include_str!("../k_quant_gemm_m8.metal"),
    include_str!("../k_quant_gemm.rs"),
);
const DENSE_LINEAR_PROVIDER_ID: &str = "provider.metal.dense_linear.f16.native";
const DENSE_LINEAR_ESTIMATOR_ID: &str = "resource-estimator.metal.dense_linear.f16.native";
const DENSE_SWIGLU_PROVIDER_ID: &str = "provider.metal.dense_swiglu.f16.native";
const DENSE_SWIGLU_ESTIMATOR_ID: &str = "resource-estimator.metal.dense_swiglu.f16.native";
const LAST_TOKEN_PROVIDER_ID: &str = "provider.metal.last_token_dense_linear.f16.native";
const LAST_TOKEN_ESTIMATOR_ID: &str = "resource-estimator.metal.last_token_dense_linear.f16.native";
const LAST_TOKEN_F32_PROVIDER_ID: &str = "provider.metal.last_token_dense_linear.f32.native";
const LAST_TOKEN_F32_ESTIMATOR_ID: &str =
    "resource-estimator.metal.last_token_dense_linear.f32.native";
const SWIGLU_SCRATCH_PARTS: u64 = 3;
const QUANTIZED_TILED_GEMM_MIN_ROWS: u32 = 8;
// Amortize float tile loading while retaining enough independent output tiles.
const NATIVE_TILED_GEMM_MIN_ROWS: u32 = 32;
const NATIVE_TILED_GEMM_MIN_OUTPUT_FEATURES: u32 = 1024;
// Short IQ4_XS waves need a wider output grid; retain GEMV for narrow grids.
const NATIVE_SHORT_TILED_GEMM_MIN_ROWS: u32 = 8;
const NATIVE_SHORT_TILED_GEMM_MIN_OUTPUT_FEATURES: u32 = 4096;
// Conservative lower end of measured IQ4_XS M64 prefill; not a crossover estimate.
const NATIVE_M64_GEMM_MIN_ROWS: u32 = 1024;
// Small output grids do not provide enough parallelism after sharing weights.
// Keep the existing GEMV there; the opt-in microbench covers both regimes.
const SHARED_WEIGHT_GEMV_MIN_OUTPUT_FEATURES: u32 = 1024;
const METAL_BLIT_ALIGNMENT_BYTES: u64 = 4;
const LAST_TOKEN_SCRATCH_PADDING_BYTES: u64 = VALUE_ALIGNMENT_BYTES - 1;

mod half_head;
mod plain_prefill;
mod small_batch;
pub(super) mod staged_prefill;
mod transformed_prefill;
use half_head::LastTokenProjection;
pub(super) use half_head::MetalHalfHeadProvider;
use plain_prefill::PlainLinearPlan;
use transformed_prefill::TransformedLinearPlan;

const LINEAR_DENSE_KERNEL: &str = "vnext_linear_dense_f16";
const LINEAR_DENSE_NARROW_KERNEL: &str = "vnext_linear_dense_narrow_f16_256";
const NARROW_DENSE_THREADS: u64 = 256;
const LINEAR_Q8_0_KERNEL: &str = "vnext_linear_q8_0_f16";
const LINEAR_DENSE_F32_KERNEL: &str = "vnext_linear_dense_f32";
const LINEAR_Q8_0_F32_KERNEL: &str = "vnext_linear_q8_0_f32";
const SWIGLU_KERNEL: &str = "vnext_swiglu_f16";
pub(super) const ALL_LINEAR_QUANTIZATION_FORMATS: &[&str] = &[
    GgufBlockFormat::Q3K.format_id(),
    Q4_K_FORMAT_ID,
    Q5_K_FORMAT_ID,
    Q6_K_FORMAT_ID,
    Q8_0_FORMAT_ID,
    GgufBlockFormat::Iq3S.format_id(),
    GgufBlockFormat::Iq4Nl.format_id(),
    GgufBlockFormat::Iq4Xs.format_id(),
    GgufBlockFormat::Pq2_0.format_id(),
];
const F32_LINEAR_QUANTIZATION_FORMATS: &[&str] = ALL_LINEAR_QUANTIZATION_FORMATS;

fn hadamard_tiled_gemm_supported(
    format: GgufBlockFormat,
    activation_type: ElementType,
    params: LinearParams,
) -> bool {
    // The transform scratch is F32; the declared activation ABI determines
    // the final output store. Keep F32 outputs on their existing GEMV path.
    activation_type == ElementType::F16
        && ((params.rows >= NATIVE_TILED_GEMM_MIN_ROWS
            && params.out_features >= NATIVE_TILED_GEMM_MIN_OUTPUT_FEATURES)
            || (format == GgufBlockFormat::Pq2_0
                && params.rows >= 20
                && params.out_features >= 4096))
}

fn pq2_mixed_prefill_m64_supported(
    format: GgufBlockFormat,
    params: LinearParams,
    pipeline_available: bool,
) -> bool {
    // M64 reuses decoded weights across twice as many input rows. Partial
    // row tiles and narrow output matrices lose that benefit in measured
    // PQ2 mixed-input workloads, so retain M32 for those shapes.
    format == GgufBlockFormat::Pq2_0
        && params.rows >= 64
        && params.rows.is_multiple_of(64)
        && params.out_features >= 4096
        && pipeline_available
}

pub(super) struct MetalLinearPipelines {
    structured_capture: ferrum_types::SloStructuredCostCapture,
    hadamard: MetalHadamardPipelines,
    dense: ComputePipelineState,
    dense_narrow: Option<ComputePipelineState>,
    dense_f32: ComputePipelineState,
    dense_f32_f16: ComputePipelineState,
    q4_k_gemv: ComputePipelineState,
    q4_k_gemv_f32: ComputePipelineState,
    q5_k_gemv: ComputePipelineState,
    q6_k_gemv: ComputePipelineState,
    q6_k_gemv_f32: ComputePipelineState,
    k_quant_gemm: MetalKQuantGemmPipelines,
    small_batch: small_batch::SmallBatchPipelines,
    q8_0: ComputePipelineState,
    q8_0_f32: ComputePipelineState,
    swiglu: ComputePipelineState,
    native: MetalNativeBlockPipelines,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinearDispatchKind {
    CooperativeGemv,
    NarrowDenseGemv,
    Pq2CooperativeGemv,
    SharedWeightGemv,
    TiledGemm,
    TiledGemmM8,
    NativeTiledGemm,
    NativeTiledGemmM64,
}

impl MetalLinearPipelines {
    pub(super) fn with_structured_capture(
        mut self,
        capture: ferrum_types::SloStructuredCostCapture,
    ) -> Self {
        self.structured_capture = capture;
        self
    }

    pub(super) fn structured_capture(&self) -> ferrum_types::SloStructuredCostCapture {
        self.structured_capture
    }

    pub(super) fn new(device: &Device) -> Result<Self, MetalDeviceRuntimeError> {
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(|error| {
                MetalDeviceRuntimeError::contract(format!(
                    "compile Metal vNext linear library: {error}"
                ))
            })?;
        let pipeline = |name: &str| {
            let function = library.get_function(name, None).map_err(|error| {
                MetalDeviceRuntimeError::contract(format!(
                    "load Metal vNext linear `{name}`: {error}"
                ))
            })?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| {
                    MetalDeviceRuntimeError::contract(format!(
                        "build Metal vNext linear `{name}`: {error}"
                    ))
                })
        };
        // The optional specialization must not remove support for devices that
        // cannot run its fixed eight-SIMD reduction.
        let dense_narrow = if device.max_threads_per_threadgroup().width >= NARROW_DENSE_THREADS {
            pipeline(LINEAR_DENSE_NARROW_KERNEL)
                .ok()
                .filter(|pipeline| {
                    supports_narrow_dense_threadgroup(
                        pipeline.thread_execution_width(),
                        pipeline.max_total_threads_per_threadgroup(),
                        pipeline.static_threadgroup_memory_length(),
                        device.max_threadgroup_memory_length(),
                    )
                })
        } else {
            None
        };
        Ok(Self {
            structured_capture: ferrum_types::SloStructuredCostCapture::Disabled,
            hadamard: MetalHadamardPipelines::new(device)?,
            dense: pipeline(LINEAR_DENSE_KERNEL)?,
            dense_narrow,
            dense_f32: pipeline(LINEAR_DENSE_F32_KERNEL)?,
            dense_f32_f16: pipeline("vnext_linear_dense_f32_f16")?,
            q4_k_gemv: crate::backend::metal::q4_k_gemv_v2::new_f16_batched_pipeline(device)
                .map_err(MetalDeviceRuntimeError::contract)?,
            q4_k_gemv_f32: crate::backend::metal::q4_k_gemv_v2::new_f32_batched_pipeline(device)
                .map_err(MetalDeviceRuntimeError::contract)?,
            q5_k_gemv: crate::backend::metal::q5_k_gemv::new_f16_batched_pipeline(device)
                .map_err(MetalDeviceRuntimeError::contract)?,
            q6_k_gemv: crate::backend::metal::q6_k_gemv::new_f16_batched_pipeline(device)
                .map_err(MetalDeviceRuntimeError::contract)?,
            q6_k_gemv_f32: crate::backend::metal::q6_k_gemv::new_f32_batched_pipeline(device)
                .map_err(MetalDeviceRuntimeError::contract)?,
            k_quant_gemm: MetalKQuantGemmPipelines::new(device)
                .map_err(MetalDeviceRuntimeError::contract)?,
            small_batch: small_batch::SmallBatchPipelines::new(device)?,
            q8_0: pipeline(LINEAR_Q8_0_KERNEL)?,
            q8_0_f32: pipeline(LINEAR_Q8_0_F32_KERNEL)?,
            swiglu: pipeline(SWIGLU_KERNEL)?,
            native: MetalNativeBlockPipelines::new(device)?,
        })
    }

    fn linear_pipeline(
        &self,
        format: LinearPhysicalFormat,
        rows: u32,
        out_features: u32,
    ) -> (&ComputePipelineState, LinearDispatchKind) {
        if out_features >= SHARED_WEIGHT_GEMV_MIN_OUTPUT_FEATURES {
            if format == LinearPhysicalFormat::Native(GgufBlockFormat::Iq4Xs) {
                if let Some(pipeline) = self.native.iq4xs_group_dot(rows) {
                    return (
                        pipeline,
                        if rows == 1 {
                            LinearDispatchKind::CooperativeGemv
                        } else {
                            LinearDispatchKind::SharedWeightGemv
                        },
                    );
                }
            }
            if let Some(pipeline) = self.small_batch.pipeline(format, rows) {
                return (pipeline, LinearDispatchKind::SharedWeightGemv);
            }
            if let Some(pipeline) = format
                .native_block(ElementType::F16)
                .and_then(|format| self.native.shared_linear(format, rows, ElementType::F16))
            {
                return (pipeline, LinearDispatchKind::SharedWeightGemv);
            }
        }
        // An eight-row wave needs only one M8 fragment. Keep the original
        // half operands and K8 accumulation order while avoiding M32 padding.
        if rows == 8 {
            let pipeline = match format {
                LinearPhysicalFormat::Q4K => Some(&self.k_quant_gemm.q4_k_m8),
                LinearPhysicalFormat::Q5K => Some(&self.k_quant_gemm.q5_k_m8),
                LinearPhysicalFormat::Q6K => Some(&self.k_quant_gemm.q6_k_m8),
                _ => None,
            };
            if let Some(pipeline) = pipeline {
                return (pipeline, LinearDispatchKind::TiledGemmM8);
            }
        }
        let tiled = rows >= QUANTIZED_TILED_GEMM_MIN_ROWS;
        match (format, tiled) {
            (LinearPhysicalFormat::Q4K, true) => {
                (&self.k_quant_gemm.q4_k, LinearDispatchKind::TiledGemm)
            }
            (LinearPhysicalFormat::Q5K, true) => {
                (&self.k_quant_gemm.q5_k, LinearDispatchKind::TiledGemm)
            }
            (LinearPhysicalFormat::Q6K, true) => {
                (&self.k_quant_gemm.q6_k, LinearDispatchKind::TiledGemm)
            }
            (LinearPhysicalFormat::Q8_0, true) => {
                (&self.k_quant_gemm.q8_0, LinearDispatchKind::TiledGemm)
            }
            (LinearPhysicalFormat::Q4K, false) => {
                (&self.q4_k_gemv, LinearDispatchKind::CooperativeGemv)
            }
            (LinearPhysicalFormat::Q5K, false) => {
                (&self.q5_k_gemv, LinearDispatchKind::CooperativeGemv)
            }
            (LinearPhysicalFormat::Q6K, false) => {
                (&self.q6_k_gemv, LinearDispatchKind::CooperativeGemv)
            }
            (LinearPhysicalFormat::DenseF16, _) => {
                (&self.dense, LinearDispatchKind::CooperativeGemv)
            }
            (LinearPhysicalFormat::Q8_0, false) => {
                (&self.q8_0, LinearDispatchKind::CooperativeGemv)
            }
            (LinearPhysicalFormat::Native(format), _)
                if (rows >= NATIVE_TILED_GEMM_MIN_ROWS
                    && out_features >= NATIVE_TILED_GEMM_MIN_OUTPUT_FEATURES)
                    || (format == GgufBlockFormat::Iq4Xs
                        && rows >= NATIVE_SHORT_TILED_GEMM_MIN_ROWS
                        && out_features >= NATIVE_SHORT_TILED_GEMM_MIN_OUTPUT_FEATURES) =>
            {
                if format == GgufBlockFormat::Iq4Xs {
                    if rows >= NATIVE_M64_GEMM_MIN_ROWS {
                        if let Some(pipeline) = self.native.iq4xs_gemm_f16_f32_m64.as_ref() {
                            return (pipeline, LinearDispatchKind::NativeTiledGemmM64);
                        }
                    }
                    return (
                        &self.native.iq4xs_gemm_f16_f32,
                        LinearDispatchKind::NativeTiledGemm,
                    );
                }
                (
                    &self.native.gemm_f16_f32,
                    LinearDispatchKind::NativeTiledGemm,
                )
            }
            (LinearPhysicalFormat::Native(format), _) => (
                self.native.linear_f16(format),
                LinearDispatchKind::CooperativeGemv,
            ),
        }
    }

    fn plain_linear_dispatch(
        &self,
        format: LinearPhysicalFormat,
        activation_type: ElementType,
        params: LinearParams,
    ) -> (&ComputePipelineState, LinearDispatchKind) {
        if activation_type == ElementType::F16
            && format == LinearPhysicalFormat::DenseF16
            && narrow_dense_shape(params)
        {
            if let Some(pipeline) = &self.dense_narrow {
                return (pipeline, LinearDispatchKind::NarrowDenseGemv);
            }
        }
        match activation_type {
            ElementType::F16 => self.linear_pipeline(format, params.rows, params.out_features),
            ElementType::F32 => self
                .f32_linear_dispatch(format, params.rows, params.out_features)
                .expect("validated Metal F32 linear format"),
            _ => unreachable!("validated Metal linear activation ABI"),
        }
    }

    fn hadamard_native_dispatch(
        &self,
        format: GgufBlockFormat,
        activation_type: ElementType,
        params: LinearParams,
    ) -> (&ComputePipelineState, LinearDispatchKind) {
        if hadamard_tiled_gemm_supported(format, activation_type, params) {
            return self.mixed_input_tiled_pipeline(format, params);
        }
        if format == GgufBlockFormat::Pq2_0 && params.rows < NATIVE_TILED_GEMM_MIN_ROWS {
            if pq2_complete_outputs_supported(params.rows, params.in_features, params.out_features)
            {
                let complete = match activation_type {
                    ElementType::F32 => self.native.pq2_linear_f32_complete.as_ref(),
                    ElementType::F16 => self.native.pq2_linear_f32_f16_complete.as_ref(),
                    _ => None,
                };
                if let Some(pipeline) = complete {
                    return (pipeline, LinearDispatchKind::Pq2CooperativeGemv);
                }
            }
            return (
                if activation_type == ElementType::F32 {
                    &self.native.pq2_linear_f32
                } else {
                    &self.native.pq2_linear_f32_f16
                },
                LinearDispatchKind::Pq2CooperativeGemv,
            );
        }
        if activation_type == ElementType::F32 {
            (
                self.native.linear_f32(format),
                LinearDispatchKind::CooperativeGemv,
            )
        } else {
            (
                self.native.linear_f32_f16(format),
                LinearDispatchKind::CooperativeGemv,
            )
        }
    }

    fn mixed_input_tiled_pipeline(
        &self,
        format: GgufBlockFormat,
        params: LinearParams,
    ) -> (&ComputePipelineState, LinearDispatchKind) {
        let m64 = self.native.pq2_gemm_input_f32_output_f16_m64.as_ref();
        if pq2_mixed_prefill_m64_supported(format, params, m64.is_some()) {
            // Complete tiles can omit boundary predicates without changing
            // operand precision, accumulation order or the output ABI. Keep
            // the guarded M64 pipeline as the capability/compilation fallback.
            if pq2_full_tiles_supported(64, params.rows, params.in_features, params.out_features) {
                if let Some(pipeline) = &self.native.pq2_gemm_input_f32_output_f16_m64_full_tiles {
                    return (pipeline, LinearDispatchKind::NativeTiledGemmM64);
                }
            }
            return (
                m64.expect("PQ2 M64 selection requires an available pipeline"),
                LinearDispatchKind::NativeTiledGemmM64,
            );
        }
        if format == GgufBlockFormat::Pq2_0 {
            (
                &self.native.pq2_gemm_input_f32_output_f16,
                LinearDispatchKind::NativeTiledGemm,
            )
        } else {
            (
                &self.native.gemm_input_f32_output_f16,
                LinearDispatchKind::NativeTiledGemm,
            )
        }
    }

    fn hadamard_native_dispatch_for_bindings(
        &self,
        format: GgufBlockFormat,
        activation_type: ElementType,
        params: LinearParams,
        input: &MetalBufferRegion,
        input_offset_bytes: u64,
        weight: &MetalBufferRegion,
    ) -> (&ComputePipelineState, LinearDispatchKind) {
        let selected = self.hadamard_native_dispatch_for_input(
            format,
            activation_type,
            params,
            input,
            input_offset_bytes,
        );
        // Refine only the original complete-eight route; prefill priority and
        // optional-pipeline fallbacks remain owned by the existing selector.
        // The bound component starts at its retained region's actual offset.
        if weight.offset_bytes().is_multiple_of(2) {
            let (complete, aligned) = match activation_type {
                ElementType::F16 => (
                    self.native.pq2_linear_f32_f16_complete.as_ref(),
                    self.native
                        .pq2_linear_f32_f16_complete_aligned_scale
                        .as_ref(),
                ),
                ElementType::F32 => (
                    self.native.pq2_linear_f32_complete.as_ref(),
                    self.native.pq2_linear_f32_complete_aligned_scale.as_ref(),
                ),
                _ => (None, None),
            };
            if let (Some(complete), Some(aligned)) = (complete, aligned) {
                if std::ptr::eq(selected.0, complete) {
                    return (aligned, selected.1);
                }
            }
        }
        selected
    }

    fn hadamard_native_dispatch_for_input(
        &self,
        format: GgufBlockFormat,
        activation_type: ElementType,
        params: LinearParams,
        input: &MetalBufferRegion,
        input_offset_bytes: u64,
    ) -> (&ComputePipelineState, LinearDispatchKind) {
        // Shape alone cannot authorize vector loads: the transform workspace
        // may be a slice with its own base and an inner suballocation offset.
        // Retain the existing full-M64 cohort and all scalar fallbacks.
        if format == GgufBlockFormat::Pq2_0 && activation_type == ElementType::F16 {
            if let Some(pipeline) =
                self.pq2_vector_input_prefill_pipeline(params, input, input_offset_bytes)
            {
                return (pipeline, LinearDispatchKind::NativeTiledGemmM64);
            }
        }
        self.hadamard_native_dispatch(format, activation_type, params)
    }

    fn pq2_vector_input_prefill_pipeline(
        &self,
        params: LinearParams,
        input: &MetalBufferRegion,
        input_offset_bytes: u64,
    ) -> Option<&ComputePipelineState> {
        if !pq2_mixed_prefill_m64_supported(
            GgufBlockFormat::Pq2_0,
            params,
            self.native.pq2_gemm_input_f32_output_f16_m64.is_some(),
        ) || self
            .native
            .pq2_gemm_input_f32_output_f16_m64_full_tiles
            .is_none()
            || !pq2_full_tiles_vector_input_supported(
                params.rows,
                params.in_features,
                params.out_features,
                input.offset_bytes(),
                input_offset_bytes,
                input.length_bytes(),
            )
        {
            return None;
        }
        self.native
            .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input
            .as_ref()
    }

    fn f32_linear_pipeline(&self, format: LinearPhysicalFormat) -> Option<&ComputePipelineState> {
        match format {
            LinearPhysicalFormat::DenseF16 => Some(&self.dense_f32),
            LinearPhysicalFormat::Q4K => Some(&self.q4_k_gemv_f32),
            LinearPhysicalFormat::Q6K => Some(&self.q6_k_gemv_f32),
            LinearPhysicalFormat::Q8_0 => Some(&self.q8_0_f32),
            LinearPhysicalFormat::Q5K => Some(self.native.linear_f32(GgufBlockFormat::Q5K)),
            LinearPhysicalFormat::Native(format) => Some(self.native.linear_f32(format)),
        }
    }

    fn f32_linear_dispatch(
        &self,
        format: LinearPhysicalFormat,
        rows: u32,
        out_features: u32,
    ) -> Option<(&ComputePipelineState, LinearDispatchKind)> {
        if out_features >= SHARED_WEIGHT_GEMV_MIN_OUTPUT_FEATURES {
            if let Some(pipeline) = self.small_batch.f32_pipeline(format, rows) {
                return Some((pipeline, LinearDispatchKind::SharedWeightGemv));
            }
            if let Some(pipeline) = format
                .native_block(ElementType::F32)
                .and_then(|format| self.native.shared_linear(format, rows, ElementType::F32))
            {
                return Some((pipeline, LinearDispatchKind::SharedWeightGemv));
            }
        }
        self.f32_linear_pipeline(format)
            .map(|pipeline| (pipeline, LinearDispatchKind::CooperativeGemv))
    }
}

fn supports_narrow_dense_threadgroup(
    simd_width: u64,
    max_threads: u64,
    static_bytes: u64,
    device_bytes: u64,
) -> bool {
    simd_width == 32 && max_threads >= NARROW_DENSE_THREADS && static_bytes <= device_bytes
}

fn narrow_dense_shape(params: LinearParams) -> bool {
    // Limit the first rollout to measured long reductions and narrow output
    // grids. Prefill and short K retain the existing cooperative kernel.
    params.rows == 1 && params.in_features >= 1024 && (1..=128).contains(&params.out_features)
}

pub(super) struct MetalDenseLinearProvider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<MetalLinearPipelines>,
}

impl MetalDenseLinearProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalLinearPipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = dense_linear_contract().map_err(super::contract_error)?;
        let descriptor = linear_provider_descriptor(
            runtime,
            &contract,
            DENSE_LINEAR_PROVIDER_ID,
            DENSE_LINEAR_F16_CAPABILITY_ID,
            DENSE_LINEAR_ESTIMATOR_ID,
            2,
            ALL_LINEAR_QUANTIZATION_FORMATS,
        )?;
        Ok(Self {
            descriptor,
            pipelines,
        })
    }
}

impl OperationResourceEstimator for MetalDenseLinearProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        hadamard::estimate_token_workspace(&self.descriptor, &request, DENSE_LINEAR_OPERATION_ID)
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalDenseLinearProvider {
    fn eager_cost_route(
        &self,
        request: ferrum_interfaces::vnext::OperationCostRouteRequest<'_>,
    ) -> Result<Option<ferrum_interfaces::vnext::OperationCostRoute>, VNextError> {
        cost_route::dense_route(&self.pipelines, request)
    }

    fn reusable_execution_topology(
        &self,
        _request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        authorize_reusable_topology(self.descriptor.execution_semantics(), || {
            Ok(ReusableExecutionTopology::Static)
        })
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_dense_linear(Arc::clone(&self.pipelines), invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| provider_failure(identity, "metal.dense_linear.encode", message))
    }
}

pub(super) struct MetalDenseSwiGluProvider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<MetalLinearPipelines>,
}

impl MetalDenseSwiGluProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalLinearPipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = dense_swiglu_contract().map_err(super::contract_error)?;
        let descriptor = linear_provider_descriptor(
            runtime,
            &contract,
            DENSE_SWIGLU_PROVIDER_ID,
            DENSE_SWIGLU_F16_CAPABILITY_ID,
            DENSE_SWIGLU_ESTIMATOR_ID,
            6,
            ALL_LINEAR_QUANTIZATION_FORMATS,
        )?
        // Gate/up and activation scratch are fully produced by this invocation.
        // No state or previous scratch contents participate in its result.
        .with_checkpoint_capability(ProviderCheckpointCapability::CompletedBoundary(
            ProviderCheckpointContract::new(
                CheckpointInputDependency::ExactTokenPrefix,
                CheckpointBoundaryConstraint::any_positive(),
                CheckpointPartitionNumerics::CapturedExecutionContinuation,
            )
            .with_completed_input_capture(CheckpointCompletedInputCapture::Supported),
        ));
        Ok(Self {
            descriptor,
            pipelines,
        })
    }
}

impl OperationResourceEstimator for MetalDenseSwiGluProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id.as_str() != DENSE_SWIGLU_OPERATION_ID
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan(format!(
                "Metal estimator `{}` received another operation",
                self.descriptor.resource_estimator_id()
            )));
        }
        let intermediate_size =
            unsigned_attribute(request.attributes(), "intermediate_size").map_err(invalid_plan)?;
        let hidden_size =
            unsigned_attribute(request.attributes(), "hidden_size").map_err(invalid_plan)?;
        let staging_bytes =
            staged_prefill::workspace_bytes(request.values(), hidden_size, intermediate_size)
                .map_err(invalid_plan)?;
        let transform_bytes =
            hadamard::workspace_bytes_per_token(request.values()).map_err(invalid_plan)?;
        let bytes_per_token = intermediate_size
            .checked_mul(SWIGLU_SCRATCH_PARTS)
            .and_then(|elements| elements.checked_mul(ElementType::F16.size_bytes()))
            .and_then(|bytes| bytes.checked_add(transform_bytes))
            .ok_or_else(|| invalid_plan("Metal dense SwiGLU scratch size overflows"))?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::affine(
                staging_bytes
                    .checked_add(if transform_bytes > 0 {
                        VALUE_ALIGNMENT_BYTES - 1
                    } else {
                        0
                    })
                    .ok_or_else(|| invalid_plan("Metal SwiGLU transform alignment overflows"))?,
                0,
                bytes_per_token,
            )?,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            VALUE_ALIGNMENT_BYTES,
            Some(scratch),
            None,
        ))
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalDenseSwiGluProvider {
    fn eager_cost_route(
        &self,
        request: ferrum_interfaces::vnext::OperationCostRouteRequest<'_>,
    ) -> Result<Option<ferrum_interfaces::vnext::OperationCostRoute>, VNextError> {
        cost_route::swiglu_route(&self.pipelines, request)
    }

    fn reusable_execution_topology(
        &self,
        _request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        authorize_reusable_topology(self.descriptor.execution_semantics(), || {
            Ok(ReusableExecutionTopology::Static)
        })
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_dense_swiglu(Arc::clone(&self.pipelines), invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| provider_failure(identity, "metal.dense_swiglu.encode", message))
    }
}

pub(super) struct MetalLastTokenDenseLinearProvider {
    descriptor: OperationProviderDescriptor,
    operation_id: &'static str,
    activation_type: ElementType,
    failure_stage: &'static str,
    pipelines: Arc<MetalLinearPipelines>,
}

impl MetalLastTokenDenseLinearProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalLinearPipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        Self::new_with_activation_type(runtime, pipelines, ElementType::F16)
    }

    pub(super) fn new_f32(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalLinearPipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let mut provider = Self::new_with_activation_type(runtime, pipelines, ElementType::F32)?;
        // The selected final row and gather/output scratch are invocation-local.
        provider.descriptor = provider.descriptor.with_checkpoint_capability(
            ProviderCheckpointCapability::CompletedBoundary(
                ProviderCheckpointContract::new(
                    CheckpointInputDependency::ExactTokenPrefix,
                    CheckpointBoundaryConstraint::any_positive(),
                    CheckpointPartitionNumerics::CapturedExecutionContinuation,
                )
                .with_completed_input_capture(CheckpointCompletedInputCapture::Supported),
            ),
        );
        Ok(provider)
    }

    fn new_with_activation_type(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalLinearPipelines>,
        activation_type: ElementType,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let (
            contract,
            operation_id,
            provider_id,
            capability_id,
            estimator_id,
            quant_formats,
            failure_stage,
        ) = match activation_type {
            ElementType::F16 => (
                last_token_dense_linear_contract().map_err(super::contract_error)?,
                LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
                LAST_TOKEN_PROVIDER_ID,
                LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID,
                LAST_TOKEN_ESTIMATOR_ID,
                ALL_LINEAR_QUANTIZATION_FORMATS,
                "metal.last_token_dense_linear.encode",
            ),
            ElementType::F32 => (
                last_token_dense_linear_f32_contract().map_err(super::contract_error)?,
                LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID,
                LAST_TOKEN_F32_PROVIDER_ID,
                LAST_TOKEN_DENSE_LINEAR_F32_CAPABILITY_ID,
                LAST_TOKEN_F32_ESTIMATOR_ID,
                F32_LINEAR_QUANTIZATION_FORMATS,
                "metal.last_token_dense_linear.f32.encode",
            ),
            _ => {
                return Err(MetalDeviceRuntimeError::contract(
                    "Metal last-token linear activation ABI supports only F16 or F32",
                ));
            }
        };
        let descriptor = linear_provider_descriptor(
            runtime,
            &contract,
            provider_id,
            capability_id,
            estimator_id,
            2,
            quant_formats,
        )?;
        Ok(Self {
            descriptor,
            operation_id,
            activation_type,
            failure_stage,
            pipelines,
        })
    }
}

impl OperationResourceEstimator for MetalLastTokenDenseLinearProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id.as_str() != self.operation_id
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan(format!(
                "Metal estimator `{}` received another operation",
                self.descriptor.resource_estimator_id()
            )));
        }
        let out_features =
            unsigned_attribute(request.attributes(), "out_features").map_err(invalid_plan)?;
        let hidden_size =
            unsigned_attribute(request.attributes(), "hidden_size").map_err(invalid_plan)?;
        let transform_bytes =
            hadamard::workspace_bytes_per_token(request.values()).map_err(invalid_plan)?;
        let bytes_per_sequence =
            last_token_scratch_bytes_per_sequence(hidden_size, out_features, self.activation_type)
                .map_err(invalid_plan)?
                .checked_add(transform_bytes)
                .ok_or_else(|| invalid_plan("Metal last-token transform workspace overflows"))?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::affine(
                LAST_TOKEN_SCRATCH_PADDING_BYTES
                    + if transform_bytes > 0 {
                        VALUE_ALIGNMENT_BYTES - 1
                    } else {
                        0
                    },
                bytes_per_sequence,
                0,
            )?,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            VALUE_ALIGNMENT_BYTES,
            Some(scratch),
            None,
        ))
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalLastTokenDenseLinearProvider {
    fn eager_cost_route(
        &self,
        request: ferrum_interfaces::vnext::OperationCostRouteRequest<'_>,
    ) -> Result<Option<ferrum_interfaces::vnext::OperationCostRoute>, VNextError> {
        head_cost_route::route(
            request,
            self.operation_id,
            self.activation_type,
            &LastTokenProjection::Strict(Arc::clone(&self.pipelines)),
        )
    }

    fn reusable_execution_topology(
        &self,
        _request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        authorize_reusable_topology(self.descriptor.execution_semantics(), || {
            Ok(ReusableExecutionTopology::Static)
        })
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_last_token_dense_linear(
            LastTokenProjection::Strict(Arc::clone(&self.pipelines)),
            self.operation_id,
            self.activation_type,
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| provider_failure(identity, self.failure_stage, message))
    }
}

fn linear_provider_descriptor(
    runtime: &MetalDeviceRuntime,
    contract: &dyn ferrum_interfaces::vnext::OperationContract,
    provider_id: &str,
    capability_id: &str,
    estimator_id: &str,
    input_count: u32,
    accepted_quantization_formats: &[&str],
) -> Result<OperationProviderDescriptor, MetalDeviceRuntimeError> {
    provider_descriptor(
        runtime,
        contract,
        provider_id,
        capability_id,
        estimator_id,
        contiguous_bindings(input_count),
        &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID],
        accepted_quantization_formats,
        implementation_fingerprint(&[
            FINGERPRINT_SOURCE.as_bytes(),
            super::native_blocks::FINGERPRINT_SOURCE.as_bytes(),
            provider_id.as_bytes(),
        ]),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinearPhysicalFormat {
    DenseF16,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
    Native(GgufBlockFormat),
}

impl LinearPhysicalFormat {
    fn native_block(self, activation_type: ElementType) -> Option<GgufBlockFormat> {
        match self {
            Self::Native(format) => Some(format),
            Self::Q5K if activation_type == ElementType::F32 => Some(GgufBlockFormat::Q5K),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PreparedLinearPart {
    region: usize,
    format: LinearPhysicalFormat,
    output_offset: u32,
    out_features: u32,
    transform: Option<HadamardTransform>,
}

impl PreparedLinearPart {
    fn relocated(self, base: usize) -> Result<Self, String> {
        Ok(Self {
            region: self
                .region
                .checked_add(base)
                .ok_or_else(|| "Metal linear region index overflows".to_owned())?,
            transform: self
                .transform
                .map(|transform| transform.relocate(base))
                .transpose()?,
            ..self
        })
    }
}

struct PreparedLinearWeight {
    regions: Vec<MetalBufferRegion>,
    parts: Vec<PreparedLinearPart>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct LinearParams {
    rows: u32,
    in_features: u32,
    out_features: u32,
    output_stride: u32,
    output_column_offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct SwiGluParams {
    rows: u32,
    intermediate_size: u32,
    gate_up_stride: u32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SwiGluLaunch {
    gate_up_offset_bytes: u64,
    activation_offset_bytes: u64,
    params: SwiGluParams,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LinearLaunch {
    input_region: usize,
    weight_region: usize,
    output_region: usize,
    input_offset_bytes: u64,
    output_offset_bytes: u64,
    activation_type: ElementType,
    format: LinearPhysicalFormat,
    params: LinearParams,
    transform: Option<HadamardTransform>,
    transform_workspace: Option<(usize, u64)>,
    // Prepared with the retained workspace; encoding and physical accounting
    // consume this same decision without reselecting the partition.
    transformed_plan: TransformedLinearPlan,
    plain_plan: PlainLinearPlan,
}

impl LinearLaunch {
    pub(super) fn activation_bytes(self) -> Result<(u64, u64), String> {
        let bytes = |width, name| {
            u64::from(self.params.rows)
                .checked_mul(u64::from(width))
                .and_then(|elements| elements.checked_mul(self.activation_type.size_bytes()))
                .ok_or_else(|| format!("Metal linear {name} byte size overflows"))
        };
        Ok((
            bytes(self.params.in_features, "input")?,
            bytes(self.params.output_stride, "output")?,
        ))
    }

    pub(super) fn dispatch_count(self) -> u64 {
        if self.transform.is_some() {
            self.transformed_plan.projection_dispatch_count() + 1
        } else {
            self.plain_plan.dispatch_count()
        }
    }

    pub(super) fn bind_hadamard_workspace(
        &mut self,
        pipelines: &MetalLinearPipelines,
        regions: &[MetalBufferRegion],
        workspace_region: usize,
        workspace_offset: u64,
    ) -> Result<(), String> {
        self.transformed_plan = TransformedLinearPlan::Single;
        let Some(transform) = self.transform else {
            return Ok(());
        };
        pipelines.hadamard.validate_dispatch(
            transform,
            self.params.in_features,
            self.activation_type,
            ElementType::F32,
        )?;
        let workspace = regions
            .get(workspace_region)
            .ok_or_else(|| "Metal linear transform workspace is absent".to_owned())?;
        let bytes = u64::from(self.params.rows)
            .checked_mul(u64::from(self.params.in_features))
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| "Metal linear transform workspace overflows".to_owned())?;
        validate_region_span(
            workspace,
            workspace_offset,
            bytes,
            "Metal linear transform workspace",
        )?;
        if workspace_offset % VALUE_ALIGNMENT_BYTES != 0 {
            return Err("Metal linear transform workspace is unaligned".to_owned());
        }
        for (index, offset, elements) in [
            (
                self.input_region,
                self.input_offset_bytes,
                u64::from(self.params.rows) * u64::from(self.params.in_features),
            ),
            (
                self.output_region,
                self.output_offset_bytes,
                u64::from(self.params.rows) * u64::from(self.params.output_stride),
            ),
        ] {
            let region = regions
                .get(index)
                .ok_or_else(|| "Metal linear transform activation is absent".to_owned())?;
            let start = region.offset_bytes() + offset;
            let end = start + elements * self.activation_type.size_bytes();
            let workspace_start = workspace.offset_bytes() + workspace_offset;
            if std::ptr::eq(region.buffer(), workspace.buffer())
                && start < workspace_start + bytes
                && workspace_start < end
            {
                return Err("Metal linear transform workspace overlaps an activation".to_owned());
            }
        }
        self.transform_workspace = Some((workspace_region, workspace_offset));
        self.transformed_plan = TransformedLinearPlan::for_launch(pipelines, regions, *self);
        Ok(())
    }
}

fn encode_dense_linear(
    pipelines: Arc<MetalLinearPipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, DENSE_LINEAR_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let in_features = unsigned_attribute(first.attributes(), "in_features")?;
    let out_features = unsigned_attribute(first.attributes(), "out_features")?;
    validate_dense_linear_participant(first, in_features, out_features)?;
    let resolved = resolve_weight(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, 1)?,
    )?;
    for participant in &invocation.participants()[1..] {
        if unsigned_attribute(participant.attributes(), "in_features")? != in_features
            || unsigned_attribute(participant.attributes(), "out_features")? != out_features
        {
            return Err("Metal dense linear participant attributes disagree".to_owned());
        }
        validate_dense_linear_participant(participant, in_features, out_features)?;
        let candidate = resolve_weight(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
        )?;
        if !same_resolved_weight(&resolved, &candidate) {
            return Err("Metal dense linear participants do not share one weight".to_owned());
        }
    }
    let prepared = prepare_matrix_weight(resolved, out_features, in_features)?;
    let [part] = prepared.parts.as_slice() else {
        return Err("Metal dense linear requires one physical matrix".to_owned());
    };
    let part = *part;
    let mut regions = prepared.regions;
    let input_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let output_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Output, 0)?;
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("Metal dense linear participant ranges are incomplete".to_owned());
    }
    let mut launches = Vec::new();
    if input_packed && output_packed {
        let rows = invocation.work_shape().immediate_tokens();
        let input_region = regions.len();
        regions.push(shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            rows,
        )?);
        let output_region = regions.len();
        regions.push(shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            rows,
        )?);
        launches.push(linear_launch(
            part,
            input_region,
            output_region,
            rows,
            in_features,
            out_features,
            0,
            0,
        )?);
    } else {
        for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
            let rows = token_range.immediate_tokens();
            let input_start = if input_packed {
                token_range.immediate_token_range().start
            } else {
                token_range.source_token_range().start
            };
            let output_start = if output_packed {
                token_range.immediate_token_range().start
            } else {
                token_range.source_token_range().start
            };
            let input_region = regions.len();
            regions.push(contiguous_token_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
                ElementType::F16,
                input_start,
                rows,
            )?);
            let output_region = regions.len();
            regions.push(contiguous_token_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
                ElementType::F16,
                output_start,
                rows,
            )?);
            launches.push(linear_launch(
                part,
                input_region,
                output_region,
                rows,
                in_features,
                out_features,
                0,
                0,
            )?);
        }
    }
    let transform_bytes = hadamard::workspace_bytes_per_token(first.bindings())?;
    if transform_bytes != 0 {
        let bytes = transform_bytes
            .checked_mul(invocation.work_shape().immediate_tokens())
            .ok_or_else(|| "Metal dense-linear transform workspace overflows".to_owned())?;
        let index = regions.len();
        regions.push(shared_scratch_region(&invocation, bytes)?);
        for launch in &mut launches {
            launch.bind_hadamard_workspace(&pipelines, &regions, index, 0)?;
        }
    }
    validate_launch_regions(&regions, &launches)?;
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "Metal dense linear participant count",
    )?;
    let token_count = invocation.work_shape().immediate_tokens();
    let route =
        cost_route::dense_command_selected(&pipelines, participant_count, token_count, &launches)
            .map_err(|error| error.to_string())?;
    let dispatch_count = route.compute_dispatch_count();
    MetalDeviceCommand::operation(
        route.native_operation(),
        regions,
        move |encoder, regions| {
            encoder.record_compute_dispatches(dispatch_count);
            for launch in &launches {
                dispatch_linear(&pipelines, encoder.compute_encoder(), regions, *launch);
            }
            Ok(())
        },
    )
    .map_err(|error| error.to_string())?
    .with_work_shape(route.batching(), participant_count, token_count)
    .map(|command| command.with_statistical_evidence(route.statistical_evidence().cloned()))
    .map_err(|error| error.to_string())
}

fn encode_last_token_dense_linear(
    pipelines: LastTokenProjection,
    operation_id: &'static str,
    activation_type: ElementType,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, operation_id)?;
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let out_features = unsigned_attribute(first.attributes(), "out_features")?;
    validate_last_token_participant(first, hidden_size, out_features, activation_type)?;
    let resolved = resolve_weight(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, 1)?,
    )?;
    for participant in &invocation.participants()[1..] {
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || unsigned_attribute(participant.attributes(), "out_features")? != out_features
        {
            return Err("Metal last-token linear participant attributes disagree".to_owned());
        }
        validate_last_token_participant(participant, hidden_size, out_features, activation_type)?;
        let candidate = resolve_weight(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
        )?;
        if !same_resolved_weight(&resolved, &candidate) {
            return Err("Metal last-token linear participants do not share one weight".to_owned());
        }
    }
    let prepared = prepare_matrix_weight(resolved, out_features, hidden_size)?;
    let [part] = prepared.parts.as_slice() else {
        return Err("Metal last-token linear requires one physical matrix".to_owned());
    };
    let part = *part;
    pipelines.validate_part(part)?;
    let mut regions = prepared.regions;
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("Metal last-token linear participant ranges are incomplete".to_owned());
    }
    let input_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let participant_count = invocation.participants().len();
    let participant_count_u32 = checked_u32(
        participant_count as u64,
        "Metal last-token linear participant count",
    )?;
    let token_count = invocation.work_shape().immediate_tokens();
    let mut scratch_layout = LastTokenPackedScratchLayout::new(
        participant_count as u64,
        hidden_size,
        out_features,
        activation_type,
    )?;
    let transform_bytes_per_sequence = hadamard::workspace_bytes_per_token(first.bindings())?;
    let transform_offset = if transform_bytes_per_sequence != 0 {
        align_up_bytes(scratch_layout.required_bytes, VALUE_ALIGNMENT_BYTES)?
    } else {
        scratch_layout.required_bytes
    };
    scratch_layout.required_bytes = transform_bytes_per_sequence
        .checked_mul(participant_count as u64)
        .and_then(|bytes| transform_offset.checked_add(bytes))
        .ok_or_else(|| "Metal packed last-token transform workspace overflows".to_owned())?;
    let shared_packed_input = packed_last_token_rows(
        input_packed,
        participant_count,
        token_ranges
            .iter()
            .map(|token_range| token_range.immediate_token_range()),
    );

    if head_cost_route::packed_eligible(participant_count, scratch_layout) {
        let packed_inputs = if shared_packed_input {
            Vec::new()
        } else {
            invocation
                .participants()
                .iter()
                .zip(token_ranges)
                .map(|(participant, token_range)| {
                    let selected = if input_packed {
                        token_range.immediate_token_range()
                    } else {
                        token_range.source_token_range()
                    };
                    if selected.is_empty() {
                        return Err(
                            "Metal last-token linear cannot select an empty span".to_owned()
                        );
                    }
                    contiguous_token_region(
                        participant,
                        binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
                        activation_type,
                        selected.end - 1,
                        1,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let packed_outputs = invocation
            .participants()
            .iter()
            .map(|participant| {
                contiguous_region(
                    participant,
                    binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
                    activation_type,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let blit_compatible = packed_inputs.iter().all(|input| {
            input.length_bytes() == scratch_layout.input_row_bytes
                && input.offset_bytes() % METAL_BLIT_ALIGNMENT_BYTES == 0
        }) && packed_outputs.iter().all(|output| {
            output.length_bytes() == scratch_layout.output_row_bytes
                && output.offset_bytes() % METAL_BLIT_ALIGNMENT_BYTES == 0
        });
        if blit_compatible {
            let shared_input_region = if shared_packed_input {
                let region = regions.len();
                regions.push(shared_token_region(
                    &invocation,
                    ResolvedValueRole::Input,
                    0,
                    activation_type,
                    participant_count as u64,
                )?);
                Some(region)
            } else {
                None
            };
            let gathered_input_region_base = (!packed_inputs.is_empty()).then_some(regions.len());
            regions.extend(packed_inputs);
            let output_region_base = regions.len();
            regions.extend(packed_outputs);
            let scratch_region = regions.len();
            let scratch = shared_scratch_region(&invocation, scratch_layout.required_bytes)?;
            if scratch.offset_bytes() % METAL_BLIT_ALIGNMENT_BYTES != 0 {
                return Err("Metal packed last-token scratch is not blit aligned".to_owned());
            }
            regions.push(scratch);
            let mut launch = linear_launch_typed(
                part,
                shared_input_region.unwrap_or(scratch_region),
                scratch_region,
                participant_count as u64,
                hidden_size,
                out_features,
                0,
                scratch_layout.output_offset_bytes,
                activation_type,
            )?;
            pipelines.bind_hadamard_workspace(
                &mut launch,
                &regions,
                scratch_region,
                transform_offset,
            )?;
            validate_launch_regions_with_raw_workspace(&regions, &[launch], &[scratch_region])?;
            pipelines.validate_launch(&regions, launch, &[scratch_region])?;
            validate_region_span(
                &regions[scratch_region],
                0,
                scratch_layout.required_bytes,
                "Metal packed last-token scratch",
            )?;
            let statistical = head_selected::evidence(
                &pipelines,
                &[launch],
                token_count,
                Some((scratch_layout, participant_count_u32, shared_packed_input)),
            );
            let operation_label = pipelines.operation_label(activation_type);
            return MetalDeviceCommand::operation(
                operation_label,
                regions,
                move |encoder, regions| {
                    if let Some(input_region_base) = gathered_input_region_base {
                        encoder.with_blit_commands(participant_count as u64, |blit| {
                            for participant_index in 0..participant_count {
                                blit.copy_from_buffer(
                                    regions[input_region_base + participant_index].buffer(),
                                    regions[input_region_base + participant_index].offset_bytes(),
                                    regions[scratch_region].buffer(),
                                    regions[scratch_region].offset_bytes()
                                        + participant_index as u64 * scratch_layout.input_row_bytes,
                                    scratch_layout.input_row_bytes,
                                );
                            }
                        });
                    }
                    encoder.record_compute_dispatches(pipelines.dispatch_count(launch));
                    pipelines.dispatch(encoder.compute_encoder(), regions, launch);
                    encoder.with_blit_commands(participant_count as u64, |blit| {
                        for participant_index in 0..participant_count {
                            blit.copy_from_buffer(
                                regions[scratch_region].buffer(),
                                regions[scratch_region].offset_bytes()
                                    + scratch_layout.output_offset_bytes
                                    + participant_index as u64 * scratch_layout.output_row_bytes,
                                regions[output_region_base + participant_index].buffer(),
                                regions[output_region_base + participant_index].offset_bytes(),
                                scratch_layout.output_row_bytes,
                            );
                        }
                    });
                    Ok(())
                },
            )
            .map_err(|error| error.to_string())?
            .with_statistical_evidence(statistical)
            .with_work_shape(
                DeviceBatchingForm::Packed,
                participant_count_u32,
                token_count,
            )
            .map_err(|error| error.to_string());
        }
    }

    let mut launches = Vec::with_capacity(invocation.participants().len());
    for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
        let selected = if input_packed {
            token_range.immediate_token_range()
        } else {
            token_range.source_token_range()
        };
        if selected.is_empty() {
            return Err("Metal last-token linear cannot select an empty span".to_owned());
        }
        let input_region = regions.len();
        regions.push(contiguous_token_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            activation_type,
            selected.end - 1,
            1,
        )?);
        let output_region = regions.len();
        regions.push(contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            activation_type,
        )?);
        launches.push(linear_launch_typed(
            part,
            input_region,
            output_region,
            1,
            hidden_size,
            out_features,
            0,
            0,
            activation_type,
        )?);
    }
    if transform_bytes_per_sequence != 0 {
        let bytes = transform_bytes_per_sequence
            .checked_mul(participant_count as u64)
            .ok_or_else(|| "Metal last-token transform workspace overflows".to_owned())?;
        let index = regions.len();
        regions.push(shared_scratch_region(&invocation, bytes)?);
        for launch in &mut launches {
            pipelines.bind_hadamard_workspace(launch, &regions, index, 0)?;
        }
    }
    validate_launch_regions(&regions, &launches)?;
    for launch in &launches {
        pipelines.validate_launch(&regions, *launch, &[])?;
    }
    let dispatch_count = launches
        .iter()
        .map(|launch| pipelines.dispatch_count(*launch))
        .sum();
    let statistical = head_selected::evidence(&pipelines, &launches, token_count, None);
    let operation_label = pipelines.operation_label(activation_type);
    MetalDeviceCommand::operation(operation_label, regions, move |encoder, regions| {
        encoder.record_compute_dispatches(dispatch_count);
        for launch in &launches {
            pipelines.dispatch(encoder.compute_encoder(), regions, *launch);
        }
        Ok(())
    })
    .map_err(|error| error.to_string())?
    .with_statistical_evidence(statistical)
    .with_work_shape(
        if participant_count_u32 == 1 {
            DeviceBatchingForm::Scalar
        } else {
            DeviceBatchingForm::ParticipantLoop
        },
        participant_count_u32,
        token_count,
    )
    .map_err(|error| error.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LastTokenPackedScratchLayout {
    input_row_bytes: u64,
    output_row_bytes: u64,
    output_offset_bytes: u64,
    required_bytes: u64,
}

impl LastTokenPackedScratchLayout {
    fn new(
        participant_count: u64,
        hidden_size: u64,
        out_features: u64,
        activation_type: ElementType,
    ) -> Result<Self, String> {
        if participant_count == 0 {
            return Err("Metal packed last-token scratch has zero participants".to_owned());
        }
        let input_row_bytes = hidden_size
            .checked_mul(activation_type.size_bytes())
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| "Metal last-token input row size is zero or overflows".to_owned())?;
        let output_row_bytes = out_features
            .checked_mul(activation_type.size_bytes())
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| "Metal last-token output row size is zero or overflows".to_owned())?;
        let input_bytes = input_row_bytes
            .checked_mul(participant_count)
            .ok_or_else(|| "Metal packed last-token input size overflows".to_owned())?;
        let output_offset_bytes = align_up_bytes(input_bytes, VALUE_ALIGNMENT_BYTES)?;
        let output_bytes = output_row_bytes
            .checked_mul(participant_count)
            .ok_or_else(|| "Metal packed last-token output size overflows".to_owned())?;
        let required_bytes = output_offset_bytes
            .checked_add(output_bytes)
            .ok_or_else(|| "Metal packed last-token scratch size overflows".to_owned())?;
        Ok(Self {
            input_row_bytes,
            output_row_bytes,
            output_offset_bytes,
            required_bytes,
        })
    }
}

fn last_token_scratch_bytes_per_sequence(
    hidden_size: u64,
    out_features: u64,
    activation_type: ElementType,
) -> Result<u64, String> {
    hidden_size
        .checked_add(out_features)
        .and_then(|elements| elements.checked_mul(activation_type.size_bytes()))
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| "Metal last-token scratch bytes per sequence overflow".to_owned())
}

fn align_up_bytes(value: u64, alignment: u64) -> Result<u64, String> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err("Metal scratch alignment is invalid".to_owned());
    }
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned & !(alignment - 1))
        .ok_or_else(|| "Metal scratch alignment overflows".to_owned())
}

fn packed_last_token_rows(
    input_packed: bool,
    participant_count: usize,
    ranges: impl IntoIterator<Item = Range<u64>>,
) -> bool {
    if !input_packed || participant_count < 2 {
        return false;
    }
    let mut next_row = 0_u64;
    let mut observed = 0_usize;
    for range in ranges {
        let Some(expected_end) = next_row.checked_add(1) else {
            return false;
        };
        if range.start != next_row || range.end != expected_end {
            return false;
        }
        next_row = expected_end;
        observed += 1;
    }
    observed == participant_count && next_row == participant_count as u64
}

fn encode_dense_swiglu(
    pipelines: Arc<MetalLinearPipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, DENSE_SWIGLU_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let intermediate_size = unsigned_attribute(first.attributes(), "intermediate_size")?;
    validate_swiglu_participant(first, hidden_size, intermediate_size)?;
    let gate_up = resolve_weight(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, 1)?,
    )?;
    let down = resolve_weight(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, 2)?,
    )?;
    for participant in &invocation.participants()[1..] {
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || unsigned_attribute(participant.attributes(), "intermediate_size")?
                != intermediate_size
        {
            return Err("Metal dense SwiGLU participant attributes disagree".to_owned());
        }
        validate_swiglu_participant(participant, hidden_size, intermediate_size)?;
        let candidate_gate_up = resolve_weight(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
        )?;
        let candidate_down = resolve_weight(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 2)?,
        )?;
        if !same_resolved_weight(&gate_up, &candidate_gate_up)
            || !same_resolved_weight(&down, &candidate_down)
        {
            return Err("Metal dense SwiGLU participants do not share weights".to_owned());
        }
    }
    let gate_up = prepare_gate_up_weight(gate_up, intermediate_size, hidden_size)?;
    let down = prepare_matrix_weight(down, hidden_size, intermediate_size)?;
    let [down_part] = down.parts.as_slice() else {
        return Err("Metal dense SwiGLU down projection requires one matrix".to_owned());
    };
    let down_part = *down_part;
    let tokens = invocation.work_shape().immediate_tokens();
    let activation_elements = tokens
        .checked_mul(intermediate_size)
        .ok_or_else(|| "Metal dense SwiGLU activation size overflows".to_owned())?;
    let gate_up_bytes = activation_elements
        .checked_mul(2)
        .and_then(|elements| elements.checked_mul(ElementType::F16.size_bytes()))
        .ok_or_else(|| "Metal dense SwiGLU gate/up scratch size overflows".to_owned())?;
    let activation_bytes = activation_elements
        .checked_mul(ElementType::F16.size_bytes())
        .ok_or_else(|| "Metal dense SwiGLU activation scratch size overflows".to_owned())?;
    let activation_scratch_bytes = gate_up_bytes
        .checked_add(activation_bytes)
        .ok_or_else(|| "Metal dense SwiGLU total scratch size overflows".to_owned())?;
    let staging_bytes =
        staged_prefill::workspace_bytes(first.bindings(), hidden_size, intermediate_size)?;
    let transform_bytes = hadamard::workspace_bytes_per_token(first.bindings())?;
    let existing_scratch_bytes = activation_scratch_bytes
        .checked_add(staging_bytes)
        .ok_or_else(|| "Metal dense SwiGLU staged scratch size overflows".to_owned())?;
    let transform_offset = if transform_bytes != 0 {
        align_up_bytes(existing_scratch_bytes, VALUE_ALIGNMENT_BYTES)?
    } else {
        existing_scratch_bytes
    };
    let required_scratch_bytes = transform_bytes
        .checked_mul(tokens)
        .and_then(|bytes| transform_offset.checked_add(bytes))
        .ok_or_else(|| "Metal dense SwiGLU transform scratch size overflows".to_owned())?;

    if gate_up.regions.is_empty() || down.regions.is_empty() {
        return Err("Metal dense SwiGLU resolved empty weight storage".to_owned());
    }
    let mut regions = gate_up.regions;
    let down_region_base = regions.len();
    regions.extend(down.regions);
    let input_region = regions.len();
    regions.push(shared_token_region(
        &invocation,
        ResolvedValueRole::Input,
        0,
        ElementType::F16,
        tokens,
    )?);
    let output_region = regions.len();
    regions.push(shared_token_region(
        &invocation,
        ResolvedValueRole::Output,
        0,
        ElementType::F16,
        tokens,
    )?);
    let scratch_region = regions.len();
    regions.push(shared_scratch_region(&invocation, required_scratch_bytes)?);

    let packed_width = intermediate_size
        .checked_mul(2)
        .ok_or_else(|| "Metal dense SwiGLU packed width overflows".to_owned())?;
    let mut gate_launches = Vec::with_capacity(gate_up.parts.len());
    for part in gate_up.parts {
        let adjusted = PreparedLinearPart {
            region: part.region,
            ..part
        };
        gate_launches.push(linear_launch(
            adjusted,
            input_region,
            scratch_region,
            tokens,
            hidden_size,
            packed_width,
            0,
            0,
        )?);
    }
    let adjusted_down = down_part.relocated(down_region_base)?;
    let mut down_launch = linear_launch(
        adjusted_down,
        scratch_region,
        output_region,
        tokens,
        intermediate_size,
        hidden_size,
        gate_up_bytes,
        0,
    )?;
    for launch in &mut gate_launches {
        launch.bind_hadamard_workspace(&pipelines, &regions, scratch_region, transform_offset)?;
    }
    down_launch.bind_hadamard_workspace(&pipelines, &regions, scratch_region, transform_offset)?;
    validate_launch_regions_with_raw_workspace(&regions, &gate_launches, &[scratch_region])?;
    validate_launch_regions_with_raw_workspace(&regions, &[down_launch], &[scratch_region])?;
    validate_region_span(
        &regions[scratch_region],
        0,
        required_scratch_bytes,
        "Metal dense SwiGLU scratch",
    )?;
    let swiglu = swiglu_launch(0, gate_up_bytes, tokens, intermediate_size, packed_width)?;
    let staging = staged_prefill::Workspace::new(
        &regions,
        scratch_region,
        activation_scratch_bytes,
        staging_bytes,
        gate_launches.iter().copied().chain([down_launch]),
    )?;
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "Metal dense SwiGLU participant count",
    )?;
    let sequence = staged_prefill::Sequence {
        gate_up: gate_launches,
        down: down_launch,
        activation: swiglu,
        scratch_region,
        workspace: staging,
    };
    let statistics = activation_scratch_bytes
        .checked_add(staging_bytes)
        .and_then(|scratch| sequence.statistical_evidence(&pipelines, tokens, scratch));
    MetalDeviceCommand::operation("vnext_dense_swiglu", regions, move |encoder, regions| {
        encoder.record_compute_dispatches(sequence.dispatch_count(regions));
        sequence.encode(&pipelines, regions, |subwork, encode| {
            encoder.begin_compute_subwork(subwork);
            encode(encoder.compute_encoder());
        });
        Ok(())
    })
    .map_err(|error| error.to_string())?
    .with_work_shape(
        if participant_count == 1 {
            DeviceBatchingForm::Scalar
        } else {
            DeviceBatchingForm::Packed
        },
        participant_count,
        tokens,
    )
    .map(|command| command.with_statistical_evidence(statistics))
    .map_err(|error| error.to_string())
}

pub(super) fn linear_launch(
    part: PreparedLinearPart,
    input_region: usize,
    output_region: usize,
    rows: u64,
    in_features: u64,
    output_stride: u64,
    input_offset_bytes: u64,
    output_offset_bytes: u64,
) -> Result<LinearLaunch, String> {
    linear_launch_typed(
        part,
        input_region,
        output_region,
        rows,
        in_features,
        output_stride,
        input_offset_bytes,
        output_offset_bytes,
        ElementType::F16,
    )
}

#[allow(clippy::too_many_arguments)]
fn linear_launch_typed(
    part: PreparedLinearPart,
    input_region: usize,
    output_region: usize,
    rows: u64,
    in_features: u64,
    output_stride: u64,
    input_offset_bytes: u64,
    output_offset_bytes: u64,
    activation_type: ElementType,
) -> Result<LinearLaunch, String> {
    if !matches!(activation_type, ElementType::F16 | ElementType::F32) {
        return Err("Metal linear activation ABI supports only F16 or F32".to_owned());
    }
    let mut launch = LinearLaunch {
        input_region,
        weight_region: part.region,
        output_region,
        input_offset_bytes,
        output_offset_bytes,
        activation_type,
        format: part.format,
        transform: part.transform,
        transform_workspace: None,
        transformed_plan: TransformedLinearPlan::Single,
        plain_plan: PlainLinearPlan::Single,
        params: LinearParams {
            rows: checked_u32(rows, "Metal linear row count")?,
            in_features: checked_u32(in_features, "Metal linear input width")?,
            out_features: part.out_features,
            output_stride: checked_u32(output_stride, "Metal linear output stride")?,
            output_column_offset: part.output_offset,
        },
    };
    launch.plain_plan = PlainLinearPlan::for_launch(launch);
    Ok(launch)
}

pub(super) fn validate_launch_regions(
    regions: &[MetalBufferRegion],
    launches: &[LinearLaunch],
) -> Result<(), String> {
    validate_launch_regions_with_raw_workspace(regions, launches, &[])
}

pub(super) fn validate_launch_regions_with_raw_workspace(
    regions: &[MetalBufferRegion],
    launches: &[LinearLaunch],
    raw_workspace_regions: &[usize],
) -> Result<(), String> {
    for launch in launches {
        if launch.transform.is_some() && launch.transform_workspace.is_none() {
            return Err("Metal linear Hadamard transform has no declared workspace".to_owned());
        }
        let input = regions
            .get(launch.input_region)
            .ok_or_else(|| "Metal linear input region index is invalid".to_owned())?;
        let weight = regions
            .get(launch.weight_region)
            .ok_or_else(|| "Metal linear weight region index is invalid".to_owned())?;
        let output = regions
            .get(launch.output_region)
            .ok_or_else(|| "Metal linear output region index is invalid".to_owned())?;
        let (input_bytes, output_bytes) = launch.activation_bytes()?;
        validate_region_span(
            input,
            launch.input_offset_bytes,
            input_bytes,
            "Metal linear input",
        )?;
        validate_region_span(
            output,
            launch.output_offset_bytes,
            output_bytes,
            "Metal linear output",
        )?;
        if !linear_activation_region_matches(
            input.element_type(),
            launch.input_region,
            launch.activation_type,
            raw_workspace_regions,
        ) || !linear_activation_region_matches(
            output.element_type(),
            launch.output_region,
            launch.activation_type,
            raw_workspace_regions,
        ) {
            return Err("Metal linear activation regions differ from the typed launch".to_owned());
        }
        if weight.length_bytes() == 0 {
            return Err("Metal linear weight region is empty".to_owned());
        }
        if u64::from(launch.params.output_column_offset)
            .checked_add(u64::from(launch.params.out_features))
            .is_none_or(|end| end > u64::from(launch.params.output_stride))
        {
            return Err("Metal linear output columns exceed their stride".to_owned());
        }
    }
    Ok(())
}

fn linear_activation_region_matches(
    region_type: ElementType,
    region_index: usize,
    activation_type: ElementType,
    raw_workspace_regions: &[usize],
) -> bool {
    region_type == activation_type
        || (region_type == ElementType::U8 && raw_workspace_regions.contains(&region_index))
}

fn validate_region_span(
    region: &MetalBufferRegion,
    offset: u64,
    length: u64,
    context: &str,
) -> Result<(), String> {
    if length == 0
        || offset
            .checked_add(length)
            .is_none_or(|end| end > region.length_bytes())
    {
        return Err(format!("{context} exceeds its retained physical region"));
    }
    Ok(())
}

pub(super) fn dispatch_linear(
    pipelines: &MetalLinearPipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    launch: LinearLaunch,
) {
    if let Some(transform) = launch.transform {
        let (workspace_region, workspace_offset) = launch
            .transform_workspace
            .expect("validated linear transform workspace");
        let workspace = &regions[workspace_region];
        pipelines.hadamard.dispatch(
            encoder,
            transform,
            &regions[launch.input_region],
            launch.input_offset_bytes,
            launch.activation_type,
            workspace,
            workspace_offset,
            ElementType::F32,
            regions,
            launch.params.rows,
            launch.params.in_features,
        );
        dispatch_transformed_linear(pipelines, encoder, regions, launch);
        return;
    }
    if let Some(parts) = launch.plain_plan.grouped_parts(launch) {
        for part in parts {
            dispatch_single_plain_linear(pipelines, encoder, regions, part);
        }
    } else if let Some([head, tail]) = launch.plain_plan.parts(launch) {
        dispatch_single_plain_linear(pipelines, encoder, regions, head);
        dispatch_single_plain_linear(pipelines, encoder, regions, tail);
    } else {
        dispatch_single_plain_linear(pipelines, encoder, regions, launch);
    }
}

fn dispatch_single_plain_linear(
    pipelines: &MetalLinearPipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    launch: LinearLaunch,
) {
    let (pipeline, dispatch_kind) =
        pipelines.plain_linear_dispatch(launch.format, launch.activation_type, launch.params);
    encoder.set_compute_pipeline_state(pipeline);
    set_region_offset(
        encoder,
        0,
        &regions[launch.input_region],
        launch.input_offset_bytes,
    );
    set_region_offset(encoder, 1, &regions[launch.weight_region], 0);
    set_region_offset(
        encoder,
        2,
        &regions[launch.output_region],
        launch.output_offset_bytes,
    );
    bind_linear_params(
        encoder,
        launch.params,
        launch.format,
        launch.activation_type,
    );
    dispatch_linear_grid(encoder, launch.params, dispatch_kind);
}

// The caller has already produced this launch's Hadamard result. Its input
// remains F32 while activation_type still declares the original output ABI.
fn dispatch_transformed_linear(
    pipelines: &MetalLinearPipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    launch: LinearLaunch,
) {
    if let Some([head, tail]) = launch.transformed_plan.parts(launch) {
        // Binding validated this plan against these immutable pipelines and
        // retained regions. Both projections read the same completed transform.
        dispatch_single_transformed_linear(pipelines, encoder, regions, head);
        transformed_prefill::dispatch_tail(pipelines, encoder, regions, tail);
    } else {
        dispatch_single_transformed_linear(pipelines, encoder, regions, launch);
    }
}

fn dispatch_single_transformed_linear(
    pipelines: &MetalLinearPipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    launch: LinearLaunch,
) {
    let (workspace_region, workspace_offset) = launch
        .transform_workspace
        .expect("validated linear transform workspace");
    let workspace = &regions[workspace_region];
    let native = match launch.format {
        LinearPhysicalFormat::DenseF16 => None,
        LinearPhysicalFormat::Q4K => Some(GgufBlockFormat::Q4K),
        LinearPhysicalFormat::Q5K => Some(GgufBlockFormat::Q5K),
        LinearPhysicalFormat::Q6K => Some(GgufBlockFormat::Q6K),
        LinearPhysicalFormat::Q8_0 => Some(GgufBlockFormat::Q8_0),
        LinearPhysicalFormat::Native(format) => Some(format),
    };
    let (pipeline, dispatch_kind) = if let Some(format) = native {
        pipelines.hadamard_native_dispatch_for_bindings(
            format,
            launch.activation_type,
            launch.params,
            workspace,
            workspace_offset,
            &regions[launch.weight_region],
        )
    } else if launch.activation_type == ElementType::F32 {
        (&pipelines.dense_f32, LinearDispatchKind::CooperativeGemv)
    } else {
        (
            &pipelines.dense_f32_f16,
            LinearDispatchKind::CooperativeGemv,
        )
    };
    encoder.set_compute_pipeline_state(pipeline);
    set_region_offset(encoder, 0, workspace, workspace_offset);
    set_region_offset(encoder, 1, &regions[launch.weight_region], 0);
    set_region_offset(
        encoder,
        2,
        &regions[launch.output_region],
        launch.output_offset_bytes,
    );
    encoder.set_bytes(
        3,
        std::mem::size_of::<LinearParams>() as u64,
        &launch.params as *const _ as *const c_void,
    );
    if let Some(format) = native {
        bind_native_block(encoder, format, 4);
    }
    dispatch_linear_grid(encoder, launch.params, dispatch_kind);
}

fn bind_linear_params(
    encoder: &ComputeCommandEncoderRef,
    params: LinearParams,
    format: LinearPhysicalFormat,
    activation_type: ElementType,
) {
    encoder.set_bytes(
        3,
        std::mem::size_of::<LinearParams>() as u64,
        &params as *const _ as *const c_void,
    );
    if let Some(format) = format.native_block(activation_type) {
        bind_native_block(encoder, format, 4);
    }
}

fn dispatch_linear_grid(
    encoder: &ComputeCommandEncoderRef,
    params: LinearParams,
    dispatch_kind: LinearDispatchKind,
) {
    match dispatch_kind {
        LinearDispatchKind::NarrowDenseGemv => encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(params.out_features), u64::from(params.rows), 1),
            MTLSize::new(NARROW_DENSE_THREADS, 1, 1),
        ),
        LinearDispatchKind::Pq2CooperativeGemv => encoder.dispatch_thread_groups(
            MTLSize::new(
                u64::from(params.out_features).div_ceil(16),
                u64::from(params.rows),
                1,
            ),
            MTLSize::new(32, 2, 1),
        ),
        LinearDispatchKind::CooperativeGemv => encoder.dispatch_thread_groups(
            MTLSize::new(
                u64::from(params.out_features).div_ceil(4),
                u64::from(params.rows),
                1,
            ),
            MTLSize::new(32, 2, 1),
        ),
        LinearDispatchKind::SharedWeightGemv => encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(params.out_features).div_ceil(4), 1, 1),
            MTLSize::new(32, 2, 1),
        ),
        LinearDispatchKind::TiledGemm | LinearDispatchKind::TiledGemmM8 => {
            encoder.set_threadgroup_memory_length(0, 8192);
            let tile_rows = if dispatch_kind == LinearDispatchKind::TiledGemmM8 {
                8
            } else {
                32
            };
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    u64::from(params.rows).div_ceil(tile_rows),
                    u64::from(params.out_features).div_ceil(64),
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
        }
        LinearDispatchKind::NativeTiledGemm => {
            // F32 X[32][32] and W[32][64]; W is reused for the output tile.
            encoder.set_threadgroup_memory_length(0, 12288);
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    u64::from(params.rows).div_ceil(32),
                    u64::from(params.out_features).div_ceil(64),
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
        }
        LinearDispatchKind::NativeTiledGemmM64 => {
            dispatch_m64_grid(encoder, params.rows, params.out_features);
        }
    }
}

pub(super) fn swiglu_launch(
    gate_up_offset_bytes: u64,
    activation_offset_bytes: u64,
    rows: u64,
    intermediate_size: u64,
    gate_up_stride: u64,
) -> Result<SwiGluLaunch, String> {
    Ok(SwiGluLaunch {
        gate_up_offset_bytes,
        activation_offset_bytes,
        params: SwiGluParams {
            rows: checked_u32(rows, "Metal SwiGLU row count")?,
            intermediate_size: checked_u32(intermediate_size, "Metal SwiGLU intermediate size")?,
            gate_up_stride: checked_u32(gate_up_stride, "Metal SwiGLU gate/up stride")?,
        },
    })
}

pub(super) fn dispatch_swiglu(
    pipelines: &MetalLinearPipelines,
    encoder: &ComputeCommandEncoderRef,
    scratch: &MetalBufferRegion,
    launch: SwiGluLaunch,
) {
    encoder.set_compute_pipeline_state(&pipelines.swiglu);
    set_region_offset(encoder, 0, scratch, launch.gate_up_offset_bytes);
    set_region_offset(encoder, 1, scratch, launch.activation_offset_bytes);
    encoder.set_bytes(
        2,
        std::mem::size_of::<SwiGluParams>() as u64,
        &launch.params as *const _ as *const c_void,
    );
    let elements = u64::from(launch.params.rows) * u64::from(launch.params.intermediate_size);
    encoder.dispatch_thread_groups(
        MTLSize::new(elements.div_ceil(THREADS_PER_GROUP), 1, 1),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );
}

fn set_region_offset(
    encoder: &ComputeCommandEncoderRef,
    index: u64,
    region: &MetalBufferRegion,
    extra_offset_bytes: u64,
) {
    encoder.set_buffer(
        index,
        Some(region.buffer()),
        region.offset_bytes() + extra_offset_bytes,
    );
}

fn prepare_matrix_weight(
    weight: MetalResolvedWeight,
    out_features: u64,
    in_features: u64,
) -> Result<PreparedLinearWeight, String> {
    if weight.logical_element_type() != ElementType::F16
        || weight.logical_dimensions() != [out_features, in_features]
    {
        return Err("Metal linear logical weight differs from its contract".to_owned());
    }
    let (regions, components, layout) = weight.into_command_parts();
    let part = prepare_leaf_part(
        &regions,
        &components,
        &layout,
        out_features,
        in_features,
        1,
        0,
    )?;
    Ok(PreparedLinearWeight {
        regions,
        parts: vec![part],
    })
}

pub(super) fn append_shared_matrix_weight(
    regions: &mut Vec<MetalBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ordinal: u32,
    out_features: u64,
    in_features: u64,
    context: &str,
) -> Result<PreparedLinearPart, String> {
    let first = &invocation.participants()[0];
    let resolved = resolve_weight(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, ordinal)?,
    )?;
    for participant in &invocation.participants()[1..] {
        let candidate = resolve_weight(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
        )?;
        if !same_resolved_weight(&resolved, &candidate) {
            return Err(format!("{context} participants do not share one weight"));
        }
    }
    let prepared = prepare_matrix_weight(resolved, out_features, in_features)?;
    let [part] = prepared.parts.as_slice() else {
        return Err(format!("{context} requires one physical matrix"));
    };
    let part = part.relocated(regions.len())?;
    regions.extend(prepared.regions);
    Ok(part)
}

pub(super) fn append_shared_partitioned_matrix_weight(
    regions: &mut Vec<MetalBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ordinal: u32,
    out_features: u64,
    in_features: u64,
    context: &str,
) -> Result<Vec<PreparedLinearPart>, String> {
    let first = &invocation.participants()[0];
    let resolved = resolve_weight(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, ordinal)?,
    )?;
    for participant in &invocation.participants()[1..] {
        let candidate = resolve_weight(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
        )?;
        if !same_resolved_weight(&resolved, &candidate) {
            return Err(format!("{context} participants do not share one weight"));
        }
    }
    let prepared = prepare_partitioned_matrix_weight(resolved, out_features, in_features)?;
    let region_base = regions.len();
    let parts = prepared
        .parts
        .into_iter()
        .map(|part| part.relocated(region_base))
        .collect::<Result<Vec<_>, String>>()?;
    if parts.is_empty() {
        return Err(format!("{context} resolved no physical matrix"));
    }
    regions.extend(prepared.regions);
    Ok(parts)
}

fn prepare_partitioned_matrix_weight(
    weight: MetalResolvedWeight,
    out_features: u64,
    in_features: u64,
) -> Result<PreparedLinearWeight, String> {
    if weight.logical_element_type() != ElementType::F16
        || weight.logical_dimensions() != [out_features, in_features]
    {
        return Err("Metal partitioned linear logical weight differs from its contract".to_owned());
    }
    let (regions, components, layout) = weight.into_command_parts();
    let parts = prepare_matrix_partition(
        &layout,
        out_features,
        in_features,
        |layout, width, offset| {
            prepare_leaf_part(&regions, &components, layout, width, in_features, 1, offset)
        },
    )?;
    Ok(PreparedLinearWeight { regions, parts })
}

pub(super) fn prepare_matrix_partition(
    layout: &MetalResolvedWeightLayout,
    out_features: u64,
    in_features: u64,
    mut prepare: impl FnMut(&MetalResolvedWeightLayout, u64, u64) -> Result<PreparedLinearPart, String>,
) -> Result<Vec<PreparedLinearPart>, String> {
    let mut parts = match layout {
        MetalResolvedWeightLayout::Composite { parts } => {
            let mut prepared = Vec::with_capacity(parts.len());
            for part in parts {
                if part.logical_offsets.len() != 2
                    || part.extents.len() != 2
                    || part.logical_offsets[1] != 0
                    || part.extents[1] != in_features
                {
                    return Err(
                        "Metal partitioned linear composite has invalid row placement".to_owned(),
                    );
                }
                prepared.push(prepare(
                    &part.layout,
                    part.extents[0],
                    part.logical_offsets[0],
                )?);
            }
            prepared
        }
        _ => vec![prepare(layout, out_features, 0)?],
    };
    parts.sort_by_key(|part| part.output_offset);
    let mut next_output = 0_u64;
    for part in &parts {
        if u64::from(part.output_offset) != next_output {
            return Err(
                "Metal partitioned linear composite overlaps or leaves a row gap".to_owned(),
            );
        }
        next_output = next_output
            .checked_add(u64::from(part.out_features))
            .ok_or_else(|| "Metal partitioned linear output width overflows".to_owned())?;
    }
    if next_output != out_features {
        return Err(
            "Metal partitioned linear composite does not cover its output width".to_owned(),
        );
    }
    Ok(parts)
}

pub(super) fn append_shared_gate_up_weight(
    regions: &mut Vec<MetalBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ordinal: u32,
    intermediate_size: u64,
    hidden_size: u64,
    context: &str,
) -> Result<Vec<PreparedLinearPart>, String> {
    let first = &invocation.participants()[0];
    let resolved = resolve_weight(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, ordinal)?,
    )?;
    for participant in &invocation.participants()[1..] {
        let candidate = resolve_weight(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
        )?;
        if !same_resolved_weight(&resolved, &candidate) {
            return Err(format!("{context} participants do not share one weight"));
        }
    }
    let prepared = prepare_gate_up_weight(resolved, intermediate_size, hidden_size)?;
    let region_base = regions.len();
    let parts = prepared
        .parts
        .into_iter()
        .map(|part| part.relocated(region_base))
        .collect::<Result<Vec<_>, String>>()?;
    if parts.is_empty() {
        return Err(format!("{context} resolved no physical matrix"));
    }
    regions.extend(prepared.regions);
    Ok(parts)
}

fn prepare_gate_up_weight(
    weight: MetalResolvedWeight,
    intermediate_size: u64,
    hidden_size: u64,
) -> Result<PreparedLinearWeight, String> {
    if weight.logical_element_type() != ElementType::F16
        || weight.logical_dimensions() != [2, intermediate_size, hidden_size]
    {
        return Err("Metal dense SwiGLU gate/up logical weight differs".to_owned());
    }
    let (regions, components, layout) = weight.into_command_parts();
    let parts = match &layout {
        MetalResolvedWeightLayout::Composite { parts } => {
            prepare_gate_up_composite(&regions, &components, parts, intermediate_size, hidden_size)?
        }
        _ => {
            let packed = intermediate_size
                .checked_mul(2)
                .ok_or_else(|| "Metal dense SwiGLU packed rows overflow".to_owned())?;
            vec![prepare_leaf_part(
                &regions,
                &components,
                &layout,
                packed,
                hidden_size,
                2,
                0,
            )?]
        }
    };
    Ok(PreparedLinearWeight { regions, parts })
}

fn prepare_gate_up_composite(
    regions: &[MetalBufferRegion],
    components: &[MetalResolvedWeightComponent],
    parts: &[MetalResolvedCompositePart],
    intermediate_size: u64,
    hidden_size: u64,
) -> Result<Vec<PreparedLinearPart>, String> {
    prepare_gate_up_partition(
        parts,
        intermediate_size,
        hidden_size,
        |layout, output_offset| {
            prepare_leaf_part(
                regions,
                components,
                layout,
                intermediate_size,
                hidden_size,
                2,
                output_offset,
            )
        },
    )
}

fn prepare_gate_up_partition(
    parts: &[MetalResolvedCompositePart],
    intermediate_size: u64,
    hidden_size: u64,
    mut prepare: impl FnMut(&MetalResolvedWeightLayout, u64) -> Result<PreparedLinearPart, String>,
) -> Result<Vec<PreparedLinearPart>, String> {
    if parts.len() != 2 {
        return Err("Metal dense SwiGLU gate/up composite must have two parts".to_owned());
    }
    let mut prepared = Vec::with_capacity(2);
    for part in parts {
        if part.logical_offsets.len() != 3
            || part.extents != [1, intermediate_size, hidden_size]
            || part.logical_offsets[1..] != [0, 0]
            || part.logical_offsets[0] > 1
        {
            return Err("Metal dense SwiGLU gate/up composite partition differs".to_owned());
        }
        let output_offset = part.logical_offsets[0]
            .checked_mul(intermediate_size)
            .ok_or_else(|| "Metal dense SwiGLU partition offset overflows".to_owned())?;
        prepared.push(prepare(&part.layout, output_offset)?);
    }
    prepared.sort_by_key(|part| part.output_offset);
    let expected_second = checked_u32(
        intermediate_size,
        "Metal dense SwiGLU second partition offset",
    )?;
    if prepared[0].output_offset != 0 || prepared[1].output_offset != expected_second {
        return Err("Metal dense SwiGLU gate/up partitions overlap or leave a gap".to_owned());
    }
    Ok(prepared)
}

#[allow(clippy::too_many_arguments)]
fn prepare_leaf_part(
    regions: &[MetalBufferRegion],
    components: &[MetalResolvedWeightComponent],
    layout: &MetalResolvedWeightLayout,
    out_features: u64,
    in_features: u64,
    expected_block_axis: u32,
    output_offset: u64,
) -> Result<PreparedLinearPart, String> {
    let (layout, transform) = match layout {
        MetalResolvedWeightLayout::Hadamard { values, transform } => {
            if transform.inverse {
                return Err("Metal linear requires a forward Hadamard transform".to_owned());
            }
            validate_hadamard_transform(*transform, in_features, components, regions)?;
            (values.as_ref(), Some(*transform))
        }
        layout => (layout, None),
    };
    let mut part = prepare_leaf_encoding(
        components,
        layout,
        out_features,
        in_features,
        expected_block_axis,
        output_offset,
    )?;
    if part.region >= regions.len() {
        return Err("Metal linear physical component is absent".to_owned());
    }
    part.transform = transform;
    Ok(part)
}

/// Pure physical ABI selection shared with read-only route projection. It
/// never certifies retained regions, live aliases, or transform workspace.
pub(super) fn prepare_leaf_encoding(
    components: &[MetalResolvedWeightComponent],
    layout: &MetalResolvedWeightLayout,
    out_features: u64,
    in_features: u64,
    expected_block_axis: u32,
    output_offset: u64,
) -> Result<PreparedLinearPart, String> {
    let (component, format) = match layout {
        MetalResolvedWeightLayout::Dense { component }
        | MetalResolvedWeightLayout::Stored { component } => {
            let metadata = component_metadata(components, *component)?;
            if metadata.encoding()
                != &(WeightEncoding::Dense {
                    element_type: ElementType::F16,
                })
                || !physical_matrix_shape_matches(
                    metadata.physical_dimensions(),
                    out_features,
                    in_features,
                )
            {
                return Err("Metal dense linear physical ABI differs".to_owned());
            }
            (*component, LinearPhysicalFormat::DenseF16)
        }
        MetalResolvedWeightLayout::BlockQuantized {
            component,
            spec,
            block_axis,
            block_padding,
        } => {
            if *block_axis != expected_block_axis
                || block_padding != &PhysicalWeightPadding::Exact
                || !in_features.is_multiple_of(u64::from(spec.logical_values_per_block))
            {
                return Err("Metal quantized linear physical ABI differs".to_owned());
            }
            let format = match GgufBlockFormat::from_spec(spec)? {
                GgufBlockFormat::Q4K => LinearPhysicalFormat::Q4K,
                GgufBlockFormat::Q5K => LinearPhysicalFormat::Q5K,
                GgufBlockFormat::Q6K => LinearPhysicalFormat::Q6K,
                GgufBlockFormat::Q8_0 => LinearPhysicalFormat::Q8_0,
                native => LinearPhysicalFormat::Native(native),
            };
            let blocks_per_row = in_features / u64::from(spec.logical_values_per_block);
            let metadata = component_metadata(components, *component)?;
            if metadata.encoding() != &WeightEncoding::BlockQuantized(spec.clone())
                || !physical_matrix_shape_matches(
                    metadata.physical_dimensions(),
                    out_features,
                    blocks_per_row,
                )
            {
                return Err("Metal quantized linear component shape differs".to_owned());
            }
            (*component, format)
        }
        _ => return Err("Metal linear weight is not one matrix leaf".to_owned()),
    };
    Ok(PreparedLinearPart {
        region: component,
        format,
        transform: None,
        output_offset: checked_u32(output_offset, "Metal linear output offset")?,
        out_features: checked_u32(out_features, "Metal linear output width")?,
    })
}

fn component_metadata(
    components: &[MetalResolvedWeightComponent],
    component: usize,
) -> Result<&MetalResolvedWeightComponent, String> {
    components
        .get(component)
        .ok_or_else(|| "Metal linear component metadata is absent".to_owned())
}

fn physical_matrix_shape_matches(dimensions: &[u64], rows: u64, columns: u64) -> bool {
    dimensions.last() == Some(&columns)
        && dimensions
            .iter()
            .try_fold(1_u64, |total, extent| total.checked_mul(*extent))
            == rows.checked_mul(columns)
}

pub(super) fn same_resolved_weight(
    left: &MetalResolvedWeight,
    right: &MetalResolvedWeight,
) -> bool {
    left.logical_dimensions() == right.logical_dimensions()
        && left.logical_element_type() == right.logical_element_type()
        && left.components() == right.components()
        && left.layout() == right.layout()
        && left.regions().len() == right.regions().len()
        && left
            .regions()
            .iter()
            .zip(right.regions())
            .all(|(left, right)| left.same_physical_region(right))
}

fn validate_dense_linear_participant(
    participant: &ferrum_interfaces::vnext::OperationInvocation<'_, MetalDeviceBuffer>,
    in_features: u64,
    out_features: u64,
) -> Result<(), String> {
    validate_dense_linear_bindings(participant.bindings(), in_features, out_features)
}

fn validate_dense_linear_bindings(
    bindings: &[ferrum_interfaces::vnext::ResolvedValueBinding],
    in_features: u64,
    out_features: u64,
) -> Result<(), String> {
    let input = binding(bindings, ResolvedValueRole::Input, 0)?;
    let weight = binding(bindings, ResolvedValueRole::Input, 1)?;
    let output = binding(bindings, ResolvedValueRole::Output, 0)?;
    let dimensions = input.tensor().dimensions();
    if dimensions.len() != 2
        || dimensions[1] != in_features
        || weight.tensor().dimensions() != [out_features, in_features]
        || output.tensor().dimensions() != [dimensions[0], out_features]
        || !f16_contiguous(input)
        || !f16_contiguous(weight)
        || !f16_contiguous(output)
    {
        return Err("Metal dense linear invocation differs from its signature".to_owned());
    }
    Ok(())
}

fn validate_last_token_participant(
    participant: &ferrum_interfaces::vnext::OperationInvocation<'_, MetalDeviceBuffer>,
    hidden_size: u64,
    out_features: u64,
    activation_type: ElementType,
) -> Result<(), String> {
    validate_last_token_bindings(
        participant.bindings(),
        hidden_size,
        out_features,
        activation_type,
    )
}

fn validate_last_token_bindings(
    bindings: &[ferrum_interfaces::vnext::ResolvedValueBinding],
    hidden_size: u64,
    out_features: u64,
    activation_type: ElementType,
) -> Result<(), String> {
    let input = binding(bindings, ResolvedValueRole::Input, 0)?;
    let weight = binding(bindings, ResolvedValueRole::Input, 1)?;
    let output = binding(bindings, ResolvedValueRole::Output, 0)?;
    let dimensions = input.tensor().dimensions();
    if dimensions.len() != 2
        || dimensions[0] == 0
        || dimensions[1] != hidden_size
        || weight.tensor().dimensions() != [out_features, hidden_size]
        || output.tensor().dimensions() != [1, out_features]
        || input.tensor().element_type() != activation_type
        || !matches!(input.tensor().layout(), ResolvedTensorLayout::Contiguous)
        || !f16_contiguous(weight)
        || output.tensor().element_type() != activation_type
        || !matches!(output.tensor().layout(), ResolvedTensorLayout::Contiguous)
    {
        return Err("Metal last-token linear invocation differs from its signature".to_owned());
    }
    Ok(())
}

fn validate_swiglu_participant(
    participant: &ferrum_interfaces::vnext::OperationInvocation<'_, MetalDeviceBuffer>,
    hidden_size: u64,
    intermediate_size: u64,
) -> Result<(), String> {
    validate_swiglu_bindings(participant.bindings(), hidden_size, intermediate_size)
}

fn validate_swiglu_bindings(
    bindings: &[ferrum_interfaces::vnext::ResolvedValueBinding],
    hidden_size: u64,
    intermediate_size: u64,
) -> Result<(), String> {
    let input = binding(bindings, ResolvedValueRole::Input, 0)?;
    let gate_up = binding(bindings, ResolvedValueRole::Input, 1)?;
    let down = binding(bindings, ResolvedValueRole::Input, 2)?;
    let output = binding(bindings, ResolvedValueRole::Output, 0)?;
    let dimensions = input.tensor().dimensions();
    if dimensions.len() != 2
        || dimensions[1] != hidden_size
        || gate_up.tensor().dimensions() != [2, intermediate_size, hidden_size]
        || down.tensor().dimensions() != [hidden_size, intermediate_size]
        || output.tensor().dimensions() != dimensions
        || !f16_contiguous(input)
        || !f16_contiguous(gate_up)
        || !f16_contiguous(down)
        || !f16_contiguous(output)
        || !matches!(input.tensor().layout(), ResolvedTensorLayout::Contiguous)
    {
        return Err("Metal dense SwiGLU invocation differs from its signature".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod native_tests;

#[cfg(test)]
mod hadamard_tests;

#[cfg(test)]
mod pq2_tests;

#[cfg(test)]
mod narrow_dense_tests;
#[cfg(test)]
mod pq2_decode_tests;

#[cfg(test)]
mod pq2_prefill_tests;

#[cfg(test)]
mod microbench;

#[cfg(test)]
mod plain_prefill_tests;

#[cfg(test)]
mod tests {
    use super::super::numerical_tolerance;
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device as CandleDevice, Tensor};
    use half::f16;
    use metal::{BufferRef, MTLCommandBufferStatus, MTLResourceOptions};

    const DENSE_LINEAR_TOLERANCE_ID: &str =
        "runtime-vnext.metal.dense-linear.v1.operation.fp16.gguf-q4-k.hidden-2560";
    const DENSE_SWIGLU_TOLERANCE_ID: &str =
        "runtime-vnext.metal.dense-swiglu.v1.operation.fp16.gguf-q4-k-q6-k.full-pipeline";
    const LAST_TOKEN_LINEAR_TOLERANCE_ID: &str =
        "runtime-vnext.metal.last-token-dense-linear.v1.operation.fp16.gguf-q6-k.final-row";
    const PACKED_LAST_TOKEN_LINEAR_TOLERANCE_ID: &str =
        "runtime-vnext.metal.last-token-dense-linear.v1_1.operation.fp16.gguf-q6-k.packed-15";

    fn shared_buffer<T>(device: &Device, values: &[T]) -> metal::Buffer {
        device.new_buffer_with_data(
            values.as_ptr() as *const c_void,
            std::mem::size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn output_buffer<T>(device: &Device, elements: usize) -> metal::Buffer {
        device.new_buffer(
            (elements * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn read_f16(buffer: &BufferRef, elements: usize) -> Vec<f32> {
        let values: &[f16] =
            unsafe { std::slice::from_raw_parts(buffer.contents() as *const f16, elements) };
        values.iter().map(|value| value.to_f32()).collect()
    }

    fn read_f32(buffer: &BufferRef, elements: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, elements) }.to_vec()
    }

    #[test]
    fn packed_last_token_shape_requires_shared_consecutive_unit_rows() {
        assert!(!packed_last_token_rows(true, 1, [0..1]));
        assert!(!packed_last_token_rows(false, 2, [0..1, 1..2]));
        assert!(packed_last_token_rows(true, 2, [0..1, 1..2]));
        assert!(packed_last_token_rows(
            true,
            15,
            (0_u64..15).map(|row| row..row + 1),
        ));
        assert!(!packed_last_token_rows(true, 2, [0..1, 2..3]));
        assert!(!packed_last_token_rows(true, 2, [0..2, 2..3]));
        assert!(!packed_last_token_rows(true, 3, [0..1, 1..2]));
    }

    #[test]
    fn raw_linear_workspace_requires_explicit_region_authorization() {
        assert!(linear_activation_region_matches(
            ElementType::F32,
            3,
            ElementType::F32,
            &[],
        ));
        assert!(!linear_activation_region_matches(
            ElementType::U8,
            3,
            ElementType::F32,
            &[],
        ));
        assert!(linear_activation_region_matches(
            ElementType::U8,
            3,
            ElementType::F32,
            &[3],
        ));
        assert!(!linear_activation_region_matches(
            ElementType::U8,
            4,
            ElementType::F32,
            &[3],
        ));
    }

    #[test]
    fn packed_last_token_scratch_formula_covers_1_2_and_15_participants() {
        let hidden_size = 2560_u64;
        let out_features = 248_320_u64;
        let bytes_per_sequence =
            last_token_scratch_bytes_per_sequence(hidden_size, out_features, ElementType::F16)
                .unwrap();
        for participant_count in [1_u64, 2, 15] {
            let layout = LastTokenPackedScratchLayout::new(
                participant_count,
                hidden_size,
                out_features,
                ElementType::F16,
            )
            .unwrap();
            let admitted = align_up_bytes(
                LAST_TOKEN_SCRATCH_PADDING_BYTES + participant_count * bytes_per_sequence,
                VALUE_ALIGNMENT_BYTES,
            )
            .unwrap();
            assert_eq!(layout.output_offset_bytes % VALUE_ALIGNMENT_BYTES, 0);
            assert!(layout.required_bytes <= admitted);
        }
    }

    #[test]
    fn native_linear_formats_match_cpu_oracles_on_real_metal() {
        let device = Device::system_default().expect("linear conformance requires Metal");
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let rows = 2_usize;
        let input_width = 2560_usize;
        let output_width = 32_usize;
        let input = (0..rows * input_width)
            .map(|index| f16::from_f32(((index as f32) * 0.0017).sin() * 0.25))
            .collect::<Vec<_>>();
        let dense = (0..output_width * input_width)
            .map(|index| f16::from_f32(((index as f32) * 0.0023).cos() * 0.125))
            .collect::<Vec<_>>();
        let input_f32 = input.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let dense_f32 = dense.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let cpu = CandleDevice::Cpu;
        let input_tensor = Tensor::from_vec(input_f32, (rows, input_width), &cpu).unwrap();
        let dense_tensor = Tensor::from_vec(dense_f32, (output_width, input_width), &cpu).unwrap();
        let input_buffer = shared_buffer(&device, &input);

        let mut cases = Vec::new();
        cases.push((
            LinearPhysicalFormat::DenseF16,
            shared_buffer(&device, &dense),
            input_tensor
                .matmul(&dense_tensor.transpose(0, 1).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        ));
        for (dtype, format) in [
            (GgmlDType::Q4K, LinearPhysicalFormat::Q4K),
            (GgmlDType::Q5K, LinearPhysicalFormat::Q5K),
            (GgmlDType::Q6K, LinearPhysicalFormat::Q6K),
            (GgmlDType::Q8_0, LinearPhysicalFormat::Q8_0),
        ] {
            let quantized = QTensor::quantize(&dense_tensor, dtype).unwrap();
            let reference = input_tensor
                .matmul(&quantized.dequantize(&cpu).unwrap().transpose(0, 1).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            cases.push((
                format,
                shared_buffer(&device, &quantized.data().unwrap()),
                reference,
            ));
        }

        for (format, weight, reference) in cases {
            let output = output_buffer::<f16>(&device, rows * output_width);
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            dispatch_raw_linear(
                &pipelines,
                encoder,
                format,
                &input_buffer,
                &weight,
                &output,
                LinearParams {
                    rows: rows as u32,
                    in_features: input_width as u32,
                    out_features: output_width as u32,
                    output_stride: output_width as u32,
                    output_column_offset: 0,
                },
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            let actual = read_f16(&output, rows * output_width);
            if format == LinearPhysicalFormat::Q4K {
                numerical_tolerance::assert_matches(
                    "Metal/CPU Q4_K dense linear",
                    &actual,
                    &[rows, output_width],
                    &reference,
                    &[rows, output_width],
                    numerical_tolerance::LogicalDtype::Fp16,
                    DENSE_LINEAR_TOLERANCE_ID,
                )
                .expect("reviewed dense-linear numerical contract");
            } else {
                assert_linear_diagnostic_close(&format!("{format:?}"), &actual, &reference);
            }
        }
    }

    #[test]
    fn shared_k_quant_gemv_honors_batch_offsets_strides_and_tail_rows() {
        let device = Device::system_default().expect("K-quant GEMV conformance requires Metal");
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let rows = 3_usize;
        let input_width = 256_usize;
        let output_width = 5_usize;
        let output_stride = 11_usize;
        let output_column_offset = 3_usize;
        let input_prefix = 7_usize;
        let output_prefix = 5_usize;
        let sentinel = f16::from_f32(123.0);

        let input = (0..rows * input_width)
            .map(|index| f16::from_f32(((index as f32) * 0.013).sin() * 0.2))
            .collect::<Vec<_>>();
        let mut prefixed_input = vec![f16::from_f32(-17.0); input_prefix];
        prefixed_input.extend_from_slice(&input);
        let input_buffer = shared_buffer(&device, &prefixed_input);
        let cpu = CandleDevice::Cpu;
        let input_tensor = Tensor::from_vec(
            input.iter().map(|value| value.to_f32()).collect::<Vec<_>>(),
            (rows, input_width),
            &cpu,
        )
        .unwrap();
        let dense = Tensor::from_vec(
            (0..output_width * input_width)
                .map(|index| ((index as f32) * 0.0071).cos() * 0.15)
                .collect::<Vec<_>>(),
            (output_width, input_width),
            &cpu,
        )
        .unwrap();

        for (dtype, format) in [
            (GgmlDType::Q4K, LinearPhysicalFormat::Q4K),
            (GgmlDType::Q5K, LinearPhysicalFormat::Q5K),
            (GgmlDType::Q6K, LinearPhysicalFormat::Q6K),
            (GgmlDType::Q8_0, LinearPhysicalFormat::Q8_0),
        ] {
            let quantized = QTensor::quantize(&dense, dtype).unwrap();
            let reference = input_tensor
                .matmul(&quantized.dequantize(&cpu).unwrap().transpose(0, 1).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let weight = shared_buffer(&device, &quantized.data().unwrap());
            let output_elements = output_prefix + rows * output_stride + output_width;
            let output_values = vec![sentinel; output_elements];
            let output = shared_buffer(&device, &output_values);
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            dispatch_raw_linear_with_offsets(
                &pipelines,
                encoder,
                format,
                &input_buffer,
                (input_prefix * std::mem::size_of::<f16>()) as u64,
                &weight,
                &output,
                (output_prefix * std::mem::size_of::<f16>()) as u64,
                LinearParams {
                    rows: rows as u32,
                    in_features: input_width as u32,
                    out_features: output_width as u32,
                    output_stride: output_stride as u32,
                    output_column_offset: output_column_offset as u32,
                },
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

            let actual = read_f16(&output, output_elements);
            let mut compact = Vec::with_capacity(rows * output_width);
            for row in 0..rows {
                let start = output_prefix + row * output_stride + output_column_offset;
                compact.extend_from_slice(&actual[start..start + output_width]);
            }
            assert_linear_diagnostic_close(&format!("{format:?} strided"), &compact, &reference);

            for (index, value) in actual.iter().enumerate() {
                let relative = index.saturating_sub(output_prefix);
                let row = relative / output_stride;
                let column = relative % output_stride;
                let written = index >= output_prefix
                    && row < rows
                    && column >= output_column_offset
                    && column < output_column_offset + output_width;
                if !written {
                    assert_eq!(
                        *value,
                        sentinel.to_f32(),
                        "{format:?} clobbered index {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn shared_quantized_tiled_gemm_matches_prefill_shape_and_preserves_output_guards() {
        let device = Device::system_default().expect("K-quant GEMM conformance requires Metal");
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let rows = 17_usize;
        let input_width = 256_usize;
        let output_width = 67_usize;
        let output_stride = 79_usize;
        let output_column_offset = 5_usize;
        let input_prefix = 3_usize;
        let output_prefix = 7_usize;
        let sentinel = f16::from_f32(123.0);

        let input = (0..rows * input_width)
            .map(|index| f16::from_f32(((index as f32) * 0.013).sin() * 0.2))
            .collect::<Vec<_>>();
        let mut prefixed_input = vec![f16::from_f32(-17.0); input_prefix];
        prefixed_input.extend_from_slice(&input);
        let input_buffer = shared_buffer(&device, &prefixed_input);
        let cpu = CandleDevice::Cpu;
        let input_tensor = Tensor::from_vec(
            input.iter().map(|value| value.to_f32()).collect::<Vec<_>>(),
            (rows, input_width),
            &cpu,
        )
        .unwrap();
        let dense = Tensor::from_vec(
            (0..output_width * input_width)
                .map(|index| ((index as f32) * 0.0071).cos() * 0.15)
                .collect::<Vec<_>>(),
            (output_width, input_width),
            &cpu,
        )
        .unwrap();

        for (dtype, format) in [
            (GgmlDType::Q4K, LinearPhysicalFormat::Q4K),
            (GgmlDType::Q5K, LinearPhysicalFormat::Q5K),
            (GgmlDType::Q6K, LinearPhysicalFormat::Q6K),
            (GgmlDType::Q8_0, LinearPhysicalFormat::Q8_0),
        ] {
            let (_, dispatch_kind) =
                pipelines.linear_pipeline(format, rows as u32, output_width as u32);
            assert_eq!(dispatch_kind, LinearDispatchKind::TiledGemm);
            let quantized = QTensor::quantize(&dense, dtype).unwrap();
            let reference = input_tensor
                .matmul(&quantized.dequantize(&cpu).unwrap().transpose(0, 1).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let weight = shared_buffer(&device, &quantized.data().unwrap());
            let output_elements = output_prefix + rows * output_stride + output_width;
            let output = shared_buffer(&device, &vec![sentinel; output_elements]);
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            dispatch_raw_linear_with_offsets(
                &pipelines,
                encoder,
                format,
                &input_buffer,
                (input_prefix * std::mem::size_of::<f16>()) as u64,
                &weight,
                &output,
                (output_prefix * std::mem::size_of::<f16>()) as u64,
                LinearParams {
                    rows: rows as u32,
                    in_features: input_width as u32,
                    out_features: output_width as u32,
                    output_stride: output_stride as u32,
                    output_column_offset: output_column_offset as u32,
                },
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

            let actual = read_f16(&output, output_elements);
            let mut compact = Vec::with_capacity(rows * output_width);
            for row in 0..rows {
                let start = output_prefix + row * output_stride + output_column_offset;
                compact.extend_from_slice(&actual[start..start + output_width]);
            }
            assert_linear_diagnostic_close(&format!("{format:?} tiled"), &compact, &reference);

            for (index, value) in actual.iter().enumerate() {
                let relative = index.saturating_sub(output_prefix);
                let row = relative / output_stride;
                let column = relative % output_stride;
                let written = index >= output_prefix
                    && row < rows
                    && column >= output_column_offset
                    && column < output_column_offset + output_width;
                if !written {
                    assert_eq!(
                        *value,
                        sentinel.to_f32(),
                        "{format:?} tiled GEMM clobbered index {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn shared_k_quant_tiled_gemm_supports_width_above_i16_max() {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping wide K-quant GEMM ABI test");
            return;
        };
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let rows = QUANTIZED_TILED_GEMM_MIN_ROWS as usize;
        let input_width = 256_usize;
        let output_width = i16::MAX as usize + 66;
        let input = (0..rows * input_width)
            .map(|index| f16::from_f32(((index as f32) * 0.013).sin() * 0.2))
            .collect::<Vec<_>>();
        let cpu = CandleDevice::Cpu;
        let input_tensor = Tensor::from_vec(
            input.iter().map(|value| value.to_f32()).collect::<Vec<_>>(),
            (rows, input_width),
            &cpu,
        )
        .unwrap();
        let dense = Tensor::from_vec(
            (0..output_width * input_width)
                .map(|index| ((index as f32) * 0.0071).cos() * 0.15)
                .collect::<Vec<_>>(),
            (output_width, input_width),
            &cpu,
        )
        .unwrap();
        let quantized = QTensor::quantize(&dense, GgmlDType::Q6K).unwrap();
        let reference = input_tensor
            .matmul(&quantized.dequantize(&cpu).unwrap().transpose(0, 1).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let input_buffer = shared_buffer(&device, &input);
        let weight_buffer = shared_buffer(&device, &quantized.data().unwrap());
        let output = output_buffer::<f16>(&device, rows * output_width);
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let (_, dispatch_kind) =
            pipelines.linear_pipeline(LinearPhysicalFormat::Q6K, rows as u32, output_width as u32);
        assert_eq!(dispatch_kind, LinearDispatchKind::TiledGemmM8);
        dispatch_raw_linear(
            &pipelines,
            encoder,
            LinearPhysicalFormat::Q6K,
            &input_buffer,
            &weight_buffer,
            &output,
            LinearParams {
                rows: rows as u32,
                in_features: input_width as u32,
                out_features: output_width as u32,
                output_stride: output_width as u32,
                output_column_offset: 0,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        let actual = read_f16(&output, rows * output_width);
        assert_linear_diagnostic_close("Q6_K wide tiled GEMM", &actual, &reference);
    }

    #[test]
    fn native_swiglu_uses_packed_gate_then_up_order() {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping SwiGLU conformance");
            return;
        };
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let rows = 2_u32;
        let intermediate = 64_u32;
        let stride = intermediate * 2;
        let packed = (0..rows * stride)
            .map(|index| f16::from_f32((index as f32 * 0.013).sin()))
            .collect::<Vec<_>>();
        let input = shared_buffer(&device, &packed);
        let output = output_buffer::<f16>(&device, (rows * intermediate) as usize);
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipelines.swiglu);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&output), 0);
        let params = SwiGluParams {
            rows,
            intermediate_size: intermediate,
            gate_up_stride: stride,
        };
        encoder.set_bytes(
            2,
            std::mem::size_of::<SwiGluParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(
                (u64::from(rows) * u64::from(intermediate)).div_ceil(THREADS_PER_GROUP),
                1,
                1,
            ),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let actual = read_f16(&output, (rows * intermediate) as usize);
        for row in 0..rows as usize {
            for column in 0..intermediate as usize {
                let gate = packed[row * stride as usize + column].to_f32();
                let up = packed[row * stride as usize + intermediate as usize + column].to_f32();
                let expected = f16::from_f32(gate / (1.0 + (-gate).exp()) * up).to_f32();
                assert!((actual[row * intermediate as usize + column] - expected).abs() <= 0.002);
            }
        }
    }

    #[test]
    fn native_dense_swiglu_q4k_q6k_matches_full_cpu_oracle_on_real_metal() {
        let device = Device::system_default().expect("SwiGLU conformance requires Metal");
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let rows = 2_usize;
        let hidden = 256_usize;
        let intermediate = 256_usize;
        let input = (0..rows * hidden)
            .map(|index| f16::from_f32((index as f32 * 0.017).sin() * 0.125))
            .collect::<Vec<_>>();
        let gate_up = (0..2 * intermediate * hidden)
            .map(|index| (index as f32 * 0.0031).cos() * 0.0625)
            .collect::<Vec<_>>();
        let down = (0..hidden * intermediate)
            .map(|index| (index as f32 * 0.0043).sin() * 0.0625)
            .collect::<Vec<_>>();
        let cpu = CandleDevice::Cpu;
        let input_tensor = Tensor::from_vec(
            input.iter().map(|value| value.to_f32()).collect::<Vec<_>>(),
            (rows, hidden),
            &cpu,
        )
        .unwrap();
        let gate_up_tensor = Tensor::from_vec(gate_up, (2 * intermediate, hidden), &cpu).unwrap();
        let down_tensor = Tensor::from_vec(down, (hidden, intermediate), &cpu).unwrap();
        let gate_up_quantized = QTensor::quantize(&gate_up_tensor, GgmlDType::Q4K).unwrap();
        let down_quantized = QTensor::quantize(&down_tensor, GgmlDType::Q6K).unwrap();

        let cpu_gate_up = input_tensor
            .matmul(
                &gate_up_quantized
                    .dequantize(&cpu)
                    .unwrap()
                    .transpose(0, 1)
                    .unwrap(),
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut cpu_activated = vec![0.0_f32; rows * intermediate];
        for row in 0..rows {
            for column in 0..intermediate {
                let gate = cpu_gate_up[row * 2 * intermediate + column];
                let up = cpu_gate_up[row * 2 * intermediate + intermediate + column];
                cpu_activated[row * intermediate + column] = gate / (1.0 + (-gate).exp()) * up;
            }
        }
        let cpu_output = Tensor::from_vec(cpu_activated, (rows, intermediate), &cpu)
            .unwrap()
            .matmul(
                &down_quantized
                    .dequantize(&cpu)
                    .unwrap()
                    .transpose(0, 1)
                    .unwrap(),
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let input_buffer = shared_buffer(&device, &input);
        let gate_up_buffer = shared_buffer(&device, &gate_up_quantized.data().unwrap());
        let down_buffer = shared_buffer(&device, &down_quantized.data().unwrap());
        let projected = output_buffer::<f16>(&device, rows * 2 * intermediate);
        let activated = output_buffer::<f16>(&device, rows * intermediate);
        let output = output_buffer::<f16>(&device, rows * hidden);
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        dispatch_raw_linear(
            &pipelines,
            encoder,
            LinearPhysicalFormat::Q4K,
            &input_buffer,
            &gate_up_buffer,
            &projected,
            LinearParams {
                rows: rows as u32,
                in_features: hidden as u32,
                out_features: (2 * intermediate) as u32,
                output_stride: (2 * intermediate) as u32,
                output_column_offset: 0,
            },
        );
        encoder.set_compute_pipeline_state(&pipelines.swiglu);
        encoder.set_buffer(0, Some(&projected), 0);
        encoder.set_buffer(1, Some(&activated), 0);
        let swiglu = SwiGluParams {
            rows: rows as u32,
            intermediate_size: intermediate as u32,
            gate_up_stride: (2 * intermediate) as u32,
        };
        encoder.set_bytes(
            2,
            std::mem::size_of::<SwiGluParams>() as u64,
            &swiglu as *const _ as *const c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(
                (rows as u64 * intermediate as u64).div_ceil(THREADS_PER_GROUP),
                1,
                1,
            ),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
        dispatch_raw_linear(
            &pipelines,
            encoder,
            LinearPhysicalFormat::Q6K,
            &activated,
            &down_buffer,
            &output,
            LinearParams {
                rows: rows as u32,
                in_features: intermediate as u32,
                out_features: hidden as u32,
                output_stride: hidden as u32,
                output_column_offset: 0,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        let actual = read_f16(&output, rows * hidden);
        assert!(actual.iter().any(|value| value.abs() > 1.0e-5));
        numerical_tolerance::assert_matches(
            "Metal/CPU dense SwiGLU Q4_K/Q6_K",
            &actual,
            &[rows, hidden],
            &cpu_output,
            &[rows, hidden],
            numerical_tolerance::LogicalDtype::Fp16,
            DENSE_SWIGLU_TOLERANCE_ID,
        )
        .expect("reviewed dense-SwiGLU numerical contract");
    }

    #[test]
    fn native_last_token_q6k_linear_selects_final_row_on_real_metal() {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping last-token linear conformance");
            return;
        };
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let rows = 3_usize;
        let hidden = 256_usize;
        let output_width = 64_usize;
        let input = (0..rows * hidden)
            .map(|index| {
                let row = index / hidden;
                f16::from_f32((index as f32 * 0.011).sin() * 0.125 + row as f32 * 0.03125)
            })
            .collect::<Vec<_>>();
        let weight = (0..output_width * hidden)
            .map(|index| (index as f32 * 0.0071).cos() * 0.0625)
            .collect::<Vec<_>>();
        let cpu = CandleDevice::Cpu;
        let weight_tensor = Tensor::from_vec(weight, (output_width, hidden), &cpu).unwrap();
        let quantized = QTensor::quantize(&weight_tensor, GgmlDType::Q6K).unwrap();
        let dequantized = quantized.dequantize(&cpu).unwrap().transpose(0, 1).unwrap();
        let cpu_row = |row: usize| {
            Tensor::from_vec(
                input[row * hidden..(row + 1) * hidden]
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>(),
                (1, hidden),
                &cpu,
            )
            .unwrap()
            .matmul(&dequantized)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
        };
        let first_row = cpu_row(0);
        let final_row = cpu_row(rows - 1);
        assert!(first_row
            .iter()
            .zip(&final_row)
            .any(|(first, last)| (first - last).abs() > 1.0e-3));

        let input_buffer = shared_buffer(&device, &input);
        let weight_buffer = shared_buffer(&device, &quantized.data().unwrap());
        let output = output_buffer::<f16>(&device, output_width);
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        dispatch_raw_linear_at(
            &pipelines,
            encoder,
            LinearPhysicalFormat::Q6K,
            &input_buffer,
            ((rows - 1) * hidden * std::mem::size_of::<f16>()) as u64,
            &weight_buffer,
            &output,
            LinearParams {
                rows: 1,
                in_features: hidden as u32,
                out_features: output_width as u32,
                output_stride: output_width as u32,
                output_column_offset: 0,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        let actual = read_f16(&output, output_width);
        numerical_tolerance::assert_matches(
            "Metal/CPU last-token Q6_K dense linear",
            &actual,
            &[1, output_width],
            &final_row,
            &[1, output_width],
            numerical_tolerance::LogicalDtype::Fp16,
            LAST_TOKEN_LINEAR_TOLERANCE_ID,
        )
        .expect("reviewed last-token dense-linear numerical contract");
    }

    #[test]
    fn native_last_token_q6k_f32_linear_preserves_f32_head_boundary_on_real_metal() {
        let device =
            Device::system_default().expect("F32 last-token linear conformance requires Metal");
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let rows = 3_usize;
        let hidden = 256_usize;
        let output_width = 64_usize;
        let input = (0..rows * hidden)
            .map(|index| {
                let row = index / hidden;
                (index as f32 * 0.011).sin() * 0.125 + row as f32 * 0.03125 + 0.000_123
            })
            .collect::<Vec<_>>();
        let weight = (0..output_width * hidden)
            .map(|index| (index as f32 * 0.0071).cos() * 0.0625)
            .collect::<Vec<_>>();
        let cpu = CandleDevice::Cpu;
        let weight_tensor = Tensor::from_vec(weight, (output_width, hidden), &cpu).unwrap();
        let quantized = QTensor::quantize(&weight_tensor, GgmlDType::Q6K).unwrap();
        let reference = Tensor::from_vec(input[(rows - 1) * hidden..].to_vec(), (1, hidden), &cpu)
            .unwrap()
            .matmul(&quantized.dequantize(&cpu).unwrap().transpose(0, 1).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let input_buffer = shared_buffer(&device, &input);
        let weight_buffer = shared_buffer(&device, &quantized.data().unwrap());
        let output = output_buffer::<f32>(&device, output_width);
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let params = LinearParams {
            rows: 1,
            in_features: hidden as u32,
            out_features: output_width as u32,
            output_stride: output_width as u32,
            output_column_offset: 0,
        };
        encoder.set_compute_pipeline_state(
            pipelines
                .f32_linear_pipeline(LinearPhysicalFormat::Q6K)
                .unwrap(),
        );
        encoder.set_buffer(
            0,
            Some(&input_buffer),
            ((rows - 1) * hidden * std::mem::size_of::<f32>()) as u64,
        );
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<LinearParams>() as u64,
            &params as *const _ as *const c_void,
        );
        dispatch_linear_grid(encoder, params, LinearDispatchKind::CooperativeGemv);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        let actual = read_f32(&output, output_width);
        for (index, (actual, expected)) in actual.iter().zip(&reference).enumerate() {
            let tolerance = 2.0e-4_f32.max(expected.abs() * 2.0e-4);
            assert!(
                (actual - expected).abs() <= tolerance,
                "F32 Q6_K head[{index}] {actual} != {expected}"
            );
        }
    }

    #[test]
    fn native_packed_last_token_q6k_linear_gathers_and_scatters_on_real_metal() {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping packed last-token linear conformance");
            return;
        };
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let participant_count = 15_usize;
        let hidden = 256_usize;
        let output_width = 64_usize;
        let token_counts = (0..participant_count)
            .map(|participant| participant % 4 + 1)
            .collect::<Vec<_>>();
        let total_rows = token_counts.iter().sum::<usize>();
        let input = (0..total_rows * hidden)
            .map(|index| {
                let row = index / hidden;
                f16::from_f32((index as f32 * 0.011).sin() * 0.125 + row as f32 * 0.003)
            })
            .collect::<Vec<_>>();

        let mut final_row_offsets = Vec::with_capacity(participant_count);
        let mut selected_input = Vec::with_capacity(participant_count * hidden);
        let mut row_cursor = 0_usize;
        for token_count in token_counts {
            let final_row = row_cursor + token_count - 1;
            final_row_offsets.push(final_row * hidden * std::mem::size_of::<f16>());
            selected_input.extend(
                input[final_row * hidden..(final_row + 1) * hidden]
                    .iter()
                    .map(|value| value.to_f32()),
            );
            row_cursor += token_count;
        }

        let weight = (0..output_width * hidden)
            .map(|index| (index as f32 * 0.0071).cos() * 0.0625)
            .collect::<Vec<_>>();
        let cpu = CandleDevice::Cpu;
        let weight_tensor = Tensor::from_vec(weight, (output_width, hidden), &cpu).unwrap();
        let quantized = QTensor::quantize(&weight_tensor, GgmlDType::Q6K).unwrap();
        let reference = Tensor::from_vec(selected_input, (participant_count, hidden), &cpu)
            .unwrap()
            .matmul(&quantized.dequantize(&cpu).unwrap().transpose(0, 1).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let input_buffer = shared_buffer(&device, &input);
        let weight_buffer = shared_buffer(&device, &quantized.data().unwrap());
        let layout = LastTokenPackedScratchLayout::new(
            participant_count as u64,
            hidden as u64,
            output_width as u64,
            ElementType::F16,
        )
        .unwrap();
        let scratch = output_buffer::<u8>(&device, layout.required_bytes as usize);
        let sentinel = f16::from_f32(-37.0);
        let outputs = (0..participant_count)
            .map(|_| shared_buffer(&device, &vec![sentinel; output_width + 2]))
            .collect::<Vec<_>>();

        let command = queue.new_command_buffer();
        let gather = command.new_blit_command_encoder();
        for (participant, source_offset) in final_row_offsets.into_iter().enumerate() {
            gather.copy_from_buffer(
                &input_buffer,
                source_offset as u64,
                &scratch,
                participant as u64 * layout.input_row_bytes,
                layout.input_row_bytes,
            );
        }
        gather.end_encoding();

        let encoder = command.new_compute_command_encoder();
        let (_, dispatch_kind) = pipelines.linear_pipeline(
            LinearPhysicalFormat::Q6K,
            participant_count as u32,
            output_width as u32,
        );
        assert_eq!(dispatch_kind, LinearDispatchKind::TiledGemm);
        dispatch_raw_linear_with_offsets(
            &pipelines,
            encoder,
            LinearPhysicalFormat::Q6K,
            &scratch,
            0,
            &weight_buffer,
            &scratch,
            layout.output_offset_bytes,
            LinearParams {
                rows: participant_count as u32,
                in_features: hidden as u32,
                out_features: output_width as u32,
                output_stride: output_width as u32,
                output_column_offset: 0,
            },
        );
        encoder.end_encoding();

        let scatter = command.new_blit_command_encoder();
        for (participant, output) in outputs.iter().enumerate() {
            scatter.copy_from_buffer(
                &scratch,
                layout.output_offset_bytes + participant as u64 * layout.output_row_bytes,
                output,
                std::mem::size_of::<f16>() as u64,
                layout.output_row_bytes,
            );
        }
        scatter.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        let mut actual = Vec::with_capacity(participant_count * output_width);
        for output in &outputs {
            let guarded = read_f16(output, output_width + 2);
            assert_eq!(guarded[0], sentinel.to_f32());
            assert_eq!(guarded[output_width + 1], sentinel.to_f32());
            actual.extend_from_slice(&guarded[1..output_width + 1]);
        }
        numerical_tolerance::assert_matches(
            "Metal/CPU packed last-token Q6_K dense linear",
            &actual,
            &[participant_count, output_width],
            &reference,
            &[participant_count, output_width],
            numerical_tolerance::LogicalDtype::Fp16,
            PACKED_LAST_TOKEN_LINEAR_TOLERANCE_ID,
        )
        .expect("reviewed packed last-token dense-linear numerical contract");
    }

    fn assert_linear_diagnostic_close(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 0.02_f32.max(expected.abs() * 0.01);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{label}[{index}] {actual} != {expected}"
            );
        }
    }

    fn dispatch_raw_linear(
        pipelines: &MetalLinearPipelines,
        encoder: &ComputeCommandEncoderRef,
        format: LinearPhysicalFormat,
        input: &BufferRef,
        weight: &BufferRef,
        output: &BufferRef,
        params: LinearParams,
    ) {
        dispatch_raw_linear_at(pipelines, encoder, format, input, 0, weight, output, params);
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_raw_linear_at(
        pipelines: &MetalLinearPipelines,
        encoder: &ComputeCommandEncoderRef,
        format: LinearPhysicalFormat,
        input: &BufferRef,
        input_offset_bytes: u64,
        weight: &BufferRef,
        output: &BufferRef,
        params: LinearParams,
    ) {
        dispatch_raw_linear_with_offsets(
            pipelines,
            encoder,
            format,
            input,
            input_offset_bytes,
            weight,
            output,
            0,
            params,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_raw_linear_with_offsets(
        pipelines: &MetalLinearPipelines,
        encoder: &ComputeCommandEncoderRef,
        format: LinearPhysicalFormat,
        input: &BufferRef,
        input_offset_bytes: u64,
        weight: &BufferRef,
        output: &BufferRef,
        output_offset_bytes: u64,
        params: LinearParams,
    ) {
        let (pipeline, dispatch_kind) =
            pipelines.linear_pipeline(format, params.rows, params.out_features);
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input), input_offset_bytes);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), output_offset_bytes);
        bind_linear_params(encoder, params, format, ElementType::F16);
        dispatch_linear_grid(encoder, params, dispatch_kind);
    }
}

/// Append evidence from the exact immutable linear selector used by this command.
pub(super) fn append_selected_projection(
    builder: &mut ferrum_interfaces::execution_cost::SelectedCommandCostBuilderV1,
    pipelines: &MetalLinearPipelines,
    launch: LinearLaunch,
    policy: Option<staged_prefill::StagingPolicy>,
    scratch: u64,
) -> Option<()> {
    selected::projection(builder, pipelines, launch, policy, scratch)
}
