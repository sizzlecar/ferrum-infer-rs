//! Native F16 Metal providers for the smallest standard transformer ops.

use std::ffi::c_void;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    last_token_masked_argmax_contract, last_token_masked_argmax_f32_contract,
    residual_add_contract, residual_add_f32_f16_contract, rms_norm_contract, rms_norm_f32_contract,
    rms_norm_f32_to_f16_contract, token_embedding_contract, token_embedding_f32_master_contract,
    BatchedOperationInvocation, CheckpointBoundaryConstraint, CheckpointCompletedInputCapture,
    CheckpointInputDependency, CheckpointPartitionNumerics, DynamicStorageRequirement, ElementType,
    EncodedDeviceOperation, OperationCostRoute, OperationCostRouteRequest, OperationFailure,
    OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, PhysicalWeightPadding,
    ProviderCheckpointCapability, ProviderCheckpointContract, ProviderWorkspaceRequirement,
    ProviderWorkspaceReusePolicy, ProviderWorkspaceScope, ProviderWorkspaceSizeFormula,
    ResolvedTensorLayout, ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, VNextError, WeightEncoding,
    LAST_TOKEN_MASKED_ARGMAX_F16_CAPABILITY_ID, LAST_TOKEN_MASKED_ARGMAX_F32_CAPABILITY_ID,
    LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID, LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID,
    RESIDUAL_ADD_F16_CAPABILITY_ID, RESIDUAL_ADD_F32_F16_CAPABILITY_ID,
    RESIDUAL_ADD_F32_F16_OPERATION_ID, RESIDUAL_ADD_OPERATION_ID, RMS_NORM_F16_CAPABILITY_ID,
    RMS_NORM_F32_CAPABILITY_ID, RMS_NORM_F32_OPERATION_ID, RMS_NORM_F32_TO_F16_CAPABILITY_ID,
    RMS_NORM_F32_TO_F16_OPERATION_ID, RMS_NORM_OPERATION_ID, TOKEN_EMBEDDING_F16_CAPABILITY_ID,
    TOKEN_EMBEDDING_F32_MASTER_CAPABILITY_ID, TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID,
    TOKEN_EMBEDDING_OPERATION_ID,
};
use metal::{
    CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, FunctionConstantValues,
    MTLDataType, MTLSize,
};

use crate::gguf_blocks::GgufBlockFormat;

use super::super::vnext_runtime::{
    MetalBufferRegion, MetalDeviceBuffer, MetalDeviceCommand, MetalDeviceRuntime,
    MetalDeviceRuntimeError,
};
use super::hadamard::{self, HadamardTransform, MetalHadamardPipelines};
use super::weights::{resolve_weight, validate_hadamard_transform};
use super::{
    authorize_reusable_topology, binding, checked_u32, contiguous_bindings, contiguous_region,
    contiguous_token_region, ensure_invocation, estimate_without_workspace, f16_contiguous,
    implementation_fingerprint, invalid_plan, provider_descriptor, provider_failure,
    rational_attribute, shared_full_region, shared_scratch_region, shared_token_region,
    token_binding_is_packed, unsigned_attribute, DENSE_SAFETENSORS_FORMAT_ID,
    GGUF_NATIVE_BLOCK_FORMAT_ID, Q4_K_FORMAT_ID, Q6_K_FORMAT_ID, Q8_0_FORMAT_ID, THREADS_PER_GROUP,
    VALUE_ALIGNMENT_BYTES,
};

mod cost_route;
mod selected;
use cost_route::PrimitiveRoute;

const SHADER_SOURCE: &str = include_str!("primitives.metal");
const TOKEN_EMBEDDING_PROVIDER_ID: &str = "provider.metal.token_embedding.f16";
const TOKEN_EMBEDDING_ESTIMATOR_ID: &str = "resource-estimator.metal.token_embedding.f16";
const RMS_NORM_PROVIDER_ID: &str = "provider.metal.rms_norm.f16";
const RMS_NORM_ESTIMATOR_ID: &str = "resource-estimator.metal.rms_norm.f16";
const RESIDUAL_ADD_PROVIDER_ID: &str = "provider.metal.residual_add.f16";
const RESIDUAL_ADD_ESTIMATOR_ID: &str = "resource-estimator.metal.residual_add.f16";
const LAST_TOKEN_MASKED_ARGMAX_PROVIDER_ID: &str = "provider.metal.last_token_masked_argmax.f16";
const LAST_TOKEN_MASKED_ARGMAX_ESTIMATOR_ID: &str =
    "resource-estimator.metal.last_token_masked_argmax.f16";
const TOKEN_EMBEDDING_F32_MASTER_PROVIDER_ID: &str = "provider.metal.token_embedding.f32-master";
const TOKEN_EMBEDDING_F32_MASTER_ESTIMATOR_ID: &str =
    "resource-estimator.metal.token_embedding.f32-master";
const RMS_NORM_F32_TO_F16_PROVIDER_ID: &str = "provider.metal.rms_norm.f32-to-f16";
const RMS_NORM_F32_TO_F16_ESTIMATOR_ID: &str = "resource-estimator.metal.rms_norm.f32-to-f16";
const RMS_NORM_F32_PROVIDER_ID: &str = "provider.metal.rms_norm.f32";
const RMS_NORM_F32_ESTIMATOR_ID: &str = "resource-estimator.metal.rms_norm.f32";
const RESIDUAL_ADD_F32_F16_PROVIDER_ID: &str = "provider.metal.residual_add.f32-f16";
const RESIDUAL_ADD_F32_F16_ESTIMATOR_ID: &str = "resource-estimator.metal.residual_add.f32-f16";
const LAST_TOKEN_MASKED_ARGMAX_F32_PROVIDER_ID: &str =
    "provider.metal.last_token_masked_argmax.f32";
const LAST_TOKEN_MASKED_ARGMAX_F32_ESTIMATOR_ID: &str =
    "resource-estimator.metal.last_token_masked_argmax.f32";

const PQ2_0_FORMAT_ID: &str = GgufBlockFormat::Pq2_0.format_id();
const TOKEN_EMBEDDING_QUANTIZATION_FORMATS: &[&str] = &[
    Q4_K_FORMAT_ID,
    Q6_K_FORMAT_ID,
    Q8_0_FORMAT_ID,
    PQ2_0_FORMAT_ID,
];
const EMBEDDING_DENSE_KERNEL: &str = "vnext_embedding_dense_f16";
const EMBEDDING_Q4_K_KERNEL: &str = "vnext_embedding_q4_k_f16";
const EMBEDDING_Q6_K_KERNEL: &str = "vnext_embedding_q6_k_f16";
const EMBEDDING_Q8_0_KERNEL: &str = "vnext_embedding_q8_0_f16";
const EMBEDDING_PQ2_0_KERNEL: &str = "vnext_embedding_pq2_0_f16";
const RMS_NORM_KERNEL: &str = "vnext_rms_norm_f16";
const RESIDUAL_ADD_KERNEL: &str = "vnext_residual_add_f16";
const LAST_TOKEN_MASKED_ARGMAX_KERNEL: &str = "vnext_last_token_masked_argmax_f16";
const EMBEDDING_DENSE_F32_KERNEL: &str = "vnext_embedding_dense_f32";
const EMBEDDING_Q4_K_F32_KERNEL: &str = "vnext_embedding_q4_k_f32";
const EMBEDDING_Q6_K_F32_KERNEL: &str = "vnext_embedding_q6_k_f32";
const EMBEDDING_Q8_0_F32_KERNEL: &str = "vnext_embedding_q8_0_f32";
const EMBEDDING_PQ2_0_F32_KERNEL: &str = "vnext_embedding_pq2_0_f32";
const RMS_NORM_F32_TO_F16_KERNEL: &str = "vnext_rms_norm_f32_to_f16";
const RMS_NORM_F32_KERNEL: &str = "vnext_rms_norm_f32";
const RESIDUAL_ADD_F32_F16_KERNEL: &str = "vnext_residual_add_f32_f16";
const LAST_TOKEN_MASKED_ARGMAX_F32_KERNEL: &str = "vnext_last_token_masked_argmax_f32";
const MASKED_ARGMAX_PARTITIONS: u64 = 32;
const MASKED_ARGMAX_PARALLEL_MIN_VOCAB: u32 = 8192;

pub(super) struct MetalPrimitivePipelines {
    structured_capture: ferrum_types::SloStructuredCostCapture,
    hadamard: MetalHadamardPipelines,
    embedding_dense: ComputePipelineState,
    embedding_q4_k: ComputePipelineState,
    embedding_q6_k: ComputePipelineState,
    embedding_q8_0: ComputePipelineState,
    embedding_pq2_0: ComputePipelineState,
    rms_norm: ComputePipelineState,
    residual_add: ComputePipelineState,
    last_token_masked_argmax: ComputePipelineState,
    embedding_dense_f32: ComputePipelineState,
    embedding_q4_k_f32: ComputePipelineState,
    embedding_q6_k_f32: ComputePipelineState,
    embedding_q8_0_f32: ComputePipelineState,
    embedding_pq2_0_f32: ComputePipelineState,
    rms_norm_f32_to_f16: ComputePipelineState,
    rms_norm_f32: ComputePipelineState,
    residual_add_f32_f16: ComputePipelineState,
    last_token_masked_argmax_f32: ComputePipelineState,
    parallel_masked_argmax: ComputePipelineState,
    parallel_masked_argmax_f32: ComputePipelineState,
    masked_argmax_finalize: ComputePipelineState,
}

impl MetalPrimitivePipelines {
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
                    "compile Metal vNext primitive library: {error}"
                ))
            })?;
        let specialized_pipeline = |name: &str, parallel: bool| {
            let constants = matches!(
                name,
                LAST_TOKEN_MASKED_ARGMAX_KERNEL | LAST_TOKEN_MASKED_ARGMAX_F32_KERNEL
            )
            .then(|| {
                let constants = FunctionConstantValues::new();
                constants.set_constant_value_at_index(
                    &parallel as *const bool as *const c_void,
                    MTLDataType::Bool,
                    0,
                );
                constants
            });
            let function = library.get_function(name, constants).map_err(|error| {
                MetalDeviceRuntimeError::contract(format!(
                    "load Metal vNext primitive `{name}`: {error}"
                ))
            })?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| {
                    MetalDeviceRuntimeError::contract(format!(
                        "build Metal vNext primitive `{name}`: {error}"
                    ))
                })
        };
        let pipeline = |name: &str| specialized_pipeline(name, false);
        Ok(Self {
            structured_capture: ferrum_types::SloStructuredCostCapture::Disabled,
            hadamard: MetalHadamardPipelines::new(device)?,
            embedding_dense: pipeline(EMBEDDING_DENSE_KERNEL)?,
            embedding_q4_k: pipeline(EMBEDDING_Q4_K_KERNEL)?,
            embedding_q6_k: pipeline(EMBEDDING_Q6_K_KERNEL)?,
            embedding_q8_0: pipeline(EMBEDDING_Q8_0_KERNEL)?,
            embedding_pq2_0: pipeline(EMBEDDING_PQ2_0_KERNEL)?,
            rms_norm: pipeline(RMS_NORM_KERNEL)?,
            residual_add: pipeline(RESIDUAL_ADD_KERNEL)?,
            last_token_masked_argmax: pipeline(LAST_TOKEN_MASKED_ARGMAX_KERNEL)?,
            embedding_dense_f32: pipeline(EMBEDDING_DENSE_F32_KERNEL)?,
            embedding_q4_k_f32: pipeline(EMBEDDING_Q4_K_F32_KERNEL)?,
            embedding_q6_k_f32: pipeline(EMBEDDING_Q6_K_F32_KERNEL)?,
            embedding_q8_0_f32: pipeline(EMBEDDING_Q8_0_F32_KERNEL)?,
            embedding_pq2_0_f32: pipeline(EMBEDDING_PQ2_0_F32_KERNEL)?,
            rms_norm_f32_to_f16: pipeline(RMS_NORM_F32_TO_F16_KERNEL)?,
            rms_norm_f32: pipeline(RMS_NORM_F32_KERNEL)?,
            residual_add_f32_f16: pipeline(RESIDUAL_ADD_F32_F16_KERNEL)?,
            last_token_masked_argmax_f32: pipeline(LAST_TOKEN_MASKED_ARGMAX_F32_KERNEL)?,
            parallel_masked_argmax: specialized_pipeline(LAST_TOKEN_MASKED_ARGMAX_KERNEL, true)?,
            parallel_masked_argmax_f32: specialized_pipeline(
                LAST_TOKEN_MASKED_ARGMAX_F32_KERNEL,
                true,
            )?,
            masked_argmax_finalize: pipeline("vnext_masked_argmax_finalize")?,
        })
    }
}

pub(super) struct MetalTokenEmbeddingProvider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<MetalPrimitivePipelines>,
}

impl MetalTokenEmbeddingProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalPrimitivePipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = token_embedding_contract().map_err(super::contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            TOKEN_EMBEDDING_PROVIDER_ID,
            TOKEN_EMBEDDING_F16_CAPABILITY_ID,
            TOKEN_EMBEDDING_ESTIMATOR_ID,
            contiguous_bindings(2),
            &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID],
            TOKEN_EMBEDDING_QUANTIZATION_FORMATS,
            implementation_fingerprint(&[
                include_str!("primitives.rs").as_bytes(),
                include_str!("primitives/cost_route.rs").as_bytes(),
                include_str!("primitives/selected.rs").as_bytes(),
                SHADER_SOURCE.as_bytes(),
                hadamard::FINGERPRINT_SOURCE.as_bytes(),
                TOKEN_EMBEDDING_PROVIDER_ID.as_bytes(),
            ]),
        )?
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

impl OperationResourceEstimator for MetalTokenEmbeddingProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        hadamard::estimate_token_workspace(&self.descriptor, &request, TOKEN_EMBEDDING_OPERATION_ID)
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalTokenEmbeddingProvider {
    fn eager_cost_route(
        &self,
        request: OperationCostRouteRequest<'_>,
    ) -> Result<Option<OperationCostRoute>, VNextError> {
        cost_route::eager_route(request, &self.pipelines)
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
        encode_token_embedding(Arc::clone(&self.pipelines), invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| provider_failure(identity, "metal.token_embedding.encode", message))
    }
}

pub(super) struct MetalRmsNormProvider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<MetalPrimitivePipelines>,
}

impl MetalRmsNormProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalPrimitivePipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = rms_norm_contract().map_err(super::contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            RMS_NORM_PROVIDER_ID,
            RMS_NORM_F16_CAPABILITY_ID,
            RMS_NORM_ESTIMATOR_ID,
            contiguous_bindings(2),
            &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID],
            &[],
            implementation_fingerprint(&[
                include_str!("primitives.rs").as_bytes(),
                include_str!("primitives/cost_route.rs").as_bytes(),
                include_str!("primitives/selected.rs").as_bytes(),
                SHADER_SOURCE.as_bytes(),
                RMS_NORM_PROVIDER_ID.as_bytes(),
            ]),
        )?;
        Ok(Self {
            descriptor,
            pipelines,
        })
    }
}

impl OperationResourceEstimator for MetalRmsNormProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        estimate_without_workspace(&self.descriptor, &request, RMS_NORM_OPERATION_ID)
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalRmsNormProvider {
    fn eager_cost_route(
        &self,
        request: OperationCostRouteRequest<'_>,
    ) -> Result<Option<OperationCostRoute>, VNextError> {
        cost_route::eager_route(request, &self.pipelines)
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
        encode_rms_norm(Arc::clone(&self.pipelines), invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| provider_failure(identity, "metal.rms_norm.encode", message))
    }
}

pub(super) struct MetalResidualAddProvider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<MetalPrimitivePipelines>,
}

impl MetalResidualAddProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalPrimitivePipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = residual_add_contract().map_err(super::contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            RESIDUAL_ADD_PROVIDER_ID,
            RESIDUAL_ADD_F16_CAPABILITY_ID,
            RESIDUAL_ADD_ESTIMATOR_ID,
            contiguous_bindings(2),
            &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID],
            &[],
            implementation_fingerprint(&[
                include_str!("primitives.rs").as_bytes(),
                include_str!("primitives/cost_route.rs").as_bytes(),
                include_str!("primitives/selected.rs").as_bytes(),
                SHADER_SOURCE.as_bytes(),
                RESIDUAL_ADD_PROVIDER_ID.as_bytes(),
            ]),
        )?;
        Ok(Self {
            descriptor,
            pipelines,
        })
    }
}

impl OperationResourceEstimator for MetalResidualAddProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        estimate_without_workspace(&self.descriptor, &request, RESIDUAL_ADD_OPERATION_ID)
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalResidualAddProvider {
    fn eager_cost_route(
        &self,
        request: OperationCostRouteRequest<'_>,
    ) -> Result<Option<OperationCostRoute>, VNextError> {
        cost_route::eager_route(request, &self.pipelines)
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
        encode_residual_add(Arc::clone(&self.pipelines), invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| provider_failure(identity, "metal.residual_add.encode", message))
    }
}

pub(super) struct MetalLastTokenMaskedArgmaxProvider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<MetalPrimitivePipelines>,
}

impl MetalLastTokenMaskedArgmaxProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalPrimitivePipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = last_token_masked_argmax_contract().map_err(super::contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            LAST_TOKEN_MASKED_ARGMAX_PROVIDER_ID,
            LAST_TOKEN_MASKED_ARGMAX_F16_CAPABILITY_ID,
            LAST_TOKEN_MASKED_ARGMAX_ESTIMATOR_ID,
            contiguous_bindings(5),
            &[],
            &[],
            implementation_fingerprint(&[
                include_str!("primitives.rs").as_bytes(),
                include_str!("primitives/cost_route.rs").as_bytes(),
                include_str!("primitives/selected.rs").as_bytes(),
                SHADER_SOURCE.as_bytes(),
                LAST_TOKEN_MASKED_ARGMAX_PROVIDER_ID.as_bytes(),
            ]),
        )?;
        Ok(Self {
            descriptor,
            pipelines,
        })
    }
}

impl OperationResourceEstimator for MetalLastTokenMaskedArgmaxProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        estimate_masked_argmax_resources(
            &self.descriptor,
            request,
            LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID,
            ElementType::F16,
        )
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalLastTokenMaskedArgmaxProvider {
    fn eager_cost_route(
        &self,
        request: OperationCostRouteRequest<'_>,
    ) -> Result<Option<OperationCostRoute>, VNextError> {
        cost_route::eager_route(request, &self.pipelines)
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
        encode_last_token_masked_argmax(Arc::clone(&self.pipelines), invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| {
                provider_failure(identity, "metal.last_token_masked_argmax.encode", message)
            })
    }
}

macro_rules! no_workspace_primitive_provider {
    (
        $provider:ident,
        $contract:ident,
        $provider_id:ident,
        $capability_id:ident,
        $estimator_id:ident,
        $operation_id:ident,
        $bindings:expr,
        $physical_formats:expr,
        $quantization_formats:expr,
        $checkpoint:expr,
        $encode:ident,
        $failure_stage:literal
    ) => {
        pub(super) struct $provider {
            descriptor: OperationProviderDescriptor,
            pipelines: Arc<MetalPrimitivePipelines>,
        }

        impl $provider {
            pub(super) fn new(
                runtime: &MetalDeviceRuntime,
                pipelines: Arc<MetalPrimitivePipelines>,
            ) -> Result<Self, MetalDeviceRuntimeError> {
                let contract = $contract().map_err(super::contract_error)?;
                let descriptor = provider_descriptor(
                    runtime,
                    &contract,
                    $provider_id,
                    $capability_id,
                    $estimator_id,
                    contiguous_bindings($bindings),
                    $physical_formats,
                    $quantization_formats,
                    implementation_fingerprint(&[
                        include_str!("primitives.rs").as_bytes(),
                        include_str!("primitives/cost_route.rs").as_bytes(),
                        include_str!("primitives/selected.rs").as_bytes(),
                        SHADER_SOURCE.as_bytes(),
                        hadamard::FINGERPRINT_SOURCE.as_bytes(),
                        $provider_id.as_bytes(),
                    ]),
                )?
                .with_checkpoint_capability($checkpoint);
                Ok(Self {
                    descriptor,
                    pipelines,
                })
            }
        }

        impl OperationResourceEstimator for $provider {
            fn descriptor(&self) -> &OperationProviderDescriptor {
                &self.descriptor
            }

            fn estimate_resources(
                &self,
                request: OperationResourceEstimateRequest<'_>,
            ) -> Result<OperationResourceEstimate, VNextError> {
                if $operation_id == TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID {
                    hadamard::estimate_token_workspace(&self.descriptor, &request, $operation_id)
                } else {
                    estimate_without_workspace(&self.descriptor, &request, $operation_id)
                }
            }
        }

        impl OperationProvider<MetalDeviceRuntime> for $provider {
            fn eager_cost_route(
                &self,
                request: OperationCostRouteRequest<'_>,
            ) -> Result<Option<OperationCostRoute>, VNextError> {
                cost_route::eager_route(request, &self.pipelines)
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
                $encode(Arc::clone(&self.pipelines), invocation)
                    .map(EncodedDeviceOperation::compute)
                    .map_err(|message| provider_failure(identity, $failure_stage, message))
            }
        }
    };
}

no_workspace_primitive_provider!(
    MetalTokenEmbeddingF32MasterProvider,
    token_embedding_f32_master_contract,
    TOKEN_EMBEDDING_F32_MASTER_PROVIDER_ID,
    TOKEN_EMBEDDING_F32_MASTER_CAPABILITY_ID,
    TOKEN_EMBEDDING_F32_MASTER_ESTIMATOR_ID,
    TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID,
    2,
    &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID],
    TOKEN_EMBEDDING_QUANTIZATION_FORMATS,
    ProviderCheckpointCapability::CompletedBoundary(
        ProviderCheckpointContract::new(
            CheckpointInputDependency::ExactTokenPrefix,
            CheckpointBoundaryConstraint::any_positive(),
            CheckpointPartitionNumerics::CapturedExecutionContinuation,
        )
        .with_completed_input_capture(CheckpointCompletedInputCapture::Supported)
    ),
    encode_token_embedding_f32_master,
    "metal.token_embedding_f32_master.encode"
);
no_workspace_primitive_provider!(
    MetalRmsNormF32ToF16Provider,
    rms_norm_f32_to_f16_contract,
    RMS_NORM_F32_TO_F16_PROVIDER_ID,
    RMS_NORM_F32_TO_F16_CAPABILITY_ID,
    RMS_NORM_F32_TO_F16_ESTIMATOR_ID,
    RMS_NORM_F32_TO_F16_OPERATION_ID,
    2,
    &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID],
    &[],
    ProviderCheckpointCapability::CompletedBoundary(
        ProviderCheckpointContract::new(
            CheckpointInputDependency::ExactTokenPrefix,
            CheckpointBoundaryConstraint::any_positive(),
            CheckpointPartitionNumerics::CapturedExecutionContinuation,
        )
        .with_completed_input_capture(CheckpointCompletedInputCapture::Supported)
    ),
    encode_rms_norm_f32_to_f16,
    "metal.rms_norm_f32_to_f16.encode"
);
no_workspace_primitive_provider!(
    MetalRmsNormF32Provider,
    rms_norm_f32_contract,
    RMS_NORM_F32_PROVIDER_ID,
    RMS_NORM_F32_CAPABILITY_ID,
    RMS_NORM_F32_ESTIMATOR_ID,
    RMS_NORM_F32_OPERATION_ID,
    2,
    &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID],
    &[],
    ProviderCheckpointCapability::CompletedBoundary(
        ProviderCheckpointContract::new(
            CheckpointInputDependency::ExactTokenPrefix,
            CheckpointBoundaryConstraint::any_positive(),
            CheckpointPartitionNumerics::CapturedExecutionContinuation,
        )
        .with_completed_input_capture(CheckpointCompletedInputCapture::Supported)
    ),
    encode_rms_norm_f32,
    "metal.rms_norm_f32.encode"
);
no_workspace_primitive_provider!(
    MetalResidualAddF32F16Provider,
    residual_add_f32_f16_contract,
    RESIDUAL_ADD_F32_F16_PROVIDER_ID,
    RESIDUAL_ADD_F32_F16_CAPABILITY_ID,
    RESIDUAL_ADD_F32_F16_ESTIMATOR_ID,
    RESIDUAL_ADD_F32_F16_OPERATION_ID,
    2,
    &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID],
    &[],
    ProviderCheckpointCapability::CompletedBoundary(
        ProviderCheckpointContract::new(
            CheckpointInputDependency::ExactTokenPrefix,
            CheckpointBoundaryConstraint::any_positive(),
            CheckpointPartitionNumerics::CapturedExecutionContinuation,
        )
        .with_completed_input_capture(CheckpointCompletedInputCapture::Supported)
    ),
    encode_residual_add_f32_f16,
    "metal.residual_add_f32_f16.encode"
);

pub(super) struct MetalLastTokenMaskedArgmaxF32Provider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<MetalPrimitivePipelines>,
}

impl MetalLastTokenMaskedArgmaxF32Provider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalPrimitivePipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = last_token_masked_argmax_f32_contract().map_err(super::contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            LAST_TOKEN_MASKED_ARGMAX_F32_PROVIDER_ID,
            LAST_TOKEN_MASKED_ARGMAX_F32_CAPABILITY_ID,
            LAST_TOKEN_MASKED_ARGMAX_F32_ESTIMATOR_ID,
            contiguous_bindings(5),
            &[],
            &[],
            implementation_fingerprint(&[
                include_str!("primitives.rs").as_bytes(),
                include_str!("primitives/cost_route.rs").as_bytes(),
                include_str!("primitives/selected.rs").as_bytes(),
                SHADER_SOURCE.as_bytes(),
                LAST_TOKEN_MASKED_ARGMAX_F32_PROVIDER_ID.as_bytes(),
            ]),
        )?
        // The output-policy inputs affect only this output, not retained state.
        // Scratch is invocation-scoped and copied from logits before use;
        // unique repetition IDs are required by the owning operation contract.
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

impl OperationResourceEstimator for MetalLastTokenMaskedArgmaxF32Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        estimate_masked_argmax_resources(
            &self.descriptor,
            request,
            LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID,
            ElementType::F32,
        )
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalLastTokenMaskedArgmaxF32Provider {
    fn eager_cost_route(
        &self,
        request: OperationCostRouteRequest<'_>,
    ) -> Result<Option<OperationCostRoute>, VNextError> {
        cost_route::eager_route(request, &self.pipelines)
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
        encode_last_token_masked_argmax_f32(Arc::clone(&self.pipelines), invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| {
                provider_failure(
                    identity,
                    "metal.last_token_masked_argmax_f32.encode",
                    message,
                )
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbeddingPhysicalFormat {
    DenseF16,
    Q4K,
    Q6K,
    Q8_0,
    Pq2_0,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct EmbeddingParams {
    token_count: u32,
    hidden_size: u32,
    vocabulary_size: u32,
}

#[derive(Debug, Clone, Copy)]
struct EmbeddingLaunch {
    table_region: usize,
    tokens_region: usize,
    output_region: usize,
    transform: Option<HadamardTransform>,
    scratch_offset_bytes: u64,
    format: EmbeddingPhysicalFormat,
    params: EmbeddingParams,
}

fn encode_token_embedding(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    encode_token_embedding_typed(
        pipelines,
        invocation,
        TOKEN_EMBEDDING_OPERATION_ID,
        ElementType::F16,
    )
}

fn encode_token_embedding_f32_master(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    encode_token_embedding_typed(
        pipelines,
        invocation,
        TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID,
        ElementType::F32,
    )
}

fn encode_token_embedding_typed(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    operation_id: &'static str,
    output_type: ElementType,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, operation_id)?;
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("Metal token embedding participant ranges are incomplete".to_owned());
    }
    let input_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let mut regions = Vec::with_capacity(invocation.participants().len() * 3);
    let mut launches = Vec::with_capacity(invocation.participants().len());
    let mut scratch_bytes = 0_u64;
    for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
        let token_ids = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let table = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        let hidden_size = unsigned_attribute(participant.attributes(), "hidden_size")?;
        let vocabulary_size = unsigned_attribute(participant.attributes(), "vocab_size")?;
        validate_embedding_signature(
            token_ids,
            table,
            output,
            vocabulary_size,
            hidden_size,
            output_type,
        )?;
        let (format, table_component, transform) =
            cost_route::embedding_weight_metadata(table, vocabulary_size, hidden_size)?;
        let weight = resolve_weight(participant, table)?;
        if let Some(transform) = transform {
            validate_hadamard_transform(
                transform,
                hidden_size,
                weight.components(),
                weight.regions(),
            )?;
        }
        let (mut table_regions, _, _) = weight.into_command_parts();
        let first_region = regions.len();
        let transform = transform
            .map(|transform| transform.relocate(first_region))
            .transpose()?;
        let scratch_offset_bytes = scratch_bytes;
        if let Some(transform) = transform {
            pipelines.hadamard.validate_dispatch(
                transform,
                checked_u32(hidden_size, "Metal embedding hidden size")?,
                ElementType::F32,
                output_type,
            )?;
            scratch_bytes = cost_route::embedding_scratch_bytes(
                scratch_bytes,
                token_range.immediate_tokens(),
                hidden_size,
            )?;
        }
        regions.append(&mut table_regions);
        let tokens_region = regions.len();
        regions.push(contiguous_token_region(
            participant,
            token_ids,
            ElementType::U32,
            if input_packed {
                token_range.immediate_token_range().start
            } else {
                token_range.source_token_range().start
            },
            token_range.immediate_tokens(),
        )?);
        let output_region = regions.len();
        regions.push(contiguous_token_region(
            participant,
            output,
            output_type,
            token_range.immediate_token_range().start,
            token_range.immediate_tokens(),
        )?);
        launches.push(EmbeddingLaunch {
            table_region: first_region + table_component,
            tokens_region,
            output_region,
            transform,
            scratch_offset_bytes,
            format,
            params: cost_route::embedding_params(
                token_range.immediate_tokens(),
                hidden_size,
                vocabulary_size,
            )?,
        });
    }
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "Metal embedding participant count",
    )?;
    let token_count = invocation.work_shape().immediate_tokens();
    let scratch_region = if scratch_bytes == 0 {
        None
    } else {
        let index = regions.len();
        regions.push(shared_scratch_region(&invocation, scratch_bytes)?);
        Some(index)
    };
    let transformed_participants = launches
        .iter()
        .filter(|launch| launch.transform.is_some())
        .count() as u32;
    let route = cost_route::compute_command(
        PrimitiveRoute::TokenEmbedding {
            transformed_participants,
        },
        participant_count,
        token_count,
    )
    .map_err(|error| error.to_string())?;
    let statistical = selected::embedding_evidence(&pipelines, &launches, output_type, token_count);
    let dispatch_count = route.compute_dispatch_count();
    let batching = route.batching();
    MetalDeviceCommand::operation(
        route.native_operation(),
        regions,
        move |encoder, regions| {
            encoder.record_compute_dispatches(dispatch_count);
            let compute = encoder.compute_encoder();
            for launch in &launches {
                let output = &regions[launch.output_region];
                if let Some(transform) = launch.transform {
                    let scratch =
                        &regions[scratch_region.expect("validated embedding inverse workspace")];
                    dispatch_embedding_at(
                        &pipelines,
                        compute,
                        launch.format,
                        &regions[launch.table_region],
                        &regions[launch.tokens_region],
                        scratch,
                        launch.scratch_offset_bytes,
                        launch.params,
                        ElementType::F32,
                    );
                    pipelines.hadamard.dispatch(
                        compute,
                        transform,
                        scratch,
                        launch.scratch_offset_bytes,
                        ElementType::F32,
                        output,
                        0,
                        output_type,
                        regions,
                        launch.params.token_count,
                        launch.params.hidden_size,
                    );
                } else {
                    dispatch_embedding(
                        &pipelines,
                        compute,
                        launch.format,
                        &regions[launch.table_region],
                        &regions[launch.tokens_region],
                        output,
                        launch.params,
                        output_type,
                    );
                }
            }
            Ok(())
        },
    )
    .map_err(|error| error.to_string())?
    .with_statistical_evidence(statistical)
    .with_work_shape(batching, participant_count, token_count)
    .map_err(|error| error.to_string())
}

fn validate_embedding_signature(
    token_ids: &ferrum_interfaces::vnext::ResolvedValueBinding,
    table: &ferrum_interfaces::vnext::ResolvedValueBinding,
    output: &ferrum_interfaces::vnext::ResolvedValueBinding,
    vocabulary_size: u64,
    hidden_size: u64,
    output_type: ElementType,
) -> Result<(), String> {
    let token_dimensions = token_ids.tensor().dimensions();
    if token_ids.tensor().element_type() != ElementType::U32
        || table.tensor().element_type() != ElementType::F16
        || output.tensor().element_type() != output_type
        || token_dimensions.len() != 1
        || table.tensor().dimensions() != [vocabulary_size, hidden_size]
        || output.tensor().dimensions() != [token_dimensions[0], hidden_size]
        || !matches!(
            token_ids.tensor().layout(),
            ferrum_interfaces::vnext::ResolvedTensorLayout::Contiguous
        )
        || !matches!(
            table.tensor().layout(),
            ferrum_interfaces::vnext::ResolvedTensorLayout::Contiguous
        )
        || !matches!(
            output.tensor().layout(),
            ferrum_interfaces::vnext::ResolvedTensorLayout::Contiguous
        )
    {
        return Err("Metal token embedding invocation differs from its signature".to_owned());
    }
    Ok(())
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct RmsNormParams {
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
}

fn encode_rms_norm(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    encode_rms_norm_typed(
        pipelines,
        invocation,
        RMS_NORM_OPERATION_ID,
        ElementType::F16,
        ElementType::F16,
    )
}

fn encode_rms_norm_f32_to_f16(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    encode_rms_norm_typed(
        pipelines,
        invocation,
        RMS_NORM_F32_TO_F16_OPERATION_ID,
        ElementType::F32,
        ElementType::F16,
    )
}

fn encode_rms_norm_f32(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    encode_rms_norm_typed(
        pipelines,
        invocation,
        RMS_NORM_F32_OPERATION_ID,
        ElementType::F32,
        ElementType::F32,
    )
}

fn encode_rms_norm_typed(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    operation_id: &'static str,
    input_type: ElementType,
    output_type: ElementType,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, operation_id)?;
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let epsilon = rational_attribute(first.attributes(), "epsilon")?;
    for participant in invocation.participants() {
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let weight = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || rational_attribute(participant.attributes(), "epsilon")? != epsilon
            || !valid_rms_norm(input, weight, output, hidden_size, input_type, output_type)
        {
            return Err("Metal RMSNorm participants disagree with the signature".to_owned());
        }
    }
    let tokens = invocation.work_shape().immediate_tokens();
    let regions = vec![
        shared_token_region(&invocation, ResolvedValueRole::Input, 0, input_type, tokens)?,
        shared_full_region(&invocation, ResolvedValueRole::Input, 1, ElementType::F16)?,
        shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            output_type,
            tokens,
        )?,
    ];
    let params = cost_route::rms_norm_params(tokens, hidden_size, epsilon)?;
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "Metal RMSNorm participant count",
    )?;
    let route = cost_route::compute_command(PrimitiveRoute::RmsNorm, participant_count, tokens)
        .map_err(|error| error.to_string())?;
    let statistical = selected::rms_evidence(&pipelines, params, input_type, output_type);
    let batching = route.batching();
    let dispatches = route.compute_dispatch_count();
    MetalDeviceCommand::operation(
        route.native_operation(),
        regions,
        move |encoder, regions| {
            encoder.record_compute_dispatches(dispatches);
            dispatch_rms_norm_typed(
                &pipelines,
                encoder.compute_encoder(),
                &regions[0],
                &regions[1],
                &regions[2],
                params,
                input_type,
                output_type,
            );
            Ok(())
        },
    )
    .map_err(|error| error.to_string())?
    .with_statistical_evidence(statistical)
    .with_work_shape(batching, participant_count, tokens)
    .map_err(|error| error.to_string())
}

fn valid_rms_norm(
    input: &ferrum_interfaces::vnext::ResolvedValueBinding,
    weight: &ferrum_interfaces::vnext::ResolvedValueBinding,
    output: &ferrum_interfaces::vnext::ResolvedValueBinding,
    hidden_size: u64,
    input_type: ElementType,
    output_type: ElementType,
) -> bool {
    let [rows, input_hidden] = input.tensor().dimensions() else {
        return false;
    };
    *input_hidden == hidden_size
        && weight.tensor().dimensions() == [hidden_size]
        && output.tensor().dimensions() == [*rows, hidden_size]
        && input.tensor().element_type() == input_type
        && matches!(input.tensor().layout(), ResolvedTensorLayout::Contiguous)
        && f16_contiguous(weight)
        && output.tensor().element_type() == output_type
        && matches!(output.tensor().layout(), ResolvedTensorLayout::Contiguous)
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ResidualAddParams {
    elements: u32,
}

fn encode_residual_add(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    encode_residual_add_typed(
        pipelines,
        invocation,
        RESIDUAL_ADD_OPERATION_ID,
        ElementType::F16,
        ElementType::F16,
        ElementType::F16,
    )
}

fn encode_residual_add_f32_f16(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    encode_residual_add_typed(
        pipelines,
        invocation,
        RESIDUAL_ADD_F32_F16_OPERATION_ID,
        ElementType::F32,
        ElementType::F16,
        ElementType::F32,
    )
}

fn encode_residual_add_typed(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    operation_id: &'static str,
    left_type: ElementType,
    right_type: ElementType,
    output_type: ElementType,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, operation_id)?;
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    for participant in invocation.participants() {
        let left = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let right = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || !valid_residual_add(
                left,
                right,
                output,
                hidden_size,
                left_type,
                right_type,
                output_type,
            )
        {
            return Err("Metal residual-add participants disagree with the signature".to_owned());
        }
    }
    let tokens = invocation.work_shape().immediate_tokens();
    let params = cost_route::residual_add_params(tokens, hidden_size)?;
    let regions = vec![
        shared_token_region(&invocation, ResolvedValueRole::Input, 0, left_type, tokens)?,
        shared_token_region(&invocation, ResolvedValueRole::Input, 1, right_type, tokens)?,
        shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            output_type,
            tokens,
        )?,
    ];
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "Metal residual-add participant count",
    )?;
    let route = cost_route::compute_command(PrimitiveRoute::ResidualAdd, participant_count, tokens)
        .map_err(|error| error.to_string())?;
    let statistical = selected::residual_evidence(
        &pipelines,
        params,
        left_type,
        right_type,
        output_type,
        tokens,
    );
    let batching = route.batching();
    let dispatches = route.compute_dispatch_count();
    MetalDeviceCommand::operation(
        route.native_operation(),
        regions,
        move |encoder, regions| {
            encoder.record_compute_dispatches(dispatches);
            dispatch_residual_add_typed(
                &pipelines,
                encoder.compute_encoder(),
                &regions[0],
                &regions[1],
                &regions[2],
                params,
                left_type,
                right_type,
                output_type,
            );
            Ok(())
        },
    )
    .map_err(|error| error.to_string())?
    .with_statistical_evidence(statistical)
    .with_work_shape(batching, participant_count, tokens)
    .map_err(|error| error.to_string())
}

fn valid_residual_add(
    left: &ferrum_interfaces::vnext::ResolvedValueBinding,
    right: &ferrum_interfaces::vnext::ResolvedValueBinding,
    output: &ferrum_interfaces::vnext::ResolvedValueBinding,
    hidden_size: u64,
    left_type: ElementType,
    right_type: ElementType,
    output_type: ElementType,
) -> bool {
    let [tokens, input_hidden] = left.tensor().dimensions() else {
        return false;
    };
    *input_hidden == hidden_size
        && right.tensor().dimensions() == [*tokens, hidden_size]
        && output.tensor().dimensions() == [*tokens, hidden_size]
        && left.tensor().element_type() == left_type
        && right.tensor().element_type() == right_type
        && output.tensor().element_type() == output_type
        && matches!(left.tensor().layout(), ResolvedTensorLayout::Contiguous)
        && matches!(right.tensor().layout(), ResolvedTensorLayout::Contiguous)
        && matches!(output.tensor().layout(), ResolvedTensorLayout::Contiguous)
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct LastTokenMaskedArgmaxParams {
    vocabulary_size: u32,
    repetition_capacity: u32,
}

#[derive(Debug, Clone, Copy)]
struct LastTokenMaskedArgmaxLaunch {
    first_region: usize,
    scratch_offset_bytes: u64,
    params: LastTokenMaskedArgmaxParams,
}

fn encode_last_token_masked_argmax(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    encode_last_token_masked_argmax_typed(
        pipelines,
        invocation,
        LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID,
        ElementType::F16,
    )
}

fn encode_last_token_masked_argmax_f32(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    encode_last_token_masked_argmax_typed(
        pipelines,
        invocation,
        LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID,
        ElementType::F32,
    )
}

fn encode_last_token_masked_argmax_typed(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    operation_id: &'static str,
    logits_type: ElementType,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, operation_id)?;
    let first_vocabulary_size =
        unsigned_attribute(invocation.participants()[0].attributes(), "vocab_size")?;
    let scratch_stride = masked_argmax_scratch_stride(first_vocabulary_size, logits_type)?;
    let required_scratch_bytes = scratch_stride
        .checked_mul(invocation.participants().len() as u64)
        .ok_or_else(|| "Metal masked argmax scratch size overflows".to_owned())?;
    let mut regions = Vec::with_capacity(invocation.participants().len() * 6 + 1);
    let mut launches = Vec::with_capacity(invocation.participants().len());
    for (participant_index, participant) in invocation.participants().iter().enumerate() {
        let logits = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let valid_mask = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let repetition_token_ids = binding(participant.bindings(), ResolvedValueRole::Input, 2)?;
        let repetition_offsets = binding(participant.bindings(), ResolvedValueRole::Input, 3)?;
        let repetition_penalty = binding(participant.bindings(), ResolvedValueRole::Input, 4)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        let vocabulary_size = unsigned_attribute(participant.attributes(), "vocab_size")?;
        if vocabulary_size != first_vocabulary_size {
            return Err("Metal masked argmax participants disagree on vocabulary size".to_owned());
        }
        let Some(repetition_capacity) = valid_last_token_masked_argmax(
            logits,
            valid_mask,
            repetition_token_ids,
            repetition_offsets,
            repetition_penalty,
            output,
            vocabulary_size,
            logits_type,
        ) else {
            return Err(
                "Metal masked argmax participant differs from its resolved signature".to_owned(),
            );
        };
        let first_region = regions.len();
        regions.push(contiguous_region(participant, logits, logits_type)?);
        regions.push(contiguous_region(participant, valid_mask, ElementType::U8)?);
        regions.push(contiguous_region(
            participant,
            repetition_token_ids,
            ElementType::U32,
        )?);
        regions.push(contiguous_region(
            participant,
            repetition_offsets,
            ElementType::U32,
        )?);
        regions.push(contiguous_region(
            participant,
            repetition_penalty,
            ElementType::F32,
        )?);
        regions.push(contiguous_region(participant, output, ElementType::U32)?);
        launches.push(LastTokenMaskedArgmaxLaunch {
            first_region,
            scratch_offset_bytes: scratch_stride
                .checked_mul(participant_index as u64)
                .ok_or_else(|| "Metal masked argmax scratch offset overflows".to_owned())?,
            params: cost_route::masked_argmax_params(vocabulary_size, repetition_capacity)?,
        });
    }
    let scratch_region = regions.len();
    regions.push(shared_scratch_region(&invocation, required_scratch_bytes)?);
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "Metal masked argmax participant count",
    )?;
    let route = cost_route::compute_command(
        PrimitiveRoute::MaskedArgmax {
            vocabulary_size: launches[0].params.vocabulary_size,
        },
        participant_count,
        invocation.work_shape().immediate_tokens(),
    )
    .map_err(|error| error.to_string())?;
    let statistical =
        selected::argmax_evidence(&pipelines, &launches, logits_type, required_scratch_bytes);
    let dispatch_count = route.compute_dispatch_count();
    let batching = route.batching();
    MetalDeviceCommand::operation(
        route.native_operation(),
        regions,
        move |encoder, regions| {
            encoder.record_compute_dispatches(dispatch_count);
            for launch in &launches {
                dispatch_last_token_masked_argmax(
                    &pipelines,
                    encoder.compute_encoder(),
                    &regions[launch.first_region],
                    &regions[launch.first_region + 1],
                    &regions[launch.first_region + 2],
                    &regions[launch.first_region + 3],
                    &regions[launch.first_region + 4],
                    &regions[launch.first_region + 5],
                    &regions[scratch_region],
                    launch.scratch_offset_bytes,
                    launch.params,
                    logits_type,
                );
            }
            Ok(())
        },
    )
    .map_err(|error| error.to_string())?
    .with_statistical_evidence(statistical)
    .with_work_shape(batching, participant_count, u64::from(participant_count))
    .map_err(|error| error.to_string())
}

fn valid_last_token_masked_argmax(
    logits: &ResolvedValueBinding,
    valid_mask: &ResolvedValueBinding,
    repetition_token_ids: &ResolvedValueBinding,
    repetition_offsets: &ResolvedValueBinding,
    repetition_penalty: &ResolvedValueBinding,
    output: &ResolvedValueBinding,
    vocabulary_size: u64,
    logits_type: ElementType,
) -> Option<u32> {
    let contiguous = |binding: &ResolvedValueBinding| {
        matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
    };
    let valid = logits.tensor().element_type() == logits_type
        && valid_mask.tensor().element_type() == ElementType::U8
        && repetition_token_ids.tensor().element_type() == ElementType::U32
        && repetition_offsets.tensor().element_type() == ElementType::U32
        && repetition_penalty.tensor().element_type() == ElementType::F32
        && output.tensor().element_type() == ElementType::U32
        && logits.tensor().dimensions() == [1, vocabulary_size]
        && valid_mask.tensor().dimensions() == [vocabulary_size]
        && repetition_token_ids.tensor().dimensions().len() == 1
        && repetition_token_ids.tensor().dimensions()[0] != 0
        && repetition_offsets.tensor().dimensions() == [2]
        && repetition_penalty.tensor().dimensions() == [1]
        && output.tensor().dimensions() == [1]
        && contiguous(logits)
        && contiguous(valid_mask)
        && contiguous(repetition_token_ids)
        && contiguous(repetition_offsets)
        && contiguous(repetition_penalty)
        && contiguous(output);
    valid
        .then(|| {
            checked_u32(
                repetition_token_ids.tensor().dimensions()[0],
                "Metal masked argmax repetition capacity",
            )
            .ok()
        })
        .flatten()
}

fn set_region(encoder: &ComputeCommandEncoderRef, index: u64, region: &MetalBufferRegion) {
    encoder.set_buffer(index, Some(region.buffer()), region.offset_bytes());
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

fn dispatch_last_token_masked_argmax(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    logits: &MetalBufferRegion,
    valid_mask: &MetalBufferRegion,
    repetition_token_ids: &MetalBufferRegion,
    repetition_offsets: &MetalBufferRegion,
    repetition_penalty: &MetalBufferRegion,
    output: &MetalBufferRegion,
    scratch: &MetalBufferRegion,
    scratch_offset_bytes: u64,
    params: LastTokenMaskedArgmaxParams,
    logits_type: ElementType,
) {
    let parallel = masked_argmax_dispatch_count(params.vocabulary_size) == 2;
    let selection = selected::argmax(pipelines, logits_type, parallel)
        .expect("validated Metal masked argmax type");
    let pipeline = selection.pipeline;
    encoder.set_compute_pipeline_state(pipeline);
    set_region(encoder, 0, logits);
    set_region_offset(encoder, 1, scratch, scratch_offset_bytes);
    set_region(encoder, 2, valid_mask);
    set_region(encoder, 3, repetition_token_ids);
    set_region(encoder, 4, repetition_offsets);
    set_region(encoder, 5, repetition_penalty);
    set_region(encoder, 6, output);
    encoder.set_bytes(
        7,
        std::mem::size_of::<LastTokenMaskedArgmaxParams>() as u64,
        &params as *const _ as *const c_void,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(
            if parallel {
                MASKED_ARGMAX_PARTITIONS
            } else {
                1
            },
            1,
            1,
        ),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );
    if parallel {
        encoder.set_compute_pipeline_state(&pipelines.masked_argmax_finalize);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
    }
}

fn masked_argmax_dispatch_count(vocabulary_size: u32) -> u64 {
    // Each partition writes an F32 maximum plus U32 index (256 bytes total).
    // The existing per-participant vocabulary scratch already covers this even
    // for F16. An active repetition penalty retains the original one-group
    // algorithm on the GPU; its finalization dispatch does no work.
    if vocabulary_size >= MASKED_ARGMAX_PARALLEL_MIN_VOCAB {
        2
    } else {
        1
    }
}

fn estimate_masked_argmax_resources(
    descriptor: &OperationProviderDescriptor,
    request: OperationResourceEstimateRequest<'_>,
    operation_id: &str,
    logits_type: ElementType,
) -> Result<OperationResourceEstimate, VNextError> {
    if request.operation().id.as_str() != operation_id
        || request.operation().fingerprint()? != descriptor.operation_fingerprint()
    {
        return Err(invalid_plan(format!(
            "Metal estimator `{}` received another operation",
            descriptor.resource_estimator_id()
        )));
    }
    let vocabulary_size =
        unsigned_attribute(request.attributes(), "vocab_size").map_err(invalid_plan)?;
    let scratch_bytes =
        masked_argmax_scratch_stride(vocabulary_size, logits_type).map_err(invalid_plan)?;
    let scratch = ProviderWorkspaceRequirement::from_formula(
        ProviderWorkspaceSizeFormula::actual_sequences(scratch_bytes)?,
        VALUE_ALIGNMENT_BYTES,
        ProviderWorkspaceScope::Invocation,
        ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
        DynamicStorageRequirement::contiguous(),
    )?;
    Ok(OperationResourceEstimate::new(
        descriptor.resource_estimator_id(),
        descriptor.resource_estimator_version(),
        descriptor.resource_estimator_implementation_fingerprint(),
        request.input_fingerprint(),
        VALUE_ALIGNMENT_BYTES,
        Some(scratch),
        None,
    ))
}

fn masked_argmax_scratch_stride(
    vocabulary_size: u64,
    logits_type: ElementType,
) -> Result<u64, String> {
    let bytes = vocabulary_size
        .checked_mul(logits_type.size_bytes())
        .ok_or_else(|| "Metal masked argmax scratch size overflows".to_owned())?;
    bytes
        .checked_add(VALUE_ALIGNMENT_BYTES - 1)
        .map(|value| value & !(VALUE_ALIGNMENT_BYTES - 1))
        .filter(|value| *value != 0)
        .ok_or_else(|| "Metal masked argmax scratch alignment overflows".to_owned())
}

fn dispatch_embedding(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    format: EmbeddingPhysicalFormat,
    table: &MetalBufferRegion,
    token_ids: &MetalBufferRegion,
    output: &MetalBufferRegion,
    params: EmbeddingParams,
    output_type: ElementType,
) {
    dispatch_embedding_at(
        pipelines,
        encoder,
        format,
        table,
        token_ids,
        output,
        0,
        params,
        output_type,
    );
}

#[allow(clippy::too_many_arguments)]
fn dispatch_embedding_at(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    format: EmbeddingPhysicalFormat,
    table: &MetalBufferRegion,
    token_ids: &MetalBufferRegion,
    output: &MetalBufferRegion,
    output_offset_bytes: u64,
    params: EmbeddingParams,
    output_type: ElementType,
) {
    let selection = selected::embedding(pipelines, format, output_type)
        .expect("validated Metal embedding type");
    let pipeline = selection.pipeline;
    encoder.set_compute_pipeline_state(pipeline);
    set_region(encoder, 0, table);
    set_region(encoder, 1, token_ids);
    set_region_offset(encoder, 2, output, output_offset_bytes);
    encoder.set_bytes(
        3,
        std::mem::size_of::<EmbeddingParams>() as u64,
        &params as *const _ as *const c_void,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(
            u64::from(params.hidden_size).div_ceil(THREADS_PER_GROUP),
            u64::from(params.token_count),
            1,
        ),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );
}

fn dispatch_rms_norm_typed(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    input: &MetalBufferRegion,
    weight: &MetalBufferRegion,
    output: &MetalBufferRegion,
    params: RmsNormParams,
    input_type: ElementType,
    output_type: ElementType,
) {
    dispatch_rms_norm_typed_at(
        pipelines,
        encoder,
        input,
        0,
        weight,
        output,
        0,
        params.rows,
        params.hidden_size,
        params.epsilon,
        input_type,
        output_type,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_rms_norm_at(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    input: &MetalBufferRegion,
    input_offset_bytes: u64,
    weight: &MetalBufferRegion,
    output: &MetalBufferRegion,
    output_offset_bytes: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
) {
    dispatch_rms_norm_typed_at(
        pipelines,
        encoder,
        input,
        input_offset_bytes,
        weight,
        output,
        output_offset_bytes,
        rows,
        hidden_size,
        epsilon,
        ElementType::F16,
        ElementType::F16,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_rms_norm_f32_to_f16_at(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    input: &MetalBufferRegion,
    input_offset_bytes: u64,
    weight: &MetalBufferRegion,
    output: &MetalBufferRegion,
    output_offset_bytes: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
) {
    dispatch_rms_norm_typed_at(
        pipelines,
        encoder,
        input,
        input_offset_bytes,
        weight,
        output,
        output_offset_bytes,
        rows,
        hidden_size,
        epsilon,
        ElementType::F32,
        ElementType::F16,
    );
}

#[allow(clippy::too_many_arguments)]
fn dispatch_rms_norm_typed_at(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    input: &MetalBufferRegion,
    input_offset_bytes: u64,
    weight: &MetalBufferRegion,
    output: &MetalBufferRegion,
    output_offset_bytes: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
    input_type: ElementType,
    output_type: ElementType,
) {
    let params = RmsNormParams {
        rows,
        hidden_size,
        epsilon,
    };
    let selection =
        selected::rms(pipelines, input_type, output_type).expect("validated Metal RMSNorm types");
    let pipeline = selection.pipeline;
    encoder.set_compute_pipeline_state(pipeline);
    set_region_offset(encoder, 0, input, input_offset_bytes);
    set_region(encoder, 1, weight);
    set_region_offset(encoder, 2, output, output_offset_bytes);
    encoder.set_bytes(
        3,
        std::mem::size_of::<RmsNormParams>() as u64,
        &params as *const _ as *const c_void,
    );
    encoder.set_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>() as u64);
    encoder.dispatch_thread_groups(
        MTLSize::new(u64::from(params.rows), 1, 1),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );
}

#[allow(clippy::too_many_arguments)]
fn dispatch_residual_add_typed(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    left: &MetalBufferRegion,
    right: &MetalBufferRegion,
    output: &MetalBufferRegion,
    params: ResidualAddParams,
    left_type: ElementType,
    right_type: ElementType,
    output_type: ElementType,
) {
    dispatch_residual_add_typed_at(
        pipelines,
        encoder,
        left,
        0,
        right,
        0,
        output,
        0,
        params.elements,
        left_type,
        right_type,
        output_type,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_residual_add_at(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    left: &MetalBufferRegion,
    left_offset_bytes: u64,
    right: &MetalBufferRegion,
    right_offset_bytes: u64,
    output: &MetalBufferRegion,
    output_offset_bytes: u64,
    elements: u32,
) {
    dispatch_residual_add_typed_at(
        pipelines,
        encoder,
        left,
        left_offset_bytes,
        right,
        right_offset_bytes,
        output,
        output_offset_bytes,
        elements,
        ElementType::F16,
        ElementType::F16,
        ElementType::F16,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_residual_add_f32_f16_at(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    left: &MetalBufferRegion,
    left_offset_bytes: u64,
    right: &MetalBufferRegion,
    right_offset_bytes: u64,
    output: &MetalBufferRegion,
    output_offset_bytes: u64,
    elements: u32,
) {
    dispatch_residual_add_typed_at(
        pipelines,
        encoder,
        left,
        left_offset_bytes,
        right,
        right_offset_bytes,
        output,
        output_offset_bytes,
        elements,
        ElementType::F32,
        ElementType::F16,
        ElementType::F32,
    );
}

#[allow(clippy::too_many_arguments)]
fn dispatch_residual_add_typed_at(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    left: &MetalBufferRegion,
    left_offset_bytes: u64,
    right: &MetalBufferRegion,
    right_offset_bytes: u64,
    output: &MetalBufferRegion,
    output_offset_bytes: u64,
    elements: u32,
    left_type: ElementType,
    right_type: ElementType,
    output_type: ElementType,
) {
    let params = ResidualAddParams { elements };
    let selection = selected::residual(pipelines, left_type, right_type, output_type)
        .expect("validated Metal residual types");
    let pipeline = selection.pipeline;
    encoder.set_compute_pipeline_state(pipeline);
    set_region_offset(encoder, 0, left, left_offset_bytes);
    set_region_offset(encoder, 1, right, right_offset_bytes);
    set_region_offset(encoder, 2, output, output_offset_bytes);
    encoder.set_bytes(
        3,
        std::mem::size_of::<ResidualAddParams>() as u64,
        &params as *const _ as *const c_void,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(u64::from(params.elements).div_ceil(THREADS_PER_GROUP), 1, 1),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );
}

#[cfg(test)]
#[path = "primitives_argmax_tests.rs"]
mod parallel_argmax_tests;

#[cfg(test)]
mod tests {
    use super::super::numerical_tolerance;
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device as CandleDevice, Tensor};
    use half::f16;
    use metal::{BufferRef, MTLCommandBufferStatus, MTLResourceOptions};

    const TOKEN_EMBEDDING_TOLERANCE_ID: &str =
        "runtime-vnext.metal.token-embedding.v1.operation.fp16.gguf-q6-k.padding";
    const RMS_NORM_TOLERANCE_ID: &str =
        "runtime-vnext.metal.rms-norm.v1.operation.fp16.none.hidden-2560";
    const RESIDUAL_ADD_TOLERANCE_ID: &str =
        "runtime-vnext.metal.residual-add.v1.operation.fp16.none.hidden-2560";

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
    fn token_embedding_capability_includes_q4_k() {
        assert!(TOKEN_EMBEDDING_QUANTIZATION_FORMATS.contains(&Q4_K_FORMAT_ID));
    }

    #[test]
    fn pq2_0_token_embedding_preserves_codes_blocks_and_f32_range_on_real_metal() {
        let device = Device::system_default().expect("PQ2_0 embedding conformance requires Metal");
        let pipelines = MetalPrimitivePipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let format = GgufBlockFormat::Pq2_0;
        assert!(TOKEN_EMBEDDING_QUANTIZATION_FORMATS.contains(&format.format_id()));
        let vocabulary = 3;
        let blocks_per_row = 3;
        let hidden = blocks_per_row * format.block_values();
        let mut bytes = vec![0_u8; vocabulary * blocks_per_row * format.block_bytes()];
        for (index, block) in bytes.chunks_exact_mut(format.block_bytes()).enumerate() {
            let scale = [0.125, -0.21875, f16::MAX.to_f32()][index % 3];
            block[..2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
            for (position, packed) in block[2..].iter_mut().enumerate() {
                // Every block contains all four codes; rotate their positions
                // by row and byte so token and lane permutations are visible.
                *packed = (0..4).fold(0, |value, lane| {
                    value | (((index + position + lane) % 4) as u8) << (2 * lane)
                });
            }
        }
        let mut decoded = vec![0.0; vocabulary * hidden];
        format.decode(&bytes, &mut decoded).unwrap();
        assert!(decoded.iter().any(|value| *value > f16::MAX.to_f32()));
        let tokens = [2_u32, 0, u32::MAX, 1, 2];
        let count = tokens.len() * hidden;
        let table = shared_buffer(&device, &bytes);
        let token_buffer = shared_buffer(&device, &tokens);
        let guard = -123.0_f32;
        let output_f16 = shared_buffer(&device, &vec![f16::from_f32(guard); count + 17]);
        let output_f32 = shared_buffer(&device, &vec![guard; count + 17]);
        let params = EmbeddingParams {
            token_count: tokens.len() as u32,
            hidden_size: hidden as u32,
            vocabulary_size: vocabulary as u32,
        };
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        dispatch_raw_embedding(
            &pipelines,
            encoder,
            EmbeddingPhysicalFormat::Pq2_0,
            &table,
            &token_buffer,
            &output_f16,
            params,
        );
        dispatch_raw_embedding_f32(
            &pipelines,
            encoder,
            EmbeddingPhysicalFormat::Pq2_0,
            &table,
            &token_buffer,
            &output_f32,
            params,
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let f16_values = read_f16(&output_f16, count + 17);
        let f32_values = read_f32(&output_f32, count + 17);
        for index in 0..count + 17 {
            let expected = if index >= count {
                guard
            } else {
                let token = tokens[index / hidden] as usize;
                if token >= vocabulary {
                    0.0
                } else {
                    decoded[token * hidden + index % hidden]
                }
            };
            assert_eq!(f32_values[index], expected, "F32 embedding[{index}]");
            assert_eq!(
                f16_values[index],
                f16::from_f32(expected).to_f32(),
                "F16 embedding[{index}]"
            );
        }
        // SAFETY: Shared inputs are still alive and the command has completed.
        let table_after =
            unsafe { std::slice::from_raw_parts(table.contents().cast::<u8>(), bytes.len()) };
        let tokens_after = unsafe {
            std::slice::from_raw_parts(token_buffer.contents().cast::<u32>(), tokens.len())
        };
        assert_eq!(table_after, bytes);
        assert_eq!(tokens_after, tokens);
    }

    #[test]
    fn q4_k_token_embedding_matches_cpu_for_f16_and_f32_on_real_metal() {
        let device = Device::system_default().expect("token-embedding conformance requires Metal");
        assert_quantized_token_embedding(&device, EmbeddingPhysicalFormat::Q4K, GgmlDType::Q4K);
    }

    #[test]
    fn q6_k_and_q8_token_embeddings_preserve_float_boundaries_on_real_metal() {
        let device = Device::system_default().expect("token-embedding conformance requires Metal");
        for (format, dtype) in [
            (EmbeddingPhysicalFormat::Q6K, GgmlDType::Q6K),
            (EmbeddingPhysicalFormat::Q8_0, GgmlDType::Q8_0),
        ] {
            assert_quantized_token_embedding(&device, format, dtype);
        }
    }

    fn assert_quantized_token_embedding(
        device: &Device,
        format: EmbeddingPhysicalFormat,
        dtype: GgmlDType,
    ) {
        let pipelines = MetalPrimitivePipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let vocabulary = 3_usize;
        let hidden = 512_usize;
        let raw_table = (0..vocabulary * hidden)
            .map(|index| ((index as f32) * 0.017).sin() * 0.75)
            .collect::<Vec<_>>();
        let cpu = CandleDevice::Cpu;
        let table = Tensor::from_vec(raw_table, (vocabulary, hidden), &cpu).unwrap();
        let quantized = QTensor::quantize(&table, dtype).unwrap();
        let reference = quantized
            .dequantize(&cpu)
            .unwrap()
            .get(2)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let table_buffer = shared_buffer(&device, &quantized.data().unwrap());
        let token_buffer = shared_buffer(&device, &[2_u32, u32::MAX]);
        let f16_output = output_buffer::<f16>(&device, hidden * 2);
        let f32_output = output_buffer::<f32>(&device, hidden * 2);

        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let params = EmbeddingParams {
            token_count: 2,
            hidden_size: hidden as u32,
            vocabulary_size: vocabulary as u32,
        };
        dispatch_raw_embedding(
            &pipelines,
            encoder,
            format,
            &table_buffer,
            &token_buffer,
            &f16_output,
            params,
        );
        dispatch_raw_embedding_f32(
            &pipelines,
            encoder,
            format,
            &table_buffer,
            &token_buffer,
            &f32_output,
            params,
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        for (index, (observed, expected)) in read_f16(&f16_output, hidden)[..hidden]
            .iter()
            .zip(reference.iter())
            .enumerate()
        {
            let expected = f16::from_f32(*expected).to_f32();
            assert!(
                (*observed - expected).abs() <= 0.002,
                "{dtype:?} F16 token embedding differs at column {index}: observed={observed} expected={expected}"
            );
        }
        for (index, (observed, expected)) in read_f32(&f32_output, hidden)[..hidden]
            .iter()
            .zip(reference.iter())
            .enumerate()
        {
            assert!(
                (*observed - expected).abs() <= 0.00002,
                "{dtype:?} F32 token embedding differs at column {index}: observed={observed} expected={expected}"
            );
        }
        assert!(read_f16(&f16_output, hidden * 2)[hidden..]
            .iter()
            .all(|value| *value == 0.0));
        assert!(read_f32(&f32_output, hidden * 2)[hidden..]
            .iter()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn f32_master_primitives_preserve_precision_and_residual_aliasing_on_real_metal() {
        let device = Device::system_default().expect("F32 primitive conformance requires Metal");
        let pipelines = MetalPrimitivePipelines::new(&device).unwrap();
        let queue = device.new_command_queue();

        let table = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let table_buffer = shared_buffer(&device, &table);
        let token_buffer = shared_buffer(&device, &[0_u32]);
        let embedding_output = output_buffer::<f32>(&device, 4);

        let rms_input = [1.0_f32, 2.0, 3.0, 4.0];
        let rms_weight = [f16::from_f32(1.0); 4];
        let rms_input_buffer = shared_buffer(&device, &rms_input);
        let rms_weight_buffer = shared_buffer(&device, &rms_weight);
        let rms_f16_output = output_buffer::<f16>(&device, 4);
        let rms_f32_output = output_buffer::<f32>(&device, 4);

        let residual_left = [1.0_f32, -2.0, 3.25, -4.5];
        let residual_right = [
            f16::from_f32(0.5),
            f16::from_f32(1.0),
            f16::from_f32(-0.25),
            f16::from_f32(2.0),
        ];
        let residual_left_buffer = shared_buffer(&device, &residual_left);
        let residual_alias_buffer = shared_buffer(&device, &residual_left);
        let residual_right_buffer = shared_buffer(&device, &residual_right);
        let residual_output = output_buffer::<f32>(&device, 4);

        let close_logits = [21.930_f32, 21.934_f32];
        assert_eq!(
            f16::from_f32(close_logits[0]),
            f16::from_f32(close_logits[1]),
            "adversarial logits must demonstrate the precision lost by the legacy F16 boundary"
        );
        let logits_buffer = shared_buffer(&device, &close_logits);
        let mask_buffer = shared_buffer(&device, &[1_u8, 1]);
        let repetition_ids_buffer = shared_buffer(&device, &[0_u32]);
        let repetition_offsets_buffer = shared_buffer(&device, &[0_u32, 0]);
        let repetition_penalty_buffer = shared_buffer(&device, &[1.0_f32]);
        let argmax_output = output_buffer::<u32>(&device, 1);
        let argmax_scratch = output_buffer::<f32>(&device, 2);

        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        dispatch_raw_embedding_f32(
            &pipelines,
            encoder,
            EmbeddingPhysicalFormat::DenseF16,
            &table_buffer,
            &token_buffer,
            &embedding_output,
            EmbeddingParams {
                token_count: 1,
                hidden_size: 4,
                vocabulary_size: 1,
            },
        );
        dispatch_raw_rms_norm_f32(
            &pipelines,
            encoder,
            &rms_input_buffer,
            &rms_weight_buffer,
            &rms_f16_output,
            RmsNormParams {
                rows: 1,
                hidden_size: 4,
                epsilon: 1e-6,
            },
            false,
        );
        dispatch_raw_rms_norm_f32(
            &pipelines,
            encoder,
            &rms_input_buffer,
            &rms_weight_buffer,
            &rms_f32_output,
            RmsNormParams {
                rows: 1,
                hidden_size: 4,
                epsilon: 1e-6,
            },
            true,
        );
        dispatch_raw_residual_add_f32_f16(
            &pipelines,
            encoder,
            &residual_left_buffer,
            &residual_right_buffer,
            &residual_output,
            ResidualAddParams { elements: 4 },
        );
        dispatch_raw_residual_add_f32_f16(
            &pipelines,
            encoder,
            &residual_alias_buffer,
            &residual_right_buffer,
            &residual_alias_buffer,
            ResidualAddParams { elements: 4 },
        );
        dispatch_raw_last_token_masked_argmax_f32(
            &pipelines,
            encoder,
            &logits_buffer,
            &mask_buffer,
            &repetition_ids_buffer,
            &repetition_offsets_buffer,
            &repetition_penalty_buffer,
            &argmax_output,
            &argmax_scratch,
            LastTokenMaskedArgmaxParams {
                vocabulary_size: 2,
                repetition_capacity: 1,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        assert_eq!(read_f32(&embedding_output, 4), [1.0, 2.0, 3.0, 4.0]);
        let inverse_rms = (7.5_f32 + 1e-6).sqrt().recip();
        let expected_rms = rms_input.map(|value| value * inverse_rms);
        for (observed, expected) in read_f32(&rms_f32_output, 4).iter().zip(expected_rms) {
            assert!((observed - expected).abs() <= 1e-5);
        }
        for (observed, expected) in read_f16(&rms_f16_output, 4).iter().zip(expected_rms) {
            assert!((observed - expected).abs() <= 1e-3);
        }
        let expected_residual = [1.5_f32, -1.0, 3.0, -2.5];
        assert_eq!(read_f32(&residual_output, 4), expected_residual);
        assert_eq!(read_f32(&residual_alias_buffer, 4), expected_residual);
        assert_eq!(unsafe { *(argmax_output.contents() as *const u32) }, 1);
    }

    #[test]
    fn native_f16_primitives_match_cpu_references_on_real_metal() {
        let device = Device::system_default().expect("F16 primitive conformance requires Metal");
        let pipelines = MetalPrimitivePipelines::new(&device).unwrap();
        let queue = device.new_command_queue();

        let vocabulary = 4_usize;
        let hidden = 2560_usize;
        let raw_table = (0..vocabulary * hidden)
            .map(|index| ((index as f32) * 0.013).sin())
            .collect::<Vec<_>>();
        let cpu = CandleDevice::Cpu;
        let table = Tensor::from_vec(raw_table, (vocabulary, hidden), &cpu).unwrap();
        let quantized = QTensor::quantize(&table, GgmlDType::Q6K).unwrap();
        let reference = quantized
            .dequantize(&cpu)
            .unwrap()
            .get(2)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let quantized_bytes = quantized.data().unwrap();
        let table_buffer = shared_buffer(&device, &quantized_bytes);
        let q8_quantized = QTensor::quantize(&table, GgmlDType::Q8_0).unwrap();
        let q8_reference = q8_quantized
            .dequantize(&cpu)
            .unwrap()
            .get(2)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let q8_bytes = q8_quantized.data().unwrap();
        let q8_table_buffer = shared_buffer(&device, &q8_bytes);
        let token_buffer = shared_buffer(&device, &[2_u32, u32::MAX]);
        let embedding_output = output_buffer::<f16>(&device, hidden * 2);
        let q8_embedding_output = output_buffer::<f16>(&device, hidden * 2);

        let rms_input = (0..hidden)
            .map(|index| f16::from_f32((index as f32 + 1.0) / hidden as f32))
            .collect::<Vec<_>>();
        let rms_weight = (0..hidden)
            .map(|index| f16::from_f32(0.75 + index as f32 / (hidden * 2) as f32))
            .collect::<Vec<_>>();
        let rms_input_buffer = shared_buffer(&device, &rms_input);
        let rms_weight_buffer = shared_buffer(&device, &rms_weight);
        let rms_output = output_buffer::<f16>(&device, hidden);

        let residual_right = (0..hidden)
            .map(|index| f16::from_f32(-0.25 + index as f32 / hidden as f32))
            .collect::<Vec<_>>();
        let residual_right_buffer = shared_buffer(&device, &residual_right);
        let residual_output = output_buffer::<f16>(&device, hidden);
        let argmax_logits = [-4.0_f32, 3.0, 3.0, 10.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let argmax_logits_buffer = shared_buffer(&device, &argmax_logits);
        let argmax_mask_buffer = shared_buffer(&device, &[1_u8, 1, 1, 0, 0]);
        let argmax_empty_mask_buffer = shared_buffer(&device, &[0_u8; 5]);
        let argmax_repetition_token_ids_buffer = shared_buffer(&device, &[0_u32]);
        let argmax_repetition_disabled_offsets_buffer = shared_buffer(&device, &[0_u32, 0_u32]);
        let argmax_repetition_disabled_penalty_buffer = shared_buffer(&device, &[1.0_f32]);
        let argmax_output = output_buffer::<u32>(&device, 1);
        let argmax_empty_output = output_buffer::<u32>(&device, 1);
        let argmax_scratch = output_buffer::<f16>(&device, 5);
        let repetition_logits = [4.0_f32, 3.0, 2.0]
            .into_iter()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let repetition_logits_buffer = shared_buffer(&device, &repetition_logits);
        let repetition_mask_buffer = shared_buffer(&device, &[1_u8; 3]);
        let repetition_offsets_buffer = shared_buffer(&device, &[0_u32, 1_u32]);
        let repetition_penalty_buffer = shared_buffer(&device, &[2.0_f32]);
        let repetition_output = output_buffer::<u32>(&device, 1);
        let repetition_scratch = output_buffer::<f16>(&device, 3);

        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        dispatch_raw_embedding(
            &pipelines,
            encoder,
            EmbeddingPhysicalFormat::Q6K,
            &table_buffer,
            &token_buffer,
            &embedding_output,
            EmbeddingParams {
                token_count: 2,
                hidden_size: hidden as u32,
                vocabulary_size: vocabulary as u32,
            },
        );
        dispatch_raw_embedding(
            &pipelines,
            encoder,
            EmbeddingPhysicalFormat::Q8_0,
            &q8_table_buffer,
            &token_buffer,
            &q8_embedding_output,
            EmbeddingParams {
                token_count: 2,
                hidden_size: hidden as u32,
                vocabulary_size: vocabulary as u32,
            },
        );
        dispatch_raw_rms_norm(
            &pipelines,
            encoder,
            &rms_input_buffer,
            &rms_weight_buffer,
            &rms_output,
            RmsNormParams {
                rows: 1,
                hidden_size: hidden as u32,
                epsilon: 1e-6,
            },
        );
        dispatch_raw_residual_add(
            &pipelines,
            encoder,
            &rms_output,
            &residual_right_buffer,
            &residual_output,
            ResidualAddParams {
                elements: hidden as u32,
            },
        );
        dispatch_raw_last_token_masked_argmax(
            &pipelines,
            encoder,
            &argmax_logits_buffer,
            &argmax_mask_buffer,
            &argmax_repetition_token_ids_buffer,
            &argmax_repetition_disabled_offsets_buffer,
            &argmax_repetition_disabled_penalty_buffer,
            &argmax_output,
            &argmax_scratch,
            LastTokenMaskedArgmaxParams {
                vocabulary_size: 5,
                repetition_capacity: 1,
            },
        );
        dispatch_raw_last_token_masked_argmax(
            &pipelines,
            encoder,
            &argmax_logits_buffer,
            &argmax_empty_mask_buffer,
            &argmax_repetition_token_ids_buffer,
            &argmax_repetition_disabled_offsets_buffer,
            &argmax_repetition_disabled_penalty_buffer,
            &argmax_empty_output,
            &argmax_scratch,
            LastTokenMaskedArgmaxParams {
                vocabulary_size: 5,
                repetition_capacity: 1,
            },
        );
        dispatch_raw_last_token_masked_argmax(
            &pipelines,
            encoder,
            &repetition_logits_buffer,
            &repetition_mask_buffer,
            &argmax_repetition_token_ids_buffer,
            &repetition_offsets_buffer,
            &repetition_penalty_buffer,
            &repetition_output,
            &repetition_scratch,
            LastTokenMaskedArgmaxParams {
                vocabulary_size: 3,
                repetition_capacity: 1,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        let embedding = read_f16(&embedding_output, hidden * 2);
        let mut embedding_reference = Vec::with_capacity(hidden * 2);
        embedding_reference.extend_from_slice(&reference);
        embedding_reference.resize(hidden * 2, 0.0);
        numerical_tolerance::assert_matches(
            "Metal/CPU Q6_K token embedding",
            &embedding,
            &[2, hidden],
            &embedding_reference,
            &[2, hidden],
            numerical_tolerance::LogicalDtype::Fp16,
            TOKEN_EMBEDDING_TOLERANCE_ID,
        )
        .expect("reviewed token-embedding numerical contract");
        assert!(embedding[hidden..].iter().all(|value| *value == 0.0));

        let q8_embedding = read_f16(&q8_embedding_output, hidden * 2);
        for (index, (observed, reference)) in q8_embedding[..hidden]
            .iter()
            .zip(q8_reference.iter())
            .enumerate()
        {
            let expected = f16::from_f32(*reference).to_f32();
            assert!(
                (*observed - expected).abs() <= 0.002,
                "Q8_0 token embedding differs at column {index}: observed={observed} expected={expected}"
            );
        }
        assert!(q8_embedding[hidden..].iter().all(|value| *value == 0.0));

        let rms = read_f16(&rms_output, hidden);
        let mean_square = rms_input
            .iter()
            .map(|value| value.to_f32().powi(2))
            .sum::<f32>()
            / hidden as f32;
        let inverse_rms = (mean_square + 1e-6).sqrt().recip();
        let rms_reference = (0..hidden)
            .map(|index| rms_input[index].to_f32() * inverse_rms * rms_weight[index].to_f32())
            .collect::<Vec<_>>();
        numerical_tolerance::assert_matches(
            "Metal/CPU RMSNorm",
            &rms,
            &[1, hidden],
            &rms_reference,
            &[1, hidden],
            numerical_tolerance::LogicalDtype::Fp16,
            RMS_NORM_TOLERANCE_ID,
        )
        .expect("reviewed RMSNorm numerical contract");
        let residual = read_f16(&residual_output, hidden);
        let residual_reference = (0..hidden)
            .map(|index| rms[index] + residual_right[index].to_f32())
            .collect::<Vec<_>>();
        numerical_tolerance::assert_matches(
            "Metal/CPU residual add",
            &residual,
            &[1, hidden],
            &residual_reference,
            &[1, hidden],
            numerical_tolerance::LogicalDtype::Fp16,
            RESIDUAL_ADD_TOLERANCE_ID,
        )
        .expect("reviewed residual-add numerical contract");
        let selected = unsafe { *(argmax_output.contents() as *const u32) };
        assert_eq!(
            selected, 1,
            "mask and lower-index tie-break must both apply"
        );
        let empty_selected = unsafe { *(argmax_empty_output.contents() as *const u32) };
        assert_eq!(
            empty_selected,
            u32::MAX,
            "an all-invalid selection mask must return the typed sentinel"
        );
        let repetition_selected = unsafe { *(repetition_output.contents() as *const u32) };
        assert_eq!(
            repetition_selected, 1,
            "sparse positive-logit repetition penalty must apply before argmax"
        );
        assert_eq!(
            read_f16(&repetition_logits_buffer, repetition_logits.len()),
            repetition_logits
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>(),
            "masked argmax must not mutate semantic logits"
        );
    }

    fn set_raw(encoder: &ComputeCommandEncoderRef, index: u64, buffer: &BufferRef) {
        encoder.set_buffer(index, Some(buffer), 0);
    }

    fn dispatch_raw_embedding(
        pipelines: &MetalPrimitivePipelines,
        encoder: &ComputeCommandEncoderRef,
        format: EmbeddingPhysicalFormat,
        table: &BufferRef,
        token_ids: &BufferRef,
        output: &BufferRef,
        params: EmbeddingParams,
    ) {
        encoder.set_compute_pipeline_state(match format {
            EmbeddingPhysicalFormat::DenseF16 => &pipelines.embedding_dense,
            EmbeddingPhysicalFormat::Q4K => &pipelines.embedding_q4_k,
            EmbeddingPhysicalFormat::Q6K => &pipelines.embedding_q6_k,
            EmbeddingPhysicalFormat::Q8_0 => &pipelines.embedding_q8_0,
            EmbeddingPhysicalFormat::Pq2_0 => &pipelines.embedding_pq2_0,
        });
        set_raw(encoder, 0, table);
        set_raw(encoder, 1, token_ids);
        set_raw(encoder, 2, output);
        encoder.set_bytes(
            3,
            std::mem::size_of::<EmbeddingParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(
                u64::from(params.hidden_size).div_ceil(THREADS_PER_GROUP),
                u64::from(params.token_count),
                1,
            ),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
    }

    fn dispatch_raw_embedding_f32(
        pipelines: &MetalPrimitivePipelines,
        encoder: &ComputeCommandEncoderRef,
        format: EmbeddingPhysicalFormat,
        table: &BufferRef,
        token_ids: &BufferRef,
        output: &BufferRef,
        params: EmbeddingParams,
    ) {
        encoder.set_compute_pipeline_state(match format {
            EmbeddingPhysicalFormat::DenseF16 => &pipelines.embedding_dense_f32,
            EmbeddingPhysicalFormat::Q4K => &pipelines.embedding_q4_k_f32,
            EmbeddingPhysicalFormat::Q6K => &pipelines.embedding_q6_k_f32,
            EmbeddingPhysicalFormat::Q8_0 => &pipelines.embedding_q8_0_f32,
            EmbeddingPhysicalFormat::Pq2_0 => &pipelines.embedding_pq2_0_f32,
        });
        set_raw(encoder, 0, table);
        set_raw(encoder, 1, token_ids);
        set_raw(encoder, 2, output);
        encoder.set_bytes(
            3,
            std::mem::size_of::<EmbeddingParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(
                u64::from(params.hidden_size).div_ceil(THREADS_PER_GROUP),
                u64::from(params.token_count),
                1,
            ),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
    }

    fn dispatch_raw_rms_norm(
        pipelines: &MetalPrimitivePipelines,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        weight: &BufferRef,
        output: &BufferRef,
        params: RmsNormParams,
    ) {
        encoder.set_compute_pipeline_state(&pipelines.rms_norm);
        set_raw(encoder, 0, input);
        set_raw(encoder, 1, weight);
        set_raw(encoder, 2, output);
        encoder.set_bytes(
            3,
            std::mem::size_of::<RmsNormParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder.set_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(params.rows), 1, 1),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
    }

    fn dispatch_raw_rms_norm_f32(
        pipelines: &MetalPrimitivePipelines,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        weight: &BufferRef,
        output: &BufferRef,
        params: RmsNormParams,
        output_f32: bool,
    ) {
        encoder.set_compute_pipeline_state(if output_f32 {
            &pipelines.rms_norm_f32
        } else {
            &pipelines.rms_norm_f32_to_f16
        });
        set_raw(encoder, 0, input);
        set_raw(encoder, 1, weight);
        set_raw(encoder, 2, output);
        encoder.set_bytes(
            3,
            std::mem::size_of::<RmsNormParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder.set_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(params.rows), 1, 1),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
    }

    fn dispatch_raw_residual_add(
        pipelines: &MetalPrimitivePipelines,
        encoder: &ComputeCommandEncoderRef,
        left: &BufferRef,
        right: &BufferRef,
        output: &BufferRef,
        params: ResidualAddParams,
    ) {
        encoder.set_compute_pipeline_state(&pipelines.residual_add);
        set_raw(encoder, 0, left);
        set_raw(encoder, 1, right);
        set_raw(encoder, 2, output);
        encoder.set_bytes(
            3,
            std::mem::size_of::<ResidualAddParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(params.elements).div_ceil(THREADS_PER_GROUP), 1, 1),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
    }

    fn dispatch_raw_residual_add_f32_f16(
        pipelines: &MetalPrimitivePipelines,
        encoder: &ComputeCommandEncoderRef,
        left: &BufferRef,
        right: &BufferRef,
        output: &BufferRef,
        params: ResidualAddParams,
    ) {
        encoder.set_compute_pipeline_state(&pipelines.residual_add_f32_f16);
        set_raw(encoder, 0, left);
        set_raw(encoder, 1, right);
        set_raw(encoder, 2, output);
        encoder.set_bytes(
            3,
            std::mem::size_of::<ResidualAddParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(params.elements).div_ceil(THREADS_PER_GROUP), 1, 1),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
    }

    fn dispatch_raw_last_token_masked_argmax(
        pipelines: &MetalPrimitivePipelines,
        encoder: &ComputeCommandEncoderRef,
        logits: &BufferRef,
        valid_mask: &BufferRef,
        repetition_token_ids: &BufferRef,
        repetition_offsets: &BufferRef,
        repetition_penalty: &BufferRef,
        output: &BufferRef,
        scratch: &BufferRef,
        params: LastTokenMaskedArgmaxParams,
    ) {
        encoder.set_compute_pipeline_state(&pipelines.last_token_masked_argmax);
        set_raw(encoder, 0, logits);
        set_raw(encoder, 1, scratch);
        set_raw(encoder, 2, valid_mask);
        set_raw(encoder, 3, repetition_token_ids);
        set_raw(encoder, 4, repetition_offsets);
        set_raw(encoder, 5, repetition_penalty);
        set_raw(encoder, 6, output);
        encoder.set_bytes(
            7,
            std::mem::size_of::<LastTokenMaskedArgmaxParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder
            .dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(THREADS_PER_GROUP, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_raw_last_token_masked_argmax_f32(
        pipelines: &MetalPrimitivePipelines,
        encoder: &ComputeCommandEncoderRef,
        logits: &BufferRef,
        valid_mask: &BufferRef,
        repetition_token_ids: &BufferRef,
        repetition_offsets: &BufferRef,
        repetition_penalty: &BufferRef,
        output: &BufferRef,
        scratch: &BufferRef,
        params: LastTokenMaskedArgmaxParams,
    ) {
        encoder.set_compute_pipeline_state(&pipelines.last_token_masked_argmax_f32);
        set_raw(encoder, 0, logits);
        set_raw(encoder, 1, scratch);
        set_raw(encoder, 2, valid_mask);
        set_raw(encoder, 3, repetition_token_ids);
        set_raw(encoder, 4, repetition_offsets);
        set_raw(encoder, 5, repetition_penalty);
        set_raw(encoder, 6, output);
        encoder.set_bytes(
            7,
            std::mem::size_of::<LastTokenMaskedArgmaxParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder
            .dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(THREADS_PER_GROUP, 1, 1));
    }
}

#[cfg(test)]
pub(crate) use selected::tests::runtime_fixtures as selected_runtime_fixtures;

pub(super) fn append_selected_rms(
    builder: &mut ferrum_interfaces::execution_cost::SelectedCommandCostBuilderV1,
    p: &MetalPrimitivePipelines,
    input: ElementType,
    output: ElementType,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
    scratch: u64,
) -> Option<()> {
    selected::push_rms(
        builder,
        p,
        RmsNormParams {
            rows,
            hidden_size,
            epsilon,
        },
        input,
        output,
        scratch,
    )
}
pub(super) fn append_selected_residual(
    builder: &mut ferrum_interfaces::execution_cost::SelectedCommandCostBuilderV1,
    p: &MetalPrimitivePipelines,
    hidden: ElementType,
    elements: u32,
    scratch: u64,
) -> Option<()> {
    selected::push_residual(
        builder,
        p,
        ResidualAddParams { elements },
        hidden,
        ElementType::F16,
        hidden,
        scratch,
    )
}
