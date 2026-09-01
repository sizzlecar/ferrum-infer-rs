//! CUDA implementations of backend-neutral dense transformer operations.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;

use cudarc::cublas::{
    result::gemm_ex,
    sys::{cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType_t},
    CudaBlas,
};
#[cfg(feature = "vllm-marlin")]
use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
#[cfg(feature = "vllm-marlin")]
use ferrum_interfaces::vnext::PhysicalWeightLayout;
use ferrum_interfaces::vnext::{
    constant_scale_contract, dense_geglu_tanh_contract, dense_linear_contract,
    dense_swiglu_contract, logit_softcap_contract, residual_add_contract, rms_norm_contract,
    AttributeId, BatchedOperationInvocation, CanonicalRational, CapabilityId, ContractVersion,
    DeviceBatchingForm, DeviceRuntime, DynamicStorageRequirement, ElementType,
    EncodedDeviceOperation, OperationContract, OperationFailure, OperationInvocation,
    OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, ProfilePhase, ProviderId,
    ProviderStorageBindingRequirement, ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy,
    ProviderWorkspaceScope, ProviderWorkspaceSizeFormula, QuantizationFormatId,
    ResolvedTensorLayout, ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, ReusableExecutionValueAddress,
    ReusableExecutionWorkspaceAddress, SemanticValue, VNextError, WeightFormatId,
    CONSTANT_SCALE_F16_CAPABILITY_ID, CONSTANT_SCALE_OPERATION_ID,
    DENSE_GEGLU_TANH_F16_CAPABILITY_ID, DENSE_GEGLU_TANH_OPERATION_ID,
    DENSE_LINEAR_F16_CAPABILITY_ID, DENSE_LINEAR_OPERATION_ID, DENSE_SWIGLU_F16_CAPABILITY_ID,
    DENSE_SWIGLU_OPERATION_ID, LOGIT_SOFTCAP_F16_CAPABILITY_ID, LOGIT_SOFTCAP_OPERATION_ID,
    RESIDUAL_ADD_F16_CAPABILITY_ID, RESIDUAL_ADD_OPERATION_ID, RMS_NORM_F16_CAPABILITY_ID,
    RMS_NORM_OPERATION_ID,
};

use super::super::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime,
    CudaDeviceRuntimeError, CudaProgramBindingWrite,
};
use super::{
    binding, contiguous_region, contiguous_token_region, contract_error,
    implementation_fingerprint, same_physical_region, DENSE_SAFETENSORS_FORMAT_ID,
    THREADS_PER_BLOCK, VALUE_ALIGNMENT_BYTES,
};
#[cfg(feature = "vllm-marlin")]
use crate::backend::cuda::vllm_marlin::{
    launch_marlin_mm_f16_weight, MarlinF16WeightType, MarlinMmBuffers, MarlinMmExecution,
    MarlinMmF16WeightRequest, MarlinMmProblem,
};
use crate::backend::cuda::vnext_replay::CudaCommandReplayKeyBuilder;
#[cfg(feature = "vllm-marlin")]
use crate::marlin_fp8_materializer::{
    marlin_fp8_projection_shape_supported, MARLIN_FP8_CAPABILITY_ID,
    MARLIN_FP8_GROUP128_QUANTIZATION_FORMAT_ID, MARLIN_FP8_GROUP128_WEIGHT_FORMAT_ID,
    MARLIN_FP8_QUANTIZATION_FORMAT_ID, MARLIN_FP8_WEIGHT_FORMAT_ID,
};
#[cfg(feature = "vllm-marlin")]
use moe_weights::{
    resolve_compressed_tensors_marlin_layout, resolve_compressed_tensors_marlin_matrix_weight,
    resolve_compressed_tensors_symmetric_marlin_matrix_weight,
    COMPRESSED_TENSORS_MARLIN_QUANTIZATION_FORMAT_ID,
    COMPRESSED_TENSORS_MARLIN_SYMMETRIC_QUANTIZATION_FORMAT_ID,
    COMPRESSED_TENSORS_MARLIN_SYMMETRIC_WEIGHT_FORMAT_ID,
    COMPRESSED_TENSORS_MARLIN_WEIGHT_FORMAT_ID,
};

mod attention;
mod causal_attention;
mod gpt_oss_attention;
#[cfg(feature = "vllm-moe-marlin")]
mod gpt_oss_moe;
#[cfg(feature = "vllm-marlin")]
mod marlin_fp8_weights;
#[cfg(feature = "vllm-moe-marlin")]
mod moe;
#[cfg(feature = "vllm-moe-marlin")]
mod moe_launch;
#[cfg(feature = "vllm-moe-marlin")]
mod moe_routed;
#[cfg(feature = "vllm-marlin")]
mod moe_weights;
#[cfg(feature = "vllm-moe-marlin")]
mod moe_workspace;

pub(super) use attention::CudaGatedDeltaRecurrentAttentionProvider;
pub(super) use causal_attention::CudaCausalPagedAttentionProvider;
pub(super) use gpt_oss_attention::CudaGptOssCausalPagedAttentionProvider;
#[cfg(feature = "vllm-moe-marlin")]
pub(super) use gpt_oss_moe::CudaGptOssRoutedClampedSwiGluMoeProvider;
#[cfg(feature = "vllm-moe-marlin")]
pub(super) use moe::CudaRoutedSharedSwiGluMoeProvider;
#[cfg(feature = "vllm-moe-marlin")]
pub(super) use moe_routed::CudaRoutedSwiGluMoeProvider;
#[cfg(feature = "vllm-marlin")]
pub(super) use moe_weights::{
    COMPRESSED_TENSORS_MARLIN_CAPABILITY_ID, COMPRESSED_TENSORS_MARLIN_SYMMETRIC_CAPABILITY_ID,
    GPTQ_MARLIN_CAPABILITY_ID,
};

const RMS_NORM_PROVIDER_ID: &str = "provider.cuda.rms_norm.f16";
const RMS_NORM_ESTIMATOR_ID: &str = "resource-estimator.cuda.rms_norm.f16";
const DENSE_LINEAR_PROVIDER_ID: &str = "provider.cuda.dense_linear.f16.cublas";
const DENSE_LINEAR_ESTIMATOR_ID: &str = "resource-estimator.cuda.dense_linear.f16.cublas";
#[cfg(feature = "vllm-marlin")]
const MARLIN_FP8_DENSE_LINEAR_PROVIDER_ID: &str = "provider.cuda.dense_linear.f16.marlin-fp8-w8a16";
#[cfg(feature = "vllm-marlin")]
const MARLIN_FP8_DENSE_LINEAR_ESTIMATOR_ID: &str =
    "resource-estimator.cuda.dense_linear.f16.marlin-fp8-w8a16";
const DENSE_SWIGLU_PROVIDER_ID: &str = "provider.cuda.dense_swiglu.f16.cublas";
const DENSE_SWIGLU_ESTIMATOR_ID: &str = "resource-estimator.cuda.dense_swiglu.f16.cublas";
const DENSE_GEGLU_TANH_PROVIDER_ID: &str = "provider.cuda.dense_geglu_tanh.f16.cublas";
const DENSE_GEGLU_TANH_ESTIMATOR_ID: &str = "resource-estimator.cuda.dense_geglu_tanh.f16.cublas";
const CONSTANT_SCALE_PROVIDER_ID: &str = "provider.cuda.constant_scale.f16";
const CONSTANT_SCALE_ESTIMATOR_ID: &str = "resource-estimator.cuda.constant_scale.f16";
const LOGIT_SOFTCAP_PROVIDER_ID: &str = "provider.cuda.logit_softcap.f16";
const LOGIT_SOFTCAP_ESTIMATOR_ID: &str = "resource-estimator.cuda.logit_softcap.f16";
const RESIDUAL_ADD_PROVIDER_ID: &str = "provider.cuda.residual_add.f16";
const RESIDUAL_ADD_ESTIMATOR_ID: &str = "resource-estimator.cuda.residual_add.f16";

const RMS_NORM_FUNCTION_NAME: &str = "rms_norm_f16";
const SILU_MUL_FUNCTION_NAME: &str = "fused_silu_mul_interleaved_f16";
#[cfg(feature = "vllm-marlin")]
const PLANAR_SILU_MUL_FUNCTION_NAME: &str = "fused_silu_mul_f16";
const PLANAR_GELU_TANH_MUL_FUNCTION_NAME: &str = "fused_gelu_tanh_mul_f16";
const SCALE_INPLACE_FUNCTION_NAME: &str = "scale_inplace_f16";
const LOGIT_SOFTCAP_INPLACE_FUNCTION_NAME: &str = "logit_softcap_inplace_f16";
const RESIDUAL_ADD_FUNCTION_NAME: &str = "residual_add_f16";
const SWIGLU_SCRATCH_PARTS: u64 = 3;
const GEGLU_SCRATCH_PARTS: u64 = 3;
static CUDA_GEMM_ALPHA_F32: f32 = 1.0;
static CUDA_GEMM_BETA_F32: f32 = 0.0;

fn attach_invocation_binding<C>(
    operation: EncodedDeviceOperation<C>,
    binding_command: C,
    has_compiled_program_slot: bool,
) -> EncodedDeviceOperation<C> {
    if has_compiled_program_slot {
        operation.with_program_binding(binding_command)
    } else {
        operation.with_dynamic_binding(binding_command)
    }
}

#[derive(Clone, Copy)]
pub(super) enum CapturedProviderWorkspace {
    Scratch,
    Binding,
    Persistent,
}

pub(super) fn captured_contiguous_addresses_are_reusable(
    request: &ReusableExecutionTopologyRequest<'_>,
    input_count: u32,
    workspaces: &[CapturedProviderWorkspace],
) -> Result<bool, VNextError> {
    let mut values = (0..input_count)
        .map(|ordinal| ReusableExecutionValueAddress::captured(ResolvedValueRole::Input, ordinal))
        .collect::<Vec<_>>();
    values.push(ReusableExecutionValueAddress::captured(
        ResolvedValueRole::Output,
        0,
    ));
    let workspaces = workspaces
        .iter()
        .map(|workspace| match workspace {
            CapturedProviderWorkspace::Scratch => ReusableExecutionWorkspaceAddress::Scratch,
            CapturedProviderWorkspace::Binding => ReusableExecutionWorkspaceAddress::Binding,
            CapturedProviderWorkspace::Persistent => ReusableExecutionWorkspaceAddress::Persistent,
        })
        .collect::<Vec<_>>();
    request
        .reusable_address_scope(&values, &workspaces)
        .map(|scope| scope.is_some())
}

pub(super) fn static_contiguous_reusable_topology(
    request: &ReusableExecutionTopologyRequest<'_>,
    input_count: u32,
    workspaces: &[CapturedProviderWorkspace],
) -> Result<ReusableExecutionTopology, VNextError> {
    if captured_contiguous_addresses_are_reusable(request, input_count, workspaces)? {
        Ok(ReusableExecutionTopology::Static)
    } else {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
}

pub(super) struct CudaRmsNormProvider {
    descriptor: OperationProviderDescriptor,
    function: CudaFunction,
}

impl CudaRmsNormProvider {
    pub(super) fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = rms_norm_contract().map_err(contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            RMS_NORM_PROVIDER_ID,
            RMS_NORM_F16_CAPABILITY_ID,
            RMS_NORM_ESTIMATOR_ID,
            contiguous_bindings(2),
            implementation_fingerprint(&[
                include_str!("transformer.rs").as_bytes(),
                crate::ptx::RMS_NORM.as_bytes(),
                RMS_NORM_FUNCTION_NAME.as_bytes(),
            ]),
        )?;
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::RMS_NORM.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("RMSNorm module load", error))?;
        let function = module
            .load_function(RMS_NORM_FUNCTION_NAME)
            .map_err(|error| CudaDeviceRuntimeError::driver("RMSNorm function load", error))?;
        Ok(Self {
            descriptor,
            function,
        })
    }
}

impl OperationResourceEstimator for CudaRmsNormProvider {
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

impl OperationProvider<CudaDeviceRuntime> for CudaRmsNormProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        static_contiguous_reusable_topology(&request, 2, &[])
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_rms_norm(
            self.descriptor.provider_implementation_fingerprint(),
            &self.function,
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| provider_failure(identity, "cuda.rms_norm.encode", message))
    }
}

pub(super) struct CudaDenseLinearProvider {
    descriptor: OperationProviderDescriptor,
}

impl CudaDenseLinearProvider {
    pub(super) fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = dense_linear_contract().map_err(contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            DENSE_LINEAR_PROVIDER_ID,
            DENSE_LINEAR_F16_CAPABILITY_ID,
            DENSE_LINEAR_ESTIMATOR_ID,
            contiguous_bindings(2),
            implementation_fingerprint(&[
                include_str!("transformer.rs").as_bytes(),
                DENSE_LINEAR_PROVIDER_ID.as_bytes(),
            ]),
        )?;
        Ok(Self { descriptor })
    }
}

impl OperationResourceEstimator for CudaDenseLinearProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        estimate_without_workspace(&self.descriptor, &request, DENSE_LINEAR_OPERATION_ID)
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaDenseLinearProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        static_contiguous_reusable_topology(&request, 2, &[])
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_dense_linear(
            self.descriptor.provider_implementation_fingerprint(),
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| provider_failure(identity, "cuda.dense_linear.encode", message))
    }
}

#[cfg(feature = "vllm-marlin")]
pub(super) struct CudaMarlinFp8DenseLinearProvider {
    descriptor: OperationProviderDescriptor,
    projection_runtime: MarlinProjectionRuntime,
}

#[cfg(feature = "vllm-marlin")]
impl CudaMarlinFp8DenseLinearProvider {
    pub(super) fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = dense_linear_contract().map_err(contract_error)?;
        let operation_capability =
            CapabilityId::new(DENSE_LINEAR_F16_CAPABILITY_ID).map_err(contract_error)?;
        let marlin_capability =
            CapabilityId::new(MARLIN_FP8_CAPABILITY_ID).map_err(contract_error)?;
        if !runtime
            .descriptor()
            .capabilities
            .contains(&operation_capability)
            || !runtime
                .descriptor()
                .capabilities
                .contains(&marlin_capability)
        {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA runtime does not advertise dense-linear Marlin FP8 capabilities",
            ));
        }
        let provider_fingerprint = implementation_fingerprint(&[
            include_str!("transformer.rs").as_bytes(),
            include_str!("transformer/marlin_fp8_weights.rs").as_bytes(),
            include_str!("../vllm_marlin.rs").as_bytes(),
            MARLIN_FP8_DENSE_LINEAR_PROVIDER_ID.as_bytes(),
        ]);
        let estimator_fingerprint = implementation_fingerprint(&[
            include_str!("transformer.rs").as_bytes(),
            MARLIN_FP8_DENSE_LINEAR_ESTIMATOR_ID.as_bytes(),
            provider_fingerprint.as_bytes(),
        ]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(MARLIN_FP8_DENSE_LINEAR_PROVIDER_ID).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            provider_fingerprint,
            ferrum_interfaces::vnext::ProviderExecutionSemantics::bitwise_eager_and_replay(),
            contract.descriptor().version,
            runtime.descriptor().id.clone(),
            BTreeSet::from([operation_capability, marlin_capability]),
            BTreeSet::from([
                WeightFormatId::new(MARLIN_FP8_WEIGHT_FORMAT_ID).map_err(contract_error)?,
                WeightFormatId::new(MARLIN_FP8_GROUP128_WEIGHT_FORMAT_ID)
                    .map_err(contract_error)?,
            ]),
            BTreeSet::from([
                QuantizationFormatId::new(MARLIN_FP8_QUANTIZATION_FORMAT_ID)
                    .map_err(contract_error)?,
                QuantizationFormatId::new(MARLIN_FP8_GROUP128_QUANTIZATION_FORMAT_ID)
                    .map_err(contract_error)?,
            ]),
            contiguous_bindings(2),
            MARLIN_FP8_DENSE_LINEAR_ESTIMATOR_ID,
            ContractVersion::new(1, 0),
            estimator_fingerprint,
        )
        .map_err(contract_error)?;
        let projection_runtime = MarlinProjectionRuntime::query(runtime)?;
        Ok(Self {
            descriptor,
            projection_runtime,
        })
    }

    fn workspace_bytes(&self) -> Result<u64, VNextError> {
        self.projection_runtime
            .workspace_bytes()
            .map_err(invalid_plan)
    }
}

#[cfg(feature = "vllm-marlin")]
impl OperationResourceEstimator for CudaMarlinFp8DenseLinearProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        ensure_estimator_request(&self.descriptor, &request, DENSE_LINEAR_OPERATION_ID)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::fixed(self.workspace_bytes()?)?,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(estimate(
            &self.descriptor,
            request.input_fingerprint(),
            Some(scratch),
        ))
    }
}

#[cfg(feature = "vllm-marlin")]
impl OperationProvider<CudaDeviceRuntime> for CudaMarlinFp8DenseLinearProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        static_contiguous_reusable_topology(&request, 2, &[CapturedProviderWorkspace::Scratch])
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_marlin_fp8_dense_linear(
            self.descriptor.provider_implementation_fingerprint(),
            self.projection_runtime,
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| {
            provider_failure(identity, "cuda.dense_linear.marlin_fp8.encode", message)
        })
    }
}

#[cfg(feature = "vllm-marlin")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenseSwiGluProjection {
    F16,
    MarlinFp8,
    CompressedTensorsMarlin,
}

#[cfg(feature = "vllm-marlin")]
fn dense_swiglu_projection_for_formats(
    gate_up: &BTreeSet<QuantizationFormatId>,
    down: &BTreeSet<QuantizationFormatId>,
) -> Result<DenseSwiGluProjection, String> {
    let classify = |label: &str, formats: &BTreeSet<QuantizationFormatId>| {
        if formats.is_empty() {
            return Ok(DenseSwiGluProjection::F16);
        }
        if formats.len() != 1 {
            return Err(format!(
                "dense SwiGLU {label} has more than one quantization format"
            ));
        }
        match formats.iter().next().map(|format| format.as_str()) {
            Some(MARLIN_FP8_QUANTIZATION_FORMAT_ID)
            | Some(MARLIN_FP8_GROUP128_QUANTIZATION_FORMAT_ID) => {
                Ok(DenseSwiGluProjection::MarlinFp8)
            }
            Some(COMPRESSED_TENSORS_MARLIN_QUANTIZATION_FORMAT_ID) => {
                Ok(DenseSwiGluProjection::CompressedTensorsMarlin)
            }
            Some(format) => Err(format!(
                "dense SwiGLU {label} uses unsupported quantization format `{format}`"
            )),
            None => unreachable!("non-empty quantization set has one entry"),
        }
    };
    let gate_up = classify("gate/up", gate_up)?;
    let down = classify("down", down)?;
    if gate_up != down {
        return Err(format!(
            "dense SwiGLU cannot mix {gate_up:?} gate/up with {down:?} down weights"
        ));
    }
    Ok(gate_up)
}

#[cfg(feature = "vllm-marlin")]
fn participant_dense_swiglu_projection(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<DenseSwiGluProjection, String> {
    let formats = |ordinal| {
        binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?
            .weight()
            .map(|weight| weight.quantization_formats())
            .ok_or_else(|| format!("dense SwiGLU input {ordinal} has no physical weight layout"))
    };
    dense_swiglu_projection_for_formats(&formats(1)?, &formats(2)?)
}

#[cfg(feature = "vllm-marlin")]
fn dense_swiglu_projection(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<DenseSwiGluProjection, String> {
    let first = participant_dense_swiglu_projection(&invocation.participants()[0])?;
    for participant in &invocation.participants()[1..] {
        if participant_dense_swiglu_projection(participant)? != first {
            return Err("dense SwiGLU participants disagree on their projection ABI".to_owned());
        }
    }
    Ok(first)
}

pub(super) struct CudaDenseSwiGluProvider {
    descriptor: OperationProviderDescriptor,
    silu_mul: CudaFunction,
    #[cfg(feature = "vllm-marlin")]
    planar_silu_mul: CudaFunction,
    #[cfg(feature = "vllm-marlin")]
    projection_runtime: MarlinProjectionRuntime,
}

impl CudaDenseSwiGluProvider {
    pub(super) fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = dense_swiglu_contract().map_err(contract_error)?;
        let provider_fingerprint = implementation_fingerprint(&[
            include_str!("transformer.rs").as_bytes(),
            crate::ptx::FUSED_SILU_MUL.as_bytes(),
            SILU_MUL_FUNCTION_NAME.as_bytes(),
            #[cfg(feature = "vllm-marlin")]
            PLANAR_SILU_MUL_FUNCTION_NAME.as_bytes(),
            #[cfg(feature = "vllm-marlin")]
            include_str!("transformer/marlin_fp8_weights.rs").as_bytes(),
            #[cfg(feature = "vllm-marlin")]
            include_str!("transformer/moe_weights.rs").as_bytes(),
            #[cfg(feature = "vllm-marlin")]
            include_str!("../vllm_marlin.rs").as_bytes(),
        ]);
        #[cfg(not(feature = "vllm-marlin"))]
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            DENSE_SWIGLU_PROVIDER_ID,
            DENSE_SWIGLU_F16_CAPABILITY_ID,
            DENSE_SWIGLU_ESTIMATOR_ID,
            contiguous_bindings(3),
            provider_fingerprint,
        )?;
        #[cfg(feature = "vllm-marlin")]
        let descriptor =
            marlin_swiglu_provider_descriptor(runtime, &contract, provider_fingerprint)?;
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("SwiGLU module load", error))?;
        let silu_mul = module
            .load_function(SILU_MUL_FUNCTION_NAME)
            .map_err(|error| CudaDeviceRuntimeError::driver("SwiGLU function load", error))?;
        #[cfg(feature = "vllm-marlin")]
        let planar_silu_mul = module
            .load_function(PLANAR_SILU_MUL_FUNCTION_NAME)
            .map_err(|error| {
                CudaDeviceRuntimeError::driver("planar SwiGLU function load", error)
            })?;
        #[cfg(feature = "vllm-marlin")]
        let projection_runtime = MarlinProjectionRuntime::query(runtime)?;
        Ok(Self {
            descriptor,
            silu_mul,
            #[cfg(feature = "vllm-marlin")]
            planar_silu_mul,
            #[cfg(feature = "vllm-marlin")]
            projection_runtime,
        })
    }
}

impl OperationResourceEstimator for CudaDenseSwiGluProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        ensure_estimator_request(&self.descriptor, &request, DENSE_SWIGLU_OPERATION_ID)?;
        let intermediate_size =
            unsigned_attribute(request.attributes(), "intermediate_size").map_err(invalid_plan)?;
        let bytes_per_token = intermediate_size
            .checked_mul(SWIGLU_SCRATCH_PARTS)
            .and_then(|elements| elements.checked_mul(ElementType::F16.size_bytes()))
            .ok_or_else(|| invalid_plan("CUDA dense SwiGLU scratch size overflows"))?;
        #[cfg(not(feature = "vllm-marlin"))]
        let formula = ProviderWorkspaceSizeFormula::tokens(bytes_per_token)?;
        #[cfg(feature = "vllm-marlin")]
        let formula = ProviderWorkspaceSizeFormula::affine(
            self.projection_runtime
                .workspace_bytes()
                .map_err(invalid_plan)?
                .checked_add(VALUE_ALIGNMENT_BYTES - 1)
                .ok_or_else(|| invalid_plan("CUDA dense SwiGLU Marlin scratch overflows"))?,
            0,
            bytes_per_token,
        )?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            formula,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(estimate(
            &self.descriptor,
            request.input_fingerprint(),
            Some(scratch),
        ))
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaDenseSwiGluProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        static_contiguous_reusable_topology(&request, 3, &[CapturedProviderWorkspace::Scratch])
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        #[cfg(feature = "vllm-marlin")]
        {
            let projection = dense_swiglu_projection(&invocation).map_err(|message| {
                provider_failure(identity.clone(), "cuda.dense_swiglu.select", message)
            })?;
            match projection {
                DenseSwiGluProjection::CompressedTensorsMarlin => {
                    return encode_compressed_tensors_dense_swiglu(
                        self.descriptor.provider_implementation_fingerprint(),
                        &self.planar_silu_mul,
                        self.projection_runtime,
                        invocation,
                    )
                    .map(EncodedDeviceOperation::compute)
                    .map_err(|message| {
                        provider_failure(
                            identity,
                            "cuda.dense_swiglu.compressed_tensors.encode",
                            message,
                        )
                    });
                }
                DenseSwiGluProjection::MarlinFp8 => {
                    return encode_marlin_fp8_dense_swiglu(
                        self.descriptor.provider_implementation_fingerprint(),
                        &self.planar_silu_mul,
                        self.projection_runtime,
                        invocation,
                    )
                    .map(EncodedDeviceOperation::compute)
                    .map_err(|message| {
                        provider_failure(identity, "cuda.dense_swiglu.marlin_fp8.encode", message)
                    });
                }
                DenseSwiGluProjection::F16 => {}
            }
        }
        encode_dense_swiglu(
            self.descriptor.provider_implementation_fingerprint(),
            &self.silu_mul,
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| provider_failure(identity, "cuda.dense_swiglu.encode", message))
    }
}

/// Physical projection ABI implemented by the dense GeGLU provider.
///
/// Keep the dense and symmetric compressed-tensors paths distinct so neither
/// can be accidentally executed with the other's physical weight ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenseGeGluProjection {
    F16,
    #[cfg(feature = "vllm-marlin")]
    CompressedTensorsSymmetricMarlin,
}

fn dense_geglu_projection_for_formats(
    gate: &BTreeSet<QuantizationFormatId>,
    up: &BTreeSet<QuantizationFormatId>,
    down: &BTreeSet<QuantizationFormatId>,
) -> Result<DenseGeGluProjection, String> {
    let classify = |label: &str, formats: &BTreeSet<QuantizationFormatId>| {
        if formats.is_empty() {
            return Ok(DenseGeGluProjection::F16);
        }
        if formats.len() != 1 {
            return Err(format!(
                "dense GeGLU {label} has more than one quantization format"
            ));
        }
        match formats.iter().next().map(|format| format.as_str()) {
            #[cfg(feature = "vllm-marlin")]
            Some(COMPRESSED_TENSORS_MARLIN_SYMMETRIC_QUANTIZATION_FORMAT_ID) => {
                Ok(DenseGeGluProjection::CompressedTensorsSymmetricMarlin)
            }
            Some(format) => Err(format!(
                "dense GeGLU {label} uses unsupported quantization format `{format}`"
            )),
            None => unreachable!("non-empty quantization set has one entry"),
        }
    };
    let gate = classify("gate", gate)?;
    let up = classify("up", up)?;
    let down = classify("down", down)?;
    if gate != up || gate != down {
        return Err(format!(
            "dense GeGLU cannot mix {gate:?} gate, {up:?} up, and {down:?} down weights"
        ));
    }
    Ok(gate)
}

fn participant_dense_geglu_projection(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<DenseGeGluProjection, String> {
    let formats = |ordinal| {
        binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?
            .weight()
            .map(|weight| weight.quantization_formats())
            .ok_or_else(|| format!("dense GeGLU input {ordinal} has no physical weight layout"))
    };
    dense_geglu_projection_for_formats(&formats(1)?, &formats(2)?, &formats(3)?)
}

fn dense_geglu_projection(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<DenseGeGluProjection, String> {
    let first = participant_dense_geglu_projection(&invocation.participants()[0])?;
    for participant in &invocation.participants()[1..] {
        if participant_dense_geglu_projection(participant)? != first {
            return Err("dense GeGLU participants disagree on their projection ABI".to_owned());
        }
    }
    Ok(first)
}

pub(super) struct CudaDenseGeGluTanhProvider {
    descriptor: OperationProviderDescriptor,
    gelu_tanh_mul: CudaFunction,
    #[cfg(feature = "vllm-marlin")]
    projection_runtime: MarlinProjectionRuntime,
}

impl CudaDenseGeGluTanhProvider {
    pub(super) fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = dense_geglu_tanh_contract().map_err(contract_error)?;
        let provider_fingerprint = implementation_fingerprint(&[
            include_str!("transformer.rs").as_bytes(),
            crate::ptx::FUSED_SILU_MUL.as_bytes(),
            PLANAR_GELU_TANH_MUL_FUNCTION_NAME.as_bytes(),
            #[cfg(feature = "vllm-marlin")]
            include_str!("transformer/moe_weights.rs").as_bytes(),
            #[cfg(feature = "vllm-marlin")]
            include_str!("../vllm_marlin.rs").as_bytes(),
        ]);
        #[cfg(not(feature = "vllm-marlin"))]
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            DENSE_GEGLU_TANH_PROVIDER_ID,
            DENSE_GEGLU_TANH_F16_CAPABILITY_ID,
            DENSE_GEGLU_TANH_ESTIMATOR_ID,
            contiguous_bindings(4),
            provider_fingerprint,
        )?;
        #[cfg(feature = "vllm-marlin")]
        let descriptor =
            marlin_geglu_provider_descriptor(runtime, &contract, provider_fingerprint)?;
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("GeGLU module load", error))?;
        let gelu_tanh_mul = module
            .load_function(PLANAR_GELU_TANH_MUL_FUNCTION_NAME)
            .map_err(|error| CudaDeviceRuntimeError::driver("planar GeGLU function load", error))?;
        #[cfg(feature = "vllm-marlin")]
        let projection_runtime = MarlinProjectionRuntime::query(runtime)?;
        Ok(Self {
            descriptor,
            gelu_tanh_mul,
            #[cfg(feature = "vllm-marlin")]
            projection_runtime,
        })
    }
}

impl OperationResourceEstimator for CudaDenseGeGluTanhProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        ensure_estimator_request(&self.descriptor, &request, DENSE_GEGLU_TANH_OPERATION_ID)?;
        let intermediate_size =
            unsigned_attribute(request.attributes(), "intermediate_size").map_err(invalid_plan)?;
        let bytes_per_token = intermediate_size
            .checked_mul(GEGLU_SCRATCH_PARTS)
            .and_then(|elements| elements.checked_mul(ElementType::F16.size_bytes()))
            .ok_or_else(|| invalid_plan("CUDA dense GeGLU scratch size overflows"))?;
        #[cfg(not(feature = "vllm-marlin"))]
        let formula = ProviderWorkspaceSizeFormula::tokens(bytes_per_token)?;
        #[cfg(feature = "vllm-marlin")]
        let formula = ProviderWorkspaceSizeFormula::affine(
            self.projection_runtime
                .workspace_bytes()
                .map_err(invalid_plan)?
                .checked_add(VALUE_ALIGNMENT_BYTES - 1)
                .ok_or_else(|| invalid_plan("CUDA dense GeGLU Marlin scratch overflows"))?,
            0,
            bytes_per_token,
        )?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            formula,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(estimate(
            &self.descriptor,
            request.input_fingerprint(),
            Some(scratch),
        ))
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaDenseGeGluTanhProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        static_contiguous_reusable_topology(&request, 4, &[CapturedProviderWorkspace::Scratch])
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        let projection = dense_geglu_projection(&invocation).map_err(|message| {
            provider_failure(identity.clone(), "cuda.dense_geglu_tanh.select", message)
        })?;
        match projection {
            DenseGeGluProjection::F16 => encode_dense_geglu_tanh(
                self.descriptor.provider_implementation_fingerprint(),
                &self.gelu_tanh_mul,
                invocation,
            ),
            #[cfg(feature = "vllm-marlin")]
            DenseGeGluProjection::CompressedTensorsSymmetricMarlin => {
                encode_compressed_tensors_symmetric_dense_geglu(
                    self.descriptor.provider_implementation_fingerprint(),
                    &self.gelu_tanh_mul,
                    self.projection_runtime,
                    invocation,
                )
            }
        }
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| provider_failure(identity, "cuda.dense_geglu_tanh.encode", message))
    }
}

pub(super) struct CudaConstantScaleProvider {
    descriptor: OperationProviderDescriptor,
    function: CudaFunction,
}

impl CudaConstantScaleProvider {
    pub(super) fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = constant_scale_contract().map_err(contract_error)?;
        let descriptor = weightless_provider_descriptor(
            runtime,
            &contract,
            CONSTANT_SCALE_PROVIDER_ID,
            CONSTANT_SCALE_F16_CAPABILITY_ID,
            CONSTANT_SCALE_ESTIMATOR_ID,
            contiguous_bindings(1),
            implementation_fingerprint(&[
                include_str!("transformer.rs").as_bytes(),
                crate::ptx::FUSED_SILU_MUL.as_bytes(),
                SCALE_INPLACE_FUNCTION_NAME.as_bytes(),
            ]),
        )?;
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("constant scale module load", error))?;
        let function = module
            .load_function(SCALE_INPLACE_FUNCTION_NAME)
            .map_err(|error| {
                CudaDeviceRuntimeError::driver("constant scale function load", error)
            })?;
        Ok(Self {
            descriptor,
            function,
        })
    }
}

impl OperationResourceEstimator for CudaConstantScaleProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        estimate_without_workspace(&self.descriptor, &request, CONSTANT_SCALE_OPERATION_ID)
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaConstantScaleProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        static_contiguous_reusable_topology(&request, 1, &[])
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_constant_scale(
            self.descriptor.provider_implementation_fingerprint(),
            &self.function,
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| provider_failure(identity, "cuda.constant_scale.encode", message))
    }
}

pub(super) struct CudaLogitSoftcapProvider {
    descriptor: OperationProviderDescriptor,
    function: CudaFunction,
}

impl CudaLogitSoftcapProvider {
    pub(super) fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = logit_softcap_contract().map_err(contract_error)?;
        let descriptor = weightless_provider_descriptor(
            runtime,
            &contract,
            LOGIT_SOFTCAP_PROVIDER_ID,
            LOGIT_SOFTCAP_F16_CAPABILITY_ID,
            LOGIT_SOFTCAP_ESTIMATOR_ID,
            contiguous_bindings(1),
            implementation_fingerprint(&[
                include_str!("transformer.rs").as_bytes(),
                crate::ptx::FUSED_SILU_MUL.as_bytes(),
                LOGIT_SOFTCAP_INPLACE_FUNCTION_NAME.as_bytes(),
            ]),
        )?;
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("logit softcap module load", error))?;
        let function = module
            .load_function(LOGIT_SOFTCAP_INPLACE_FUNCTION_NAME)
            .map_err(|error| {
                CudaDeviceRuntimeError::driver("logit softcap function load", error)
            })?;
        Ok(Self {
            descriptor,
            function,
        })
    }
}

impl OperationResourceEstimator for CudaLogitSoftcapProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        estimate_without_workspace(&self.descriptor, &request, LOGIT_SOFTCAP_OPERATION_ID)
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaLogitSoftcapProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        static_contiguous_reusable_topology(&request, 1, &[])
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_logit_softcap(
            self.descriptor.provider_implementation_fingerprint(),
            &self.function,
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| provider_failure(identity, "cuda.logit_softcap.encode", message))
    }
}

pub(super) struct CudaResidualAddProvider {
    descriptor: OperationProviderDescriptor,
    function: CudaFunction,
}

impl CudaResidualAddProvider {
    pub(super) fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = residual_add_contract().map_err(contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            RESIDUAL_ADD_PROVIDER_ID,
            RESIDUAL_ADD_F16_CAPABILITY_ID,
            RESIDUAL_ADD_ESTIMATOR_ID,
            contiguous_bindings(2),
            implementation_fingerprint(&[
                include_str!("transformer.rs").as_bytes(),
                crate::ptx::RESIDUAL_ADD.as_bytes(),
                RESIDUAL_ADD_FUNCTION_NAME.as_bytes(),
            ]),
        )?;
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::RESIDUAL_ADD.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("residual add module load", error))?;
        let function = module
            .load_function(RESIDUAL_ADD_FUNCTION_NAME)
            .map_err(|error| CudaDeviceRuntimeError::driver("residual add function load", error))?;
        Ok(Self {
            descriptor,
            function,
        })
    }
}

impl OperationResourceEstimator for CudaResidualAddProvider {
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

impl OperationProvider<CudaDeviceRuntime> for CudaResidualAddProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        static_contiguous_reusable_topology(&request, 2, &[])
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_residual_add(
            self.descriptor.provider_implementation_fingerprint(),
            &self.function,
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| provider_failure(identity, "cuda.residual_add.encode", message))
    }
}

pub(super) fn provider_descriptor(
    runtime: &CudaDeviceRuntime,
    contract: &dyn OperationContract,
    provider_id: &str,
    capability_id: &str,
    estimator_id: &str,
    bindings: Vec<ProviderStorageBindingRequirement>,
    provider_fingerprint: String,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    provider_descriptor_with_formats(
        runtime,
        contract,
        provider_id,
        capability_id,
        estimator_id,
        bindings,
        BTreeSet::from([WeightFormatId::new(DENSE_SAFETENSORS_FORMAT_ID).map_err(contract_error)?]),
        provider_fingerprint,
    )
}

fn weightless_provider_descriptor(
    runtime: &CudaDeviceRuntime,
    contract: &dyn OperationContract,
    provider_id: &str,
    capability_id: &str,
    estimator_id: &str,
    bindings: Vec<ProviderStorageBindingRequirement>,
    provider_fingerprint: String,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    provider_descriptor_with_formats(
        runtime,
        contract,
        provider_id,
        capability_id,
        estimator_id,
        bindings,
        BTreeSet::new(),
        provider_fingerprint,
    )
}

#[allow(clippy::too_many_arguments)]
fn provider_descriptor_with_formats(
    runtime: &CudaDeviceRuntime,
    contract: &dyn OperationContract,
    provider_id: &str,
    capability_id: &str,
    estimator_id: &str,
    bindings: Vec<ProviderStorageBindingRequirement>,
    accepted_weight_formats: BTreeSet<WeightFormatId>,
    provider_fingerprint: String,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(capability_id).map_err(contract_error)?;
    if !runtime.descriptor().capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(format!(
            "CUDA runtime does not advertise capability `{capability_id}`"
        )));
    }
    let estimator_fingerprint = implementation_fingerprint(&[
        include_str!("transformer.rs").as_bytes(),
        estimator_id.as_bytes(),
        provider_fingerprint.as_bytes(),
    ]);
    OperationProviderDescriptor::new(
        ProviderId::new(provider_id).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        provider_fingerprint,
        ferrum_interfaces::vnext::ProviderExecutionSemantics::bitwise_eager_and_replay(),
        contract.descriptor().version,
        runtime.descriptor().id.clone(),
        BTreeSet::from([capability]),
        accepted_weight_formats,
        BTreeSet::new(),
        bindings,
        estimator_id,
        ContractVersion::new(1, 0),
        estimator_fingerprint,
    )
    .map_err(contract_error)
}

#[cfg(feature = "vllm-marlin")]
fn marlin_swiglu_provider_descriptor(
    runtime: &CudaDeviceRuntime,
    contract: &dyn OperationContract,
    provider_fingerprint: String,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let operation_capability =
        CapabilityId::new(DENSE_SWIGLU_F16_CAPABILITY_ID).map_err(contract_error)?;
    let marlin_capability =
        CapabilityId::new(COMPRESSED_TENSORS_MARLIN_CAPABILITY_ID).map_err(contract_error)?;
    let marlin_fp8_capability =
        CapabilityId::new(MARLIN_FP8_CAPABILITY_ID).map_err(contract_error)?;
    if !runtime
        .descriptor()
        .capabilities
        .contains(&operation_capability)
        || !runtime
            .descriptor()
            .capabilities
            .contains(&marlin_capability)
        || !runtime
            .descriptor()
            .capabilities
            .contains(&marlin_fp8_capability)
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime does not advertise dense-SwiGLU Marlin capabilities",
        ));
    }
    let estimator_fingerprint = implementation_fingerprint(&[
        include_str!("transformer.rs").as_bytes(),
        DENSE_SWIGLU_ESTIMATOR_ID.as_bytes(),
        provider_fingerprint.as_bytes(),
    ]);
    OperationProviderDescriptor::new(
        ProviderId::new(DENSE_SWIGLU_PROVIDER_ID).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        provider_fingerprint,
        ferrum_interfaces::vnext::ProviderExecutionSemantics::bitwise_eager_and_replay(),
        contract.descriptor().version,
        runtime.descriptor().id.clone(),
        BTreeSet::from([
            operation_capability,
            marlin_capability,
            marlin_fp8_capability,
        ]),
        BTreeSet::from([
            WeightFormatId::new(DENSE_SAFETENSORS_FORMAT_ID).map_err(contract_error)?,
            WeightFormatId::new(COMPRESSED_TENSORS_MARLIN_WEIGHT_FORMAT_ID)
                .map_err(contract_error)?,
            WeightFormatId::new(MARLIN_FP8_WEIGHT_FORMAT_ID).map_err(contract_error)?,
            WeightFormatId::new(MARLIN_FP8_GROUP128_WEIGHT_FORMAT_ID).map_err(contract_error)?,
        ]),
        BTreeSet::from([
            QuantizationFormatId::new(COMPRESSED_TENSORS_MARLIN_QUANTIZATION_FORMAT_ID)
                .map_err(contract_error)?,
            QuantizationFormatId::new(MARLIN_FP8_QUANTIZATION_FORMAT_ID).map_err(contract_error)?,
            QuantizationFormatId::new(MARLIN_FP8_GROUP128_QUANTIZATION_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        contiguous_bindings(3),
        DENSE_SWIGLU_ESTIMATOR_ID,
        ContractVersion::new(2, 0),
        estimator_fingerprint,
    )
    .map_err(contract_error)
}

#[cfg(feature = "vllm-marlin")]
fn marlin_geglu_provider_descriptor(
    runtime: &CudaDeviceRuntime,
    contract: &dyn OperationContract,
    provider_fingerprint: String,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let operation_capability =
        CapabilityId::new(DENSE_GEGLU_TANH_F16_CAPABILITY_ID).map_err(contract_error)?;
    let marlin_capability = CapabilityId::new(COMPRESSED_TENSORS_MARLIN_SYMMETRIC_CAPABILITY_ID)
        .map_err(contract_error)?;
    if !runtime
        .descriptor()
        .capabilities
        .contains(&operation_capability)
        || !runtime
            .descriptor()
            .capabilities
            .contains(&marlin_capability)
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime does not advertise dense-GeGLU symmetric compressed-tensors Marlin capabilities",
        ));
    }
    let estimator_fingerprint = implementation_fingerprint(&[
        include_str!("transformer.rs").as_bytes(),
        DENSE_GEGLU_TANH_ESTIMATOR_ID.as_bytes(),
        provider_fingerprint.as_bytes(),
    ]);
    OperationProviderDescriptor::new(
        ProviderId::new(DENSE_GEGLU_TANH_PROVIDER_ID).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        provider_fingerprint,
        ferrum_interfaces::vnext::ProviderExecutionSemantics::bitwise_eager_and_replay(),
        contract.descriptor().version,
        runtime.descriptor().id.clone(),
        BTreeSet::from([operation_capability, marlin_capability]),
        BTreeSet::from([
            WeightFormatId::new(DENSE_SAFETENSORS_FORMAT_ID).map_err(contract_error)?,
            WeightFormatId::new(COMPRESSED_TENSORS_MARLIN_SYMMETRIC_WEIGHT_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::from([QuantizationFormatId::new(
            COMPRESSED_TENSORS_MARLIN_SYMMETRIC_QUANTIZATION_FORMAT_ID,
        )
        .map_err(contract_error)?]),
        contiguous_bindings(4),
        DENSE_GEGLU_TANH_ESTIMATOR_ID,
        ContractVersion::new(2, 0),
        estimator_fingerprint,
    )
    .map_err(contract_error)
}

pub(super) fn contiguous_bindings(input_count: u32) -> Vec<ProviderStorageBindingRequirement> {
    (0..input_count)
        .map(|ordinal| {
            ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Input,
                ordinal,
                DynamicStorageRequirement::contiguous(),
            )
        })
        .chain(std::iter::once(ProviderStorageBindingRequirement::new(
            ResolvedValueRole::Output,
            0,
            DynamicStorageRequirement::contiguous(),
        )))
        .collect()
}

fn estimate_without_workspace(
    descriptor: &OperationProviderDescriptor,
    request: &OperationResourceEstimateRequest<'_>,
    operation_id: &str,
) -> Result<OperationResourceEstimate, VNextError> {
    ensure_estimator_request(descriptor, request, operation_id)?;
    Ok(estimate(descriptor, request.input_fingerprint(), None))
}

pub(super) fn ensure_estimator_request(
    descriptor: &OperationProviderDescriptor,
    request: &OperationResourceEstimateRequest<'_>,
    operation_id: &str,
) -> Result<(), VNextError> {
    if request.operation().id.as_str() != operation_id
        || request.operation().fingerprint()? != descriptor.operation_fingerprint()
    {
        return Err(invalid_plan(format!(
            "CUDA estimator `{}` received another operation",
            descriptor.resource_estimator_id()
        )));
    }
    Ok(())
}

pub(super) fn estimate(
    descriptor: &OperationProviderDescriptor,
    input_fingerprint: &str,
    scratch: Option<ProviderWorkspaceRequirement>,
) -> OperationResourceEstimate {
    OperationResourceEstimate::new(
        descriptor.resource_estimator_id(),
        descriptor.resource_estimator_version(),
        descriptor.resource_estimator_implementation_fingerprint(),
        input_fingerprint,
        VALUE_ALIGNMENT_BYTES,
        scratch,
        None,
    )
}

fn encode_rms_norm(
    provider_fingerprint: &str,
    function: &CudaFunction,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    ensure_invocation(&invocation, RMS_NORM_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let first_input = binding(first.bindings(), ResolvedValueRole::Input, 0)?;
    let first_weight = binding(first.bindings(), ResolvedValueRole::Input, 1)?;
    let first_output = binding(first.bindings(), ResolvedValueRole::Output, 0)?;
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let epsilon = rational_attribute(first.attributes(), "epsilon")?;
    validate_rms_norm(first_input, first_weight, first_output, hidden_size)?;
    for participant in &invocation.participants()[1..] {
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let weight = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || rational_attribute(participant.attributes(), "epsilon")? != epsilon
        {
            return Err("CUDA RMSNorm participant attributes disagree".to_owned());
        }
        validate_rms_norm(input, weight, output, hidden_size)?;
    }
    let tokens = invocation.work_shape().immediate_tokens();
    let input = shared_token_region(
        &invocation,
        ResolvedValueRole::Input,
        0,
        ElementType::F16,
        tokens,
    )?;
    let weight = shared_full_region(&invocation, ResolvedValueRole::Input, 1, ElementType::F16)?;
    let output = shared_token_region(
        &invocation,
        ResolvedValueRole::Output,
        0,
        ElementType::F16,
        tokens,
    )?;
    let regions = vec![input, weight, output];
    let rows = checked_u32(tokens, "RMSNorm row count")?;
    let hidden_size = checked_i32(hidden_size, "RMSNorm hidden size")?;
    let function = function.clone();
    let replay_key = CudaCommandReplayKeyBuilder::new(provider_fingerprint, "vnext_rms_norm")
        .u32(rows)
        .i32(hidden_size)
        .f32(epsilon)
        .finish();
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "RMSNorm participant count",
    )?;
    CudaDeviceCommand::replayable_operation(
        "vnext_rms_norm",
        regions,
        replay_key,
        move |stream, regions| {
            let input = regions[0].device_ptr();
            let weight = regions[1].device_ptr();
            let output = regions[2].device_ptr();
            let mut builder = stream.launch_builder(&function);
            builder.arg(&input);
            builder.arg(&weight);
            builder.arg(&output);
            builder.arg(&hidden_size);
            builder.arg(&epsilon);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (rows, 1, 1),
                    block_dim: ((hidden_size as u32).min(1024), 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|error| CudaDeviceRuntimeError::driver("vNext RMSNorm launch", error))
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            DeviceBatchingForm::Packed,
            participant_count,
            u64::from(rows),
            1,
            0,
        )
    })
    .map_err(|error| error.to_string())
}

#[derive(Clone, Copy)]
struct GemmLaunch {
    input_region: usize,
    output_region: usize,
    rows: i32,
    out_features: i32,
    in_features: i32,
}

fn encode_dense_linear(
    provider_fingerprint: &str,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    ensure_invocation(&invocation, DENSE_LINEAR_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let first_input = binding(first.bindings(), ResolvedValueRole::Input, 0)?;
    let first_weight = binding(first.bindings(), ResolvedValueRole::Input, 1)?;
    let first_output = binding(first.bindings(), ResolvedValueRole::Output, 0)?;
    let in_features = unsigned_attribute(first.attributes(), "in_features")?;
    let out_features = unsigned_attribute(first.attributes(), "out_features")?;
    validate_dense_linear(
        first_input,
        first_weight,
        first_output,
        in_features,
        out_features,
    )?;
    for participant in &invocation.participants()[1..] {
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let weight = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "in_features")? != in_features
            || unsigned_attribute(participant.attributes(), "out_features")? != out_features
        {
            return Err("CUDA dense linear participant attributes disagree".to_owned());
        }
        validate_dense_linear(input, weight, output, in_features, out_features)?;
    }
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("CUDA dense linear participant ranges are incomplete".to_owned());
    }
    let input_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let output_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Output, 0)?;
    let mut regions = vec![shared_full_region(
        &invocation,
        ResolvedValueRole::Input,
        1,
        ElementType::F16,
    )?];
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
        launches.push(GemmLaunch {
            input_region,
            output_region,
            rows: checked_i32(rows, "dense linear row count")?,
            out_features: checked_i32(out_features, "dense linear output width")?,
            in_features: checked_i32(in_features, "dense linear input width")?,
        });
    } else {
        for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
            let packed = token_range.immediate_token_range();
            let source = token_range.source_token_range();
            let rows = token_range.immediate_tokens();
            let input_region = regions.len();
            regions.push(contiguous_token_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
                ElementType::F16,
                if input_packed {
                    packed.start
                } else {
                    source.start
                },
                rows,
            )?);
            let output_region = regions.len();
            regions.push(contiguous_token_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
                ElementType::F16,
                if output_packed {
                    packed.start
                } else {
                    source.start
                },
                rows,
            )?);
            launches.push(GemmLaunch {
                input_region,
                output_region,
                rows: checked_i32(rows, "dense linear row count")?,
                out_features: checked_i32(out_features, "dense linear output width")?,
                in_features: checked_i32(in_features, "dense linear input width")?,
            });
        }
    }
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "dense linear participant count",
    )?;
    let token_count = invocation.work_shape().immediate_tokens();
    let batching_form = if input_packed && output_packed {
        DeviceBatchingForm::Packed
    } else {
        DeviceBatchingForm::ParticipantLoop
    };
    let compute_dispatch_count = launches.len() as u64;
    let mut replay_key =
        CudaCommandReplayKeyBuilder::new(provider_fingerprint, "vnext_dense_linear")
            .u64(launches.len() as u64);
    for launch in &launches {
        replay_key = replay_key
            .u64(launch.input_region as u64)
            .u64(launch.output_region as u64)
            .i32(launch.rows)
            .i32(launch.out_features)
            .i32(launch.in_features);
    }
    CudaDeviceCommand::replayable_operation_with_blas(
        "vnext_dense_linear",
        regions,
        replay_key.finish(),
        move |_stream, blas, regions| {
            for launch in &launches {
                launch_gemm_f16(
                    blas,
                    regions[launch.input_region].device_ptr(),
                    regions[0].device_ptr(),
                    regions[launch.output_region].device_ptr(),
                    launch.rows,
                    launch.out_features,
                    launch.in_features,
                    "vNext dense linear GEMM",
                )?;
            }
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            batching_form,
            participant_count,
            token_count,
            compute_dispatch_count,
            0,
        )
    })
    .map_err(|error| error.to_string())
}

#[cfg(feature = "vllm-marlin")]
fn encode_marlin_fp8_dense_linear(
    provider_fingerprint: &str,
    projection_runtime: MarlinProjectionRuntime,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    use marlin_fp8_weights::resolve_marlin_fp8_weight;

    ensure_invocation(&invocation, DENSE_LINEAR_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let first_input = binding(first.bindings(), ResolvedValueRole::Input, 0)?;
    let first_weight_binding = binding(first.bindings(), ResolvedValueRole::Input, 1)?;
    let first_output = binding(first.bindings(), ResolvedValueRole::Output, 0)?;
    let in_features = unsigned_attribute(first.attributes(), "in_features")?;
    let out_features = unsigned_attribute(first.attributes(), "out_features")?;
    validate_dense_linear(
        first_input,
        first_weight_binding,
        first_output,
        in_features,
        out_features,
    )?;
    let first_weight =
        resolve_marlin_fp8_weight(first, first_weight_binding, &[out_features, in_features])?;
    for participant in &invocation.participants()[1..] {
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let weight_binding = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "in_features")? != in_features
            || unsigned_attribute(participant.attributes(), "out_features")? != out_features
        {
            return Err("CUDA Marlin FP8 dense linear participant attributes disagree".to_owned());
        }
        validate_dense_linear(input, weight_binding, output, in_features, out_features)?;
        let candidate =
            resolve_marlin_fp8_weight(participant, weight_binding, &[out_features, in_features])?;
        if !same_physical_region(first_weight.packed_region(), candidate.packed_region())
            || !same_physical_region(first_weight.scales_region(), candidate.scales_region())
            || first_weight.group_size() != candidate.group_size()
        {
            return Err(
                "CUDA Marlin FP8 dense linear participants do not share one weight".to_owned(),
            );
        }
    }

    let workspace_bytes = projection_runtime.workspace_bytes()?;
    let group_size = first_weight.group_size();
    let [packed_region, scales_region] = first_weight.into_regions();
    let mut regions = vec![
        packed_region,
        scales_region,
        shared_scratch_region(&invocation, workspace_bytes)?,
    ];
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("CUDA Marlin FP8 dense linear participant ranges are incomplete".to_owned());
    }
    let input_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let output_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Output, 0)?;
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
        launches.push(GemmLaunch {
            input_region,
            output_region,
            rows: checked_i32(rows, "Marlin FP8 dense linear row count")?,
            out_features: checked_i32(out_features, "Marlin FP8 dense linear output width")?,
            in_features: checked_i32(in_features, "Marlin FP8 dense linear input width")?,
        });
    } else {
        for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
            let packed = token_range.immediate_token_range();
            let source = token_range.source_token_range();
            let rows = token_range.immediate_tokens();
            let input_region = regions.len();
            regions.push(contiguous_token_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
                ElementType::F16,
                if input_packed {
                    packed.start
                } else {
                    source.start
                },
                rows,
            )?);
            let output_region = regions.len();
            regions.push(contiguous_token_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
                ElementType::F16,
                if output_packed {
                    packed.start
                } else {
                    source.start
                },
                rows,
            )?);
            launches.push(GemmLaunch {
                input_region,
                output_region,
                rows: checked_i32(rows, "Marlin FP8 dense linear row count")?,
                out_features: checked_i32(out_features, "Marlin FP8 dense linear output width")?,
                in_features: checked_i32(in_features, "Marlin FP8 dense linear input width")?,
            });
        }
    }

    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "Marlin FP8 dense linear participant count",
    )?;
    let token_count = invocation.work_shape().immediate_tokens();
    let batching_form = if input_packed && output_packed {
        DeviceBatchingForm::Packed
    } else {
        DeviceBatchingForm::ParticipantLoop
    };
    let compute_dispatch_count = launches.len() as u64;
    let mut replay_key =
        CudaCommandReplayKeyBuilder::new(provider_fingerprint, "vnext_dense_linear_marlin_fp8")
            .i32(projection_runtime.multiprocessor_count)
            .i32(projection_runtime.device_ordinal)
            .i32(group_size)
            .u64(launches.len() as u64);
    for launch in &launches {
        replay_key = replay_key
            .u64(launch.input_region as u64)
            .u64(launch.output_region as u64)
            .i32(launch.rows)
            .i32(launch.out_features)
            .i32(launch.in_features);
    }
    CudaDeviceCommand::replayable_operation(
        "vnext_dense_linear_marlin_fp8",
        regions,
        replay_key.finish(),
        move |stream, regions| {
            let workspace = &regions[2];
            if workspace.length_bytes() < workspace_bytes {
                return Err(CudaDeviceRuntimeError::contract(
                    "Marlin FP8 workspace is smaller than its admitted estimate",
                ));
            }
            for launch in &launches {
                projection_runtime.launch(
                    MarlinF16WeightType::E4M3Fn,
                    stream,
                    regions[launch.input_region].device_ptr(),
                    regions[0].device_ptr(),
                    regions[1].device_ptr(),
                    None,
                    regions[launch.output_region].device_ptr(),
                    workspace.device_ptr(),
                    workspace.length_bytes(),
                    launch.rows,
                    launch.out_features,
                    launch.in_features,
                    group_size,
                    "Marlin FP8 dense linear",
                )?;
            }
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            batching_form,
            participant_count,
            token_count,
            compute_dispatch_count,
            0,
        )
    })
    .map_err(|error| error.to_string())
}

pub(super) fn aligned_projection_workspace_bytes(
    workspace_bytes: u64,
    alignment: u64,
    operation: &'static str,
) -> Result<u64, String> {
    if workspace_bytes == 0 {
        return Ok(0);
    }
    if !alignment.is_power_of_two() {
        return Err(format!(
            "{operation} alignment {alignment} is not a non-zero power of two"
        ));
    }
    workspace_bytes
        .checked_add(alignment - 1)
        .map(|bytes| bytes & !(alignment - 1))
        .filter(|bytes| *bytes >= workspace_bytes)
        .ok_or_else(|| format!("{operation} aligned size overflows"))
}

#[cfg(feature = "vllm-marlin")]
#[derive(Debug, Clone, Copy)]
pub(super) struct MarlinProjectionRuntime {
    multiprocessor_count: i32,
    device_ordinal: i32,
}

#[cfg(feature = "vllm-marlin")]
fn marlin_num_groups(
    input_features: i32,
    group_size: i32,
    operation: &'static str,
) -> Result<i32, CudaDeviceRuntimeError> {
    if input_features <= 0 {
        return Err(CudaDeviceRuntimeError::contract(format!(
            "{operation} input width {input_features} must be positive"
        )));
    }
    if group_size == -1 {
        return Ok(1);
    }
    if group_size <= 0 {
        return Err(CudaDeviceRuntimeError::contract(format!(
            "{operation} group size {group_size} must be -1 for channelwise weights or positive"
        )));
    }
    if input_features % group_size != 0 {
        return Err(CudaDeviceRuntimeError::contract(format!(
            "{operation} input width {input_features} is not divisible by group size {group_size}"
        )));
    }
    let num_groups = input_features / group_size;
    if num_groups <= 0 {
        return Err(CudaDeviceRuntimeError::contract(format!(
            "{operation} input width {input_features} and group size {group_size} produce non-positive group count {num_groups}"
        )));
    }
    Ok(num_groups)
}

#[cfg(feature = "vllm-marlin")]
impl MarlinProjectionRuntime {
    pub(super) fn query(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let multiprocessor_count = runtime
            .context()
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|error| CudaDeviceRuntimeError::driver("multiprocessor count query", error))?;
        if multiprocessor_count <= 0 {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA multiprocessor count is not positive",
            ));
        }
        let device_ordinal = i32::try_from(runtime.descriptor().ordinal)
            .map_err(|_| CudaDeviceRuntimeError::contract("CUDA device ordinal exceeds i32"))?;
        Ok(Self {
            multiprocessor_count,
            device_ordinal,
        })
    }

    pub(super) fn workspace_bytes(self) -> Result<u64, String> {
        u64::try_from(self.multiprocessor_count)
            .ok()
            .and_then(|sms| sms.checked_mul(std::mem::size_of::<i32>() as u64))
            .ok_or_else(|| "CUDA Marlin FP8 workspace size overflows".to_owned())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch(
        self,
        weight_type: MarlinF16WeightType,
        stream: &CudaStream,
        input: u64,
        packed_weight: u64,
        scales: u64,
        zero_points: Option<u64>,
        output: u64,
        workspace: u64,
        workspace_length_bytes: u64,
        rows: i32,
        output_features: i32,
        input_features: i32,
        group_size: i32,
        operation: &'static str,
    ) -> Result<(), CudaDeviceRuntimeError> {
        if weight_type == MarlinF16WeightType::E4M3Fn {
            let output_features = usize::try_from(output_features).map_err(|_| {
                CudaDeviceRuntimeError::contract(format!(
                    "{operation} FP8 output width is not positive"
                ))
            })?;
            let input_features = usize::try_from(input_features).map_err(|_| {
                CudaDeviceRuntimeError::contract(format!(
                    "{operation} FP8 input width is not positive"
                ))
            })?;
            if !marlin_fp8_projection_shape_supported(output_features, input_features) {
                return Err(CudaDeviceRuntimeError::contract(format!(
                    "{operation} FP8 shape [{output_features}, {input_features}] is not supported by the shared execution provider"
                )));
            }
        }
        let num_groups = marlin_num_groups(input_features, group_size, operation)?;
        let required_workspace = self
            .workspace_bytes()
            .map_err(CudaDeviceRuntimeError::contract)?;
        if workspace_length_bytes < required_workspace {
            return Err(CudaDeviceRuntimeError::contract(format!(
                "{operation} workspace differs from its admitted estimate"
            )));
        }
        let workspace_bytes = usize::try_from(required_workspace).map_err(|_| {
            CudaDeviceRuntimeError::contract(format!("{operation} workspace exceeds usize"))
        })?;
        unsafe {
            cudarc::driver::result::memset_d8_async(
                workspace,
                0,
                workspace_bytes,
                stream.cu_stream(),
            )
        }
        .map_err(|error| CudaDeviceRuntimeError::driver(operation, error))?;
        unsafe {
            launch_marlin_mm_f16_weight(MarlinMmF16WeightRequest {
                weight_type,
                buffers: MarlinMmBuffers {
                    a: input as *const c_void,
                    b: packed_weight as *const c_void,
                    c: output as *mut c_void,
                    c_tmp: std::ptr::null_mut(),
                    a_scales: std::ptr::null_mut(),
                    b_scales: scales as *mut c_void,
                    zero_points: zero_points
                        .map_or(std::ptr::null_mut(), |pointer| pointer as *mut c_void),
                    group_index: std::ptr::null_mut(),
                    permutation: std::ptr::null_mut(),
                    a_tmp: std::ptr::null_mut(),
                    workspace: workspace as *mut c_void,
                },
                problem: MarlinMmProblem {
                    m: rows,
                    n: output_features,
                    k: input_features,
                    lda: input_features,
                    num_groups,
                    group_size,
                },
                execution: MarlinMmExecution {
                    device: self.device_ordinal,
                    stream: stream.cu_stream(),
                    sms: self.multiprocessor_count,
                    has_act_order: false,
                    is_k_full: true,
                    use_atomic_add: false,
                    use_fp32_reduce: false,
                },
            });
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "vllm-marlin"))]
mod marlin_group_count_tests {
    use super::{marlin_num_groups, CudaDeviceRuntimeError};

    #[test]
    fn channelwise_group_size_maps_to_one_native_group() {
        assert!(matches!(marlin_num_groups(4096, -1, "test"), Ok(1)));
        for input_features in [0, -1] {
            assert!(matches!(
                marlin_num_groups(input_features, -1, "test"),
                Err(CudaDeviceRuntimeError::Contract(message))
                    if message.contains("input width") && message.contains("must be positive")
            ));
        }
    }

    #[test]
    fn positive_group_size_requires_exact_positive_groups() {
        assert!(matches!(marlin_num_groups(4096, 128, "test"), Ok(32)));
        assert!(matches!(
            marlin_num_groups(4097, 128, "test"),
            Err(CudaDeviceRuntimeError::Contract(message))
                if message.contains("is not divisible")
        ));
        assert!(matches!(
            marlin_num_groups(0, 128, "test"),
            Err(CudaDeviceRuntimeError::Contract(message))
                if message.contains("input width 0 must be positive")
        ));
    }

    #[test]
    fn zero_and_other_negative_group_sizes_are_typed_failures() {
        for group_size in [0, -2, i32::MIN] {
            assert!(matches!(
                marlin_num_groups(4096, group_size, "test"),
                Err(CudaDeviceRuntimeError::Contract(message))
                    if message.contains("must be -1 for channelwise weights or positive")
            ));
        }
    }
}

#[cfg(feature = "vllm-marlin")]
fn marlin_fp8_swiglu_gate_up_layout(
    layout: &PhysicalWeightLayout,
    partition: usize,
    intermediate_size: u64,
    hidden_size: u64,
) -> Result<&PhysicalWeightLayout, String> {
    let PhysicalWeightLayout::Composite { parts } = layout else {
        return Err("Marlin FP8 SwiGLU gate/up weight must be composite".to_owned());
    };
    if parts.len() != 2 {
        return Err("Marlin FP8 SwiGLU gate/up weight must contain exactly two parts".to_owned());
    }
    for (index, part) in parts.iter().enumerate() {
        if part.logical_offsets != [index as u64, 0, 0]
            || part.extents != [1, intermediate_size, hidden_size]
            || !matches!(part.layout.as_ref(), PhysicalWeightLayout::Quantized { .. })
        {
            return Err(format!(
                "Marlin FP8 SwiGLU gate/up part {index} has invalid placement or layout"
            ));
        }
    }
    parts
        .get(partition)
        .map(|part| part.layout.as_ref())
        .ok_or_else(|| format!("Marlin FP8 SwiGLU gate/up part {partition} is absent"))
}

#[cfg(feature = "vllm-marlin")]
#[derive(Debug, Clone, Copy)]
struct SharedMarlinFp8Weight {
    packed_region: usize,
    scales_region: usize,
    group_size: i32,
}

#[cfg(feature = "vllm-marlin")]
fn push_shared_marlin_fp8_weight(
    regions: &mut Vec<CudaBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ordinal: u32,
    logical_dimensions: &[u64],
    composite_partition: Option<usize>,
) -> Result<SharedMarlinFp8Weight, String> {
    use marlin_fp8_weights::{resolve_marlin_fp8_layout, resolve_marlin_fp8_weight};

    let [output_features, input_features] = logical_dimensions else {
        return Err("Marlin FP8 SwiGLU projection must have shape [N, K]".to_owned());
    };
    let resolve = |participant: &OperationInvocation<'_, CudaDeviceBuffer>| {
        let value = binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?;
        if let Some(partition) = composite_partition {
            let weight = value.weight().ok_or_else(|| {
                format!("Marlin FP8 SwiGLU input {ordinal} has no physical weight layout")
            })?;
            let layout = marlin_fp8_swiglu_gate_up_layout(
                weight.physical_layout(),
                partition,
                *output_features,
                *input_features,
            )?;
            resolve_marlin_fp8_layout(participant, value, layout, logical_dimensions)
        } else {
            resolve_marlin_fp8_weight(participant, value, logical_dimensions)
        }
    };

    let first = resolve(&invocation.participants()[0])?;
    if first.output_features() != *output_features || first.input_features() != *input_features {
        return Err(format!(
            "Marlin FP8 SwiGLU input {ordinal} resolved inconsistent dimensions"
        ));
    }
    for participant in &invocation.participants()[1..] {
        let candidate = resolve(participant)?;
        if candidate.output_features() != first.output_features()
            || candidate.input_features() != first.input_features()
            || candidate.group_size() != first.group_size()
            || !same_physical_region(first.packed_region(), candidate.packed_region())
            || !same_physical_region(first.scales_region(), candidate.scales_region())
        {
            return Err(format!(
                "Marlin FP8 SwiGLU input {ordinal} is not shared by all participants"
            ));
        }
    }
    let group_size = first.group_size();
    let [packed, scales] = first.into_regions();
    let packed_region = regions.len();
    regions.push(packed);
    let scales_region = regions.len();
    regions.push(scales);
    Ok(SharedMarlinFp8Weight {
        packed_region,
        scales_region,
        group_size,
    })
}

#[cfg(feature = "vllm-marlin")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MarlinSwiGluScratchLayout {
    activation_elements: u64,
    activation_bytes: u64,
    workspace_offset: u64,
    workspace_bytes: u64,
    required_bytes: u64,
}

#[cfg(feature = "vllm-marlin")]
fn marlin_swiglu_scratch_layout(
    tokens: u64,
    intermediate_size: u64,
    workspace_bytes: u64,
) -> Result<MarlinSwiGluScratchLayout, String> {
    let activation_elements = tokens
        .checked_mul(intermediate_size)
        .ok_or_else(|| "Marlin SwiGLU activation size overflows".to_owned())?;
    let activation_bytes = activation_elements
        .checked_mul(ElementType::F16.size_bytes())
        .ok_or_else(|| "Marlin SwiGLU activation bytes overflow".to_owned())?;
    let activation_scratch_bytes = activation_bytes
        .checked_mul(SWIGLU_SCRATCH_PARTS)
        .ok_or_else(|| "Marlin SwiGLU activation scratch bytes overflow".to_owned())?;
    let workspace_offset = activation_scratch_bytes
        .checked_add(VALUE_ALIGNMENT_BYTES - 1)
        .map(|value| value / VALUE_ALIGNMENT_BYTES * VALUE_ALIGNMENT_BYTES)
        .ok_or_else(|| "Marlin SwiGLU workspace offset overflows".to_owned())?;
    let required_bytes = workspace_offset
        .checked_add(workspace_bytes)
        .ok_or_else(|| "Marlin SwiGLU total scratch bytes overflow".to_owned())?;
    Ok(MarlinSwiGluScratchLayout {
        activation_elements,
        activation_bytes,
        workspace_offset,
        workspace_bytes,
        required_bytes,
    })
}

#[cfg(feature = "vllm-marlin")]
#[derive(Debug, Clone, Copy)]
struct SharedSymmetricCompressedTensorsWeight {
    packed_region: usize,
    scales_region: usize,
    group_size: i32,
}

#[cfg(feature = "vllm-marlin")]
fn push_shared_symmetric_compressed_tensors_weight(
    regions: &mut Vec<CudaBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ordinal: u32,
    logical_dimensions: &[u64],
) -> Result<SharedSymmetricCompressedTensorsWeight, String> {
    let resolve = |participant: &OperationInvocation<'_, CudaDeviceBuffer>| {
        let value = binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?;
        resolve_compressed_tensors_symmetric_marlin_matrix_weight(
            participant,
            value,
            logical_dimensions,
        )
    };
    let first = resolve(&invocation.participants()[0])?;
    if first.expert_count() != 1 {
        return Err(format!(
            "symmetric compressed-tensors GeGLU input {ordinal} is not one matrix"
        ));
    }
    for participant in &invocation.participants()[1..] {
        let candidate = resolve(participant)?;
        if candidate.logical_dimensions() != first.logical_dimensions()
            || candidate.packed_physical_dimensions() != first.packed_physical_dimensions()
            || candidate.scales_physical_dimensions() != first.scales_physical_dimensions()
            || candidate.expert_count() != first.expert_count()
            || candidate.packed_expert_stride_bytes() != first.packed_expert_stride_bytes()
            || candidate.scales_expert_stride_bytes() != first.scales_expert_stride_bytes()
            || candidate.group_size() != first.group_size()
            || candidate.weight_type() != first.weight_type()
            || !same_physical_region(first.packed_region(), candidate.packed_region())
            || !same_physical_region(first.scales_region(), candidate.scales_region())
        {
            return Err(format!(
                "symmetric compressed-tensors GeGLU input {ordinal} is not shared by all participants"
            ));
        }
    }
    let group_size = first.group_size();
    let [packed, scales] = first.into_regions();
    let packed_region = regions.len();
    regions.push(packed);
    let scales_region = regions.len();
    regions.push(scales);
    Ok(SharedSymmetricCompressedTensorsWeight {
        packed_region,
        scales_region,
        group_size,
    })
}

#[cfg(feature = "vllm-marlin")]
#[allow(clippy::too_many_arguments)]
fn dense_geglu_marlin_replay_key(
    provider_fingerprint: &str,
    projection_runtime: MarlinProjectionRuntime,
    gate_group_size: i32,
    up_group_size: i32,
    down_group_size: i32,
    rows: i32,
    hidden: i32,
    intermediate: i32,
    scratch: MarlinSwiGluScratchLayout,
) -> crate::backend::cuda::vnext_replay::CudaCommandReplayKey {
    CudaCommandReplayKeyBuilder::new(
        provider_fingerprint,
        "vnext_dense_geglu_tanh_compressed_tensors_symmetric_marlin",
    )
    .i32(projection_runtime.multiprocessor_count)
    .i32(projection_runtime.device_ordinal)
    .i32(gate_group_size)
    .i32(up_group_size)
    .i32(down_group_size)
    .i32(rows)
    .i32(hidden)
    .i32(intermediate)
    .u64(scratch.activation_elements)
    .u64(scratch.activation_bytes)
    .u64(scratch.workspace_offset)
    .u64(scratch.workspace_bytes)
    .u64(scratch.required_bytes)
    .finish()
}

#[cfg(test)]
mod gemma_simple_ops_tests {
    use super::{
        constant_scale_dimensions_match, dense_geglu_dimensions_match,
        dense_geglu_projection_for_formats, dense_geglu_scratch_layout,
        logit_softcap_dimensions_match, DenseGeGluProjection,
    };
    use ferrum_interfaces::vnext::QuantizationFormatId;
    use std::collections::BTreeSet;

    #[test]
    fn dense_geglu_scratch_keeps_two_token_planes_disjoint() {
        let layout = dense_geglu_scratch_layout(2, 3).unwrap();
        assert_eq!(layout.activation_elements, 6);
        assert_eq!(layout.activation_bytes, 12);
        assert_eq!(layout.up_offset_bytes, 12);
        assert_eq!(layout.activation_offset_bytes, 24);
        assert_eq!(layout.required_bytes, 36);
        assert!(dense_geglu_scratch_layout(u64::MAX, 2).is_err());
    }

    #[test]
    fn gemma_simple_provider_shapes_fail_closed() {
        assert!(dense_geglu_dimensions_match(
            &[2, 4],
            &[6, 4],
            &[6, 4],
            &[4, 6],
            &[2, 4],
            4,
            6,
        ));
        assert!(!dense_geglu_dimensions_match(
            &[2, 4],
            &[4, 6],
            &[6, 4],
            &[4, 6],
            &[2, 4],
            4,
            6,
        ));
        assert!(!dense_geglu_dimensions_match(
            &[2, 4],
            &[6, 4],
            &[6, 4],
            &[4, 6],
            &[1, 4],
            4,
            6,
        ));

        assert!(constant_scale_dimensions_match(&[2, 4], &[2, 4], 4));
        assert!(!constant_scale_dimensions_match(&[2, 4], &[2, 5], 4));
        assert!(logit_softcap_dimensions_match(&[1, 8], &[1, 8], 8));
        assert!(!logit_softcap_dimensions_match(&[2, 8], &[2, 8], 8));
        assert!(!logit_softcap_dimensions_match(&[1, 8], &[1, 7], 8));
    }

    #[test]
    fn dense_geglu_quantization_classification_fails_closed() {
        let dense = BTreeSet::new();
        assert_eq!(
            dense_geglu_projection_for_formats(&dense, &dense, &dense),
            Ok(DenseGeGluProjection::F16)
        );
        let unknown = BTreeSet::from([QuantizationFormatId::new(
            "quantization.test.symmetric-compressed-tensors",
        )
        .unwrap()]);
        for (gate, up, down) in [
            (&unknown, &unknown, &unknown),
            (&unknown, &dense, &dense),
            (&dense, &unknown, &dense),
            (&dense, &dense, &unknown),
        ] {
            assert!(dense_geglu_projection_for_formats(gate, up, down).is_err());
        }
    }
}

#[cfg(all(test, feature = "vllm-marlin"))]
mod dense_geglu_marlin_tests {
    use super::{
        dense_geglu_marlin_replay_key, dense_geglu_projection_for_formats,
        marlin_swiglu_scratch_layout, DenseGeGluProjection, MarlinProjectionRuntime,
        COMPRESSED_TENSORS_MARLIN_SYMMETRIC_QUANTIZATION_FORMAT_ID, VALUE_ALIGNMENT_BYTES,
    };
    use ferrum_interfaces::vnext::QuantizationFormatId;
    use std::collections::BTreeSet;

    fn formats(values: &[&str]) -> BTreeSet<QuantizationFormatId> {
        values
            .iter()
            .map(|value| QuantizationFormatId::new(*value).unwrap())
            .collect()
    }

    #[test]
    fn symmetric_compressed_tensors_geglu_classification_is_exact() {
        let dense = BTreeSet::new();
        let symmetric = formats(&[COMPRESSED_TENSORS_MARLIN_SYMMETRIC_QUANTIZATION_FORMAT_ID]);
        assert_eq!(
            dense_geglu_projection_for_formats(&symmetric, &symmetric, &symmetric),
            Ok(DenseGeGluProjection::CompressedTensorsSymmetricMarlin)
        );
        for (gate, up, down) in [
            (&symmetric, &dense, &dense),
            (&dense, &symmetric, &dense),
            (&dense, &dense, &symmetric),
            (&symmetric, &symmetric, &dense),
        ] {
            assert!(dense_geglu_projection_for_formats(gate, up, down).is_err());
        }
        let multiple = formats(&[
            COMPRESSED_TENSORS_MARLIN_SYMMETRIC_QUANTIZATION_FORMAT_ID,
            "quantization.test.other",
        ]);
        assert!(dense_geglu_projection_for_formats(&multiple, &multiple, &multiple).is_err());
    }

    #[test]
    fn symmetric_compressed_tensors_geglu_uses_bounded_aligned_marlin_layout() {
        for tokens in [1, 8, 32] {
            let layout = marlin_swiglu_scratch_layout(tokens, 15_360, 512).unwrap();
            assert_eq!(layout.activation_elements, tokens * 15_360);
            assert_eq!(layout.activation_bytes, tokens * 15_360 * 2);
            assert!(layout.workspace_offset >= layout.activation_bytes * 3);
            assert_eq!(layout.workspace_offset % VALUE_ALIGNMENT_BYTES, 0);
            assert_eq!(layout.required_bytes, layout.workspace_offset + 512);
        }
        assert!(marlin_swiglu_scratch_layout(u64::MAX, 2, 512).is_err());
    }

    #[test]
    fn symmetric_compressed_tensors_geglu_replay_key_binds_execution_geometry() {
        let runtime = MarlinProjectionRuntime {
            multiprocessor_count: 128,
            device_ordinal: 0,
        };
        let scratch = marlin_swiglu_scratch_layout(8, 15_360, 512).unwrap();
        let key = dense_geglu_marlin_replay_key(
            "provider-fingerprint",
            runtime,
            32,
            32,
            32,
            8,
            3_840,
            15_360,
            scratch,
        );
        assert_eq!(
            key,
            dense_geglu_marlin_replay_key(
                "provider-fingerprint",
                runtime,
                32,
                32,
                32,
                8,
                3_840,
                15_360,
                scratch,
            )
        );
        assert_ne!(
            key,
            dense_geglu_marlin_replay_key(
                "provider-fingerprint",
                runtime,
                32,
                32,
                32,
                32,
                3_840,
                15_360,
                marlin_swiglu_scratch_layout(32, 15_360, 512).unwrap(),
            )
        );
        assert_ne!(
            key,
            dense_geglu_marlin_replay_key(
                "another-provider",
                runtime,
                32,
                32,
                32,
                8,
                3_840,
                15_360,
                scratch,
            )
        );
    }
}

#[cfg(all(test, feature = "vllm-marlin"))]
mod dense_swiglu_marlin_tests {
    use super::{
        dense_swiglu_projection_for_formats, marlin_fp8_swiglu_gate_up_layout,
        marlin_swiglu_scratch_layout, DenseSwiGluProjection, PhysicalWeightLayout,
        COMPRESSED_TENSORS_MARLIN_QUANTIZATION_FORMAT_ID,
        MARLIN_FP8_GROUP128_QUANTIZATION_FORMAT_ID, MARLIN_FP8_QUANTIZATION_FORMAT_ID,
    };
    use ferrum_interfaces::vnext::{
        CompositeWeightPart, PhysicalWeightComponentBinding, PhysicalWeightPadding,
        QuantizationFormatId, WeightId,
    };
    use std::collections::BTreeSet;

    fn formats(values: &[&str]) -> BTreeSet<QuantizationFormatId> {
        values
            .iter()
            .map(|value| QuantizationFormatId::new(*value).unwrap())
            .collect()
    }

    fn quantized_layout(label: &str) -> PhysicalWeightLayout {
        PhysicalWeightLayout::Quantized {
            packed_values: PhysicalWeightComponentBinding::exact_contiguous(
                WeightId::new(format!("component.{label}.packed")).unwrap(),
            ),
            packed_dimensions: vec![1, 256, 128],
            scales: PhysicalWeightComponentBinding::exact_contiguous(
                WeightId::new(format!("component.{label}.scales")).unwrap(),
            ),
            zero_points: None,
            zero_point_packed_dimensions: None,
            axis_indices: None,
            permutation: None,
            codebook: None,
            group_axis: 2,
            group_padding: PhysicalWeightPadding::Exact,
        }
    }

    fn gate_up_layout() -> PhysicalWeightLayout {
        PhysicalWeightLayout::Composite {
            parts: vec![
                CompositeWeightPart {
                    layout: Box::new(quantized_layout("gate")),
                    logical_offsets: vec![0, 0, 0],
                    extents: vec![1, 256, 128],
                },
                CompositeWeightPart {
                    layout: Box::new(quantized_layout("up")),
                    logical_offsets: vec![1, 0, 0],
                    extents: vec![1, 256, 128],
                },
            ],
        }
    }

    #[test]
    fn dense_swiglu_projection_classification_is_exact() {
        for (format, expected) in [
            (None, DenseSwiGluProjection::F16),
            (
                Some(MARLIN_FP8_QUANTIZATION_FORMAT_ID),
                DenseSwiGluProjection::MarlinFp8,
            ),
            (
                Some(MARLIN_FP8_GROUP128_QUANTIZATION_FORMAT_ID),
                DenseSwiGluProjection::MarlinFp8,
            ),
            (
                Some(COMPRESSED_TENSORS_MARLIN_QUANTIZATION_FORMAT_ID),
                DenseSwiGluProjection::CompressedTensorsMarlin,
            ),
        ] {
            let formats = format.map_or_else(BTreeSet::new, |format| formats(&[format]));
            assert_eq!(
                dense_swiglu_projection_for_formats(&formats, &formats),
                Ok(expected)
            );
        }

        let fp8 = formats(&[MARLIN_FP8_QUANTIZATION_FORMAT_ID]);
        let compressed = formats(&[COMPRESSED_TENSORS_MARLIN_QUANTIZATION_FORMAT_ID]);
        let dense = BTreeSet::new();
        let unknown = formats(&["quantization.test.unknown"]);
        let multiple = formats(&[
            MARLIN_FP8_QUANTIZATION_FORMAT_ID,
            COMPRESSED_TENSORS_MARLIN_QUANTIZATION_FORMAT_ID,
        ]);
        for (gate_up, down) in [
            (&fp8, &compressed),
            (&fp8, &dense),
            (&unknown, &unknown),
            (&multiple, &multiple),
        ] {
            assert!(dense_swiglu_projection_for_formats(gate_up, down).is_err());
        }
    }

    #[test]
    fn marlin_fp8_gate_up_requires_exact_two_part_placement() {
        let layout = gate_up_layout();
        assert!(marlin_fp8_swiglu_gate_up_layout(&layout, 0, 256, 128)
            .is_ok_and(|layout| matches!(layout, PhysicalWeightLayout::Quantized { .. })));
        assert!(marlin_fp8_swiglu_gate_up_layout(&layout, 1, 256, 128)
            .is_ok_and(|layout| matches!(layout, PhysicalWeightLayout::Quantized { .. })));
        assert!(marlin_fp8_swiglu_gate_up_layout(&layout, 2, 256, 128).is_err());

        let mut bad_offset = gate_up_layout();
        let PhysicalWeightLayout::Composite { parts } = &mut bad_offset else {
            unreachable!();
        };
        parts[1].logical_offsets[0] = 2;
        assert!(marlin_fp8_swiglu_gate_up_layout(&bad_offset, 0, 256, 128).is_err());

        let mut bad_extent = gate_up_layout();
        let PhysicalWeightLayout::Composite { parts } = &mut bad_extent else {
            unreachable!();
        };
        parts[0].extents[0] = 2;
        assert!(marlin_fp8_swiglu_gate_up_layout(&bad_extent, 0, 256, 128).is_err());

        let mut missing_part = gate_up_layout();
        let PhysicalWeightLayout::Composite { parts } = &mut missing_part else {
            unreachable!();
        };
        parts.pop();
        assert!(marlin_fp8_swiglu_gate_up_layout(&missing_part, 0, 256, 128).is_err());

        let mut dense_part = gate_up_layout();
        let PhysicalWeightLayout::Composite { parts } = &mut dense_part else {
            unreachable!();
        };
        parts[1].layout = Box::new(PhysicalWeightLayout::Dense {
            component_id: WeightId::new("component.up.dense").unwrap(),
        });
        assert!(marlin_fp8_swiglu_gate_up_layout(&dense_part, 0, 256, 128).is_err());
    }

    #[test]
    fn marlin_swiglu_scratch_layout_is_aligned_and_overflow_checked() {
        let layout = marlin_swiglu_scratch_layout(3, 5, 128).unwrap();
        assert_eq!(layout.activation_elements, 15);
        assert_eq!(layout.activation_bytes, 30);
        assert_eq!(layout.workspace_offset, 96);
        assert_eq!(layout.workspace_bytes, 128);
        assert_eq!(layout.required_bytes, 224);

        assert!(marlin_swiglu_scratch_layout(u64::MAX, 2, 0).is_err());
        assert!(marlin_swiglu_scratch_layout(1, u64::MAX, 0).is_err());
        assert!(marlin_swiglu_scratch_layout(1, 1, u64::MAX).is_err());
    }
}

#[cfg(feature = "vllm-marlin")]
#[derive(Debug, Clone, Copy)]
struct SharedCompressedTensorsWeight {
    packed_region: usize,
    scales_region: usize,
    zero_points_region: usize,
    group_size: i32,
}

#[cfg(feature = "vllm-marlin")]
fn push_shared_compressed_tensors_weight(
    regions: &mut Vec<CudaBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ordinal: u32,
    logical_dimensions: &[u64],
    composite_partition: Option<usize>,
) -> Result<SharedCompressedTensorsWeight, String> {
    let resolve = |participant: &OperationInvocation<'_, CudaDeviceBuffer>| {
        let value = binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?;
        if let Some(partition) = composite_partition {
            let weight = value.weight().ok_or_else(|| {
                format!("compressed-tensors SwiGLU input {ordinal} has no weight layout")
            })?;
            let ferrum_interfaces::vnext::PhysicalWeightLayout::Composite { parts } =
                weight.physical_layout()
            else {
                return Err(format!(
                    "compressed-tensors SwiGLU input {ordinal} must be a composite gate/up weight"
                ));
            };
            let part = parts.get(partition).ok_or_else(|| {
                format!("compressed-tensors SwiGLU gate/up partition {partition} is absent")
            })?;
            let [output_features, input_features] = logical_dimensions else {
                return Err("compressed-tensors SwiGLU partition must be rank two".to_owned());
            };
            if part.logical_offsets != [partition as u64, 0, 0]
                || part.extents != [1, *output_features, *input_features]
            {
                return Err(format!(
                    "compressed-tensors SwiGLU partition {partition} has invalid placement"
                ));
            }
            resolve_compressed_tensors_marlin_layout(
                participant,
                value,
                part.layout.as_ref(),
                logical_dimensions,
            )
        } else {
            resolve_compressed_tensors_marlin_matrix_weight(participant, value, logical_dimensions)
        }
    };

    let first = resolve(&invocation.participants()[0])?;
    for participant in &invocation.participants()[1..] {
        let candidate = resolve(participant)?;
        if candidate.logical_dimensions() != first.logical_dimensions()
            || candidate.packed_physical_dimensions() != first.packed_physical_dimensions()
            || candidate.scales_physical_dimensions() != first.scales_physical_dimensions()
            || candidate.zero_points_physical_dimensions()
                != first.zero_points_physical_dimensions()
            || candidate.group_size() != first.group_size()
            || !same_physical_region(first.packed_region(), candidate.packed_region())
            || !same_physical_region(first.scales_region(), candidate.scales_region())
            || !same_physical_region(first.zero_points_region(), candidate.zero_points_region())
        {
            return Err(format!(
                "compressed-tensors SwiGLU input {ordinal} is not shared by all participants"
            ));
        }
    }
    let group_size = i32::try_from(first.group_size())
        .map_err(|_| "compressed-tensors SwiGLU group size exceeds i32".to_owned())?;
    let [packed, scales, zero_points] = first.into_regions();
    let packed_region = regions.len();
    regions.push(packed);
    let scales_region = regions.len();
    regions.push(scales);
    let zero_points_region = regions.len();
    regions.push(zero_points);
    Ok(SharedCompressedTensorsWeight {
        packed_region,
        scales_region,
        zero_points_region,
        group_size,
    })
}

#[cfg(feature = "vllm-marlin")]
fn encode_compressed_tensors_dense_swiglu(
    provider_fingerprint: &str,
    planar_silu_mul: &CudaFunction,
    projection_runtime: MarlinProjectionRuntime,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    ensure_invocation(&invocation, DENSE_SWIGLU_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let intermediate_size = unsigned_attribute(first.attributes(), "intermediate_size")?;
    for participant in invocation.participants() {
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || unsigned_attribute(participant.attributes(), "intermediate_size")?
                != intermediate_size
        {
            return Err(
                "compressed-tensors dense SwiGLU participant attributes disagree".to_owned(),
            );
        }
        validate_dense_swiglu(
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            binding(participant.bindings(), ResolvedValueRole::Input, 2)?,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            hidden_size,
            intermediate_size,
        )?;
    }

    let tokens = invocation.work_shape().immediate_tokens();
    let MarlinSwiGluScratchLayout {
        activation_elements,
        activation_bytes,
        workspace_offset,
        workspace_bytes,
        required_bytes: required_scratch_bytes,
    } = marlin_swiglu_scratch_layout(
        tokens,
        intermediate_size,
        projection_runtime.workspace_bytes()?,
    )?;

    let mut regions = Vec::new();
    let gate = push_shared_compressed_tensors_weight(
        &mut regions,
        &invocation,
        1,
        &[intermediate_size, hidden_size],
        Some(0),
    )?;
    let up = push_shared_compressed_tensors_weight(
        &mut regions,
        &invocation,
        1,
        &[intermediate_size, hidden_size],
        Some(1),
    )?;
    let down = push_shared_compressed_tensors_weight(
        &mut regions,
        &invocation,
        2,
        &[hidden_size, intermediate_size],
        None,
    )?;
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

    let rows = checked_i32(tokens, "compressed-tensors SwiGLU token count")?;
    let hidden = checked_i32(hidden_size, "compressed-tensors SwiGLU hidden width")?;
    let intermediate = checked_i32(
        intermediate_size,
        "compressed-tensors SwiGLU intermediate width",
    )?;
    let planar_silu_mul = planar_silu_mul.clone();
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "compressed-tensors SwiGLU participant count",
    )?;
    let replay_key = CudaCommandReplayKeyBuilder::new(
        provider_fingerprint,
        "vnext_dense_swiglu_compressed_tensors_marlin",
    )
    .i32(rows)
    .i32(hidden)
    .i32(intermediate)
    .u64(workspace_offset)
    .u64(required_scratch_bytes)
    .finish();
    CudaDeviceCommand::replayable_operation_with_blas(
        "vnext_dense_swiglu_compressed_tensors_marlin",
        regions,
        replay_key,
        move |stream, blas, regions| {
            let scratch = &regions[scratch_region];
            if scratch.length_bytes() < required_scratch_bytes {
                return Err(CudaDeviceRuntimeError::contract(
                    "compressed-tensors SwiGLU scratch is smaller than admitted",
                ));
            }
            let gate_output = scratch.device_ptr();
            let up_output = gate_output.checked_add(activation_bytes).ok_or_else(|| {
                CudaDeviceRuntimeError::contract("compressed-tensors up pointer overflows")
            })?;
            let activation = up_output.checked_add(activation_bytes).ok_or_else(|| {
                CudaDeviceRuntimeError::contract("compressed-tensors activation pointer overflows")
            })?;
            let workspace = scratch
                .device_ptr()
                .checked_add(workspace_offset)
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "compressed-tensors workspace pointer overflows",
                    )
                })?;
            for (weight, output, output_features, input_features, input) in [
                (
                    gate,
                    gate_output,
                    intermediate,
                    hidden,
                    regions[input_region].device_ptr(),
                ),
                (
                    up,
                    up_output,
                    intermediate,
                    hidden,
                    regions[input_region].device_ptr(),
                ),
            ] {
                projection_runtime.launch(
                    MarlinF16WeightType::U4,
                    stream,
                    input,
                    regions[weight.packed_region].device_ptr(),
                    regions[weight.scales_region].device_ptr(),
                    Some(regions[weight.zero_points_region].device_ptr()),
                    output,
                    workspace,
                    workspace_bytes,
                    rows,
                    output_features,
                    input_features,
                    weight.group_size,
                    "compressed-tensors SwiGLU gate/up projection",
                )?;
            }
            launch_planar_silu_mul(
                stream,
                &planar_silu_mul,
                gate_output,
                up_output,
                activation,
                activation_elements,
            )?;
            projection_runtime.launch(
                MarlinF16WeightType::U4,
                stream,
                activation,
                regions[down.packed_region].device_ptr(),
                regions[down.scales_region].device_ptr(),
                Some(regions[down.zero_points_region].device_ptr()),
                regions[output_region].device_ptr(),
                workspace,
                workspace_bytes,
                rows,
                hidden,
                intermediate,
                down.group_size,
                "compressed-tensors SwiGLU down projection",
            )?;
            let _ = blas;
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(DeviceBatchingForm::Packed, participant_count, tokens, 4, 0)
    })
    .map_err(|error| error.to_string())
}

#[cfg(feature = "vllm-marlin")]
fn encode_marlin_fp8_dense_swiglu(
    provider_fingerprint: &str,
    planar_silu_mul: &CudaFunction,
    projection_runtime: MarlinProjectionRuntime,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    ensure_invocation(&invocation, DENSE_SWIGLU_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let intermediate_size = unsigned_attribute(first.attributes(), "intermediate_size")?;
    for participant in invocation.participants() {
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || unsigned_attribute(participant.attributes(), "intermediate_size")?
                != intermediate_size
        {
            return Err("Marlin FP8 dense SwiGLU participant attributes disagree".to_owned());
        }
        validate_dense_swiglu(
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            binding(participant.bindings(), ResolvedValueRole::Input, 2)?,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            hidden_size,
            intermediate_size,
        )?;
    }

    let tokens = invocation.work_shape().immediate_tokens();
    let MarlinSwiGluScratchLayout {
        activation_elements,
        activation_bytes,
        workspace_offset,
        workspace_bytes,
        required_bytes: required_scratch_bytes,
    } = marlin_swiglu_scratch_layout(
        tokens,
        intermediate_size,
        projection_runtime.workspace_bytes()?,
    )?;
    let mut regions = Vec::new();
    let gate = push_shared_marlin_fp8_weight(
        &mut regions,
        &invocation,
        1,
        &[intermediate_size, hidden_size],
        Some(0),
    )?;
    let up = push_shared_marlin_fp8_weight(
        &mut regions,
        &invocation,
        1,
        &[intermediate_size, hidden_size],
        Some(1),
    )?;
    let down = push_shared_marlin_fp8_weight(
        &mut regions,
        &invocation,
        2,
        &[hidden_size, intermediate_size],
        None,
    )?;
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

    let rows = checked_i32(tokens, "Marlin FP8 SwiGLU token count")?;
    let hidden = checked_i32(hidden_size, "Marlin FP8 SwiGLU hidden width")?;
    let intermediate = checked_i32(intermediate_size, "Marlin FP8 SwiGLU intermediate width")?;
    let planar_silu_mul = planar_silu_mul.clone();
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "Marlin FP8 SwiGLU participant count",
    )?;
    let replay_key =
        CudaCommandReplayKeyBuilder::new(provider_fingerprint, "vnext_dense_swiglu_marlin_fp8")
            .i32(projection_runtime.multiprocessor_count)
            .i32(projection_runtime.device_ordinal)
            .i32(gate.group_size)
            .i32(up.group_size)
            .i32(down.group_size)
            .i32(rows)
            .i32(hidden)
            .i32(intermediate)
            .u64(workspace_offset)
            .u64(required_scratch_bytes)
            .finish();
    CudaDeviceCommand::replayable_operation(
        "vnext_dense_swiglu_marlin_fp8",
        regions,
        replay_key,
        move |stream, regions| {
            let scratch = &regions[scratch_region];
            if scratch.length_bytes() < required_scratch_bytes {
                return Err(CudaDeviceRuntimeError::contract(
                    "Marlin FP8 SwiGLU scratch is smaller than admitted",
                ));
            }
            let gate_output = scratch.device_ptr();
            let up_output = gate_output.checked_add(activation_bytes).ok_or_else(|| {
                CudaDeviceRuntimeError::contract("Marlin FP8 SwiGLU up pointer overflows")
            })?;
            let activation = up_output.checked_add(activation_bytes).ok_or_else(|| {
                CudaDeviceRuntimeError::contract("Marlin FP8 SwiGLU activation pointer overflows")
            })?;
            let workspace = scratch
                .device_ptr()
                .checked_add(workspace_offset)
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "Marlin FP8 SwiGLU workspace pointer overflows",
                    )
                })?;
            for (weight, output) in [(gate, gate_output), (up, up_output)] {
                projection_runtime.launch(
                    MarlinF16WeightType::E4M3Fn,
                    stream,
                    regions[input_region].device_ptr(),
                    regions[weight.packed_region].device_ptr(),
                    regions[weight.scales_region].device_ptr(),
                    None,
                    output,
                    workspace,
                    workspace_bytes,
                    rows,
                    intermediate,
                    hidden,
                    weight.group_size,
                    "Marlin FP8 SwiGLU gate/up projection",
                )?;
            }
            launch_planar_silu_mul(
                stream,
                &planar_silu_mul,
                gate_output,
                up_output,
                activation,
                activation_elements,
            )?;
            projection_runtime.launch(
                MarlinF16WeightType::E4M3Fn,
                stream,
                activation,
                regions[down.packed_region].device_ptr(),
                regions[down.scales_region].device_ptr(),
                None,
                regions[output_region].device_ptr(),
                workspace,
                workspace_bytes,
                rows,
                hidden,
                intermediate,
                down.group_size,
                "Marlin FP8 SwiGLU down projection",
            )?;
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(DeviceBatchingForm::Packed, participant_count, tokens, 4, 0)
    })
    .map_err(|error| error.to_string())
}

fn encode_dense_swiglu(
    provider_fingerprint: &str,
    silu_mul: &CudaFunction,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    ensure_invocation(&invocation, DENSE_SWIGLU_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let first_input = binding(first.bindings(), ResolvedValueRole::Input, 0)?;
    let first_gate_up = binding(first.bindings(), ResolvedValueRole::Input, 1)?;
    let first_down = binding(first.bindings(), ResolvedValueRole::Input, 2)?;
    let first_output = binding(first.bindings(), ResolvedValueRole::Output, 0)?;
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let intermediate_size = unsigned_attribute(first.attributes(), "intermediate_size")?;
    validate_dense_swiglu(
        first_input,
        first_gate_up,
        first_down,
        first_output,
        hidden_size,
        intermediate_size,
    )?;
    for participant in &invocation.participants()[1..] {
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let gate_up = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let down = binding(participant.bindings(), ResolvedValueRole::Input, 2)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || unsigned_attribute(participant.attributes(), "intermediate_size")?
                != intermediate_size
        {
            return Err("CUDA dense SwiGLU participant attributes disagree".to_owned());
        }
        validate_dense_swiglu(input, gate_up, down, output, hidden_size, intermediate_size)?;
    }
    let tokens = invocation.work_shape().immediate_tokens();
    let activation_elements = tokens
        .checked_mul(intermediate_size)
        .ok_or_else(|| "dense SwiGLU activation element count overflows".to_owned())?;
    let gate_up_bytes = activation_elements
        .checked_mul(2)
        .and_then(|elements| elements.checked_mul(ElementType::F16.size_bytes()))
        .ok_or_else(|| "dense SwiGLU gate/up scratch size overflows".to_owned())?;
    let required_scratch_bytes = gate_up_bytes
        .checked_add(
            activation_elements
                .checked_mul(ElementType::F16.size_bytes())
                .ok_or_else(|| "dense SwiGLU activation scratch size overflows".to_owned())?,
        )
        .ok_or_else(|| "dense SwiGLU total scratch size overflows".to_owned())?;
    let scratch = shared_scratch_region(&invocation, required_scratch_bytes)?;
    let regions = vec![
        shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            tokens,
        )?,
        shared_full_region(&invocation, ResolvedValueRole::Input, 1, ElementType::F16)?,
        shared_full_region(&invocation, ResolvedValueRole::Input, 2, ElementType::F16)?,
        shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            tokens,
        )?,
        scratch,
    ];
    let token_count = tokens;
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "dense SwiGLU participant count",
    )?;
    let tokens = checked_i32(tokens, "dense SwiGLU token count")?;
    let hidden_size = checked_i32(hidden_size, "dense SwiGLU hidden size")?;
    let intermediate_size = checked_i32(intermediate_size, "dense SwiGLU intermediate size")?;
    let silu_mul = silu_mul.clone();
    let replay_key = CudaCommandReplayKeyBuilder::new(provider_fingerprint, "vnext_dense_swiglu")
        .i32(tokens)
        .i32(hidden_size)
        .i32(intermediate_size)
        .u64(gate_up_bytes)
        .u64(required_scratch_bytes)
        .finish();
    CudaDeviceCommand::replayable_operation_with_blas(
        "vnext_dense_swiglu",
        regions,
        replay_key,
        move |stream, blas, regions| {
            let input = regions[0].device_ptr();
            let gate_up_weight = regions[1].device_ptr();
            let down_weight = regions[2].device_ptr();
            let output = regions[3].device_ptr();
            let scratch = &regions[4];
            if scratch.length_bytes() < required_scratch_bytes {
                return Err(CudaDeviceRuntimeError::contract(
                    "vNext dense SwiGLU scratch is smaller than its admitted estimate",
                ));
            }
            let gate_up_output = scratch.device_ptr();
            let activation = gate_up_output.checked_add(gate_up_bytes).ok_or_else(|| {
                CudaDeviceRuntimeError::contract("vNext dense SwiGLU activation pointer overflows")
            })?;
            launch_gemm_f16(
                blas,
                input,
                gate_up_weight,
                gate_up_output,
                tokens,
                intermediate_size.checked_mul(2).ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "vNext dense SwiGLU packed width overflows i32",
                    )
                })?,
                hidden_size,
                "vNext dense SwiGLU gate/up GEMM",
            )?;
            launch_silu_mul(
                stream,
                &silu_mul,
                gate_up_output,
                activation,
                intermediate_size,
                activation_elements,
            )?;
            launch_gemm_f16(
                blas,
                activation,
                down_weight,
                output,
                tokens,
                hidden_size,
                intermediate_size,
                "vNext dense SwiGLU down GEMM",
            )?;
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            DeviceBatchingForm::Packed,
            participant_count,
            token_count,
            3,
            0,
        )
    })
    .map_err(|error| error.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DenseGeGluScratchLayout {
    activation_elements: u64,
    activation_bytes: u64,
    up_offset_bytes: u64,
    activation_offset_bytes: u64,
    required_bytes: u64,
}

fn dense_geglu_scratch_layout(
    tokens: u64,
    intermediate_size: u64,
) -> Result<DenseGeGluScratchLayout, String> {
    let activation_elements = tokens
        .checked_mul(intermediate_size)
        .ok_or_else(|| "dense GeGLU activation element count overflows".to_owned())?;
    let activation_bytes = activation_elements
        .checked_mul(ElementType::F16.size_bytes())
        .ok_or_else(|| "dense GeGLU activation byte count overflows".to_owned())?;
    let up_offset_bytes = activation_bytes;
    let activation_offset_bytes = activation_bytes
        .checked_mul(2)
        .ok_or_else(|| "dense GeGLU activation offset overflows".to_owned())?;
    let required_bytes = activation_bytes
        .checked_mul(GEGLU_SCRATCH_PARTS)
        .ok_or_else(|| "dense GeGLU scratch size overflows".to_owned())?;
    Ok(DenseGeGluScratchLayout {
        activation_elements,
        activation_bytes,
        up_offset_bytes,
        activation_offset_bytes,
        required_bytes,
    })
}

#[cfg(feature = "vllm-marlin")]
fn encode_compressed_tensors_symmetric_dense_geglu(
    provider_fingerprint: &str,
    gelu_tanh_mul: &CudaFunction,
    projection_runtime: MarlinProjectionRuntime,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    ensure_invocation(&invocation, DENSE_GEGLU_TANH_OPERATION_ID)?;
    if dense_geglu_projection(&invocation)?
        != DenseGeGluProjection::CompressedTensorsSymmetricMarlin
    {
        return Err(
            "symmetric compressed-tensors GeGLU encoder received another projection ABI".to_owned(),
        );
    }
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let intermediate_size = unsigned_attribute(first.attributes(), "intermediate_size")?;
    for participant in invocation.participants() {
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || unsigned_attribute(participant.attributes(), "intermediate_size")?
                != intermediate_size
        {
            return Err(
                "symmetric compressed-tensors GeGLU participant attributes disagree".to_owned(),
            );
        }
        validate_dense_geglu_tanh(
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            binding(participant.bindings(), ResolvedValueRole::Input, 2)?,
            binding(participant.bindings(), ResolvedValueRole::Input, 3)?,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            hidden_size,
            intermediate_size,
        )?;
    }

    let tokens = invocation.work_shape().immediate_tokens();
    let scratch_layout = marlin_swiglu_scratch_layout(
        tokens,
        intermediate_size,
        projection_runtime.workspace_bytes()?,
    )?;
    let mut regions = Vec::new();
    let gate = push_shared_symmetric_compressed_tensors_weight(
        &mut regions,
        &invocation,
        1,
        &[intermediate_size, hidden_size],
    )?;
    let up = push_shared_symmetric_compressed_tensors_weight(
        &mut regions,
        &invocation,
        2,
        &[intermediate_size, hidden_size],
    )?;
    let down = push_shared_symmetric_compressed_tensors_weight(
        &mut regions,
        &invocation,
        3,
        &[hidden_size, intermediate_size],
    )?;
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
    regions.push(shared_scratch_region(
        &invocation,
        scratch_layout.required_bytes,
    )?);

    let rows = checked_i32(tokens, "symmetric compressed-tensors GeGLU token count")?;
    let hidden = checked_i32(
        hidden_size,
        "symmetric compressed-tensors GeGLU hidden width",
    )?;
    let intermediate = checked_i32(
        intermediate_size,
        "symmetric compressed-tensors GeGLU intermediate width",
    )?;
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "symmetric compressed-tensors GeGLU participant count",
    )?;
    let gelu_tanh_mul = gelu_tanh_mul.clone();
    let replay_key = dense_geglu_marlin_replay_key(
        provider_fingerprint,
        projection_runtime,
        gate.group_size,
        up.group_size,
        down.group_size,
        rows,
        hidden,
        intermediate,
        scratch_layout,
    );
    CudaDeviceCommand::replayable_operation(
        "vnext_dense_geglu_tanh_compressed_tensors_symmetric_marlin",
        regions,
        replay_key,
        move |stream, regions| {
            let scratch = &regions[scratch_region];
            if scratch.length_bytes() < scratch_layout.required_bytes {
                return Err(CudaDeviceRuntimeError::contract(
                    "symmetric compressed-tensors GeGLU scratch is smaller than admitted",
                ));
            }
            let gate_output = scratch.device_ptr();
            let up_output = gate_output
                .checked_add(scratch_layout.activation_bytes)
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "symmetric compressed-tensors GeGLU up pointer overflows",
                    )
                })?;
            let activation = up_output
                .checked_add(scratch_layout.activation_bytes)
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "symmetric compressed-tensors GeGLU activation pointer overflows",
                    )
                })?;
            let workspace = scratch
                .device_ptr()
                .checked_add(scratch_layout.workspace_offset)
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "symmetric compressed-tensors GeGLU workspace pointer overflows",
                    )
                })?;
            for (weight, output) in [(gate, gate_output), (up, up_output)] {
                projection_runtime.launch(
                    MarlinF16WeightType::U4B8,
                    stream,
                    regions[input_region].device_ptr(),
                    regions[weight.packed_region].device_ptr(),
                    regions[weight.scales_region].device_ptr(),
                    None,
                    output,
                    workspace,
                    scratch_layout.workspace_bytes,
                    rows,
                    intermediate,
                    hidden,
                    weight.group_size,
                    "symmetric compressed-tensors GeGLU gate/up projection",
                )?;
            }
            launch_planar_gelu_tanh_mul(
                stream,
                &gelu_tanh_mul,
                gate_output,
                up_output,
                activation,
                scratch_layout.activation_elements,
            )?;
            projection_runtime.launch(
                MarlinF16WeightType::U4B8,
                stream,
                activation,
                regions[down.packed_region].device_ptr(),
                regions[down.scales_region].device_ptr(),
                None,
                regions[output_region].device_ptr(),
                workspace,
                scratch_layout.workspace_bytes,
                rows,
                hidden,
                intermediate,
                down.group_size,
                "symmetric compressed-tensors GeGLU down projection",
            )?;
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(DeviceBatchingForm::Packed, participant_count, tokens, 4, 0)
    })
    .map_err(|error| error.to_string())
}

fn encode_dense_geglu_tanh(
    provider_fingerprint: &str,
    gelu_tanh_mul: &CudaFunction,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    ensure_invocation(&invocation, DENSE_GEGLU_TANH_OPERATION_ID)?;
    if dense_geglu_projection(&invocation)? != DenseGeGluProjection::F16 {
        return Err("dense F16 GeGLU encoder received another projection ABI".to_owned());
    }
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let intermediate_size = unsigned_attribute(first.attributes(), "intermediate_size")?;
    for participant in invocation.participants() {
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || unsigned_attribute(participant.attributes(), "intermediate_size")?
                != intermediate_size
        {
            return Err("CUDA dense GeGLU participant attributes disagree".to_owned());
        }
        validate_dense_geglu_tanh(
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            binding(participant.bindings(), ResolvedValueRole::Input, 2)?,
            binding(participant.bindings(), ResolvedValueRole::Input, 3)?,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            hidden_size,
            intermediate_size,
        )?;
    }

    let token_count = invocation.work_shape().immediate_tokens();
    let scratch_layout = dense_geglu_scratch_layout(token_count, intermediate_size)?;
    let regions = vec![
        shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            token_count,
        )?,
        shared_full_region(&invocation, ResolvedValueRole::Input, 1, ElementType::F16)?,
        shared_full_region(&invocation, ResolvedValueRole::Input, 2, ElementType::F16)?,
        shared_full_region(&invocation, ResolvedValueRole::Input, 3, ElementType::F16)?,
        shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            token_count,
        )?,
        shared_scratch_region(&invocation, scratch_layout.required_bytes)?,
    ];
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "dense GeGLU participant count",
    )?;
    let tokens = checked_i32(token_count, "dense GeGLU token count")?;
    let hidden = checked_i32(hidden_size, "dense GeGLU hidden size")?;
    let intermediate = checked_i32(intermediate_size, "dense GeGLU intermediate size")?;
    let gelu_tanh_mul = gelu_tanh_mul.clone();
    let replay_key =
        CudaCommandReplayKeyBuilder::new(provider_fingerprint, "vnext_dense_geglu_tanh")
            .i32(tokens)
            .i32(hidden)
            .i32(intermediate)
            .u64(scratch_layout.activation_bytes)
            .u64(scratch_layout.up_offset_bytes)
            .u64(scratch_layout.activation_offset_bytes)
            .u64(scratch_layout.required_bytes)
            .finish();
    CudaDeviceCommand::replayable_operation_with_blas(
        "vnext_dense_geglu_tanh",
        regions,
        replay_key,
        move |stream, blas, regions| {
            let scratch = &regions[5];
            if scratch.length_bytes() < scratch_layout.required_bytes {
                return Err(CudaDeviceRuntimeError::contract(
                    "vNext dense GeGLU scratch is smaller than its admitted estimate",
                ));
            }
            let gate_output = scratch.device_ptr();
            let up_output = gate_output
                .checked_add(scratch_layout.up_offset_bytes)
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("vNext dense GeGLU up pointer overflows")
                })?;
            let activation = gate_output
                .checked_add(scratch_layout.activation_offset_bytes)
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "vNext dense GeGLU activation pointer overflows",
                    )
                })?;
            for (weight_region, output, label) in [
                (1_usize, gate_output, "vNext dense GeGLU gate GEMM"),
                (2_usize, up_output, "vNext dense GeGLU up GEMM"),
            ] {
                launch_gemm_f16(
                    blas,
                    regions[0].device_ptr(),
                    regions[weight_region].device_ptr(),
                    output,
                    tokens,
                    intermediate,
                    hidden,
                    label,
                )?;
            }
            launch_planar_gelu_tanh_mul(
                stream,
                &gelu_tanh_mul,
                gate_output,
                up_output,
                activation,
                scratch_layout.activation_elements,
            )?;
            launch_gemm_f16(
                blas,
                activation,
                regions[3].device_ptr(),
                regions[4].device_ptr(),
                tokens,
                hidden,
                intermediate,
                "vNext dense GeGLU down GEMM",
            )?;
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            DeviceBatchingForm::Packed,
            participant_count,
            token_count,
            4,
            0,
        )
    })
    .map_err(|error| error.to_string())
}

fn encode_constant_scale(
    provider_fingerprint: &str,
    function: &CudaFunction,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    ensure_invocation(&invocation, CONSTANT_SCALE_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let scale_rational = canonical_rational_attribute(first.attributes(), "scale")?;
    let scale = rational_attribute(first.attributes(), "scale")?;
    for participant in invocation.participants() {
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || canonical_rational_attribute(participant.attributes(), "scale")? != scale_rational
        {
            return Err("CUDA constant-scale participant attributes disagree".to_owned());
        }
        validate_constant_scale(
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            hidden_size,
        )?;
    }
    let tokens = invocation.work_shape().immediate_tokens();
    let elements = tokens
        .checked_mul(hidden_size)
        .ok_or_else(|| "CUDA constant-scale element count overflows".to_owned())?;
    let input = shared_token_region(
        &invocation,
        ResolvedValueRole::Input,
        0,
        ElementType::F16,
        tokens,
    )?;
    let output = shared_token_region(
        &invocation,
        ResolvedValueRole::Output,
        0,
        ElementType::F16,
        tokens,
    )?;
    if !same_physical_region(&input, &output) {
        return Err("CUDA constant-scale output does not exactly alias its input".to_owned());
    }
    let elements = checked_i32(elements, "constant-scale element count")?;
    let grid_x = checked_u32(
        u64::try_from(elements)
            .map_err(|_| "constant-scale element count is negative".to_owned())?
            .div_ceil(u64::from(THREADS_PER_BLOCK)),
        "constant-scale launch grid",
    )?;
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "constant-scale participant count",
    )?;
    let function = function.clone();
    let replay_key = CudaCommandReplayKeyBuilder::new(provider_fingerprint, "vnext_constant_scale")
        .i32(elements)
        .u32(grid_x)
        .f32(scale)
        .finish();
    CudaDeviceCommand::replayable_operation(
        "vnext_constant_scale",
        vec![input],
        replay_key,
        move |stream, regions| {
            let buffer = regions[0].device_ptr();
            let mut builder = stream.launch_builder(&function);
            builder.arg(&buffer);
            builder.arg(&scale);
            builder.arg(&elements);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|error| CudaDeviceRuntimeError::driver("vNext constant-scale launch", error))
        },
    )
    .and_then(|command| {
        command.with_work_attribution(DeviceBatchingForm::Packed, participant_count, tokens, 1, 0)
    })
    .map_err(|error| error.to_string())
}

fn encode_logit_softcap(
    provider_fingerprint: &str,
    function: &CudaFunction,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    ensure_invocation(&invocation, LOGIT_SOFTCAP_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let vocabulary_size = unsigned_attribute(first.attributes(), "vocab_size")?;
    let cap_rational = canonical_rational_attribute(first.attributes(), "cap")?;
    let cap = rational_attribute(first.attributes(), "cap")?;
    let mut regions = Vec::with_capacity(invocation.participants().len());
    for participant in invocation.participants() {
        if unsigned_attribute(participant.attributes(), "vocab_size")? != vocabulary_size
            || canonical_rational_attribute(participant.attributes(), "cap")? != cap_rational
        {
            return Err("CUDA logit-softcap participant attributes disagree".to_owned());
        }
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        validate_logit_softcap(input, output, vocabulary_size)?;
        let input_region = contiguous_region(participant, input, ElementType::F16)?;
        let output_region = contiguous_region(participant, output, ElementType::F16)?;
        if !same_physical_region(&input_region, &output_region) {
            return Err("CUDA logit-softcap output does not exactly alias its input".to_owned());
        }
        regions.push(input_region);
    }
    let elements = checked_i32(vocabulary_size, "logit-softcap vocabulary size")?;
    let grid_x = checked_u32(
        vocabulary_size.div_ceil(u64::from(THREADS_PER_BLOCK)),
        "logit-softcap launch grid",
    )?;
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "logit-softcap participant count",
    )?;
    let function = function.clone();
    let replay_key = CudaCommandReplayKeyBuilder::new(provider_fingerprint, "vnext_logit_softcap")
        .i32(elements)
        .u32(grid_x)
        .f32(cap)
        .u32(participant_count)
        .finish();
    CudaDeviceCommand::replayable_operation(
        "vnext_logit_softcap",
        regions,
        replay_key,
        move |stream, regions| {
            for region in regions {
                let buffer = region.device_ptr();
                let mut builder = stream.launch_builder(&function);
                builder.arg(&buffer);
                builder.arg(&cap);
                builder.arg(&elements);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (grid_x, 1, 1),
                        block_dim: (THREADS_PER_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("vNext logit-softcap launch", error)
                })?;
            }
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            if participant_count == 1 {
                DeviceBatchingForm::Scalar
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            participant_count,
            u64::from(participant_count),
            u64::from(participant_count),
            0,
        )
    })
    .map_err(|error| error.to_string())
}

fn encode_residual_add(
    provider_fingerprint: &str,
    function: &CudaFunction,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    ensure_invocation(&invocation, RESIDUAL_ADD_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let first_left = binding(first.bindings(), ResolvedValueRole::Input, 0)?;
    let first_right = binding(first.bindings(), ResolvedValueRole::Input, 1)?;
    let first_output = binding(first.bindings(), ResolvedValueRole::Output, 0)?;
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    validate_residual_add(first_left, first_right, first_output, hidden_size)?;
    for participant in &invocation.participants()[1..] {
        let left = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let right = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size {
            return Err("CUDA residual add participant attributes disagree".to_owned());
        }
        validate_residual_add(left, right, output, hidden_size)?;
    }
    let tokens = invocation.work_shape().immediate_tokens();
    let elements = tokens
        .checked_mul(hidden_size)
        .ok_or_else(|| "CUDA residual add element count overflows".to_owned())?;
    let regions = vec![
        shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            tokens,
        )?,
        shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            1,
            ElementType::F16,
            tokens,
        )?,
        shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            tokens,
        )?,
    ];
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "residual add participant count",
    )?;
    let token_count = tokens;
    let elements = checked_i32(elements, "residual add element count")?;
    let grid_x = checked_u32(
        u64::try_from(elements)
            .map_err(|_| "residual add element count is negative".to_owned())?
            .div_ceil(u64::from(THREADS_PER_BLOCK)),
        "residual add launch grid",
    )?;
    let function = function.clone();
    let replay_key = CudaCommandReplayKeyBuilder::new(provider_fingerprint, "vnext_residual_add")
        .i32(elements)
        .u32(grid_x)
        .finish();
    CudaDeviceCommand::replayable_operation(
        "vnext_residual_add",
        regions,
        replay_key,
        move |stream, regions| {
            let left = regions[0].device_ptr();
            let right = regions[1].device_ptr();
            let output = regions[2].device_ptr();
            let mut builder = stream.launch_builder(&function);
            builder.arg(&left);
            builder.arg(&right);
            builder.arg(&output);
            builder.arg(&elements);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|error| CudaDeviceRuntimeError::driver("vNext residual add launch", error))
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            DeviceBatchingForm::Packed,
            participant_count,
            token_count,
            1,
            0,
        )
    })
    .map_err(|error| error.to_string())
}

pub(super) fn launch_gemm_f16(
    blas: &CudaBlas,
    input: cudarc::driver::sys::CUdeviceptr,
    weight: cudarc::driver::sys::CUdeviceptr,
    output: cudarc::driver::sys::CUdeviceptr,
    rows: i32,
    out_features: i32,
    in_features: i32,
    operation: &'static str,
) -> Result<(), CudaDeviceRuntimeError> {
    unsafe {
        gemm_ex(
            *blas.handle(),
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            out_features,
            rows,
            in_features,
            &CUDA_GEMM_ALPHA_F32 as *const f32 as *const c_void,
            weight as *const c_void,
            cudaDataType_t::CUDA_R_16F,
            in_features,
            input as *const c_void,
            cudaDataType_t::CUDA_R_16F,
            in_features,
            &CUDA_GEMM_BETA_F32 as *const f32 as *const c_void,
            output as *mut c_void,
            cudaDataType_t::CUDA_R_16F,
            out_features,
            cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
    .map_err(|error| CudaDeviceRuntimeError::blas(operation, error))
}

fn launch_silu_mul(
    stream: &CudaStream,
    function: &CudaFunction,
    gate_up: cudarc::driver::sys::CUdeviceptr,
    output: cudarc::driver::sys::CUdeviceptr,
    intermediate_size: i32,
    activation_elements: u64,
) -> Result<(), CudaDeviceRuntimeError> {
    let total = checked_i32_runtime(activation_elements, "SwiGLU activation element count")?;
    let grid_x = activation_elements
        .div_ceil(u64::from(THREADS_PER_BLOCK))
        .try_into()
        .map_err(|_| CudaDeviceRuntimeError::contract("SwiGLU launch grid exceeds u32"))?;
    let mut builder = stream.launch_builder(function);
    builder.arg(&gate_up);
    builder.arg(&output);
    builder.arg(&intermediate_size);
    builder.arg(&total);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("vNext SwiGLU activation launch", error))
}

#[cfg(feature = "vllm-marlin")]
fn launch_planar_silu_mul(
    stream: &CudaStream,
    function: &CudaFunction,
    gate: cudarc::driver::sys::CUdeviceptr,
    up: cudarc::driver::sys::CUdeviceptr,
    output: cudarc::driver::sys::CUdeviceptr,
    activation_elements: u64,
) -> Result<(), CudaDeviceRuntimeError> {
    let total = checked_i32_runtime(
        activation_elements,
        "Marlin SwiGLU activation element count",
    )?;
    let grid_x = activation_elements
        .div_ceil(u64::from(THREADS_PER_BLOCK))
        .try_into()
        .map_err(|_| CudaDeviceRuntimeError::contract("Marlin SwiGLU launch grid exceeds u32"))?;
    let mut builder = stream.launch_builder(function);
    builder.arg(&gate);
    builder.arg(&up);
    builder.arg(&output);
    builder.arg(&total);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("Marlin SwiGLU activation launch", error))
}

fn launch_planar_gelu_tanh_mul(
    stream: &CudaStream,
    function: &CudaFunction,
    gate: cudarc::driver::sys::CUdeviceptr,
    up: cudarc::driver::sys::CUdeviceptr,
    output: cudarc::driver::sys::CUdeviceptr,
    activation_elements: u64,
) -> Result<(), CudaDeviceRuntimeError> {
    let total = checked_i32_runtime(activation_elements, "dense GeGLU activation element count")?;
    let grid_x = activation_elements
        .div_ceil(u64::from(THREADS_PER_BLOCK))
        .try_into()
        .map_err(|_| CudaDeviceRuntimeError::contract("dense GeGLU launch grid exceeds u32"))?;
    let mut builder = stream.launch_builder(function);
    builder.arg(&gate);
    builder.arg(&up);
    builder.arg(&output);
    builder.arg(&total);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("vNext dense GeGLU activation launch", error))
}

fn validate_rms_norm(
    input: &ResolvedValueBinding,
    weight: &ResolvedValueBinding,
    output: &ResolvedValueBinding,
    hidden_size: u64,
) -> Result<u64, String> {
    let [rows, input_hidden] = input.tensor().dimensions() else {
        return Err("CUDA RMSNorm input is not two-dimensional".to_owned());
    };
    if *input_hidden != hidden_size
        || weight.tensor().dimensions() != [hidden_size]
        || output.tensor().dimensions() != [*rows, hidden_size]
        || !f16_contiguous(input)
        || !f16_contiguous(weight)
        || !f16_contiguous(output)
    {
        return Err("CUDA RMSNorm invocation differs from its resolved signature".to_owned());
    }
    Ok(*rows)
}

fn validate_dense_linear(
    input: &ResolvedValueBinding,
    weight: &ResolvedValueBinding,
    output: &ResolvedValueBinding,
    in_features: u64,
    out_features: u64,
) -> Result<u64, String> {
    let [rows, input_width] = input.tensor().dimensions() else {
        return Err("CUDA dense linear input is not two-dimensional".to_owned());
    };
    if *input_width != in_features
        || weight.tensor().dimensions() != [out_features, in_features]
        || output.tensor().dimensions() != [*rows, out_features]
        || !f16_contiguous(input)
        || !f16_contiguous(weight)
        || !f16_contiguous(output)
    {
        return Err("CUDA dense linear invocation differs from its resolved signature".to_owned());
    }
    Ok(*rows)
}

fn validate_dense_swiglu(
    input: &ResolvedValueBinding,
    gate_up: &ResolvedValueBinding,
    down: &ResolvedValueBinding,
    output: &ResolvedValueBinding,
    hidden_size: u64,
    intermediate_size: u64,
) -> Result<u64, String> {
    let [tokens, input_hidden] = input.tensor().dimensions() else {
        return Err("CUDA dense SwiGLU input is not two-dimensional".to_owned());
    };
    if *input_hidden != hidden_size
        || gate_up.tensor().dimensions() != [2, intermediate_size, hidden_size]
        || down.tensor().dimensions() != [hidden_size, intermediate_size]
        || output.tensor().dimensions() != [*tokens, hidden_size]
        || !f16_contiguous(input)
        || !f16_contiguous(gate_up)
        || !f16_contiguous(down)
        || !f16_contiguous(output)
    {
        return Err("CUDA dense SwiGLU invocation differs from its resolved signature".to_owned());
    }
    Ok(*tokens)
}

fn dense_geglu_dimensions_match(
    input: &[u64],
    gate: &[u64],
    up: &[u64],
    down: &[u64],
    output: &[u64],
    hidden_size: u64,
    intermediate_size: u64,
) -> bool {
    let [tokens, input_hidden] = input else {
        return false;
    };
    *input_hidden == hidden_size
        && gate == [intermediate_size, hidden_size]
        && up == [intermediate_size, hidden_size]
        && down == [hidden_size, intermediate_size]
        && output == [*tokens, hidden_size]
}

fn validate_dense_geglu_tanh(
    input: &ResolvedValueBinding,
    gate: &ResolvedValueBinding,
    up: &ResolvedValueBinding,
    down: &ResolvedValueBinding,
    output: &ResolvedValueBinding,
    hidden_size: u64,
    intermediate_size: u64,
) -> Result<u64, String> {
    if !dense_geglu_dimensions_match(
        input.tensor().dimensions(),
        gate.tensor().dimensions(),
        up.tensor().dimensions(),
        down.tensor().dimensions(),
        output.tensor().dimensions(),
        hidden_size,
        intermediate_size,
    ) || !f16_contiguous(input)
        || !f16_contiguous(gate)
        || !f16_contiguous(up)
        || !f16_contiguous(down)
        || !f16_contiguous(output)
    {
        return Err("CUDA dense GeGLU invocation differs from its resolved signature".to_owned());
    }
    Ok(input.tensor().dimensions()[0])
}

fn constant_scale_dimensions_match(input: &[u64], output: &[u64], hidden_size: u64) -> bool {
    let [tokens, input_hidden] = input else {
        return false;
    };
    *input_hidden == hidden_size && output == [*tokens, hidden_size]
}

fn validate_constant_scale(
    input: &ResolvedValueBinding,
    output: &ResolvedValueBinding,
    hidden_size: u64,
) -> Result<u64, String> {
    if !constant_scale_dimensions_match(
        input.tensor().dimensions(),
        output.tensor().dimensions(),
        hidden_size,
    ) || !f16_contiguous(input)
        || !f16_contiguous(output)
        || input.storage() != output.storage()
    {
        return Err(
            "CUDA constant-scale invocation differs from its resolved in-place signature"
                .to_owned(),
        );
    }
    Ok(input.tensor().dimensions()[0])
}

fn logit_softcap_dimensions_match(input: &[u64], output: &[u64], vocabulary_size: u64) -> bool {
    input == [1, vocabulary_size] && output == [1, vocabulary_size]
}

fn validate_logit_softcap(
    input: &ResolvedValueBinding,
    output: &ResolvedValueBinding,
    vocabulary_size: u64,
) -> Result<(), String> {
    if !logit_softcap_dimensions_match(
        input.tensor().dimensions(),
        output.tensor().dimensions(),
        vocabulary_size,
    ) || !f16_contiguous(input)
        || !f16_contiguous(output)
        || input.storage() != output.storage()
    {
        return Err(
            "CUDA logit-softcap invocation differs from its resolved in-place signature".to_owned(),
        );
    }
    Ok(())
}

fn validate_residual_add(
    left: &ResolvedValueBinding,
    right: &ResolvedValueBinding,
    output: &ResolvedValueBinding,
    hidden_size: u64,
) -> Result<u64, String> {
    let [tokens, input_hidden] = left.tensor().dimensions() else {
        return Err("CUDA residual add input is not two-dimensional".to_owned());
    };
    if *input_hidden != hidden_size
        || right.tensor().dimensions() != [*tokens, hidden_size]
        || output.tensor().dimensions() != [*tokens, hidden_size]
        || !f16_contiguous(left)
        || !f16_contiguous(right)
        || !f16_contiguous(output)
    {
        return Err("CUDA residual add invocation differs from its resolved signature".to_owned());
    }
    tokens
        .checked_mul(hidden_size)
        .ok_or_else(|| "CUDA residual add element count overflows".to_owned())
}

fn f16_contiguous(binding: &ResolvedValueBinding) -> bool {
    binding.tensor().element_type() == ElementType::F16
        && matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
}

fn shared_token_region(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    role: ResolvedValueRole,
    ordinal: u32,
    element_type: ElementType,
    tokens: u64,
) -> Result<CudaBufferRegion, String> {
    let first = &invocation.participants()[0];
    let first_binding = binding(first.bindings(), role, ordinal)?;
    let region = contiguous_token_region(first, first_binding, element_type, 0, tokens)?;
    for participant in &invocation.participants()[1..] {
        let candidate = contiguous_token_region(
            participant,
            binding(participant.bindings(), role, ordinal)?,
            element_type,
            0,
            tokens,
        )?;
        if !same_physical_region(&region, &candidate) {
            return Err(format!(
                "CUDA batch {role:?} binding {ordinal} is not one shared packed-token region"
            ));
        }
    }
    Ok(region)
}

pub(super) fn token_binding_is_packed(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    role: ResolvedValueRole,
    ordinal: u32,
) -> Result<bool, String> {
    invocation
        .binding_uses_packed_batch_coordinates(role, ordinal)
        .map_err(|error| error.to_string())
}

pub(super) fn shared_full_region(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    role: ResolvedValueRole,
    ordinal: u32,
    element_type: ElementType,
) -> Result<CudaBufferRegion, String> {
    let first = &invocation.participants()[0];
    let region = contiguous_region(
        first,
        binding(first.bindings(), role, ordinal)?,
        element_type,
    )?;
    for participant in &invocation.participants()[1..] {
        let candidate = contiguous_region(
            participant,
            binding(participant.bindings(), role, ordinal)?,
            element_type,
        )?;
        if !same_physical_region(&region, &candidate) {
            return Err(format!(
                "CUDA batch {role:?} binding {ordinal} is not one shared full region"
            ));
        }
    }
    Ok(region)
}

pub(super) fn shared_scratch_region(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    required_bytes: u64,
) -> Result<CudaBufferRegion, String> {
    let region = contiguous_scratch_region(&invocation.participants()[0], required_bytes)?;
    for participant in &invocation.participants()[1..] {
        let candidate = contiguous_scratch_region(participant, required_bytes)?;
        if !same_physical_region(&region, &candidate) {
            return Err("CUDA batch scratch is not one invocation-scoped region".to_owned());
        }
    }
    Ok(region)
}

pub(super) fn shared_binding_region(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    required_bytes: u64,
) -> Result<CudaBufferRegion, String> {
    let region = contiguous_binding_region(&invocation.participants()[0], required_bytes)?;
    for participant in &invocation.participants()[1..] {
        let candidate = contiguous_binding_region(participant, required_bytes)?;
        if !same_physical_region(&region, &candidate) {
            return Err("CUDA batch binding is not one invocation-scoped region".to_owned());
        }
    }
    Ok(region)
}

fn contiguous_scratch_region(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    required_bytes: u64,
) -> Result<CudaBufferRegion, String> {
    let view = participant
        .scratch_view()
        .ok_or_else(|| "CUDA dense SwiGLU invocation has no scratch view".to_owned())?;
    if view.descriptor().element_type != ElementType::U8
        || view.descriptor().size_bytes < required_bytes
    {
        return Err("CUDA dense SwiGLU scratch differs from its estimate".to_owned());
    }
    let translated = view
        .translate(0, view.descriptor().size_bytes)
        .map_err(|error| error.to_string())?;
    let mut physical = translated.iter();
    let region = physical
        .next()
        .ok_or_else(|| "CUDA dense SwiGLU scratch has no physical region".to_owned())?;
    if physical.next().is_some() {
        return Err("CUDA dense SwiGLU scratch is not physically contiguous".to_owned());
    }
    let (buffer, range, retention) = region.buffer_and_physical_range();
    buffer
        .retained_region(range, retention)
        .map_err(|error| error.to_string())
}

fn contiguous_binding_region(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    required_bytes: u64,
) -> Result<CudaBufferRegion, String> {
    let view = participant
        .binding_view()
        .ok_or_else(|| "CUDA invocation has no binding workspace view".to_owned())?;
    if view.descriptor().element_type != ElementType::U8
        || view.descriptor().size_bytes < required_bytes
    {
        return Err("CUDA binding workspace differs from its estimate".to_owned());
    }
    let translated = view
        .translate(0, view.descriptor().size_bytes)
        .map_err(|error| error.to_string())?;
    let mut physical = translated.iter();
    let region = physical
        .next()
        .ok_or_else(|| "CUDA binding workspace has no physical region".to_owned())?;
    if physical.next().is_some() {
        return Err("CUDA binding workspace is not physically contiguous".to_owned());
    }
    let (buffer, range, retention) = region.buffer_and_physical_range();
    buffer
        .retained_region(range, retention)
        .map_err(|error| error.to_string())
}

fn ensure_invocation(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    operation_id: &str,
) -> Result<(), String> {
    if invocation.participants().is_empty() || invocation.operation().id.as_str() != operation_id {
        return Err(format!(
            "CUDA provider for `{operation_id}` received another or empty operation"
        ));
    }
    Ok(())
}

fn unsigned_attribute(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<u64, String> {
    match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Unsigned(value)) => Ok(*value),
        _ => Err(format!("CUDA provider lacks unsigned attribute {name:?}")),
    }
}

fn rational_attribute(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<f32, String> {
    let rational = canonical_rational_attribute(attributes, name)?;
    let value = rational.numerator() as f64 / rational.denominator() as f64;
    let value = value as f32;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!(
            "CUDA provider rational attribute {name:?} cannot be represented as positive f32"
        ));
    }
    Ok(value)
}

fn canonical_rational_attribute(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<CanonicalRational, String> {
    match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Rational(value)) => Ok(*value),
        _ => Err(format!("CUDA provider lacks rational attribute {name:?}")),
    }
}

fn checked_i32(value: u64, context: &str) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("{context} exceeds i32"))
}

fn checked_u32(value: u64, context: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{context} exceeds u32"))
}

fn checked_i32_runtime(value: u64, context: &'static str) -> Result<i32, CudaDeviceRuntimeError> {
    i32::try_from(value)
        .map_err(|_| CudaDeviceRuntimeError::contract(format!("{context} exceeds i32")))
}

fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

fn provider_failure(
    identity: ferrum_interfaces::vnext::ExecutionIdentityEnvelope,
    stage: &str,
    message: String,
) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        stage,
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("core-issued CUDA operation identity must form a valid provider failure")
}
