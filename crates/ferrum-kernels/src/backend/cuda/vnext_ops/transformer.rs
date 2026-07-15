//! CUDA implementations of backend-neutral dense transformer operations.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;

use cudarc::cublas::{
    result::gemm_ex,
    sys::{cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType_t},
    CudaBlas,
};
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ferrum_interfaces::vnext::{
    dense_linear_contract, dense_swiglu_contract, residual_add_contract, rms_norm_contract,
    AttributeId, BatchedOperationInvocation, CapabilityId, ContractVersion, DeviceRuntime,
    DynamicStorageRequirement, ElementType, OperationContract, OperationFailure,
    OperationInvocation, OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, ProfilePhase, ProviderId,
    ProviderStorageBindingRequirement, ProviderWorkspaceRequirement, ProviderWorkspaceScope,
    ProviderWorkspaceSizeFormula, ResolvedTensorLayout, ResolvedValueBinding, ResolvedValueRole,
    SemanticValue, VNextError, WeightFormatId, DENSE_LINEAR_F16_CAPABILITY_ID,
    DENSE_LINEAR_OPERATION_ID, DENSE_SWIGLU_F16_CAPABILITY_ID, DENSE_SWIGLU_OPERATION_ID,
    RESIDUAL_ADD_F16_CAPABILITY_ID, RESIDUAL_ADD_OPERATION_ID, RMS_NORM_F16_CAPABILITY_ID,
    RMS_NORM_OPERATION_ID,
};

use super::super::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime,
    CudaDeviceRuntimeError,
};
use super::{
    binding, contiguous_region, contiguous_token_region, contract_error,
    implementation_fingerprint, same_physical_region, DENSE_SAFETENSORS_FORMAT_ID,
    THREADS_PER_BLOCK, VALUE_ALIGNMENT_BYTES,
};

mod attention;
mod causal_attention;

pub(super) use attention::CudaGatedDeltaRecurrentAttentionProvider;
pub(super) use causal_attention::CudaCausalPagedAttentionProvider;

const RMS_NORM_PROVIDER_ID: &str = "provider.cuda.rms_norm.f16";
const RMS_NORM_ESTIMATOR_ID: &str = "resource-estimator.cuda.rms_norm.f16";
const DENSE_LINEAR_PROVIDER_ID: &str = "provider.cuda.dense_linear.f16.cublas";
const DENSE_LINEAR_ESTIMATOR_ID: &str = "resource-estimator.cuda.dense_linear.f16.cublas";
const DENSE_SWIGLU_PROVIDER_ID: &str = "provider.cuda.dense_swiglu.f16.cublas";
const DENSE_SWIGLU_ESTIMATOR_ID: &str = "resource-estimator.cuda.dense_swiglu.f16.cublas";
const RESIDUAL_ADD_PROVIDER_ID: &str = "provider.cuda.residual_add.f16";
const RESIDUAL_ADD_ESTIMATOR_ID: &str = "resource-estimator.cuda.residual_add.f16";

const RMS_NORM_FUNCTION_NAME: &str = "rms_norm_f16";
const SILU_MUL_FUNCTION_NAME: &str = "fused_silu_mul_interleaved_f16";
const RESIDUAL_ADD_FUNCTION_NAME: &str = "residual_add_f16";
const SWIGLU_SCRATCH_PARTS: u64 = 3;

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
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_rms_norm(&self.function, invocation)
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
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_dense_linear(invocation)
            .map_err(|message| provider_failure(identity, "cuda.dense_linear.encode", message))
    }
}

pub(super) struct CudaDenseSwiGluProvider {
    descriptor: OperationProviderDescriptor,
    silu_mul: CudaFunction,
}

impl CudaDenseSwiGluProvider {
    pub(super) fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = dense_swiglu_contract().map_err(contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            DENSE_SWIGLU_PROVIDER_ID,
            DENSE_SWIGLU_F16_CAPABILITY_ID,
            DENSE_SWIGLU_ESTIMATOR_ID,
            contiguous_bindings(3),
            implementation_fingerprint(&[
                include_str!("transformer.rs").as_bytes(),
                crate::ptx::FUSED_SILU_MUL.as_bytes(),
                SILU_MUL_FUNCTION_NAME.as_bytes(),
            ]),
        )?;
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("SwiGLU module load", error))?;
        let silu_mul = module
            .load_function(SILU_MUL_FUNCTION_NAME)
            .map_err(|error| CudaDeviceRuntimeError::driver("SwiGLU function load", error))?;
        Ok(Self {
            descriptor,
            silu_mul,
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
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::tokens(bytes_per_token)?,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
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
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_dense_swiglu(&self.silu_mul, invocation)
            .map_err(|message| provider_failure(identity, "cuda.dense_swiglu.encode", message))
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
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_residual_add(&self.function, invocation)
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
    let capability = CapabilityId::new(capability_id).map_err(contract_error)?;
    if !runtime.descriptor().capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(format!(
            "CUDA runtime does not advertise capability `{capability_id}`"
        )));
    }
    let estimator_fingerprint = implementation_fingerprint(&[
        include_str!("transformer.rs").as_bytes(),
        estimator_id.as_bytes(),
    ]);
    OperationProviderDescriptor::new(
        ProviderId::new(provider_id).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        provider_fingerprint,
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::from([WeightFormatId::new(DENSE_SAFETENSORS_FORMAT_ID).map_err(contract_error)?]),
        BTreeSet::new(),
        bindings,
        estimator_id,
        ContractVersion::new(1, 0),
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
    CudaDeviceCommand::operation("vnext_rms_norm", regions, move |stream, regions| {
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
    let input_shared =
        token_binding_is_shared(&invocation, ResolvedValueRole::Input, 0, ElementType::F16)?;
    let output_shared =
        token_binding_is_shared(&invocation, ResolvedValueRole::Output, 0, ElementType::F16)?;
    let mut regions = vec![shared_full_region(
        &invocation,
        ResolvedValueRole::Input,
        1,
        ElementType::F16,
    )?];
    let mut launches = Vec::new();
    if input_shared && output_shared {
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
                if input_shared {
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
                if output_shared {
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
    CudaDeviceCommand::operation_with_blas(
        "vnext_dense_linear",
        regions,
        move |_stream, blas, regions| {
            for launch in launches {
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
    .map_err(|error| error.to_string())
}

fn encode_dense_swiglu(
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
    let tokens = checked_i32(tokens, "dense SwiGLU token count")?;
    let hidden_size = checked_i32(hidden_size, "dense SwiGLU hidden size")?;
    let intermediate_size = checked_i32(intermediate_size, "dense SwiGLU intermediate size")?;
    let silu_mul = silu_mul.clone();
    CudaDeviceCommand::operation_with_blas(
        "vnext_dense_swiglu",
        regions,
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
    .map_err(|error| error.to_string())
}

fn encode_residual_add(
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
    let elements = checked_i32(elements, "residual add element count")?;
    let grid_x = checked_u32(
        u64::try_from(elements)
            .map_err(|_| "residual add element count is negative".to_owned())?
            .div_ceil(u64::from(THREADS_PER_BLOCK)),
        "residual add launch grid",
    )?;
    let function = function.clone();
    CudaDeviceCommand::operation("vnext_residual_add", regions, move |stream, regions| {
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
    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    unsafe {
        gemm_ex(
            *blas.handle(),
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            out_features,
            rows,
            in_features,
            &alpha as *const f32 as *const c_void,
            weight as *const c_void,
            cudaDataType_t::CUDA_R_16F,
            in_features,
            input as *const c_void,
            cudaDataType_t::CUDA_R_16F,
            in_features,
            &beta as *const f32 as *const c_void,
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

pub(super) fn token_binding_is_shared(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    role: ResolvedValueRole,
    ordinal: u32,
    element_type: ElementType,
) -> Result<bool, String> {
    let first = &invocation.participants()[0];
    let region = contiguous_token_region(
        first,
        binding(first.bindings(), role, ordinal)?,
        element_type,
        0,
        1,
    )?;
    for participant in &invocation.participants()[1..] {
        let candidate = contiguous_token_region(
            participant,
            binding(participant.bindings(), role, ordinal)?,
            element_type,
            0,
            1,
        )?;
        if !same_physical_region(&region, &candidate) {
            return Ok(false);
        }
    }
    Ok(true)
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
    let (buffer, range) = region.buffer_and_physical_range();
    buffer.region(range).map_err(|error| error.to_string())
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
    let rational = match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Rational(value)) => *value,
        _ => return Err(format!("CUDA provider lacks rational attribute {name:?}")),
    };
    let value = rational.numerator() as f64 / rational.denominator() as f64;
    let value = value as f32;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!(
            "CUDA provider rational attribute {name:?} cannot be represented as positive f32"
        ));
    }
    Ok(value)
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
