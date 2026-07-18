//! CUDA providers for backend-neutral vNext operation contracts.

use std::collections::BTreeSet;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ferrum_interfaces::vnext::{
    causal_paged_attention_contract, dense_linear_contract, dense_swiglu_contract,
    gated_delta_recurrent_attention_contract, last_token_dense_linear_contract,
    residual_add_contract, rms_norm_contract, token_embedding_contract, AttributeId,
    BatchedOperationInvocation, CapabilityCatalog, CapabilityId, ContractVersion, DeviceId,
    DeviceRuntime, DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageRequirement,
    DynamicStorageView, ElementType, EncodedDeviceOperation, EngineProviderDescriptor,
    OperationContract, OperationFailure, OperationInvocation, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, OperationRuntimeRegistry, ProfilePhase, ProviderId,
    ProviderStorageBindingRequirement, ResolvedTensorLayout, ResolvedValueBinding,
    ResolvedValueRole, SemanticValue, VNextError, WeightFormatId,
    CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID, DENSE_LINEAR_F16_CAPABILITY_ID,
    DENSE_SWIGLU_F16_CAPABILITY_ID, GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
    LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID, LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
    RESIDUAL_ADD_F16_CAPABILITY_ID, RMS_NORM_F16_CAPABILITY_ID, TOKEN_EMBEDDING_F16_CAPABILITY_ID,
    TOKEN_EMBEDDING_OPERATION_ID,
};
use sha2::{Digest, Sha256};

use super::vnext_replay::CudaCommandReplayKeyBuilder;
use super::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime,
    CudaDeviceRuntimeConfig, CudaDeviceRuntimeError,
};

mod transformer;

const TOKEN_EMBEDDING_PROVIDER_ID: &str = "provider.cuda.token_embedding.f16";
const TOKEN_EMBEDDING_ESTIMATOR_ID: &str = "resource-estimator.cuda.token_embedding.f16";
const LAST_TOKEN_DENSE_LINEAR_PROVIDER_ID: &str =
    "provider.cuda.last_token_dense_linear.f16.cublas";
const LAST_TOKEN_DENSE_LINEAR_ESTIMATOR_ID: &str =
    "resource-estimator.cuda.last_token_dense_linear.f16.cublas";
const CUDA_ENGINE_PROVIDER_ID: &str = "provider.engine.cuda.vnext";
const CUDA_REUSABLE_EXECUTABLE_CAPABILITY_ID: &str =
    "capability.device.cuda.reusable_executable.v1";
const DEFAULT_REUSABLE_EXECUTABLE_CACHE_ENTRIES: usize = 64;
const DENSE_SAFETENSORS_FORMAT_ID: &str = "weight-format.safetensors.dense";
const EMBEDDING_FUNCTION_NAME: &str = "vnext_embedding_lookup_f16";
const VALUE_ALIGNMENT_BYTES: u64 = 16;
const THREADS_PER_BLOCK: u32 = 256;
const MAXIMUM_TOKENS_PER_LAUNCH: u64 = u16::MAX as u64;
pub(super) const VNEXT_KV_PAGE_BYTES: u64 = 64 * 1024;

/// Typed CUDA runtime input for the currently installed vNext provider bundle.
pub fn cuda_vnext_runtime_config(
    ordinal: usize,
    device_id: DeviceId,
) -> Result<CudaDeviceRuntimeConfig, VNextError> {
    Ok(CudaDeviceRuntimeConfig {
        ordinal,
        device_id,
        runtime_implementation_fingerprint: implementation_fingerprint(&[
            include_str!("vnext_runtime.rs").as_bytes(),
            include_str!("vnext_replay.rs").as_bytes(),
            include_str!("vnext_ops.rs").as_bytes(),
            include_str!("vnext_ops/transformer.rs").as_bytes(),
            include_str!("vnext_ops/transformer/attention.rs").as_bytes(),
            include_str!("vnext_ops/transformer/causal_attention.rs").as_bytes(),
            crate::ptx::EMBEDDING_LOOKUP.as_bytes(),
            crate::ptx::RMS_NORM.as_bytes(),
            crate::ptx::FUSED_SILU_MUL.as_bytes(),
            crate::ptx::RESIDUAL_ADD.as_bytes(),
            crate::ptx::SANDWICH_NORM.as_bytes(),
            crate::ptx::LINEAR_ATTENTION.as_bytes(),
            crate::ptx::GATED_DELTA_RULE.as_bytes(),
            crate::ptx::VNEXT_CAUSAL_ATTENTION.as_bytes(),
        ]),
        capabilities: cuda_vnext_capabilities()?,
        dynamic_storage_profiles: BTreeSet::from([
            DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )?,
            DynamicStorageProfile::new(
                DynamicStorageAllocator::FixedBlockArena {
                    block_bytes: VNEXT_KV_PAGE_BYTES,
                },
                DynamicStorageView::PagedRegions {
                    block_bytes: VNEXT_KV_PAGE_BYTES,
                },
            )?,
        ]),
        maximum_reusable_executables_per_stream: DEFAULT_REUSABLE_EXECUTABLE_CACHE_ENTRIES,
    })
}

pub fn cuda_vnext_capabilities() -> Result<BTreeSet<CapabilityId>, VNextError> {
    [
        TOKEN_EMBEDDING_F16_CAPABILITY_ID,
        LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID,
        RMS_NORM_F16_CAPABILITY_ID,
        DENSE_LINEAR_F16_CAPABILITY_ID,
        DENSE_SWIGLU_F16_CAPABILITY_ID,
        RESIDUAL_ADD_F16_CAPABILITY_ID,
        GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
        CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
        CUDA_REUSABLE_EXECUTABLE_CAPABILITY_ID,
    ]
    .into_iter()
    .map(CapabilityId::new)
    .collect()
}

/// Build the exact composition root used for both planning and dispatch.
pub fn cuda_vnext_operation_registry(
    runtime: &CudaDeviceRuntime,
) -> Result<OperationRuntimeRegistry<CudaDeviceRuntime>, CudaDeviceRuntimeError> {
    let contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(token_embedding_contract().map_err(contract_error)?),
        Box::new(last_token_dense_linear_contract().map_err(contract_error)?),
        Box::new(rms_norm_contract().map_err(contract_error)?),
        Box::new(dense_linear_contract().map_err(contract_error)?),
        Box::new(dense_swiglu_contract().map_err(contract_error)?),
        Box::new(residual_add_contract().map_err(contract_error)?),
        Box::new(gated_delta_recurrent_attention_contract().map_err(contract_error)?),
        Box::new(causal_paged_attention_contract().map_err(contract_error)?),
    ];
    let providers: Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>> = vec![
        Box::new(CudaTokenEmbeddingProvider::new(runtime)?),
        Box::new(CudaLastTokenDenseLinearProvider::new(runtime)?),
        Box::new(transformer::CudaRmsNormProvider::new(runtime)?),
        Box::new(transformer::CudaDenseLinearProvider::new(runtime)?),
        Box::new(transformer::CudaDenseSwiGluProvider::new(runtime)?),
        Box::new(transformer::CudaResidualAddProvider::new(runtime)?),
        Box::new(transformer::CudaGatedDeltaRecurrentAttentionProvider::new(
            runtime,
        )?),
        Box::new(transformer::CudaCausalPagedAttentionProvider::new(runtime)?),
    ];
    OperationRuntimeRegistry::new(contracts, providers).map_err(contract_error)
}

/// One CUDA composition root shared by planning, provisioning, and dispatch.
/// The capability catalog is derived from the retained registry objects rather
/// than reconstructed from a second descriptor list.
pub struct CudaVNextComposition {
    runtime: Arc<CudaDeviceRuntime>,
    registry: OperationRuntimeRegistry<CudaDeviceRuntime>,
    catalog: CapabilityCatalog,
}

impl CudaVNextComposition {
    pub fn create(ordinal: usize, device_id: DeviceId) -> Result<Self, CudaDeviceRuntimeError> {
        let config = cuda_vnext_runtime_config(ordinal, device_id).map_err(contract_error)?;
        let runtime = Arc::new(CudaDeviceRuntime::new(config)?);
        let registry = cuda_vnext_operation_registry(&runtime)?;
        let engine = EngineProviderDescriptor::new(
            ProviderId::new(CUDA_ENGINE_PROVIDER_ID).map_err(contract_error)?,
            ContractVersion::new(1, 0),
            implementation_fingerprint(&[
                include_str!("vnext_ops.rs").as_bytes(),
                include_str!("vnext_runtime.rs").as_bytes(),
                CUDA_ENGINE_PROVIDER_ID.as_bytes(),
            ]),
            runtime.descriptor().id.clone(),
            runtime.descriptor().capabilities.clone(),
        )
        .map_err(contract_error)?;
        let catalog = registry
            .capability_catalog(runtime.descriptor().clone(), vec![engine])
            .map_err(contract_error)?;
        Ok(Self {
            runtime,
            registry,
            catalog,
        })
    }

    pub fn runtime(&self) -> &Arc<CudaDeviceRuntime> {
        &self.runtime
    }

    pub fn registry(&self) -> &OperationRuntimeRegistry<CudaDeviceRuntime> {
        &self.registry
    }

    pub fn catalog(&self) -> &CapabilityCatalog {
        &self.catalog
    }

    pub fn into_parts(
        self,
    ) -> (
        Arc<CudaDeviceRuntime>,
        OperationRuntimeRegistry<CudaDeviceRuntime>,
        CapabilityCatalog,
    ) {
        (self.runtime, self.registry, self.catalog)
    }
}

pub struct CudaTokenEmbeddingProvider {
    descriptor: OperationProviderDescriptor,
    function: CudaFunction,
}

impl CudaTokenEmbeddingProvider {
    pub fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = token_embedding_contract().map_err(contract_error)?;
        let capability =
            CapabilityId::new(TOKEN_EMBEDDING_F16_CAPABILITY_ID).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA runtime does not advertise the token embedding capability",
            ));
        }

        let provider_fingerprint = implementation_fingerprint(&[
            include_str!("vnext_ops.rs").as_bytes(),
            crate::ptx::EMBEDDING_LOOKUP.as_bytes(),
            EMBEDDING_FUNCTION_NAME.as_bytes(),
        ]);
        let estimator_fingerprint = implementation_fingerprint(&[
            include_str!("vnext_ops.rs").as_bytes(),
            TOKEN_EMBEDDING_ESTIMATOR_ID.as_bytes(),
        ]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(TOKEN_EMBEDDING_PROVIDER_ID).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            provider_fingerprint,
            ContractVersion::new(1, 0),
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::from([
                WeightFormatId::new(DENSE_SAFETENSORS_FORMAT_ID).map_err(contract_error)?
            ]),
            BTreeSet::new(),
            contiguous_bindings(),
            TOKEN_EMBEDDING_ESTIMATOR_ID,
            ContractVersion::new(1, 0),
            estimator_fingerprint,
        )
        .map_err(contract_error)?;
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::EMBEDDING_LOOKUP.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("embedding module load", error))?;
        let function = module
            .load_function(EMBEDDING_FUNCTION_NAME)
            .map_err(|error| CudaDeviceRuntimeError::driver("embedding function load", error))?;
        Ok(Self {
            descriptor,
            function,
        })
    }
}

impl OperationResourceEstimator for CudaTokenEmbeddingProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id.as_str() != TOKEN_EMBEDDING_OPERATION_ID
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "CUDA token embedding estimator received another operation".to_owned(),
            });
        }
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            VALUE_ALIGNMENT_BYTES,
            None,
            None,
        ))
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaTokenEmbeddingProvider {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_token_embedding(&self.function, invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| provider_failure(identity, "cuda.token_embedding.encode", message))
    }
}

pub struct CudaLastTokenDenseLinearProvider {
    descriptor: OperationProviderDescriptor,
}

impl CudaLastTokenDenseLinearProvider {
    pub fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = last_token_dense_linear_contract().map_err(contract_error)?;
        let descriptor = transformer::provider_descriptor(
            runtime,
            &contract,
            LAST_TOKEN_DENSE_LINEAR_PROVIDER_ID,
            LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID,
            LAST_TOKEN_DENSE_LINEAR_ESTIMATOR_ID,
            transformer::contiguous_bindings(2),
            implementation_fingerprint(&[
                include_str!("vnext_ops.rs").as_bytes(),
                LAST_TOKEN_DENSE_LINEAR_PROVIDER_ID.as_bytes(),
            ]),
        )?;
        Ok(Self { descriptor })
    }
}

impl OperationResourceEstimator for CudaLastTokenDenseLinearProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        transformer::ensure_estimator_request(
            &self.descriptor,
            &request,
            LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
        )?;
        Ok(transformer::estimate(
            &self.descriptor,
            request.input_fingerprint(),
            None,
        ))
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaLastTokenDenseLinearProvider {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_last_token_dense_linear(
            self.descriptor.provider_implementation_fingerprint(),
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| {
            provider_failure(identity, "cuda.last_token_dense_linear.encode", message)
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct LastTokenDenseLinearLaunch {
    input_region: usize,
    output_region: usize,
}

fn encode_last_token_dense_linear(
    provider_fingerprint: &str,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    if invocation.operation().id.as_str() != LAST_TOKEN_DENSE_LINEAR_OPERATION_ID
        || invocation.participants().is_empty()
    {
        return Err("CUDA last-token dense-linear received another or empty operation".to_owned());
    }
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("CUDA last-token dense-linear participant ranges are incomplete".to_owned());
    }
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let out_features = unsigned_attribute(first.attributes(), "out_features")?;
    let input_shared = transformer::token_binding_is_shared(
        &invocation,
        ResolvedValueRole::Input,
        0,
        ElementType::F16,
    )?;
    let mut regions = vec![transformer::shared_full_region(
        &invocation,
        ResolvedValueRole::Input,
        1,
        ElementType::F16,
    )?];
    let mut launches = Vec::with_capacity(invocation.participants().len());
    for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let participant_weight = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || unsigned_attribute(participant.attributes(), "out_features")? != out_features
        {
            return Err("CUDA last-token dense-linear participant attributes disagree".to_owned());
        }
        validate_last_token_dense_linear_signature(
            input,
            participant_weight,
            output,
            hidden_size,
            out_features,
        )?;
        let source_range = token_range.source_token_range();
        let packed_range = token_range.immediate_token_range();
        let selected_range = if input_shared {
            packed_range
        } else {
            source_range
        };
        if selected_range.is_empty() {
            return Err("CUDA last-token dense-linear cannot select from an empty span".to_owned());
        }
        let last_token = selected_range.end - 1;
        let input_region = regions.len();
        let source = contiguous_token_region(participant, input, ElementType::F16, last_token, 1)?;
        regions.push(source);
        let output_region = regions.len();
        let destination = contiguous_region(participant, output, ElementType::F16)?;
        regions.push(destination);
        launches.push(LastTokenDenseLinearLaunch {
            input_region,
            output_region,
        });
    }

    let rows = 1_i32;
    let hidden_size = i32::try_from(hidden_size)
        .map_err(|_| "last-token dense-linear hidden size exceeds i32".to_owned())?;
    let out_features = i32::try_from(out_features)
        .map_err(|_| "last-token dense-linear output width exceeds i32".to_owned())?;
    let mut replay_key =
        CudaCommandReplayKeyBuilder::new(provider_fingerprint, "vnext_last_token_dense_linear")
            .i32(rows)
            .i32(hidden_size)
            .i32(out_features)
            .u64(launches.len() as u64);
    for launch in &launches {
        replay_key = replay_key
            .u64(launch.input_region as u64)
            .u64(launch.output_region as u64);
    }
    CudaDeviceCommand::replayable_operation_with_blas(
        "vnext_last_token_dense_linear",
        regions,
        replay_key.finish(),
        move |_stream, blas, regions| {
            let weight = regions[0].device_ptr();
            for launch in &launches {
                transformer::launch_gemm_f16(
                    blas,
                    regions[launch.input_region].device_ptr(),
                    weight,
                    regions[launch.output_region].device_ptr(),
                    rows,
                    out_features,
                    hidden_size,
                    "vNext last-token dense-linear GEMM",
                )?;
            }
            Ok(())
        },
    )
    .map_err(|error| error.to_string())
}

fn validate_last_token_dense_linear_signature(
    input: &ResolvedValueBinding,
    weight: &ResolvedValueBinding,
    output: &ResolvedValueBinding,
    hidden_size: u64,
    out_features: u64,
) -> Result<(), String> {
    let contiguous = |binding: &ResolvedValueBinding| {
        matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
    };
    let input_dimensions = input.tensor().dimensions();
    if input.tensor().element_type() != ElementType::F16
        || weight.tensor().element_type() != ElementType::F16
        || output.tensor().element_type() != ElementType::F16
        || input_dimensions.len() != 2
        || input_dimensions[0] == 0
        || input_dimensions[1] != hidden_size
        || weight.tensor().dimensions() != [out_features, hidden_size]
        || output.tensor().dimensions() != [1, out_features]
        || !contiguous(input)
        || !contiguous(weight)
        || !contiguous(output)
    {
        return Err(
            "CUDA last-token dense-linear invocation differs from its resolved signature"
                .to_owned(),
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct EmbeddingLaunch {
    first_region: usize,
    token_count: u64,
    vocabulary_size: u32,
    hidden_size: i32,
    grid_x: u32,
}

fn encode_token_embedding(
    function: &CudaFunction,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    if invocation.operation().id.as_str() != TOKEN_EMBEDDING_OPERATION_ID
        || invocation.participants().is_empty()
    {
        return Err("CUDA token embedding received another or empty operation".to_owned());
    }

    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("CUDA token embedding participant ranges are incomplete".to_owned());
    }
    let mut regions = Vec::with_capacity(invocation.participants().len() * 3);
    let mut launches = Vec::with_capacity(invocation.participants().len());
    for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
        let token_ids = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let table = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        let hidden_size = unsigned_attribute(participant.attributes(), "hidden_size")?;
        let vocabulary_size = unsigned_attribute(participant.attributes(), "vocab_size")?;
        validate_signature(token_ids, table, output, vocabulary_size, hidden_size)?;
        let source_range = token_range.source_token_range();
        let packed_range = token_range.immediate_token_range();
        let token_count = token_range.immediate_tokens();
        let grid_x = hidden_size
            .div_ceil(THREADS_PER_BLOCK as u64)
            .try_into()
            .map_err(|_| "embedding launch grid exceeds u32".to_owned())?;

        let first_region = regions.len();
        regions.push(contiguous_region(participant, table, ElementType::F16)?);
        regions.push(contiguous_token_region(
            participant,
            token_ids,
            ElementType::U32,
            source_range.start,
            token_count,
        )?);
        regions.push(contiguous_token_region(
            participant,
            output,
            ElementType::F16,
            packed_range.start,
            token_count,
        )?);
        launches.push(EmbeddingLaunch {
            first_region,
            token_count,
            vocabulary_size: vocabulary_size
                .try_into()
                .map_err(|_| "embedding vocabulary size exceeds u32".to_owned())?,
            hidden_size: hidden_size
                .try_into()
                .map_err(|_| "embedding hidden size exceeds i32".to_owned())?,
            grid_x,
        });
    }

    let function = function.clone();
    CudaDeviceCommand::operation("vnext_token_embedding", regions, move |stream, regions| {
        for launch in &launches {
            let table = regions[launch.first_region].device_ptr();
            let token_ids_base = regions[launch.first_region + 1].device_ptr();
            let output_base = regions[launch.first_region + 2].device_ptr();
            let mut token_offset = 0_u64;
            while token_offset < launch.token_count {
                let chunk_tokens =
                    (launch.token_count - token_offset).min(MAXIMUM_TOKENS_PER_LAUNCH);
                let token_ids = checked_pointer_offset(
                    token_ids_base,
                    token_offset,
                    ElementType::U32.size_bytes(),
                    "token id",
                )?;
                let output_element_offset = token_offset
                    .checked_mul(launch.hidden_size as u64)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "vNext embedding output element offset overflows",
                        )
                    })?;
                let output = checked_pointer_offset(
                    output_base,
                    output_element_offset,
                    ElementType::F16.size_bytes(),
                    "embedding output",
                )?;
                let batch = chunk_tokens as i32;
                let mut builder = stream.launch_builder(&function);
                builder.arg(&table);
                builder.arg(&token_ids);
                builder.arg(&output);
                builder.arg(&batch);
                builder.arg(&launch.hidden_size);
                builder.arg(&launch.vocabulary_size);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (launch.grid_x, chunk_tokens as u32, 1),
                        block_dim: (THREADS_PER_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("vNext token embedding launch", error)
                })?;
                token_offset += chunk_tokens;
            }
        }
        Ok(())
    })
    .map_err(|error| error.to_string())
}

fn checked_pointer_offset(
    base: cudarc::driver::sys::CUdeviceptr,
    elements: u64,
    element_bytes: u64,
    context: &'static str,
) -> Result<cudarc::driver::sys::CUdeviceptr, CudaDeviceRuntimeError> {
    elements
        .checked_mul(element_bytes)
        .and_then(|bytes| base.checked_add(bytes))
        .ok_or_else(|| CudaDeviceRuntimeError::contract(format!("{context} pointer overflows")))
}

fn validate_signature(
    token_ids: &ResolvedValueBinding,
    table: &ResolvedValueBinding,
    output: &ResolvedValueBinding,
    vocabulary_size: u64,
    hidden_size: u64,
) -> Result<u64, String> {
    let token_dimensions = token_ids.tensor().dimensions();
    let table_dimensions = table.tensor().dimensions();
    let output_dimensions = output.tensor().dimensions();
    let contiguous = |binding: &ResolvedValueBinding| {
        matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
    };
    if token_ids.tensor().element_type() != ElementType::U32
        || table.tensor().element_type() != ElementType::F16
        || output.tensor().element_type() != ElementType::F16
        || token_dimensions.len() != 1
        || table_dimensions != [vocabulary_size, hidden_size]
        || output_dimensions != [token_dimensions[0], hidden_size]
        || !contiguous(token_ids)
        || !contiguous(table)
        || !contiguous(output)
    {
        return Err(
            "CUDA token embedding invocation differs from its resolved signature".to_owned(),
        );
    }
    Ok(token_dimensions[0])
}

fn binding(
    bindings: &[ResolvedValueBinding],
    role: ResolvedValueRole,
    ordinal: u32,
) -> Result<&ResolvedValueBinding, String> {
    bindings
        .iter()
        .find(|binding| binding.role() == role && binding.ordinal() == ordinal)
        .ok_or_else(|| format!("CUDA operation lacks {role:?} binding {ordinal}"))
}

fn unsigned_attribute(
    attributes: &std::collections::BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<u64, String> {
    match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Unsigned(value)) => Ok(*value),
        _ => Err(format!("CUDA operation lacks unsigned attribute {name:?}")),
    }
}

fn contiguous_region(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    element_type: ElementType,
) -> Result<CudaBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("CUDA operation requires one storage component per value".to_owned());
    };
    if component.element_type() != element_type {
        return Err("CUDA operation storage element type differs from its contract".to_owned());
    }
    contiguous_region_range(
        participant,
        binding,
        element_type,
        component.offset_bytes(),
        component.length_bytes(),
    )
}

fn contiguous_token_region(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    element_type: ElementType,
    token_start: u64,
    token_count: u64,
) -> Result<CudaBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("CUDA operation requires one storage component per value".to_owned());
    };
    let projection = participant
        .work()
        .token_projection(binding.role(), binding.ordinal())
        .ok_or_else(|| "CUDA operation binding has no token work projection".to_owned())?;
    let dimensions = binding.tensor().dimensions();
    if projection.axis() != 0
        || projection.rank() as usize != dimensions.len()
        || dimensions.first() != Some(&projection.canonical_extent())
        || component.offset_bytes() != 0
        || component.length_bytes() % projection.canonical_extent() != 0
    {
        return Err(
            "CUDA contiguous token projection is not a canonical leading-axis tensor".to_owned(),
        );
    }
    let bytes_per_token = component.length_bytes() / projection.canonical_extent();
    let logical_offset = token_start
        .checked_mul(bytes_per_token)
        .ok_or_else(|| "CUDA token region offset overflows".to_owned())?;
    let logical_length = token_count
        .checked_mul(bytes_per_token)
        .ok_or_else(|| "CUDA token region length overflows".to_owned())?;
    contiguous_region_range(
        participant,
        binding,
        element_type,
        logical_offset,
        logical_length,
    )
}

fn contiguous_region_range(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    element_type: ElementType,
    logical_offset_bytes: u64,
    logical_length_bytes: u64,
) -> Result<CudaBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("CUDA operation requires one storage component per value".to_owned());
    };
    if component.element_type() != element_type {
        return Err("CUDA operation storage element type differs from its contract".to_owned());
    }
    let view = participant
        .views()
        .iter()
        .find(|view| view.resource_id() == component.resource_id())
        .ok_or_else(|| "CUDA operation value has no resource view".to_owned())?;
    // The invocation construction has already proved the resource view and
    // component describe the same allocation. The provider still translates
    // the logical slice instead of assuming an arena base pointer.
    let translated = view
        .translate(logical_offset_bytes, logical_length_bytes)
        .map_err(|error| error.to_string())?;
    let mut physical = translated.iter();
    let region = physical
        .next()
        .ok_or_else(|| "CUDA operation translated to no physical region".to_owned())?;
    if physical.next().is_some() {
        return Err("CUDA operation requires contiguous physical storage".to_owned());
    }
    let (buffer, range) = region.buffer_and_physical_range();
    let region = buffer.region(range).map_err(|error| error.to_string())?;
    if region.element_type() != element_type || region.length_bytes() != logical_length_bytes {
        return Err(
            "CUDA operation physical region differs from its resolved component".to_owned(),
        );
    }
    Ok(region)
}

fn same_physical_region(left: &CudaBufferRegion, right: &CudaBufferRegion) -> bool {
    left.device_ptr() == right.device_ptr()
        && left.length_bytes() == right.length_bytes()
        && left.element_type() == right.element_type()
}

fn contiguous_bindings() -> Vec<ProviderStorageBindingRequirement> {
    [
        (ResolvedValueRole::Input, 0),
        (ResolvedValueRole::Input, 1),
        (ResolvedValueRole::Output, 0),
    ]
    .into_iter()
    .map(|(role, ordinal)| {
        ProviderStorageBindingRequirement::new(
            role,
            ordinal,
            DynamicStorageRequirement::contiguous(),
        )
    })
    .collect()
}

fn provider_failure(
    identity: ferrum_interfaces::vnext::ExecutionIdentityEnvelope,
    stage: &'static str,
    message: String,
) -> OperationFailure {
    let message = message.chars().take(2048).collect::<String>();
    OperationFailure::new(identity, ProfilePhase::Forward, stage, message, false)
        .expect("core-issued CUDA operation identity must form a valid provider failure")
}

fn implementation_fingerprint(parts: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update((part.len() as u64).to_le_bytes());
        digest.update(part);
    }
    format!("{:x}", digest.finalize())
}

fn contract_error(error: VNextError) -> CudaDeviceRuntimeError {
    CudaDeviceRuntimeError::contract(error.to_string())
}
