//! CUDA providers for backend-neutral vNext operation contracts.

use std::collections::BTreeSet;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ferrum_interfaces::vnext::{
    token_embedding_contract, AttributeId, BatchedOperationInvocation, CapabilityId,
    ContractVersion, DeviceId, DeviceRuntime, DynamicStorageAllocator, DynamicStorageProfile,
    DynamicStorageRequirement, DynamicStorageView, ElementType, OperationContract,
    OperationFailure, OperationInvocation, OperationProvider, OperationProviderDescriptor,
    OperationResourceEstimate, OperationResourceEstimateRequest, OperationResourceEstimator,
    OperationRuntimeRegistry, ProfilePhase, ProviderId, ProviderStorageBindingRequirement,
    ResolvedTensorLayout, ResolvedValueBinding, ResolvedValueRole, SemanticValue, VNextError,
    WeightFormatId, TOKEN_EMBEDDING_F16_CAPABILITY_ID, TOKEN_EMBEDDING_OPERATION_ID,
};
use sha2::{Digest, Sha256};

use super::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime,
    CudaDeviceRuntimeConfig, CudaDeviceRuntimeError,
};

const TOKEN_EMBEDDING_PROVIDER_ID: &str = "provider.cuda.token_embedding.f16";
const TOKEN_EMBEDDING_ESTIMATOR_ID: &str = "resource-estimator.cuda.token_embedding.f16";
const DENSE_SAFETENSORS_FORMAT_ID: &str = "weight-format.safetensors.dense";
const EMBEDDING_FUNCTION_NAME: &str = "vnext_embedding_lookup_f16";
const VALUE_ALIGNMENT_BYTES: u64 = 16;
const THREADS_PER_BLOCK: u32 = 256;
const MAXIMUM_TOKENS_PER_LAUNCH: u64 = u16::MAX as u64;

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
            include_str!("vnext_ops.rs").as_bytes(),
            crate::ptx::EMBEDDING_LOOKUP.as_bytes(),
        ]),
        capabilities: cuda_vnext_capabilities()?,
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )?]),
    })
}

pub fn cuda_vnext_capabilities() -> Result<BTreeSet<CapabilityId>, VNextError> {
    Ok(BTreeSet::from([CapabilityId::new(
        TOKEN_EMBEDDING_F16_CAPABILITY_ID,
    )?]))
}

/// Build the exact composition root used for both planning and dispatch.
pub fn cuda_vnext_operation_registry(
    runtime: &CudaDeviceRuntime,
) -> Result<OperationRuntimeRegistry<CudaDeviceRuntime>, CudaDeviceRuntimeError> {
    let contract = token_embedding_contract().map_err(contract_error)?;
    let provider = CudaTokenEmbeddingProvider::new(runtime)?;
    OperationRuntimeRegistry::new(vec![Box::new(contract)], vec![Box::new(provider)])
        .map_err(contract_error)
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
    ) -> Result<CudaDeviceCommand, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_token_embedding(&self.function, invocation)
            .map_err(|message| provider_failure(identity, message))
    }
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

    let mut regions = Vec::with_capacity(invocation.participants().len() * 3);
    let mut launches = Vec::with_capacity(invocation.participants().len());
    for participant in invocation.participants() {
        let token_ids = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let table = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        let hidden_size = unsigned_attribute(participant.attributes(), "hidden_size")?;
        let vocabulary_size = unsigned_attribute(participant.attributes(), "vocab_size")?;
        let token_count =
            validate_signature(token_ids, table, output, vocabulary_size, hidden_size)?;
        let grid_x = hidden_size
            .div_ceil(THREADS_PER_BLOCK as u64)
            .try_into()
            .map_err(|_| "embedding launch grid exceeds u32".to_owned())?;

        let first_region = regions.len();
        regions.push(contiguous_region(participant, table, ElementType::F16)?);
        regions.push(contiguous_region(participant, token_ids, ElementType::U32)?);
        regions.push(contiguous_region(participant, output, ElementType::F16)?);
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
        for launch in launches {
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
        .ok_or_else(|| format!("token embedding lacks {role:?} binding {ordinal}"))
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
        _ => Err(format!("token embedding lacks unsigned attribute {name:?}")),
    }
}

fn contiguous_region(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    element_type: ElementType,
) -> Result<CudaBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("CUDA token embedding requires one storage component per value".to_owned());
    };
    if component.element_type() != element_type {
        return Err(
            "CUDA token embedding storage element type differs from its contract".to_owned(),
        );
    }
    let view = participant
        .views()
        .iter()
        .find(|view| view.resource_id() == component.resource_id())
        .ok_or_else(|| "CUDA token embedding value has no resource view".to_owned())?;
    // The invocation construction has already proved the resource view and
    // component describe the same allocation. The provider still translates
    // the logical slice instead of assuming an arena base pointer.
    let translated = view
        .translate(component.offset_bytes(), component.length_bytes())
        .map_err(|error| error.to_string())?;
    let mut physical = translated.iter();
    let region = physical
        .next()
        .ok_or_else(|| "CUDA token embedding translated to no physical region".to_owned())?;
    if physical.next().is_some() {
        return Err("CUDA token embedding requires contiguous physical storage".to_owned());
    }
    let (buffer, range) = region.buffer_and_physical_range();
    let region = buffer.region(range).map_err(|error| error.to_string())?;
    if region.element_type() != element_type || region.length_bytes() != component.length_bytes() {
        return Err(
            "CUDA token embedding physical region differs from its resolved component".to_owned(),
        );
    }
    Ok(region)
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
    message: String,
) -> OperationFailure {
    let message = message.chars().take(2048).collect::<String>();
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "cuda.token_embedding.encode",
        message,
        false,
    )
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
