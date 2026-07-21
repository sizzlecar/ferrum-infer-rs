//! Metal providers for backend-neutral vNext operation contracts.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    causal_paged_attention_contract, dense_linear_contract, dense_swiglu_contract,
    gated_delta_recurrent_attention_contract, last_token_dense_linear_contract,
    residual_add_contract, rms_norm_contract, routed_shared_swiglu_moe_contract,
    token_embedding_contract, AttributeId, BatchedOperationInvocation, CapabilityCatalog,
    CapabilityId, ContractVersion, DeviceId, DeviceRuntime, DynamicStorageAllocator,
    DynamicStorageProfile, DynamicStorageRequirement, DynamicStorageView, ElementType,
    EngineProviderDescriptor, ExecutionIdentityEnvelope, OperationContract, OperationFailure,
    OperationInvocation, OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationRuntimeRegistry, ProfilePhase, ProviderId,
    ProviderStorageBindingRequirement, QuantizationFormatId, ResolvedTensorLayout,
    ResolvedValueBinding, ResolvedValueRole, SemanticValue, VNextError, WeightFormatId,
    CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID, DENSE_LINEAR_F16_CAPABILITY_ID,
    DENSE_SWIGLU_F16_CAPABILITY_ID, GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
    LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID, RESIDUAL_ADD_F16_CAPABILITY_ID,
    RMS_NORM_F16_CAPABILITY_ID, ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID,
    TOKEN_EMBEDDING_F16_CAPABILITY_ID,
};
use sha2::{Digest, Sha256};

use super::vnext_runtime::{
    MetalBufferRegion, MetalDeviceBuffer, MetalDeviceRuntime, MetalDeviceRuntimeConfig,
    MetalDeviceRuntimeError,
};

mod causal_attention;
mod gated_delta_attention;
mod linear;
mod moe;
#[cfg(test)]
mod numerical_tolerance;
mod primitives;
mod weights;

use causal_attention::{MetalCausalAttentionPipelines, MetalCausalPagedAttentionProvider};
use gated_delta_attention::{MetalGatedDeltaPipelines, MetalGatedDeltaRecurrentAttentionProvider};
use linear::{
    MetalDenseLinearProvider, MetalDenseSwiGluProvider, MetalLastTokenDenseLinearProvider,
    MetalLinearPipelines,
};
use moe::{MetalMoePipelines, MetalRoutedSharedSwiGluMoeProvider};
use primitives::{
    MetalPrimitivePipelines, MetalResidualAddProvider, MetalRmsNormProvider,
    MetalTokenEmbeddingProvider,
};

const METAL_ENGINE_PROVIDER_ID: &str = "provider.engine.metal.vnext";
pub(crate) const DENSE_SAFETENSORS_FORMAT_ID: &str = "weight-format.safetensors.dense";
pub(crate) const GGUF_NATIVE_BLOCK_FORMAT_ID: &str = "weight-format.gguf.native-block";
pub(crate) const Q4_K_FORMAT_ID: &str = "quantization.gguf.q4-k";
pub(crate) const Q5_K_FORMAT_ID: &str = "quantization.gguf.q5-k";
pub(crate) const Q6_K_FORMAT_ID: &str = "quantization.gguf.q6-k";
pub(crate) const Q8_0_FORMAT_ID: &str = "quantization.gguf.q8-0";
pub(crate) const VALUE_ALIGNMENT_BYTES: u64 = 16;
pub(crate) const THREADS_PER_GROUP: u64 = 256;
pub(crate) const VNEXT_KV_PAGE_BYTES: u64 = 64 * 1024;

pub fn metal_vnext_capabilities() -> Result<BTreeSet<CapabilityId>, VNextError> {
    [
        TOKEN_EMBEDDING_F16_CAPABILITY_ID,
        RMS_NORM_F16_CAPABILITY_ID,
        RESIDUAL_ADD_F16_CAPABILITY_ID,
        DENSE_LINEAR_F16_CAPABILITY_ID,
        DENSE_SWIGLU_F16_CAPABILITY_ID,
        LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID,
        ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID,
        GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
        CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
    ]
    .into_iter()
    .map(CapabilityId::new)
    .collect()
}

pub fn metal_vnext_runtime_config(
    device_id: DeviceId,
) -> Result<MetalDeviceRuntimeConfig, VNextError> {
    Ok(MetalDeviceRuntimeConfig {
        device_id,
        runtime_implementation_fingerprint: implementation_fingerprint(&[
            include_str!("../vnext_runtime.rs").as_bytes(),
            include_str!("mod.rs").as_bytes(),
            include_str!("weights.rs").as_bytes(),
            include_str!("primitives.rs").as_bytes(),
            include_str!("primitives.metal").as_bytes(),
            include_str!("linear.rs").as_bytes(),
            include_str!("linear.metal").as_bytes(),
            include_str!("moe.rs").as_bytes(),
            include_str!("moe.metal").as_bytes(),
            include_str!("gated_delta_attention.rs").as_bytes(),
            include_str!("gated_delta_attention.metal").as_bytes(),
            include_str!("causal_attention.rs").as_bytes(),
            include_str!("causal_attention.metal").as_bytes(),
        ]),
        capabilities: metal_vnext_capabilities()?,
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
    })
}

pub fn metal_vnext_operation_registry(
    runtime: &MetalDeviceRuntime,
) -> Result<OperationRuntimeRegistry<MetalDeviceRuntime>, MetalDeviceRuntimeError> {
    let pipelines = Arc::new(MetalPrimitivePipelines::new(runtime.device())?);
    let linear_pipelines = Arc::new(MetalLinearPipelines::new(runtime.device())?);
    let moe_pipelines = Arc::new(MetalMoePipelines::new(runtime.device())?);
    let gated_delta_pipelines = Arc::new(MetalGatedDeltaPipelines::new(runtime.device())?);
    let causal_attention_pipelines =
        Arc::new(MetalCausalAttentionPipelines::new(runtime.device())?);
    let contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(token_embedding_contract().map_err(contract_error)?),
        Box::new(rms_norm_contract().map_err(contract_error)?),
        Box::new(residual_add_contract().map_err(contract_error)?),
        Box::new(dense_linear_contract().map_err(contract_error)?),
        Box::new(dense_swiglu_contract().map_err(contract_error)?),
        Box::new(last_token_dense_linear_contract().map_err(contract_error)?),
        Box::new(routed_shared_swiglu_moe_contract().map_err(contract_error)?),
        Box::new(gated_delta_recurrent_attention_contract().map_err(contract_error)?),
        Box::new(causal_paged_attention_contract().map_err(contract_error)?),
    ];
    let providers: Vec<Box<dyn OperationProvider<MetalDeviceRuntime>>> = vec![
        Box::new(MetalTokenEmbeddingProvider::new(
            runtime,
            Arc::clone(&pipelines),
        )?),
        Box::new(MetalRmsNormProvider::new(runtime, Arc::clone(&pipelines))?),
        Box::new(MetalResidualAddProvider::new(
            runtime,
            Arc::clone(&pipelines),
        )?),
        Box::new(MetalDenseLinearProvider::new(
            runtime,
            Arc::clone(&linear_pipelines),
        )?),
        Box::new(MetalDenseSwiGluProvider::new(
            runtime,
            Arc::clone(&linear_pipelines),
        )?),
        Box::new(MetalLastTokenDenseLinearProvider::new(
            runtime,
            Arc::clone(&linear_pipelines),
        )?),
        Box::new(MetalRoutedSharedSwiGluMoeProvider::new(
            runtime,
            moe_pipelines,
            Arc::clone(&linear_pipelines),
        )?),
        Box::new(MetalGatedDeltaRecurrentAttentionProvider::new(
            runtime,
            gated_delta_pipelines,
            Arc::clone(&linear_pipelines),
            Arc::clone(&pipelines),
        )?),
        Box::new(MetalCausalPagedAttentionProvider::new(
            runtime,
            causal_attention_pipelines,
            linear_pipelines,
            pipelines,
        )?),
    ];
    OperationRuntimeRegistry::new(contracts, providers).map_err(contract_error)
}

/// Composition root shared by Metal planning, provisioning, and dispatch.
/// It deliberately advertises only providers that are implemented by this
/// bundle; missing Qwen3.5 operations therefore fail before weight allocation.
pub struct MetalVNextComposition {
    runtime: Arc<MetalDeviceRuntime>,
    registry: OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: CapabilityCatalog,
}

impl MetalVNextComposition {
    pub fn create(device_id: DeviceId) -> Result<Self, MetalDeviceRuntimeError> {
        let config = metal_vnext_runtime_config(device_id).map_err(contract_error)?;
        let runtime = Arc::new(MetalDeviceRuntime::new(config)?);
        let registry = metal_vnext_operation_registry(&runtime)?;
        let engine = EngineProviderDescriptor::new(
            ProviderId::new(METAL_ENGINE_PROVIDER_ID).map_err(contract_error)?,
            ContractVersion::new(1, 0),
            implementation_fingerprint(&[
                include_str!("mod.rs").as_bytes(),
                include_str!("../vnext_runtime.rs").as_bytes(),
                METAL_ENGINE_PROVIDER_ID.as_bytes(),
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

    pub fn runtime(&self) -> &Arc<MetalDeviceRuntime> {
        &self.runtime
    }

    pub fn registry(&self) -> &OperationRuntimeRegistry<MetalDeviceRuntime> {
        &self.registry
    }

    pub fn catalog(&self) -> &CapabilityCatalog {
        &self.catalog
    }

    pub fn into_parts(
        self,
    ) -> (
        Arc<MetalDeviceRuntime>,
        OperationRuntimeRegistry<MetalDeviceRuntime>,
        CapabilityCatalog,
    ) {
        (self.runtime, self.registry, self.catalog)
    }
}

pub(crate) fn provider_descriptor(
    runtime: &MetalDeviceRuntime,
    contract: &dyn OperationContract,
    provider_id: &str,
    capability_id: &str,
    estimator_id: &str,
    bindings: Vec<ProviderStorageBindingRequirement>,
    accepted_weight_formats: &[&str],
    accepted_quantization_formats: &[&str],
    provider_fingerprint: String,
) -> Result<OperationProviderDescriptor, MetalDeviceRuntimeError> {
    let capability = CapabilityId::new(capability_id).map_err(contract_error)?;
    if !runtime.descriptor().capabilities.contains(&capability) {
        return Err(MetalDeviceRuntimeError::contract(format!(
            "Metal runtime does not advertise capability `{capability_id}`"
        )));
    }
    let weight_formats = accepted_weight_formats
        .iter()
        .map(|format| WeightFormatId::new(*format))
        .collect::<Result<BTreeSet<_>, _>>()
        .map_err(contract_error)?;
    let quantization_formats = accepted_quantization_formats
        .iter()
        .map(|format| QuantizationFormatId::new(*format))
        .collect::<Result<BTreeSet<_>, _>>()
        .map_err(contract_error)?;
    let estimator_fingerprint = implementation_fingerprint(&[
        include_str!("mod.rs").as_bytes(),
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
        contract.descriptor().version,
        runtime.descriptor().id.clone(),
        BTreeSet::from([capability]),
        weight_formats,
        quantization_formats,
        bindings,
        estimator_id,
        ContractVersion::new(1, 0),
        estimator_fingerprint,
    )
    .map_err(contract_error)
}

pub(crate) fn contiguous_bindings(input_count: u32) -> Vec<ProviderStorageBindingRequirement> {
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

pub(crate) fn estimate_without_workspace(
    descriptor: &OperationProviderDescriptor,
    request: &OperationResourceEstimateRequest<'_>,
    operation_id: &str,
) -> Result<OperationResourceEstimate, VNextError> {
    if request.operation().id.as_str() != operation_id
        || request.operation().fingerprint()? != descriptor.operation_fingerprint()
    {
        return Err(invalid_plan(format!(
            "Metal estimator `{}` received another operation",
            descriptor.resource_estimator_id()
        )));
    }
    Ok(OperationResourceEstimate::new(
        descriptor.resource_estimator_id(),
        descriptor.resource_estimator_version(),
        descriptor.resource_estimator_implementation_fingerprint(),
        request.input_fingerprint(),
        VALUE_ALIGNMENT_BYTES,
        None,
        None,
    ))
}

pub(crate) fn ensure_invocation<B>(
    invocation: &BatchedOperationInvocation<'_, B>,
    operation_id: &str,
) -> Result<(), String> {
    if invocation.participants().is_empty() || invocation.operation().id.as_str() != operation_id {
        return Err(format!(
            "Metal provider for `{operation_id}` received another or empty operation"
        ));
    }
    Ok(())
}

pub(crate) fn binding(
    bindings: &[ResolvedValueBinding],
    role: ResolvedValueRole,
    ordinal: u32,
) -> Result<&ResolvedValueBinding, String> {
    bindings
        .iter()
        .find(|binding| binding.role() == role && binding.ordinal() == ordinal)
        .ok_or_else(|| format!("Metal operation lacks {role:?} binding {ordinal}"))
}

pub(crate) fn unsigned_attribute(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<u64, String> {
    match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Unsigned(value)) => Ok(*value),
        _ => Err(format!("Metal provider lacks unsigned attribute {name:?}")),
    }
}

pub(crate) fn rational_attribute(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<f32, String> {
    let rational = match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Rational(value)) => *value,
        _ => return Err(format!("Metal provider lacks rational attribute {name:?}")),
    };
    let value = (rational.numerator() as f64 / rational.denominator() as f64) as f32;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!(
            "Metal provider rational attribute {name:?} is not a positive f32"
        ));
    }
    Ok(value)
}

pub(crate) fn contiguous_region(
    participant: &OperationInvocation<'_, MetalDeviceBuffer>,
    binding: &ResolvedValueBinding,
    element_type: ElementType,
) -> Result<MetalBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("Metal operation requires one storage component per value".to_owned());
    };
    if component.element_type() != element_type {
        return Err("Metal operation storage element type differs from its contract".to_owned());
    }
    contiguous_region_range(
        participant,
        binding,
        element_type,
        component.offset_bytes(),
        component.length_bytes(),
    )
}

pub(crate) fn contiguous_token_region(
    participant: &OperationInvocation<'_, MetalDeviceBuffer>,
    binding: &ResolvedValueBinding,
    element_type: ElementType,
    token_start: u64,
    token_count: u64,
) -> Result<MetalBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("Metal operation requires one storage component per value".to_owned());
    };
    let projection = participant
        .work()
        .token_projection(binding.role(), binding.ordinal())
        .ok_or_else(|| "Metal operation binding has no token work projection".to_owned())?;
    let dimensions = binding.tensor().dimensions();
    if projection.axis() != 0
        || projection.rank() as usize != dimensions.len()
        || dimensions.first() != Some(&projection.canonical_extent())
        || component.offset_bytes() != 0
        || component.length_bytes() % projection.canonical_extent() != 0
    {
        return Err(
            "Metal contiguous token projection is not a canonical leading-axis tensor".to_owned(),
        );
    }
    let bytes_per_token = component.length_bytes() / projection.canonical_extent();
    let offset = token_start
        .checked_mul(bytes_per_token)
        .ok_or_else(|| "Metal token region offset overflows".to_owned())?;
    let length = token_count
        .checked_mul(bytes_per_token)
        .ok_or_else(|| "Metal token region length overflows".to_owned())?;
    contiguous_region_range(participant, binding, element_type, offset, length)
}

fn contiguous_region_range(
    participant: &OperationInvocation<'_, MetalDeviceBuffer>,
    binding: &ResolvedValueBinding,
    element_type: ElementType,
    logical_offset_bytes: u64,
    logical_length_bytes: u64,
) -> Result<MetalBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("Metal operation requires one storage component per value".to_owned());
    };
    if component.element_type() != element_type {
        return Err("Metal operation storage element type differs from its contract".to_owned());
    }
    let view = participant
        .views()
        .iter()
        .find(|view| view.resource_id() == component.resource_id())
        .ok_or_else(|| "Metal operation value has no resource view".to_owned())?;
    let translated = view
        .translate(logical_offset_bytes, logical_length_bytes)
        .map_err(|error| error.to_string())?;
    let mut physical_regions = translated.iter();
    let physical = physical_regions
        .next()
        .ok_or_else(|| "Metal operation translated to no physical region".to_owned())?;
    if physical_regions.next().is_some() {
        return Err("Metal operation requires contiguous physical storage".to_owned());
    }
    let (buffer, range, retention) = physical.buffer_and_physical_range();
    let region = buffer
        .retained_region(range, retention)
        .map_err(|error| error.to_string())?;
    if region.element_type() != element_type || region.length_bytes() != logical_length_bytes {
        return Err("Metal operation retained the wrong physical region".to_owned());
    }
    Ok(region)
}

pub(crate) fn shared_token_region(
    invocation: &BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    role: ResolvedValueRole,
    ordinal: u32,
    element_type: ElementType,
    tokens: u64,
) -> Result<MetalBufferRegion, String> {
    let first = &invocation.participants()[0];
    let region = contiguous_token_region(
        first,
        binding(first.bindings(), role, ordinal)?,
        element_type,
        0,
        tokens,
    )?;
    for participant in &invocation.participants()[1..] {
        let candidate = contiguous_token_region(
            participant,
            binding(participant.bindings(), role, ordinal)?,
            element_type,
            0,
            tokens,
        )?;
        if !region.same_physical_region(&candidate) {
            return Err(format!(
                "Metal batch {role:?} binding {ordinal} is not one shared packed-token region"
            ));
        }
    }
    Ok(region)
}

pub(crate) fn token_binding_is_shared(
    invocation: &BatchedOperationInvocation<'_, MetalDeviceBuffer>,
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
        if !region.same_physical_region(&candidate) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) fn shared_full_region(
    invocation: &BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    role: ResolvedValueRole,
    ordinal: u32,
    element_type: ElementType,
) -> Result<MetalBufferRegion, String> {
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
        if !region.same_physical_region(&candidate) {
            return Err(format!(
                "Metal batch {role:?} binding {ordinal} is not one shared full region"
            ));
        }
    }
    Ok(region)
}

pub(crate) fn shared_scratch_region(
    invocation: &BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    required_bytes: u64,
) -> Result<MetalBufferRegion, String> {
    let region = contiguous_scratch_region(&invocation.participants()[0], required_bytes)?;
    for participant in &invocation.participants()[1..] {
        let candidate = contiguous_scratch_region(participant, required_bytes)?;
        if !region.same_physical_region(&candidate) {
            return Err("Metal batch scratch is not one invocation-scoped region".to_owned());
        }
    }
    Ok(region)
}

pub(crate) fn shared_binding_region(
    invocation: &BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    required_bytes: u64,
) -> Result<MetalBufferRegion, String> {
    let region = contiguous_binding_region(&invocation.participants()[0], required_bytes)?;
    for participant in &invocation.participants()[1..] {
        let candidate = contiguous_binding_region(participant, required_bytes)?;
        if !region.same_physical_region(&candidate) {
            return Err("Metal batch binding is not one invocation-scoped region".to_owned());
        }
    }
    Ok(region)
}

fn contiguous_scratch_region(
    participant: &OperationInvocation<'_, MetalDeviceBuffer>,
    required_bytes: u64,
) -> Result<MetalBufferRegion, String> {
    let view = participant
        .scratch_view()
        .ok_or_else(|| "Metal invocation has no scratch view".to_owned())?;
    if view.descriptor().element_type != ElementType::U8
        || view.descriptor().size_bytes < required_bytes
    {
        return Err("Metal scratch differs from its admitted estimate".to_owned());
    }
    let translated = view
        .translate(0, view.descriptor().size_bytes)
        .map_err(|error| error.to_string())?;
    let mut physical = translated.iter();
    let region = physical
        .next()
        .ok_or_else(|| "Metal scratch has no physical region".to_owned())?;
    if physical.next().is_some() {
        return Err("Metal scratch is not physically contiguous".to_owned());
    }
    let (buffer, range, retention) = region.buffer_and_physical_range();
    buffer
        .retained_region(range, retention)
        .map_err(|error| error.to_string())
}

fn contiguous_binding_region(
    participant: &OperationInvocation<'_, MetalDeviceBuffer>,
    required_bytes: u64,
) -> Result<MetalBufferRegion, String> {
    let view = participant
        .binding_view()
        .ok_or_else(|| "Metal invocation has no binding workspace view".to_owned())?;
    if view.descriptor().element_type != ElementType::U8
        || view.descriptor().size_bytes < required_bytes
    {
        return Err("Metal binding workspace differs from its admitted estimate".to_owned());
    }
    let translated = view
        .translate(0, view.descriptor().size_bytes)
        .map_err(|error| error.to_string())?;
    let mut physical = translated.iter();
    let region = physical
        .next()
        .ok_or_else(|| "Metal binding workspace has no physical region".to_owned())?;
    if physical.next().is_some() {
        return Err("Metal binding workspace is not physically contiguous".to_owned());
    }
    let (buffer, range, retention) = region.buffer_and_physical_range();
    buffer
        .retained_region(range, retention)
        .map_err(|error| error.to_string())
}

pub(crate) fn f16_contiguous(binding: &ResolvedValueBinding) -> bool {
    binding.tensor().element_type() == ElementType::F16
        && matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
}

pub(crate) fn checked_u32(value: u64, context: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{context} exceeds u32"))
}

pub(crate) fn provider_failure(
    identity: ExecutionIdentityEnvelope,
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
    .expect("core-issued Metal operation identity must form a valid provider failure")
}

pub(crate) fn contract_error(error: impl std::fmt::Display) -> MetalDeviceRuntimeError {
    MetalDeviceRuntimeError::contract(error.to_string())
}

pub(crate) fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

pub(crate) fn implementation_fingerprint(parts: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update((part.len() as u64).to_le_bytes());
        digest.update(part);
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::{
        OperationId, CAUSAL_PAGED_ATTENTION_OPERATION_ID, DENSE_LINEAR_OPERATION_ID,
        DENSE_SWIGLU_OPERATION_ID, GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
        LAST_TOKEN_DENSE_LINEAR_OPERATION_ID, RESIDUAL_ADD_OPERATION_ID, RMS_NORM_OPERATION_ID,
        ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID, TOKEN_EMBEDDING_OPERATION_ID,
    };

    #[test]
    fn partial_composition_advertises_only_installed_operation_capabilities() {
        let composition = MetalVNextComposition::create(DeviceId::new("device.metal.0").unwrap())
            .expect("create Metal primitive composition");
        assert_eq!(composition.runtime().descriptor().capabilities.len(), 9);
        assert_eq!(composition.catalog().device().capabilities.len(), 9);
        for operation_id in [
            TOKEN_EMBEDDING_OPERATION_ID,
            RMS_NORM_OPERATION_ID,
            RESIDUAL_ADD_OPERATION_ID,
            DENSE_LINEAR_OPERATION_ID,
            DENSE_SWIGLU_OPERATION_ID,
            LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
            ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID,
            GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
            CAUSAL_PAGED_ATTENTION_OPERATION_ID,
        ] {
            assert_eq!(
                composition
                    .catalog()
                    .providers_for(&OperationId::new(operation_id).unwrap())
                    .expect("installed primitive provider")
                    .len(),
                1
            );
        }
    }
}
