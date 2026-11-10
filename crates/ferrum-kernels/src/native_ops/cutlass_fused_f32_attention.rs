//! Shared executable-provider descriptor and CPU-only source declaration.
//! Packaging does not initialize CUDA or claim an installed/qualified artifact.
//! The runtime separately verifies its compiled artifact and binds that actual
//! identity plus its prepared native plan into each admitted implementation.

use ferrum_native_ops::cutlass_fused_f32_attention::CUTLASS_FUSED_F32_ATTENTION_NUMERICAL_IDENTITY;
use ferrum_runtime::*;
use ferrum_types::{
    NativeOperatorBackend, NativeOperatorContractVersion, NativeOperatorProviderCatalog,
    NativeOperatorProviderCatalogRow, NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const PROVIDER_ID: &str = "provider.cuda.structured_gqa_attention_fused.f32.cutlass";
pub const ESTIMATOR_ID: &str = "resource-estimator.cuda.structured_gqa_attention_fused.f32.cutlass";
pub const NUMERICAL_PROFILE_ID: &str =
    "cuda.cutlass.structured_gqa_attention_fused.f32.fast_f32.v1";

/// Real provider/native/strict-fallback source, fixed build policy, pinned
/// native seed and numerical method. No compiled artifact lookup is needed.
pub fn source_declaration_fingerprint() -> String {
    digest(&[
        b"ferrum.cutlass-fused-f32-attention.provider-source.v1",
        PROVIDER_ID.as_bytes(),
        NUMERICAL_PROFILE_ID.as_bytes(),
        CUTLASS_FUSED_F32_ATTENTION_NUMERICAL_IDENTITY.as_bytes(),
        include_bytes!("cutlass_fused_f32_attention.rs"),
        include_bytes!("../backend/cuda/vnext_ops/tensor_silu/gqa_fused_f32.rs"),
        include_bytes!("../backend/cuda/vnext_ops/tensor_silu/gqa_fused_f32/native.rs"),
        include_bytes!("../backend/cuda/vnext_runtime/tensor_eager/fused_f32_owner.rs"),
        include_bytes!("../backend/cuda/vnext_ops/tensor_silu/gqa_f32.rs"),
        include_bytes!("../backend/cuda/vnext_ops/tensor_silu/attention.rs"),
        include_bytes!("../../kernels/vnext_bf16_silu.cu"),
        include_bytes!("../../native/cutlass-fused-f32-attention/abi.h"),
        include_bytes!("../../native/cutlass-fused-f32-attention/boundary.cu"),
        include_bytes!("../../native/cutlass-fused-f32-attention/source.seed.json"),
        include_bytes!("../../build.rs"),
    ])
}

/// The CUDA provider calls this same constructor after verifying its compiled
/// native artifact. Availability is not inferred from this source declaration.
pub fn provider_descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, VNextError> {
    let capability = CapabilityId::new(STRUCTURED_GQA_ATTENTION_FUSED_F32_CAPABILITY_ID)?;
    if !device.capabilities.contains(&capability) {
        return Err(VNextError::UnsupportedOperation {
            node_id: None,
            operation_id: STRUCTURED_GQA_ATTENTION_FUSED_F32_OPERATION_ID.into(),
            device_id: device.id.to_string(),
            reason: "fused F32 capability is absent".into(),
        });
    }
    descriptor_for_device_id(&device.id)
}

fn descriptor_for_device_id(device: &DeviceId) -> Result<OperationProviderDescriptor, VNextError> {
    let contract = structured_gqa_attention_fused_f32_contract()?;
    let implementation = source_declaration_fingerprint();
    let estimator = digest(&[ESTIMATOR_ID.as_bytes(), implementation.as_bytes()]);
    let bindings = (0..3)
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
        .collect();
    OperationProviderDescriptor::new(
        ProviderId::new(PROVIDER_ID)?,
        contract.descriptor().id.clone(),
        contract.descriptor().fingerprint()?,
        implementation,
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.clone(),
        BTreeSet::from([CapabilityId::new(
            STRUCTURED_GQA_ATTENTION_FUSED_F32_CAPABILITY_ID,
        )?]),
        BTreeSet::new(),
        BTreeSet::new(),
        bindings,
        ESTIMATOR_ID,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map(|descriptor| descriptor.with_fixed_tensor_work(true))
}

/// Project the exact shared descriptor into the device-independent catalog.
/// The placeholder ID is never published as a runtime/device receipt and no
/// synthetic device memory/capability inventory is constructed here.
pub fn provider_catalog_declaration() -> Result<NativeOperatorProviderCatalog, String> {
    let id = DeviceId::new("device.cuda.source_declaration").map_err(|e| e.to_string())?;
    let descriptor = descriptor_for_device_id(&id).map_err(|e| e.to_string())?;
    let contract = structured_gqa_attention_fused_f32_contract().map_err(|e| e.to_string())?;
    let operation = contract.descriptor();
    let catalog = NativeOperatorProviderCatalog {
        schema_version: NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
        backend: NativeOperatorBackend::Cuda,
        providers: vec![NativeOperatorProviderCatalogRow {
            operation_id: descriptor.operation_id().to_string(),
            operation_contract_version: NativeOperatorContractVersion::new(
                operation.version.major,
                operation.version.minor,
            ),
            operation_fingerprint: descriptor.operation_fingerprint().to_owned(),
            provider_id: descriptor.provider_id().to_string(),
            provider_version: NativeOperatorContractVersion::new(
                descriptor.version().major,
                descriptor.version().minor,
            ),
            provider_implementation_fingerprint: descriptor
                .provider_implementation_fingerprint()
                .to_owned(),
        }],
    };
    catalog.validate()?;
    Ok(catalog)
}

fn digest(parts: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    format!("{:x}", hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_catalog_projects_real_descriptor_without_artifact_or_device_bootstrap() {
        let catalog = provider_catalog_declaration().unwrap();
        let mut device = DeviceDescriptor {
            id: DeviceId::new("device.cuda.fixture").unwrap(),
            class: DeviceClass::Accelerator,
            ordinal: 0,
            total_memory_bytes: 1 << 20,
            runtime_implementation_fingerprint: "a".repeat(64),
            capabilities: BTreeSet::from([CapabilityId::new(
                STRUCTURED_GQA_ATTENTION_FUSED_F32_CAPABILITY_ID,
            )
            .unwrap()]),
            dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )
            .unwrap()]),
        };
        let runtime = provider_descriptor(&device).unwrap();
        let row = &catalog.providers[0];
        assert_eq!(
            row.provider_implementation_fingerprint,
            runtime.provider_implementation_fingerprint()
        );
        assert_eq!(row.operation_fingerprint, runtime.operation_fingerprint());
        assert_eq!(row.provider_id, runtime.provider_id().as_str());
        assert!(runtime.supports_fixed_tensor_work());
        assert_eq!(
            catalog.canonical_json_bytes().unwrap(),
            provider_catalog_declaration()
                .unwrap()
                .canonical_json_bytes()
                .unwrap()
        );
        device.capabilities.clear();
        assert!(provider_descriptor(&device).is_err());
    }
}
