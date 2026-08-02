use std::collections::BTreeSet;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    dense_linear_contract, CapabilityCatalog, CapabilityId, ContractVersion, DeviceClass,
    DeviceDescriptor, DeviceId, DeviceRuntime, DynamicStorageAllocator, DynamicStorageProfile,
    DynamicStorageView, EngineProviderDescriptor, OperationContract, OperationProvider,
    OperationRuntimeRegistry, ProviderId, VNextError, WeightMaterializerId,
    WeightMaterializerRegistry, DENSE_LINEAR_F16_CAPABILITY_ID, IDENTITY_WEIGHT_MATERIALIZER_ID,
};

use super::dense_linear::{implementation_fingerprint, ReferenceDenseLinearProvider};
use super::runtime::{
    ReferenceDeviceRuntime, ReferenceDeviceRuntimeConfig, ReferenceDeviceRuntimeError,
};

const REFERENCE_ENGINE_PROVIDER_ID: &str = "provider.engine.reference.vnext";
pub const REFERENCE_DENSE_SAFETENSORS_FORMAT_ID: &str = "weight-format.safetensors.dense";
const REFERENCE_MEMORY_BYTES: u64 = 64 * 1024 * 1024;

pub fn reference_vnext_capabilities() -> Result<BTreeSet<CapabilityId>, VNextError> {
    Ok(BTreeSet::from([CapabilityId::new(
        DENSE_LINEAR_F16_CAPABILITY_ID,
    )?]))
}

pub(super) fn reference_vnext_runtime_config(
    device_id: DeviceId,
) -> Result<ReferenceDeviceRuntimeConfig, VNextError> {
    let descriptor = DeviceDescriptor {
        id: device_id,
        class: DeviceClass::Reference,
        ordinal: 0,
        total_memory_bytes: REFERENCE_MEMORY_BYTES,
        runtime_implementation_fingerprint: implementation_fingerprint(&[
            include_str!("runtime.rs").as_bytes(),
            include_str!("dense_linear.rs").as_bytes(),
            include_str!("composition.rs").as_bytes(),
        ]),
        capabilities: reference_vnext_capabilities()?,
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )?]),
    };
    descriptor.validate()?;
    Ok(ReferenceDeviceRuntimeConfig { descriptor })
}

pub fn reference_vnext_operation_registry(
    runtime: &ReferenceDeviceRuntime,
) -> Result<OperationRuntimeRegistry<ReferenceDeviceRuntime>, ReferenceDeviceRuntimeError> {
    let contracts: Vec<Box<dyn OperationContract>> =
        vec![Box::new(dense_linear_contract().map_err(contract_error)?)];
    let providers: Vec<Box<dyn OperationProvider<ReferenceDeviceRuntime>>> =
        vec![Box::new(ReferenceDenseLinearProvider::new(runtime)?)];
    OperationRuntimeRegistry::new(contracts, providers).map_err(contract_error)
}

/// Single composition authority for reference planning, allocation, weight
/// initialization, provider binding, and execution.
pub struct ReferenceVNextComposition {
    runtime: Arc<ReferenceDeviceRuntime>,
    registry: OperationRuntimeRegistry<ReferenceDeviceRuntime>,
    weight_materializers: WeightMaterializerRegistry,
    weight_materializer_id: WeightMaterializerId,
    catalog: CapabilityCatalog,
}

impl ReferenceVNextComposition {
    pub fn create(device_id: DeviceId) -> Result<Self, ReferenceDeviceRuntimeError> {
        let config = reference_vnext_runtime_config(device_id).map_err(contract_error)?;
        let runtime = Arc::new(ReferenceDeviceRuntime::new(config)?);
        let registry = reference_vnext_operation_registry(&runtime)?;
        let weight_materializers =
            WeightMaterializerRegistry::identity_only().map_err(contract_error)?;
        let weight_materializer_id =
            WeightMaterializerId::new(IDENTITY_WEIGHT_MATERIALIZER_ID).map_err(contract_error)?;
        let engine = EngineProviderDescriptor::new(
            ProviderId::new(REFERENCE_ENGINE_PROVIDER_ID).map_err(contract_error)?,
            ContractVersion::new(1, 0),
            implementation_fingerprint(&[
                include_str!("composition.rs").as_bytes(),
                REFERENCE_ENGINE_PROVIDER_ID.as_bytes(),
            ]),
            runtime.descriptor().id.clone(),
            runtime.descriptor().capabilities.clone(),
        )
        .map_err(contract_error)?;
        let catalog = registry
            .capability_catalog(runtime.descriptor().clone(), vec![engine])
            .map_err(contract_error)?;
        let catalog = weight_materializers
            .augment_catalog(catalog)
            .map_err(contract_error)?;
        Ok(Self {
            runtime,
            registry,
            weight_materializers,
            weight_materializer_id,
            catalog,
        })
    }

    pub fn runtime(&self) -> &Arc<ReferenceDeviceRuntime> {
        &self.runtime
    }

    pub fn registry(&self) -> &OperationRuntimeRegistry<ReferenceDeviceRuntime> {
        &self.registry
    }

    pub fn catalog(&self) -> &CapabilityCatalog {
        &self.catalog
    }

    pub fn weight_materializers(&self) -> &WeightMaterializerRegistry {
        &self.weight_materializers
    }

    pub fn weight_materializer_id(&self) -> &WeightMaterializerId {
        &self.weight_materializer_id
    }
}

fn contract_error(error: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(error.to_string())
}
