use std::collections::BTreeSet;
use std::sync::Arc;

use ferrum_interfaces::vnext::*;
use sha2::{Digest, Sha256};

use super::super::vnext_runtime::{CpuDeviceRuntime, CpuRuntimeError};
use super::provider::{CpuOperation, CpuOperationProvider};

/// An explicit host capacity and the implemented CPU registry form one planning
/// authority. Missing operations remain absent from the catalog and fail during
/// resolution, before provisioning model weights.
pub struct CpuVNextComposition {
    runtime: Arc<CpuDeviceRuntime>,
    registry: OperationRuntimeRegistry<CpuDeviceRuntime>,
    materializers: WeightMaterializerRegistry,
    catalog: CapabilityCatalog,
}

impl CpuVNextComposition {
    pub fn for_host(device_id: DeviceId) -> Result<Self, CpuRuntimeError> {
        Self::create(
            device_id,
            super::super::vnext_runtime::host_memory_capacity()?,
        )
    }
    pub fn create(device_id: DeviceId, memory_budget_bytes: u64) -> Result<Self, CpuRuntimeError> {
        let contracts = CpuOperation::IMPLEMENTED
            .iter()
            .map(|operation| {
                operation
                    .contract()
                    .map(|contract| Box::new(contract) as Box<dyn OperationContract>)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let capabilities = contracts
            .iter()
            .flat_map(|contract| {
                contract
                    .descriptor()
                    .provider
                    .required_capabilities
                    .iter()
                    .cloned()
            })
            .collect();
        let runtime = Arc::new(CpuDeviceRuntime::new(DeviceDescriptor {
            id: device_id,
            class: DeviceClass::Host,
            ordinal: 0,
            total_memory_bytes: memory_budget_bytes,
            runtime_implementation_fingerprint: implementation_fingerprint(),
            capabilities,
            dynamic_storage_profiles: BTreeSet::from([
                DynamicStorageProfile::new(
                    DynamicStorageAllocator::LinearArena,
                    DynamicStorageView::Contiguous,
                )?,
                super::causal_attention::kv_storage_profile()?,
            ]),
        })?);
        let providers = CpuOperation::IMPLEMENTED
            .iter()
            .map(|&operation| {
                CpuOperationProvider::new(&runtime, operation).map(|provider| {
                    Box::new(provider) as Box<dyn OperationProvider<CpuDeviceRuntime>>
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let registry = OperationRuntimeRegistry::new(contracts, providers)?;
        let engine = EngineProviderDescriptor::new(
            ProviderId::new("provider.engine.cpu.vnext")?,
            ContractVersion::new(1, 0),
            implementation_fingerprint(),
            runtime.descriptor().id.clone(),
            runtime.descriptor().capabilities.clone(),
        )?;
        let materializers = WeightMaterializerRegistry::identity_only()?;
        let catalog = materializers.augment_catalog(
            registry.capability_catalog(runtime.descriptor().clone(), vec![engine])?,
        )?;
        Ok(Self {
            runtime,
            registry,
            materializers,
            catalog,
        })
    }

    pub fn runtime(&self) -> &Arc<CpuDeviceRuntime> {
        &self.runtime
    }
    pub fn registry(&self) -> &OperationRuntimeRegistry<CpuDeviceRuntime> {
        &self.registry
    }
    pub fn catalog(&self) -> &CapabilityCatalog {
        &self.catalog
    }

    pub fn into_parts(
        self,
    ) -> Result<
        (
            Arc<CpuDeviceRuntime>,
            OperationRuntimeRegistry<CpuDeviceRuntime>,
            WeightMaterializerRegistry,
            WeightMaterializerId,
            CapabilityCatalog,
        ),
        VNextError,
    > {
        Ok((
            self.runtime,
            self.registry,
            self.materializers,
            WeightMaterializerId::new(IDENTITY_WEIGHT_MATERIALIZER_ID)?,
            self.catalog,
        ))
    }
}

pub(super) fn implementation_fingerprint() -> String {
    let mut digest = Sha256::new();
    digest.update(std::env::consts::ARCH.as_bytes());
    digest.update([0]);
    digest.update(std::env::consts::OS.as_bytes());
    digest.update([0]);
    for source in [
        include_str!("composition.rs"),
        include_str!("provider.rs"),
        include_str!("lowering.rs"),
        include_str!("bindings.rs"),
        include_str!("weights.rs"),
        include_str!("launch.rs"),
        include_str!("matrix.rs"),
        include_str!("scalar.rs"),
        include_str!("elementwise.rs"),
        include_str!("gated_delta.rs"),
        include_str!("gated_delta_launch.rs"),
        include_str!("causal_attention.rs"),
        include_str!("causal_attention_launch.rs"),
        include_str!("../vnext_runtime.rs"),
        include_str!("../vnext_runtime/command.rs"),
        include_str!("../vnext_runtime/memory.rs"),
        include_str!("../vnext_runtime/host_memory.rs"),
        include_str!("../../../gguf_blocks/mod.rs"),
        include_str!("../../../gguf_blocks/iq3s_grid.rs"),
    ] {
        digest.update((source.len() as u64).to_le_bytes());
        digest.update(source.as_bytes());
    }
    format!("{:x}", digest.finalize())
}
