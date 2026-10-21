//! Shared cold setup for standalone fixed-tensor verification examples.
use anyhow::{bail, Context, Result};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub fn id<T>(value: impl Into<String>) -> Result<T>
where
    T: TryFrom<String, Error = VNextError>,
{
    Ok(T::try_from(value.into())?)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorPolicy {
    pub capacity_bytes: u64,
    pub storage_profiles: Vec<DynamicStorageProfile>,
}

impl RuntimePolicy for TensorPolicy {
    fn version(&self) -> ContractVersion {
        ContractVersion::new(1, 0)
    }
    fn memory_capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }
    fn memory_reserve_bytes(&self) -> u64 {
        0
    }
    fn maximum_active_sequences(&self) -> u32 {
        1
    }
    fn work_limits(&self) -> RuntimeWorkLimits {
        RuntimeWorkLimits::FixedTensor
    }
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }
    fn execution_determinism_requirement(&self) -> ExecutionDeterminismRequirement {
        ExecutionDeterminismRequirement::BitwiseSameRuntime
    }
    fn dynamic_storage_profile_order(&self) -> &[DynamicStorageProfile] {
        &self.storage_profiles
    }
    fn reusable_execution_policy(&self) -> Option<&ReusableExecutionPolicy> {
        None
    }
    fn validate(&self) -> std::result::Result<(), VNextError> {
        if self.capacity_bytes == 0 || self.storage_profiles.is_empty() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "projection requires positive capacity and a dynamic storage profile"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

pub fn initialize_resources<R: DeviceRuntime>(
    runtime: Arc<R>,
    family: &PreparedModelFamily,
    plan: &ExecutionPlan,
    source: &dyn WeightComponentSource,
    expected_bytes: u64,
    host: &HostMemoryAccount,
    staging_bytes: usize,
) -> Result<Arc<PlanRuntimeResources<R>>> {
    let provisioned = plan.provision_static(runtime.clone(), id("request.component.provision")?)?;
    let permit = match provisioned.into_provisioning() {
        StaticProvisioning::Required(permit) => permit,
        StaticProvisioning::NoStatic(_) => bail!("component requires real static weights"),
    };
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id("run.component.provision")?,
        id("transaction.component.provision")?,
    );
    let reserved =
        ResourceTransaction::begin(RuntimeResourceDriver::new(runtime)?, identity, permit)?
            .reserve()
            .map_err(|error| anyhow::anyhow!("static reservation failed: {:?}", error.failure()))?;
    let committed = match reserved.commit() {
        Ok(committed) => committed,
        Err(ResourceCommitTransitionError::Recoverable(error)) => {
            bail!("static commit failed: {:?}", error.failure())
        }
        Err(ResourceCommitTransitionError::Poisoned(error)) => {
            bail!("static commit indeterminate: {:?}", error.failure())
        }
    };
    let pool_ids: Vec<_> = committed
        .maintenance_controller()
        .pool_ids()
        .cloned()
        .collect();
    for pool in pool_ids {
        committed.maintenance_controller().initialize_pool(&pool)?;
    }
    let initialized = committed
        .initialize_static_managed(
            family,
            plan,
            source,
            host,
            StaticInitializationPolicy::new(u64::try_from(staging_bytes)?, 64)?,
        )
        .map_err(|error| anyhow::anyhow!("static initialization failed: {error:?}"))?;
    if initialized.receipt().uploaded_bytes() != expected_bytes {
        bail!("static initialization upload does not match the complete component weight");
    }
    initialized
        .into_plan_runtime()
        .map_err(|error| anyhow::anyhow!("runtime handoff failed: {}", error.failure().message()))
}
