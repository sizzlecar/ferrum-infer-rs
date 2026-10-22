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

/// Caller-owned source bytes plus their explicit tensor port. This descriptor
/// grants no runtime authority; bytes enter the executor only after Host charge.
pub struct ComponentInput<'a> {
    pub value_id: ProgramValueId,
    pub tensor: ResolvedTensorSpec,
    pub bytes: &'a [u8],
}

pub struct StoredComponentRuntime<'a, R: DeviceRuntime> {
    pub runtime: Arc<R>,
    pub registry: &'a OperationRuntimeRegistry<R>,
    pub catalog: &'a CapabilityCatalog,
    pub materializers: &'a WeightMaterializerRegistry,
    pub capacity_bytes: u64,
    pub staging_bytes: usize,
    pub provenance: serde_json::Value,
}

/// Shared managed execution for diagnostic Stored tensor components. Output
/// capacity comes from compiler-derived ports, never an assumed node count.
pub fn execute_stored_component<R: DeviceRuntime>(
    family: &PreparedModelFamily,
    source: &dyn WeightComponentSource,
    composition: StoredComponentRuntime<'_, R>,
    inputs: Vec<ComponentInput<'_>>,
    inspect: impl FnOnce(
        &[ferrum_execution::TensorExecutionOutput],
        &serde_json::Value,
    ) -> Result<serde_json::Value>,
) -> Result<(serde_json::Value, serde_json::Value)> {
    let specs = inputs
        .iter()
        .map(|input| {
            (
                input.value_id.clone(),
                ProgramTensorSpec {
                    dimensions: input.tensor.dimensions().to_vec(),
                    element_type: input.tensor.element_type(),
                    layout: input.tensor.layout().clone(),
                },
            )
        })
        .collect();
    let mut options = ProgramPlanCompileOptions::new(specs)?;
    options.require_weight_materializer(id(IDENTITY_WEIGHT_MATERIALIZER_ID)?);
    execute_stored_component_with_options(family, source, composition, inputs, options, inspect)
}

/// Uses the model adapter's complete compile options, including shape-dependent
/// transport outputs. Supplied input tensors must exactly match those options;
/// an intermediate spec never grants permission to change an input ABI.
pub fn execute_stored_component_with_options<R: DeviceRuntime>(
    family: &PreparedModelFamily,
    source: &dyn WeightComponentSource,
    composition: StoredComponentRuntime<'_, R>,
    inputs: Vec<ComponentInput<'_>>,
    mut options: ProgramPlanCompileOptions,
    inspect: impl FnOnce(
        &[ferrum_execution::TensorExecutionOutput],
        &serde_json::Value,
    ) -> Result<serde_json::Value>,
) -> Result<(serde_json::Value, serde_json::Value)> {
    use ferrum_execution::{PreparedTensorExecutor, TensorExecutorConfig};
    use serde_json::json;
    let mut specs = std::collections::BTreeMap::new();
    let mut input_bytes = 0_usize;
    for input in &inputs {
        if usize::try_from(input.tensor.minimum_storage_bytes()?)? != input.bytes.len()
            || input.tensor.layout() != &ResolvedTensorLayout::Contiguous
        {
            bail!("component input bytes differ from exact contiguous tensor shape");
        }
        if specs
            .insert(
                input.value_id.clone(),
                ProgramTensorSpec {
                    dimensions: input.tensor.dimensions().to_vec(),
                    element_type: input.tensor.element_type(),
                    layout: input.tensor.layout().clone(),
                },
            )
            .is_some()
        {
            bail!("duplicate component input port");
        }
        input_bytes = input_bytes
            .checked_add(input.bytes.len())
            .context("component input bytes overflow")?;
    }
    if specs.len() != family.program().inputs().len()
        || family
            .program()
            .inputs()
            .iter()
            .any(|value| !specs.contains_key(value))
    {
        bail!("component inputs do not cover the exact prepared family input ports");
    }
    for (value_id, spec) in &specs {
        if options.tensor_specs().get(value_id) != Some(spec) {
            bail!("component input differs from explicit compile options: {value_id}");
        }
    }
    for output in family.program().outputs() {
        options.retain_completion_value(output.clone());
    }
    let policy = TensorPolicy {
        capacity_bytes: composition.capacity_bytes,
        storage_profiles: composition
            .runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    };
    let compiled = ProgramPlanCompiler::compile_with_weight_materializers(
        family,
        composition.catalog,
        &policy,
        &composition.registry.planning(),
        composition.materializers,
        &options,
    )?;
    let executable = compiled.executable();
    let plan = executable.execution_plan();
    let ports = plan
        .payload()
        .tensor_ports()
        .context("component plan has no tensor ports")?;
    let mut output_bytes = 0_usize;
    for output in ports.outputs() {
        output_bytes = output_bytes
            .checked_add(usize::try_from(output.tensor().minimum_storage_bytes()?)?)
            .context("component output bytes overflow")?;
    }
    let total = input_bytes
        .checked_add(output_bytes)
        .and_then(|n| n.checked_add(composition.staging_bytes))
        .context("component Host capacity overflow")?;
    let host = HostMemoryAccount::new(HostMemoryLimits::new(
        total,
        HostMemoryClassLimits {
            input_preprocess_bytes: input_bytes,
            output_bytes,
            transfer_staging_bytes: composition.staging_bytes,
            weight_backing_bytes: 0,
        },
    )?);
    let mut weight_bytes = 0_u64;
    for component in &family.stored_weight_schema()?.components {
        weight_bytes = weight_bytes
            .checked_add(component.physical_bytes()?)
            .context("component weight bytes overflow")?;
    }
    let resources = initialize_resources(
        composition.runtime.clone(),
        family,
        plan,
        source,
        weight_bytes,
        &host,
        composition.staging_bytes,
    )?;
    // All execution owners leave this scope before explicit plan retirement,
    // including errors during binding, Host input setup or inspection. Drop
    // remains a safety fallback for nonterminal work, not a cleanup receipt.
    let execution: Result<serde_json::Value> = (|| {
        let mut executor = PreparedTensorExecutor::new(
            executable.clone(),
            composition.registry.bind_plan(executable)?,
            resources.clone(),
            host.clone(),
            TensorExecutorConfig::default(),
        )
        .map_err(|error| anyhow::anyhow!("component executor: {error}"))?;
        let provenance = json!({"family":family.family_id(),"profile":family.numerical_profile().id,"family_fingerprint":family.fingerprint()?,"canonical_config":family.canonical_config(),"plan_hash":plan.plan_hash(),"runtime_fingerprint":composition.runtime.descriptor().runtime_implementation_fingerprint,"backend":composition.provenance,"initialized_weight_bytes":weight_bytes,"managed_host_bytes":total,"managed_input_bytes":input_bytes,"managed_output_bytes":output_bytes,"managed_staging_bytes":composition.staging_bytes,"nodes":plan.payload().nodes().iter().map(|node|json!({"node":node.id(),"operation":node.operation_id(),"provider":node.provider_resources().provider_id(),"implementation_fingerprint":node.provider_implementation_fingerprint()})).collect::<Vec<_>>()});
        let mut managed_inputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            let mut storage =
                host.allocate_zeroed(HostMemoryClass::InputPreprocess, input.bytes.len())?;
            storage.as_mut().copy_from_slice(input.bytes);
            managed_inputs.push(ManagedHostTensor::new(
                input.value_id,
                input.tensor,
                storage.freeze(),
            )?);
        }
        let outputs = executor
            .execute(
                id("run.component.verify")?,
                id("request.component.verify")?,
                managed_inputs,
            )
            .map_err(|error| anyhow::anyhow!("component execution: {error}"))?;
        inspect(&outputs, &provenance)
    })();
    let closed = match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => Ok(receipt.released_static_resources()),
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => Err(anyhow::anyhow!(
            "component close retains {strong_count} owners"
        )),
        Err(error) => Err(anyhow::anyhow!(
            "component close failed: {:?}",
            error.failure()
        )),
    };
    // Check Host even when close fails. Retained nonterminal owners are
    // reported without asserting that releasing them would be safe.
    let host_cleanup: Result<_> = (|| {
        let usage = host.usage()?;
        if usage.committed_bytes() != 0 || usage.allocated_bytes() != 0 {
            bail!(
                "component Host owners remain after close: committed={}, allocated={}",
                usage.committed_bytes(),
                usage.allocated_bytes()
            );
        }
        Ok(usage)
    })();
    let cleanup = match (closed, host_cleanup) {
        (Ok(released), Ok(usage)) => Ok(
            json!({"released_static_resources":released,"managed_host_committed_bytes":usage.committed_bytes(),"managed_host_allocated_bytes":usage.allocated_bytes(),"scope":"managed input/output/static staging and plan retirement; not total Host/RSS or CUDA opaque-memory qualification"}),
        ),
        (Err(error), Ok(_)) | (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(host_error)) => {
            Err(error.context(format!("Host cleanup also failed: {host_error:#}")))
        }
    };
    match (execution, cleanup) {
        (Ok(report), Ok(cleanup)) => Ok((report, cleanup)),
        (Err(error), Ok(_)) | (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(cleanup_error)) => {
            Err(error.context(format!("component cleanup also failed: {cleanup_error:#}")))
        }
    }
}
