//! Real family, compilation, initialization and managed tensor execution.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use ferrum_execution::{PreparedTensorExecutor, TensorExecutorConfig};
use ferrum_kernels::backend::reference::{ReferenceDeviceRuntime, ReferenceVNextComposition};
use ferrum_models::qwen_image21::{
    QwenImage21Config, QwenImage21ImageProjectionFamily, IMAGE_PROJECTION_FAMILY_ID,
    IMAGE_PROJECTION_INPUT_ID, IMAGE_PROJECTION_OUTPUT_ID, IMAGE_PROJECTION_PROFILE_ID,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::oracle::ProjectionShape;

fn id<T>(value: impl Into<String>) -> Result<T>
where
    T: TryFrom<String, Error = VNextError>,
{
    Ok(T::try_from(value.into())?)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorPolicy {
    capacity_bytes: u64,
    storage_profiles: Vec<DynamicStorageProfile>,
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

pub struct ProjectionExecutor {
    executor: PreparedTensorExecutor<ReferenceDeviceRuntime>,
    resources: Arc<PlanRuntimeResources<ReferenceDeviceRuntime>>,
    composition: ReferenceVNextComposition,
    host: HostMemoryAccount,
    input_spec: ResolvedTensorSpec,
    output_spec: ResolvedTensorSpec,
    provenance: Value,
}

impl ProjectionExecutor {
    pub fn new(
        config: &QwenImage21Config,
        directory: &Path,
        shape: ProjectionShape,
    ) -> Result<Self> {
        if shape.inputs as u64 != config.in_channels()
            || shape.outputs as u64 != config.hidden_size()
        {
            bail!("tool shape does not match the checked family configuration");
        }
        let profile =
            NumericalProfileId::new(IMAGE_PROJECTION_PROFILE_ID).map_err(anyhow::Error::msg)?;
        let family = TypedFamilyRegistration::new(QwenImage21ImageProjectionFamily::new()?)
            .prepare_with_profile(&serde_json::to_value(config)?, &profile)?;
        let composition =
            ReferenceVNextComposition::create(id("device.reference.qwen_image21.projection")?)?;
        let descriptor = composition.runtime().descriptor();
        let policy = TensorPolicy {
            capacity_bytes: descriptor.total_memory_bytes,
            storage_profiles: descriptor
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        };
        let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id(IMAGE_PROJECTION_INPUT_ID)?,
            ProgramTensorSpec {
                dimensions: vec![shape.rows as u64, shape.inputs as u64],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        )]))?;
        options.require_weight_materializer(composition.weight_materializer_id().clone());
        if !options.retain_completion_value(id(IMAGE_PROJECTION_OUTPUT_ID)?) {
            bail!("projection output was already retained unexpectedly");
        }
        let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
            &family,
            composition.catalog(),
            &policy,
            &composition.registry().planning(),
            composition.weight_materializers(),
            &options,
        )?;
        let executable = compilation.executable();
        let plan = executable.execution_plan();
        let providers = composition.registry().bind_plan(executable)?;
        let source = SafetensorsArchive::open(directory)?;
        let resources = initialize_resources(
            Arc::clone(composition.runtime()),
            &family,
            plan,
            &source,
            shape.weight_bytes()? as u64,
        )?;
        let input_bytes = shape.input_bytes()?;
        let output_bytes = shape.output_bytes()?;
        let total_bytes = input_bytes
            .checked_add(output_bytes)
            .context("Host capacity overflows")?;
        let host = HostMemoryAccount::new(HostMemoryLimits::new(
            total_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: input_bytes,
                transfer_staging_bytes: 0,
                output_bytes,
                weight_backing_bytes: 0,
            },
        )?);
        let executor = PreparedTensorExecutor::new(
            executable.clone(),
            providers,
            Arc::clone(&resources),
            host.clone(),
            TensorExecutorConfig::default(),
        )
        .map_err(|error| anyhow::anyhow!("executor construction: {error}"))?;
        let tensor_spec = |width| {
            ResolvedTensorSpec::new(
                vec![shape.rows as u64, width],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )
        };
        let input_spec = tensor_spec(shape.inputs as u64)?;
        let output_spec = tensor_spec(shape.outputs as u64)?;
        let provenance = json!({
            "family": IMAGE_PROJECTION_FAMILY_ID,
            "profile": IMAGE_PROJECTION_PROFILE_ID,
            "operation": DENSE_LINEAR_BF16_OPERATION_ID,
            "operation_version": "1.0",
            "numerical_validation": "OperationDefined; independent Candle F64 error bounds, analytic BF16 bits",
            "plan_hash": plan.plan_hash(),
            "runtime_fingerprint": descriptor.runtime_implementation_fingerprint,
            "device_class": "Reference",
            "runtime_capacity_bytes": descriptor.total_memory_bytes,
            "managed_host_capacity_bytes": total_bytes,
            "managed_input_bytes": input_bytes,
            "managed_output_bytes": output_bytes,
            "initialized_weight_bytes": shape.weight_bytes()?,
            "work": "FixedTensor; no token or KV placeholder",
            "timing_scope": "no performance or throughput claim"
        });
        Ok(Self {
            executor,
            resources,
            composition,
            host,
            input_spec,
            output_spec,
            provenance,
        })
    }

    pub fn provenance(&self) -> &Value {
        &self.provenance
    }

    pub fn execute(&mut self, name: &str, input: &[u8]) -> Result<ManagedHostBytes> {
        if input.len() as u64 != self.input_spec.minimum_storage_bytes()? {
            bail!("projection input length differs from the compiled tensor port");
        }
        let mut managed = self
            .host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, input.len())?;
        managed.as_mut().copy_from_slice(input);
        let input = ManagedHostTensor::new(
            id(IMAGE_PROJECTION_INPUT_ID)?,
            self.input_spec.clone(),
            managed.freeze(),
        )?;
        let mut outputs = self
            .executor
            .execute(
                id(format!("run.qwen_image21.projection.{name}"))?,
                id(format!("request.qwen_image21.projection.{name}"))?,
                vec![input],
            )
            .map_err(|error| anyhow::anyhow!("projection execution: {error}"))?;
        if outputs.len() != 1 {
            bail!(
                "projection executor returned {} outputs instead of one",
                outputs.len()
            );
        }
        let output = outputs.pop().context("projection output is missing")?;
        if output.value_id().as_str() != IMAGE_PROJECTION_OUTPUT_ID
            || output.tensor() != &self.output_spec
        {
            bail!("projection output does not match its compiled value and shape");
        }
        // Clone only the immutable authority; this never copies tensor bytes.
        Ok(output.bytes().clone())
    }

    pub fn close(self) -> Result<Value> {
        let Self {
            executor,
            resources,
            composition,
            host,
            ..
        } = self;
        drop(executor);
        let released = match PlanRuntimeResources::close(resources) {
            Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => receipt.released_static_resources(),
            Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
                bail!("projection close retains {strong_count} plan roots")
            }
            Err(error) => bail!("projection close failed: {:?}", error.failure()),
        };
        let snapshot = composition.runtime().snapshot();
        let usage = host.usage()?;
        if snapshot.live_allocations != 0
            || usage.committed_bytes() != 0
            || usage.allocated_bytes() != 0
        {
            bail!("projection close left live runtime allocations or Host claims");
        }
        Ok(json!({
            "released_static_resources": released,
            "runtime_live_allocations": snapshot.live_allocations,
            "managed_host_committed_bytes": usage.committed_bytes(),
            "managed_host_allocated_bytes": usage.allocated_bytes(),
            "scope": "this executor's device allocations and injected Host account, not process RSS"
        }))
    }
}

fn initialize_resources(
    runtime: Arc<ReferenceDeviceRuntime>,
    family: &PreparedModelFamily,
    plan: &ExecutionPlan,
    source: &SafetensorsArchive,
    expected_bytes: u64,
) -> Result<Arc<PlanRuntimeResources<ReferenceDeviceRuntime>>> {
    let provisioned =
        plan.provision_static(runtime.clone(), id("request.projection.provision")?)?;
    let permit = match provisioned.into_provisioning() {
        StaticProvisioning::Required(permit) => permit,
        StaticProvisioning::NoStatic(_) => bail!("img_in requires real static weights"),
    };
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id("run.projection.provision")?,
        id("transaction.projection.provision")?,
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
        .initialize_static(
            family,
            plan,
            source,
            StaticInitializationPolicy::new(1 << 20, 8)?,
        )
        .map_err(|error| anyhow::anyhow!("static initialization failed: {error:?}"))?;
    if initialized.receipt().uploaded_bytes() != expected_bytes {
        bail!("static initialization upload does not match the complete projection weight");
    }
    initialized
        .into_plan_runtime()
        .map_err(|error| anyhow::anyhow!("runtime handoff failed: {}", error.failure().message()))
}
