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
use serde_json::{json, Value};

use super::arguments::Backend;
use super::oracle::ProjectionShape;
#[path = "../support/tensor_component.rs"]
mod shared;
use shared::{initialize_resources, TensorPolicy};

#[cfg(feature = "cuda")]
#[path = "cuda.rs"]
mod cuda;

fn id<T>(value: impl Into<String>) -> Result<T>
where
    T: TryFrom<String, Error = VNextError>,
{
    Ok(T::try_from(value.into())?)
}

pub enum ProjectionExecutor {
    Reference(ManagedProjection<ReferenceDeviceRuntime>),
    #[cfg(feature = "cuda")]
    Cuda(ManagedProjection<ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime>),
}

impl ProjectionExecutor {
    pub fn new(
        config: &QwenImage21Config,
        directory: &Path,
        shape: ProjectionShape,
        backend: &Backend,
    ) -> Result<Self> {
        match backend {
            Backend::Reference => {
                let composition = ReferenceVNextComposition::create(id(
                    "device.reference.qwen_image21.projection",
                )?)?;
                let settings = ExecutionSettings {
                    profile: IMAGE_PROJECTION_PROFILE_ID,
                    backend: "Reference",
                    capacity_bytes: composition.runtime().descriptor().total_memory_bytes,
                    staging_bytes: 65_536,
                    backend_configuration: json!({"backend": "reference"}),
                    live_allocations: Some(|runtime: &ReferenceDeviceRuntime| {
                        runtime.snapshot().live_allocations
                    }),
                };
                Ok(Self::Reference(ManagedProjection::new(
                    config,
                    directory,
                    shape,
                    Arc::clone(composition.runtime()),
                    composition.registry(),
                    composition.catalog(),
                    composition.weight_materializers(),
                    settings,
                )?))
            }
            Backend::Cuda(options) => {
                #[cfg(feature = "cuda")]
                {
                    cuda::create(config, directory, shape, options).map(Self::Cuda)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = options;
                    bail!("CUDA backend requires --features cuda")
                }
            }
        }
    }

    pub fn provenance(&self) -> &Value {
        match self {
            Self::Reference(executor) => executor.provenance(),
            #[cfg(feature = "cuda")]
            Self::Cuda(executor) => executor.provenance(),
        }
    }

    pub fn execute(&mut self, name: &str, input: &[u8]) -> Result<ManagedHostBytes> {
        match self {
            Self::Reference(executor) => executor.execute(name, input),
            #[cfg(feature = "cuda")]
            Self::Cuda(executor) => executor.execute(name, input),
        }
    }

    pub fn close(self) -> Result<Value> {
        match self {
            Self::Reference(executor) => executor.close(),
            #[cfg(feature = "cuda")]
            Self::Cuda(executor) => executor.close(),
        }
    }
}

struct ExecutionSettings<R> {
    profile: &'static str,
    backend: &'static str,
    capacity_bytes: u64,
    staging_bytes: usize,
    backend_configuration: Value,
    live_allocations: Option<fn(&R) -> u64>,
}

pub struct ManagedProjection<R: DeviceRuntime> {
    executor: PreparedTensorExecutor<R>,
    resources: Arc<PlanRuntimeResources<R>>,
    runtime: Arc<R>,
    live_allocations: Option<fn(&R) -> u64>,
    host: HostMemoryAccount,
    input_spec: ResolvedTensorSpec,
    output_spec: ResolvedTensorSpec,
    provenance: Value,
}

impl<R: DeviceRuntime> ManagedProjection<R> {
    fn new(
        config: &QwenImage21Config,
        directory: &Path,
        shape: ProjectionShape,
        runtime: Arc<R>,
        registry: &OperationRuntimeRegistry<R>,
        catalog: &CapabilityCatalog,
        weight_materializers: &WeightMaterializerRegistry,
        settings: ExecutionSettings<R>,
    ) -> Result<Self> {
        if shape.inputs as u64 != config.in_channels()
            || shape.outputs as u64 != config.hidden_size()
        {
            bail!("tool shape does not match the checked family configuration");
        }
        let profile = NumericalProfileId::new(settings.profile).map_err(anyhow::Error::msg)?;
        let family = TypedFamilyRegistration::new(QwenImage21ImageProjectionFamily::new()?)
            .prepare_with_profile(&serde_json::to_value(config)?, &profile)?;
        let descriptor = runtime.descriptor();
        let policy = TensorPolicy {
            capacity_bytes: settings.capacity_bytes,
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
        options.require_weight_materializer(id(IDENTITY_WEIGHT_MATERIALIZER_ID)?);
        if !options.retain_completion_value(id(IMAGE_PROJECTION_OUTPUT_ID)?) {
            bail!("projection output was already retained unexpectedly");
        }
        let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
            &family,
            catalog,
            &policy,
            &registry.planning(),
            weight_materializers,
            &options,
        )?;
        let executable = compilation.executable();
        let plan = executable.execution_plan();
        let providers = registry.bind_plan(executable)?;
        let source = SafetensorsArchive::open(directory)?;
        let input_bytes = shape.input_bytes()?;
        let output_bytes = shape.output_bytes()?;
        let total_bytes = input_bytes
            .checked_add(output_bytes)
            .and_then(|bytes| bytes.checked_add(settings.staging_bytes))
            .context("Host capacity overflows")?;
        let host = HostMemoryAccount::new(HostMemoryLimits::new(
            total_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: input_bytes,
                transfer_staging_bytes: settings.staging_bytes,
                output_bytes,
                weight_backing_bytes: 0,
            },
        )?);
        let resources = initialize_resources(
            Arc::clone(&runtime),
            &family,
            plan,
            &source,
            shape.weight_bytes()? as u64,
            &host,
            settings.staging_bytes,
        )?;
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
            "profile": settings.profile,
            "operation": DENSE_LINEAR_BF16_OPERATION_ID,
            "operation_version": "1.0",
            "numerical_validation": "OperationDefined; independent Candle F64 error bounds, analytic BF16 bits",
            "plan_hash": plan.plan_hash(),
            "runtime_fingerprint": descriptor.runtime_implementation_fingerprint,
            "providers": plan.payload().nodes().iter().map(|node| json!({
                "provider_id": node.provider_resources().provider_id(),
                "implementation_fingerprint": node.provider_implementation_fingerprint(),
            })).collect::<Vec<_>>(),
            "device_class": settings.backend,
            "backend_configuration": settings.backend_configuration,
            "runtime_capacity_bytes": settings.capacity_bytes,
            "managed_host_capacity_bytes": total_bytes,
            "managed_input_bytes": input_bytes,
            "managed_output_bytes": output_bytes,
            "managed_transfer_staging_bytes": settings.staging_bytes,
            "initialized_weight_bytes": shape.weight_bytes()?,
            "work": "FixedTensor; no token or KV placeholder",
            "timing_scope": "no performance or throughput claim"
        });
        Ok(Self {
            executor,
            resources,
            runtime,
            live_allocations: settings.live_allocations,
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
            runtime,
            live_allocations,
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
        let live_allocations = live_allocations.map(|snapshot| snapshot(&runtime));
        let usage = host.usage()?;
        if live_allocations.is_some_and(|count| count != 0)
            || usage.committed_bytes() != 0
            || usage.allocated_bytes() != 0
        {
            bail!("projection close left live runtime allocations or Host claims");
        }
        Ok(json!({
            "released_static_resources": released,
            "runtime_live_allocations": live_allocations,
            "managed_host_committed_bytes": usage.committed_bytes(),
            "managed_host_allocated_bytes": usage.allocated_bytes(),
            "scope": "plan close receipt and injected Host account; CUDA allocation count is unavailable (null), not inferred zero; no process RSS or opaque-memory qualification"
        }))
    }
}
