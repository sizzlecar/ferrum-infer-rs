use super::{
    arguments::Arguments,
    backend::Backend,
    shared::{self, id, StoredComponentRuntime, TensorPolicy},
};
use anyhow::{bail, ensure, Context, Result};
use ferrum_execution::{PreparedTensorExecutor, TensorExecutorConfig};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_models::{
    qwen3_vl_text::{Qwen3VLTextConfig, Qwen3VLTextEncoderFamily as Family},
    qwen_image21::t2i_prompt::{
        QwenImage21T2iManagedInputs, QwenImage21T2iPromptBatch, QwenImage21T2iTextFeatures,
    },
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Debug, Serialize)]
pub struct HostBudget {
    pub inputs: usize,
    pub uncropped_output: usize,
    pub cropped_features: usize,
    pub staging: usize,
    pub total: usize,
}
impl HostBudget {
    pub fn new(
        config: &Qwen3VLTextConfig,
        prompts: &QwenImage21T2iPromptBatch,
        staging: usize,
        limit: usize,
    ) -> Result<Self> {
        ensure!(
            config.layout() == prompts.layout(),
            "prompt layout differs from text configuration"
        );
        let specs = config.input_tensor_specs()?;
        let mut inputs = 0_usize;
        for spec in specs.values() {
            let t = ResolvedTensorSpec::new(
                spec.dimensions.clone(),
                spec.element_type,
                spec.layout.clone(),
            )?;
            inputs = inputs
                .checked_add(usize::try_from(t.minimum_storage_bytes()?)?)
                .context("Host input budget overflows")?;
        }
        let h = config.architecture().hidden_size();
        let uncropped_output = usize::try_from(
            ResolvedTensorSpec::new(
                vec![config.layout().rows(), h],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?
            .minimum_storage_bytes()?,
        )?;
        let cropped_rows = u64::try_from(prompts.batch())?
            .checked_mul(u64::try_from(prompts.feature_sequence_length())?)
            .context("cropped rows overflow")?;
        let cropped_features = usize::try_from(
            ResolvedTensorSpec::new(
                vec![cropped_rows, h],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?
            .minimum_storage_bytes()?,
        )?;
        let total = [inputs, uncropped_output, cropped_features, staging]
            .into_iter()
            .try_fold(0_usize, usize::checked_add)
            .context("Host total overflows")?;
        ensure!(
            staging > 0 && total <= limit && total <= isize::MAX as usize,
            "required managed Host backing {total} exceeds --maximum-host-bytes {limit}"
        );
        Ok(Self {
            inputs,
            uncropped_output,
            cropped_features,
            staging,
            total,
        })
    }
    pub fn account(&self) -> Result<HostMemoryAccount> {
        Ok(HostMemoryAccount::new(HostMemoryLimits::new(
            self.total,
            HostMemoryClassLimits {
                input_preprocess_bytes: self
                    .inputs
                    .checked_add(self.cropped_features)
                    .context("Host input/crop budget overflows")?,
                output_bytes: self.uncropped_output,
                transfer_staging_bytes: self.staging,
                weight_backing_bytes: 0,
            },
        )?))
    }
}
pub struct ExecutionResult {
    pub features: QwenImage21T2iTextFeatures,
    pub provenance: Value,
    pub retirement: Value,
}

pub fn run(
    config: &Qwen3VLTextConfig,
    prompts: &QwenImage21T2iPromptBatch,
    inputs: QwenImage21T2iManagedInputs,
    host: &HostMemoryAccount,
    budget: &HostBudget,
    source: &SafetensorsArchive,
    args: &Arguments,
    weight_bytes: u64,
) -> Result<ExecutionResult> {
    ensure!(
        inputs.config() == config,
        "prepared managed inputs differ from the canonical text configuration"
    );
    let profile = match &args.backend {
        Backend::Reference => Family::REFERENCE_PROFILE_ID,
        Backend::Cuda(_) => Family::CUDA_PROFILE_ID,
    };
    let family = TypedFamilyRegistration::new(Family::new()?).prepare_with_profile(
        &serde_json::to_value(config)?,
        &NumericalProfileId::new(profile).map_err(anyhow::Error::msg)?,
    )?;
    match &args.backend {
        Backend::Reference => {
            let c = ReferenceVNextComposition::create_with_capacity(
                id("device.reference.qwen3vl.text")?,
                args.reference_memory_bytes,
            )?;
            execute(
                &family,
                config,
                prompts,
                inputs,
                host,
                budget,
                source,
                weight_bytes,
                StoredComponentRuntime {
                    runtime: c.runtime().clone(),
                    registry: c.registry(),
                    catalog: c.catalog(),
                    materializers: c.weight_materializers(),
                    capacity_bytes: args.reference_memory_bytes,
                    staging_bytes: budget.staging,
                    provenance: json!({"backend":"Reference","capacity":args.reference_memory_bytes,"qualification":"local arithmetic profiles; no whole-text-tower numerical or Torch parity qualification"}),
                },
            )
        }
        Backend::Cuda(options) => {
            #[cfg(not(feature = "cuda"))]
            {
                let _ = options;
                bail!("CUDA requires --features cuda")
            }
            #[cfg(feature = "cuda")]
            {
                use ferrum_kernels::backend::cuda::{
                    vnext_ops::*,
                    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
                };
                use CudaTensorOperation as Op;
                let runtime = std::sync::Arc::new(CudaDeviceRuntime::new_tensor_eager(
                    CudaTensorRuntimeConfig {
                        device: cuda_tensor_runtime_config(
                            options.device,
                            id(format!("device.cuda.{}", options.device))?,
                        )?,
                        usable_memory_bytes: options.usable_memory_bytes,
                        bootstrap_memory_bytes: options.bootstrap_memory_bytes,
                        stream_memory_bytes: options.stream_memory_bytes,
                        maximum_streams: options.maximum_streams,
                        maximum_in_flight_per_stream: options.maximum_in_flight_per_stream,
                        maximum_commands_per_submission: options.maximum_commands_per_submission,
                        opaque_memory_profile: options.opaque_memory_profile.clone(),
                    },
                )?);
                let mut providers = cuda_vnext_tensor_providers(
                    &runtime,
                    &[
                        Op::HeadRmsNorm,
                        Op::HalfSplitRope,
                        Op::StructuredGqaAttention,
                        Op::PlainResidualAdd,
                        Op::SiluMul,
                    ],
                )?;
                providers.push(cuda_vnext_dense_bf16_provider(&runtime)?);
                let registry = OperationRuntimeRegistry::new(
                    vec![
                        Box::new(dense_linear_bf16_contract()?),
                        Box::new(head_rms_norm_bf16_contract()?),
                        Box::new(half_split_rope_bf16_contract()?),
                        Box::new(structured_gqa_attention_bf16_contract()?),
                        Box::new(plain_residual_add_bf16_contract()?),
                        Box::new(silu_mul_bf16_contract()?),
                    ],
                    providers,
                )?;
                let materializers = WeightMaterializerRegistry::identity_only()?;
                let descriptor = runtime.descriptor();
                let engine = EngineProviderDescriptor::new(
                    id("provider.engine.cuda.tensor.qwen3vl_text")?,
                    ContractVersion::new(1, 0),
                    descriptor.runtime_implementation_fingerprint.clone(),
                    descriptor.id.clone(),
                    descriptor.capabilities.clone(),
                )?;
                let catalog = materializers.augment_catalog(
                    registry.capability_catalog(descriptor.clone(), vec![engine])?,
                )?;
                let provenance = json!({"backend":"CUDA","allowances":options,"cublas_version":cuda_dense_bf16_cublas_version()?,
                    "dense_profile":CUDA_DENSE_BF16_NUMERICAL_PROFILE_ID,"qualification":"execution smoke only; no Torch/vLLM whole-text-tower parity"});
                execute(
                    &family,
                    config,
                    prompts,
                    inputs,
                    host,
                    budget,
                    source,
                    weight_bytes,
                    StoredComponentRuntime {
                        runtime,
                        registry: &registry,
                        catalog: &catalog,
                        materializers: &materializers,
                        capacity_bytes: options.usable_memory_bytes,
                        staging_bytes: budget.staging,
                        provenance,
                    },
                )
            }
        }
    }
}

// This is the existing TensorPolicy/compiler/initializer/executor composition,
// specialized only to retaining one Host account through the post-TE crop.
fn execute<R: DeviceRuntime>(
    family: &PreparedModelFamily,
    config: &Qwen3VLTextConfig,
    prompts: &QwenImage21T2iPromptBatch,
    inputs: QwenImage21T2iManagedInputs,
    host: &HostMemoryAccount,
    budget: &HostBudget,
    source: &SafetensorsArchive,
    weight_bytes: u64,
    composition: StoredComponentRuntime<'_, R>,
) -> Result<ExecutionResult> {
    ensure!(
        inputs.tensors().iter().all(|t| host.owns(t.bytes())),
        "managed TE inputs belong to a different Host account"
    );
    let mut options = config.compile_options()?;
    options.retain_completion_value(id(Family::OUTPUT_ID)?);
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
        .context("text plan has no tensor ports")?;
    ensure!(
        ports.outputs().len() == 1
            && ports.outputs()[0].value_id().as_str() == Family::OUTPUT_ID
            && usize::try_from(ports.outputs()[0].tensor().minimum_storage_bytes()?)?
                == budget.uncropped_output,
        "compiled text output differs from Host budget"
    );
    let resources = shared::initialize_resources(
        composition.runtime.clone(),
        family,
        plan,
        source,
        weight_bytes,
        host,
        budget.staging,
    )?;
    let execution: Result<_> = (|| {
        let mut executor = PreparedTensorExecutor::new(
            executable.clone(),
            composition.registry.bind_plan(executable)?,
            resources.clone(),
            host.clone(),
            TensorExecutorConfig::default(),
        )
        .map_err(|e| anyhow::anyhow!("text executor: {e}"))?;
        let (_, inputs) = inputs.into_parts();
        let outputs = executor
            .execute(
                id("run.qwen3vl.text")?,
                id("request.qwen3vl.text")?,
                inputs.into_iter().collect(),
            )
            .map_err(|e| anyhow::anyhow!("text execution: {e}"))?;
        let [hidden] = outputs.as_slice() else {
            bail!("text execution returned an unexpected output set")
        };
        let features = prompts.crop_hidden(host, config.architecture(), hidden)?;
        ensure!(
            features.tensor().bytes().len() == budget.cropped_features,
            "cropped text size differs from admitted budget"
        );
        let provenance = json!({"family":family.family_id(),"profile":family.numerical_profile().id,"family_fingerprint":family.fingerprint()?,"canonical_config":family.canonical_config(),
            "plan_hash":plan.plan_hash(),"runtime_fingerprint":composition.runtime.descriptor().runtime_implementation_fingerprint,
            "backend":composition.provenance,"initialized_weight_bytes":weight_bytes,"managed_host_budget":budget,
            "nodes":plan.payload().nodes().iter().map(|node|json!({"node":node.id(),"operation":node.operation_id(),"provider":node.provider_resources().provider_id(),"implementation_fingerprint":node.provider_implementation_fingerprint()})).collect::<Vec<_>>()});
        Ok((features, provenance))
    })();
    let retired = match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => Ok(receipt.released_static_resources()),
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            Err(anyhow::anyhow!("text plan retains {strong_count} owners"))
        }
        Err(error) => Err(anyhow::anyhow!(
            "text plan retirement failed: {:?}",
            error.failure()
        )),
    };
    match (execution, retired) {
        (Ok((features, provenance)), Ok(released)) => {
            let usage = host.usage()?;
            ensure!(
                usage.committed_bytes() == budget.cropped_features
                    && usage.allocated_bytes() == budget.cropped_features,
                "after text plan close only cropped features may retain Host backing"
            );
            Ok(ExecutionResult {
                features,
                provenance,
                retirement: json!({"released_static_resources":released,"managed_host_committed_after_close":usage.committed_bytes(),
                "managed_host_allocated_after_close":usage.allocated_bytes(),"remaining_owner":"cropped text feature artifact"}),
            })
        }
        (Err(e), Ok(_)) | (Ok(_), Err(e)) => Err(e),
        (Err(e), Err(retirement)) => {
            Err(e.context(format!("retirement also failed: {retirement:#}")))
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
