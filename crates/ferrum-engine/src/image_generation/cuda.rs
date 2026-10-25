//! One-context CUDA composition for the experimental cold image pipeline.
//!
//! Prepare `QwenImage21Assets` before calling this module. Construction validates
//! caller budgets and architecture-derived command counts before device work,
//! then creates one runtime, one Host account and one native provider bundle.
//! cuDNN retains its separately admitted opaque owner inside that same runtime.
//! Opaque allowances are declarations, not measured memory qualification.

use super::{
    ImageCudaAttentionPolicy, ImageCudaPrecisionPolicy, ImageFp8HostCachePolicy,
    ImagePipelineConfig, ImageStreamLimit, ImageStreamPlan, ImageStreamRequirements,
    QwenImage21Assets, QwenImage21Pipeline, QwenImage21Profiles,
};
use anyhow::{ensure, Context, Result};
use ferrum_diffusion::Bf16EulerProfile;
use ferrum_kernels::backend::cuda::{
    vnext_ops::*,
    vnext_runtime::{CudaDeviceCommand, CudaDeviceRuntime, CudaTensorRuntimeConfig},
    CudaCudnnConfig, CudaMemoryUtilization, CudaTensorMemoryObservation,
    CudaTensorRuntimeAutoConfig, CudnnSdpaPlanPolicy,
};
use ferrum_models::normal_noise::{NormalNoiseConfig, NormalNoiseFamily};
use ferrum_models::qwen3_vl_text::{
    Qwen3VLTextConfig, Qwen3VLTextEncoderFamily, Qwen3VLTextImageSpan, Qwen3VLTextKeyRun,
    Qwen3VLTextLayout,
};
use ferrum_models::qwen3_vl_vision::{
    Qwen3VLVisionArchitecture, Qwen3VLVisionConfig, Qwen3VLVisionFamily, Qwen3VLVisionGrid,
};
use ferrum_models::qwen_image21::{
    QwenImage21T2iLayout, QwenImage21TransformerConfig, QwenImage21TransformerFamily,
    QwenImage21VaeDecoderFamily, QwenImage21VaeEncoderConfig, QwenImage21VaeEncoderFamily,
};
use ferrum_runtime::*;
use std::{num::NonZeroU64, sync::Arc};

/// One total device ceiling, including opaque claims. Automatic selection
/// observes the sole runtime context; the optional maximum only tightens it.
#[derive(Debug, Clone, Copy)]
pub enum ImageCudaMemoryPolicy {
    ExplicitTotal(NonZeroU64),
    AutoAvailable {
        utilization: CudaMemoryUtilization,
        maximum_bytes: Option<NonZeroU64>,
    },
}

/// Request limits without an independent device budget. The factory supplies
/// the actual runtime ceiling to every component plan after context creation.
#[derive(Debug, Clone)]
pub struct ImageCudaPipelineLimits {
    pub maximum_user_tokens: usize,
    pub maximum_width: u32,
    pub maximum_height: u32,
    pub maximum_batch_size: usize,
    pub decoder_batch_policy: super::ImageDecoderBatchPolicy,
    pub edit_prefix_policy: super::ImageEditPrefixPolicy,
    pub plan_cache: super::ImagePlanCachePolicy,
    pub residency: super::ImageResidencyPolicy,
    pub dynamic_residency: super::ImageDynamicResidencyPolicy,
    pub static_initialization: StaticInitializationPolicy,
    pub executor: ferrum_execution::TensorExecutorConfig,
}
impl ImageCudaPipelineLimits {
    pub fn validate(&self) -> Result<()> {
        self.plan_cache.validate()?;
        self.dynamic_residency.validate(self.residency)?;
        ensure!(
            self.maximum_width > 0 && self.maximum_height > 0,
            "image dimension limits must be positive"
        );
        ensure!(
            (1..=8192).contains(&self.maximum_user_tokens),
            "user-token limit must be 1..=8192"
        );
        ensure!(
            (1..=STRUCTURED_ATTENTION_BF16_MAX_BATCH).contains(&self.maximum_batch_size),
            "image maximum_batch_size exceeds supported attention capacity"
        );
        Ok(())
    }
    pub fn with_device_capacity(self, capacity: NonZeroU64) -> ImagePipelineConfig {
        ImagePipelineConfig {
            device_capacity_bytes: capacity.get(),
            maximum_user_tokens: self.maximum_user_tokens,
            maximum_width: self.maximum_width,
            maximum_height: self.maximum_height,
            maximum_batch_size: self.maximum_batch_size,
            decoder_batch_policy: self.decoder_batch_policy,
            edit_prefix_policy: self.edit_prefix_policy,
            plan_cache: self.plan_cache,
            residency: self.residency,
            dynamic_residency: self.dynamic_residency,
            static_initialization: self.static_initialization,
            executor: self.executor,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ImageCudaCompositionReceipt {
    pub memory_policy: &'static str,
    pub stream_plan: ImageStreamPlan,
    pub device_capacity_bytes: u64,
    pub observed_memory: Option<CudaTensorMemoryObservation>,
    /// Absolute path passed to the admitted native loader. The generated
    /// component plans retain the provider's library fingerprint separately.
    pub cudnn_library: std::path::PathBuf,
    pub cudnn_workspace_limit_bytes: u64,
    pub attention_policy: ImageCudaAttentionPolicy,
    pub precision_policy: ImageCudaPrecisionPolicy,
    pub fp8_host_cache: ImageFp8HostCachePolicy,
    /// This CUTLASS bridge cannot prove native unload, so its declared module
    /// claim remains retained until process exit; tensor payloads retire normally.
    pub fp8_opaque_memory_bytes: u64,
    /// Additional experimental plan-cache allowance, not a measured peak.
    pub attention_opaque_memory_bytes: u64,
    /// These allowances are declared, not measured native memory peaks.
    pub runtime_opaque_profile: String,
    pub cudnn_opaque_profile: String,
}

/// Explicit limits or a named product preset are resolved before device work.
#[derive(Debug, Clone)]
pub struct QwenImage21CudaConfig {
    pub ordinal: usize,
    pub memory: ImageCudaMemoryPolicy,
    pub bootstrap_memory_bytes: u64,
    pub stream_memory_bytes: u64,
    pub maximum_streams: ImageStreamLimit,
    pub maximum_in_flight_per_stream: usize,
    pub maximum_commands_per_submission: usize,
    pub opaque_memory_profile: String,
    pub cudnn: CudaCudnnConfig,
    pub attention_policy: ImageCudaAttentionPolicy,
    pub precision_policy: ImageCudaPrecisionPolicy,
    pub fp8_host_cache: ImageFp8HostCachePolicy,
    pub host: HostMemoryLimits,
    pub pipeline: ImageCudaPipelineLimits,
}

impl QwenImage21CudaConfig {
    /// Pure schedule planning, before context creation or any native allocation.
    pub fn stream_plan(&self) -> Result<ImageStreamPlan> {
        self.maximum_streams
            .resolve(ImageStreamRequirements::for_serial_pipeline(
                self.pipeline.static_initialization,
                self.pipeline.executor,
            ))
    }

    /// Pure configuration checks. No driver query, context, library load,
    /// account construction, source mapping, or payload allocation occurs.
    /// Request-specific shapes still pass through the pipeline's compilation
    /// and ordinary Host/device admission before their execution.
    pub fn validate(&self) -> Result<()> {
        self.pipeline.validate()?;
        self.fp8_host_cache
            .validate(self.precision_policy, self.host)?;
        let maximum_streams = self.stream_plan()?.resolved_streams();
        ensure!(
            self.pipeline.edit_prefix_policy != super::ImageEditPrefixPolicy::RequestCache
                || self.attention_policy == ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1,
            "CUDA request-cache editing requires the explicit cuDNN SDPA attention policy"
        );
        if self.precision_policy.uses_fp8() {
            ensure!(
                ferrum_kernels::native_ops::cutlass_fp8::compiled(),
                "FP8 requires cuda-cutlass-fp8 and its locked native artifact"
            );
            ensure!(
                self.host.class_bytes(HostMemoryClass::WeightBacking) > 0,
                "FP8 requires an explicit Host WeightBacking allowance"
            );
        }
        i32::try_from(self.ordinal).context("CUDA ordinal exceeds driver range")?;
        ensure!(
            self.bootstrap_memory_bytes > 0 && self.stream_memory_bytes > 0,
            "CUDA bootstrap and per-stream allowances must be positive"
        );
        validate_profile(&self.opaque_memory_profile, "CUDA runtime")?;
        validate_profile(&self.cudnn.opaque_memory_profile, "cuDNN")?;
        ensure!(
            self.cudnn.library_path.is_absolute(),
            "cuDNN library path must be absolute"
        );
        ensure!(
            self.cudnn.opaque_memory_bytes > 0
                && self.cudnn.workspace_limit_bytes <= isize::MAX as u64,
            "cuDNN needs positive opaque bytes and addressable workspace"
        );
        ensure!(
            self.maximum_in_flight_per_stream > 0
                && self.maximum_commands_per_submission > 0
                && self.maximum_commands_per_submission <= u32::MAX as usize,
            "CUDA stream/flight limits must be positive and commands fit u32"
        );
        maximum_streams
            .checked_mul(self.maximum_in_flight_per_stream)
            .and_then(|n| n.checked_mul(self.maximum_commands_per_submission))
            .and_then(|n| {
                n.checked_mul(std::mem::size_of::<DeviceCommandEntry<CudaDeviceCommand>>())
            })
            .filter(|&n| n <= isize::MAX as usize)
            .context("CUDA command-record capacity exceeds addressable memory")?;
        self.opaque_bytes()?;
        if let Some(capacity) = self.declared_maximum() {
            self.validate_capacity(capacity.get())?;
        }

        let staging = usize::try_from(self.pipeline.static_initialization.maximum_staging_bytes())
            .context("static staging exceeds Host address range")?;
        ensure!(
            staging <= isize::MAX as usize
                && staging <= self.host.class_bytes(HostMemoryClass::TransferStaging),
            "static initialization staging exceeds its Host class ceiling"
        );
        ensure!(
            self.host.class_bytes(HostMemoryClass::InputPreprocess) >= 16
                && self.host.class_bytes(HostMemoryClass::Output) > 0,
            "Host requires a seed input and a nonzero output class"
        );
        ensure!(
            self.pipeline
                .static_initialization
                .maximum_commands_per_batch()
                <= self.maximum_commands_per_submission,
            "static command batch exceeds CUDA submission limit"
        );
        Ok(())
    }

    /// Consume already checked metadata/source ownership. Validation and four
    /// small metadata-only family preparations precede the sole CUDA context.
    /// Request-dependent provider shape estimates remain the pipeline compiler's
    /// responsibility; this method does not promise every maximum-sized request
    /// fits the selected memory quotas.
    pub fn create_pipeline(
        self,
        assets: QwenImage21Assets,
    ) -> Result<QwenImage21Pipeline<CudaDeviceRuntime>> {
        self.create_pipeline_with_receipt(assets)
            .map(|(pipeline, _)| pipeline)
    }

    pub fn create_pipeline_with_receipt(
        self,
        assets: QwenImage21Assets,
    ) -> Result<(
        QwenImage21Pipeline<CudaDeviceRuntime>,
        ImageCudaCompositionReceipt,
    )> {
        self.validate()?;
        let stream_plan = self.stream_plan()?;
        let mut profiles =
            cuda_profiles_with_precision(self.attention_policy, self.precision_policy)?;
        self.precision_policy
            .validate_denoiser_source(!assets.transformer_encodings.is_empty())?;
        if !assets.transformer_encodings.is_empty() {
            profiles.denoiser = NumericalProfileId::new(match self.attention_policy {
                ImageCudaAttentionPolicy::Sequential => {
                    QwenImage21TransformerFamily::CUDA_NATIVE_BLOCK_PROFILE_ID
                }
                ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1 => {
                    QwenImage21TransformerFamily::CUDA_NATIVE_BLOCK_FUSED_PROFILE_ID
                }
            })
            .map_err(anyhow::Error::msg)?;
        }
        let editing = assets.editing.is_some();
        let mut contracts = contracts(self.attention_policy)?;
        // Exactly one Host ledger, also used by CPU weight conversion.
        let host = HostMemoryAccount::new(self.host);
        let (materializers, execution_weight_bytes) =
            self.prepare_assets(&assets, &profiles, &host)?;
        let device_id = DeviceId::new(format!("device.cuda.{}", self.ordinal))?;
        let device = if self.precision_policy.uses_fp8() {
            contracts.push(Box::new(dense_linear_fp8_bf16_contract()?));
            cuda_tensor_runtime_config_with_cutlass_fp8(self.ordinal, device_id)?
        } else {
            cuda_tensor_runtime_config(self.ordinal, device_id)?
        };
        let runtime = Arc::new(match self.memory {
            ImageCudaMemoryPolicy::ExplicitTotal(bytes) => {
                CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
                    device,
                    usable_memory_bytes: bytes.get(),
                    bootstrap_memory_bytes: self.bootstrap_memory_bytes,
                    stream_memory_bytes: self.stream_memory_bytes,
                    maximum_streams: stream_plan.resolved_streams(),
                    maximum_in_flight_per_stream: self.maximum_in_flight_per_stream,
                    maximum_commands_per_submission: self.maximum_commands_per_submission,
                    opaque_memory_profile: self.opaque_memory_profile.clone(),
                })?
            }
            ImageCudaMemoryPolicy::AutoAvailable {
                utilization,
                maximum_bytes,
            } => CudaDeviceRuntime::new_tensor_eager_auto(CudaTensorRuntimeAutoConfig {
                device,
                utilization,
                maximum_memory_bytes: maximum_bytes.map(NonZeroU64::get),
                bootstrap_memory_bytes: self.bootstrap_memory_bytes,
                stream_memory_bytes: self.stream_memory_bytes,
                maximum_streams: stream_plan.resolved_streams(),
                maximum_in_flight_per_stream: self.maximum_in_flight_per_stream,
                maximum_commands_per_submission: self.maximum_commands_per_submission,
                opaque_memory_profile: self.opaque_memory_profile.clone(),
            })?,
        });
        let observed_memory = runtime.tensor_memory_observation().copied();
        let capacity = match self.memory {
            ImageCudaMemoryPolicy::ExplicitTotal(bytes) => bytes,
            ImageCudaMemoryPolicy::AutoAvailable { .. } => NonZeroU64::new(
                observed_memory
                    .context("automatic runtime omitted its memory observation")?
                    .effective_ceiling_bytes(),
            )
            .context("automatic runtime returned zero capacity")?,
        };
        self.validate_capacity(capacity.get())?;
        self.validate_weight_capacity(&execution_weight_bytes, capacity.get())?;
        let receipt = ImageCudaCompositionReceipt {
            stream_plan,
            memory_policy: match self.memory {
                ImageCudaMemoryPolicy::ExplicitTotal(_) => "explicit_total",
                ImageCudaMemoryPolicy::AutoAvailable { .. } => "auto_available",
            },
            device_capacity_bytes: capacity.get(),
            observed_memory,
            cudnn_library: self.cudnn.library_path.clone(),
            cudnn_workspace_limit_bytes: self.cudnn.workspace_limit_bytes,
            attention_policy: self.attention_policy,
            precision_policy: self.precision_policy,
            fp8_host_cache: self.fp8_host_cache,
            fp8_opaque_memory_bytes: self.precision_policy.additional_opaque_bytes(),
            attention_opaque_memory_bytes: self.attention_policy.additional_opaque_bytes(),
            runtime_opaque_profile: self.opaque_memory_profile.clone(),
            cudnn_opaque_profile: self.cudnn.opaque_memory_profile.clone(),
        };
        // ALL includes Philox. Do not call the standalone Philox factory:
        // it would try to acquire a second live native-module permit.
        let mut providers = match self.attention_policy {
            ImageCudaAttentionPolicy::Sequential => cuda_vnext_tensor_providers_with_cudnn(
                &runtime,
                &self.cudnn,
                &CudaTensorOperation::ALL,
            )?,
            ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1 => {
                cuda_vnext_tensor_providers_with_cudnn_cross_attention(
                    &runtime,
                    &self.cudnn,
                    CudnnSdpaPlanPolicy {
                        maximum_plans: 64,
                        maximum_live_bindings: 2048,
                        opaque_memory_bytes: self.attention_policy.additional_opaque_bytes(),
                        workspace_limit_bytes: self.cudnn.workspace_limit_bytes,
                    },
                    &CudaTensorOperation::ALL,
                )?
            }
        };
        providers.push(cuda_vnext_dense_bf16_provider(&runtime)?);
        if self.precision_policy.uses_fp8() {
            providers.push(cuda_vnext_dense_linear_fp8_bf16_provider(
                &runtime,
                &CudaCutlassFp8Config {
                    opaque_memory_bytes: self.precision_policy.additional_opaque_bytes(),
                    maximum_plans: 64,
                    opaque_memory_profile: "experimental-cutlass-fp8-process-retained-v1".into(),
                },
            )?);
        }
        let registry = OperationRuntimeRegistry::new(contracts, providers)?;
        let descriptor = runtime.descriptor();
        let engine = EngineProviderDescriptor::new(
            ProviderId::new("provider.engine.cuda.image_generation")?,
            ContractVersion::new(1, 0),
            descriptor.runtime_implementation_fingerprint.clone(),
            descriptor.id.clone(),
            descriptor.capabilities.clone(),
        )?;
        let catalog = materializers
            .augment_catalog(registry.capability_catalog(descriptor.clone(), vec![engine])?)?;
        let native_units: &[ferrum_kernels::native_ops::CudaNativeBuildUnit] =
            if self.precision_policy.uses_fp8() {
                &[ferrum_kernels::native_ops::CudaNativeBuildUnit::CutlassFp8ScaledMm]
            } else {
                &[]
            };
        ferrum_kernels::native_ops::validate_compiled_native_operator_provider_catalog_for_units(
            &catalog.native_operator_provider_catalog(ferrum_types::NativeOperatorBackend::Cuda)?,
            ferrum_kernels::native_ops::compiled_native_operator_artifacts(),
            native_units,
        )
        .map_err(anyhow::Error::msg)?;
        let device_opaque_allowance = self.opaque_bytes()?;
        let pipeline = QwenImage21Pipeline::new(
            runtime,
            host,
            registry,
            catalog,
            materializers,
            assets,
            profiles,
            self.pipeline.with_device_capacity(capacity),
        )?
        .with_device_opaque_allowance(device_opaque_allowance)?;
        let pipeline = if editing {
            pipeline.with_editing_profiles(super::QwenImage21EditProfiles {
                vision: NumericalProfileId::new(
                    ferrum_models::qwen3_vl_vision::Qwen3VLVisionFamily::CUDA_PROFILE_ID,
                )
                .map_err(anyhow::Error::msg)?,
                encoder: NumericalProfileId::new(
                    ferrum_models::qwen_image21::QwenImage21VaeEncoderFamily::CUDA_PROFILE_ID,
                )
                .map_err(anyhow::Error::msg)?,
            })?
        } else {
            pipeline
        };
        Ok((pipeline, receipt))
    }

    fn declared_maximum(&self) -> Option<NonZeroU64> {
        match self.memory {
            ImageCudaMemoryPolicy::ExplicitTotal(bytes) => Some(bytes),
            ImageCudaMemoryPolicy::AutoAvailable { maximum_bytes, .. } => maximum_bytes,
        }
    }
    fn validate_capacity(&self, capacity: u64) -> Result<()> {
        let overhead = self.opaque_bytes()?;
        ensure!(
            overhead < capacity,
            "CUDA runtime/stream/cuDNN opaque allowances leave no storage capacity"
        );
        ensure!(
            self.cudnn.workspace_limit_bytes < capacity - overhead,
            "cuDNN workspace ceiling leaves no storage for tensors"
        );
        Ok(())
    }
    fn validate_weight_capacity(&self, execution_bytes: &[u64; 3], capacity: u64) -> Result<()> {
        let largest = execution_bytes.iter().copied().max().unwrap_or(0);
        let opaque = self.opaque_bytes()?;
        let storage = capacity
            .checked_sub(opaque)
            .context("opaque allowances exceed capacity")?;
        ensure!(largest < storage, "largest component execution weights ({largest} bytes) leave no admitted activation capacity under total ceiling {capacity} with declared opaque allowances {opaque}");
        Ok(())
    }

    fn opaque_bytes(&self) -> Result<u64> {
        self.stream_memory_bytes
            .checked_mul(u64::try_from(self.stream_plan()?.resolved_streams())?)
            .and_then(|n| n.checked_add(self.bootstrap_memory_bytes))
            .and_then(|n| n.checked_add(self.cudnn.opaque_memory_bytes))
            .and_then(|n| n.checked_add(self.attention_policy.additional_opaque_bytes()))
            .and_then(|n| n.checked_add(self.precision_policy.additional_opaque_bytes()))
            .context("CUDA opaque allowance total overflows")
    }

    fn prepare_assets(
        &self,
        assets: &QwenImage21Assets,
        profiles: &QwenImage21Profiles,
        host: &HostMemoryAccount,
    ) -> Result<(WeightMaterializerRegistry, [u64; 3])> {
        let minimum_pixels = assets
            .spatial_scale()?
            .checked_mul(2)
            .context("minimum even-axis output dimension overflows")?;
        ensure!(
            self.pipeline.maximum_width >= minimum_pixels
                && self.pipeline.maximum_height >= minimum_pixels,
            "configured image bounds admit no supported even latent grid"
        );

        // Node/input counts do not depend on request width or text padding for
        // these fixed-topology families. Build their smallest legal geometry,
        // inspect the real programs, and never materialize a tensor payload.
        let text = Qwen3VLTextConfig::new(
            assets.text.clone(),
            Qwen3VLTextLayout::new(1, vec![vec![Qwen3VLTextKeyRun::new(1, true)]])?,
        )?;
        let dit = QwenImage21TransformerConfig::new(
            assets.transformer.clone(),
            QwenImage21T2iLayout::new(1, 2, 2, vec![1])?,
        )?
        .with_weight_encodings(assets.transformer_encodings.clone())?;
        let decoder = assets.decoder(2, 2)?;
        let noise = NormalNoiseConfig::new(64, 2, 2)?;
        let mut families = vec![
            TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new()?)
                .prepare_with_profile(&serde_json::to_value(&text)?, &profiles.text_encoder)?,
            TypedFamilyRegistration::new(NormalNoiseFamily::new(profiles.noise.as_str())?)
                .prepare_with_profile(&serde_json::to_value(noise)?, &profiles.noise)?,
            TypedFamilyRegistration::new(QwenImage21TransformerFamily::new()?)
                .prepare_with_profile(&serde_json::to_value(&dit)?, &profiles.denoiser)?,
            TypedFamilyRegistration::new(QwenImage21VaeDecoderFamily::new()?)
                .prepare_with_profile(&serde_json::to_value(decoder)?, &profiles.decoder)?,
        ];
        // Keep the original four indices stable for materializer selection.
        // Editing adds two stages and different TE/DiT programs/input counts.
        if let Some(edit) = &assets.editing {
            families.extend(editing_command_families(
                &text,
                &dit,
                &edit.vision,
                &edit.encoder,
                profiles,
            )?);
            if self.pipeline.edit_prefix_policy == super::ImageEditPrefixPolicy::RequestCache {
                let (prefix, copies) = prefix_command_families(&dit, profiles)?;
                let prefill_commands = component_command_count(&prefix[0])?
                    .checked_add(copies)
                    .context("prefix D2D command count overflows")?;
                ensure!(prefill_commands <= self.maximum_commands_per_submission,
                    "prefix prefill requires {prefill_commands} compute/upload/state-copy commands, exceeding configured {}",
                    self.maximum_commands_per_submission);
                families.extend(prefix);
            }
        }
        validate_family_command_limits(&families, self.maximum_commands_per_submission)?;
        let mut execution_bytes = assets.receipt().required_weight_bytes;
        let mut selected: Vec<Box<dyn WeightMaterializer>> = Vec::new();
        let mut maximum_group_bytes = 0;
        for (index, stage, family, materializer_id) in [
            (
                0,
                super::ImageStage::TextEncoder,
                &families[0],
                super::precision::TEXT_MATERIALIZER,
            ),
            (
                1,
                super::ImageStage::Denoiser,
                &families[2],
                super::precision::DIT_MATERIALIZER,
            ),
        ] {
            if !self.precision_policy.stage_uses_fp8(stage) {
                continue;
            }
            let groups = match stage {
                super::ImageStage::TextEncoder => {
                    Qwen3VLTextEncoderFamily::fp8_weight_groups(&text)?
                }
                super::ImageStage::Denoiser => {
                    QwenImage21TransformerFamily::fp8_weight_groups(&dit)?
                }
                _ => unreachable!("only materialized stages are enumerated"),
            };
            let mut materializer = super::precision::materializer(materializer_id, host, groups)?;
            super::precision::validate_backing(family, &materializer, host)?;
            maximum_group_bytes =
                maximum_group_bytes.max(materializer.maximum_group_backing_bytes(family)?);
            // Schema bytes include scales, but are not a peak/activation estimate.
            execution_bytes[index] = materializer.execution_weight_bytes(family)?;
            if let Some(capacity) = self.fp8_host_cache.maximum_bytes(stage) {
                self.fp8_host_cache
                    .validate_stage(stage, materializer.converted_cache_bytes(family)?)?;
                materializer = materializer.with_host_cache(capacity)?;
            }
            selected.push(Box::new(materializer));
        }
        self.fp8_host_cache
            .validate_peak(self.host, maximum_group_bytes)?;
        let materializers = WeightMaterializerRegistry::new(selected)?;
        if let Some(capacity) = self.declared_maximum() {
            self.validate_weight_capacity(&execution_bytes, capacity.get())?;
        }
        Ok((materializers, execution_bytes))
    }
}

fn component_command_count(family: &PreparedModelFamily) -> Result<usize> {
    family
        .program()
        .blocks()
        .iter()
        .try_fold(family.program().inputs().len(), |n, block| {
            n.checked_add(block.nodes.len())
        })
        .context("component command count overflows")
}

fn prefix_command_families(
    dit: &QwenImage21TransformerConfig,
    profiles: &QwenImage21Profiles,
) -> Result<([PreparedModelFamily; 2], usize)> {
    use ferrum_models::qwen_image21::QwenImage21TransformerPhase;
    let config = QwenImage21TransformerConfig::new(
        dit.transformer().clone(),
        QwenImage21T2iLayout::new(3, 2, 2, vec![3])?,
    )?
    .with_weight_encodings(dit.weight_encodings().clone())?
    .with_reference_image(1, 1, 2, 2)?;
    let copies = config.prefix_cache_ports()?.len();
    let family = TypedFamilyRegistration::new(QwenImage21TransformerFamily::new()?);
    let prefill = config
        .clone()
        .with_execution_phase(QwenImage21TransformerPhase::PrefixPrefill)?;
    let decode = config.with_execution_phase(QwenImage21TransformerPhase::PrefixDecode)?;
    Ok((
        [
            family.prepare_with_profile(&serde_json::to_value(prefill)?, &profiles.denoiser)?,
            family.prepare_with_profile(&serde_json::to_value(decode)?, &profiles.denoiser)?,
        ],
        copies,
    ))
}

fn validate_family_command_limits(families: &[PreparedModelFamily], maximum: usize) -> Result<()> {
    for family in families {
        let commands = component_command_count(family)?;
        ensure!(commands <= maximum,
            "{} requires at least {commands} compute/upload commands, exceeding configured {maximum}",
            family.family_id());
    }
    Ok(())
}

/// Small metadata-only representatives of every additional editing topology.
/// Tensor contents and request dimensions cannot reduce these command counts.
fn editing_command_families(
    text: &Qwen3VLTextConfig,
    dit: &QwenImage21TransformerConfig,
    vision: &Qwen3VLVisionArchitecture,
    encoder: &QwenImage21VaeEncoderConfig,
    profiles: &QwenImage21Profiles,
) -> Result<[PreparedModelFamily; 4]> {
    let merge = vision.merge_size();
    let visual = Qwen3VLVisionConfig::new(
        vision.clone(),
        vec![Qwen3VLVisionGrid::new(1, merge, merge)?],
    )?;
    let text = text
        .clone()
        .with_image_spans(&[Qwen3VLTextImageSpan::new(0, merge, merge, merge)?])?;
    let dit = QwenImage21TransformerConfig::new(
        dit.transformer().clone(),
        QwenImage21T2iLayout::new(3, 2, 2, vec![3])?,
    )?
    .with_weight_encodings(dit.weight_encodings().clone())?
    .with_reference_image(1, 1, 2, 2)?;
    Ok([
        TypedFamilyRegistration::new(Qwen3VLVisionFamily::new()?).prepare_with_profile(
            &serde_json::to_value(visual)?,
            &NumericalProfileId::new(Qwen3VLVisionFamily::CUDA_PROFILE_ID)
                .map_err(anyhow::Error::msg)?,
        )?,
        TypedFamilyRegistration::new(QwenImage21VaeEncoderFamily::new()?).prepare_with_profile(
            &serde_json::to_value(encoder)?,
            &NumericalProfileId::new(QwenImage21VaeEncoderFamily::CUDA_PROFILE_ID)
                .map_err(anyhow::Error::msg)?,
        )?,
        TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new()?)
            .prepare_with_profile(&serde_json::to_value(text)?, &profiles.text_encoder)?,
        TypedFamilyRegistration::new(QwenImage21TransformerFamily::new()?)
            .prepare_with_profile(&serde_json::to_value(dit)?, &profiles.denoiser)?,
    ])
}

fn validate_profile(profile: &str, label: &str) -> Result<()> {
    ensure!(
        !profile.trim().is_empty()
            && profile.len() <= 256
            && profile.is_ascii()
            && !profile.bytes().any(|b| b.is_ascii_control()),
        "{label} requires an explicit bounded ASCII opaque-memory profile"
    );
    Ok(())
}

#[cfg(test)]
fn cuda_profiles(attention: ImageCudaAttentionPolicy) -> Result<QwenImage21Profiles> {
    cuda_profiles_with_precision(attention, ImageCudaPrecisionPolicy::Bf16)
}
fn cuda_profiles_with_precision(
    attention: ImageCudaAttentionPolicy,
    precision: ImageCudaPrecisionPolicy,
) -> Result<QwenImage21Profiles> {
    use super::{ImageDenoiserPrecision as Denoiser, ImageTextPrecision as Text};
    let id = |value: &str| NumericalProfileId::new(value).map_err(anyhow::Error::msg);
    Ok(QwenImage21Profiles {
        text_encoder: id(match precision.text_encoder() {
            Text::Bf16 => Qwen3VLTextEncoderFamily::CUDA_PROFILE_ID,
            Text::Fp8LinearExperimentalV1 => Qwen3VLTextEncoderFamily::CUDA_FP8_PROFILE_ID,
        })?,
        noise: id(CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_ID)?,
        denoiser: id(match (attention, precision.denoiser()) {
            (ImageCudaAttentionPolicy::Sequential, Denoiser::Bf16) => {
                QwenImage21TransformerFamily::CUDA_PROFILE_ID
            }
            (ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1, Denoiser::Bf16) => {
                QwenImage21TransformerFamily::CUDA_FUSED_PROFILE_ID
            }
            (ImageCudaAttentionPolicy::Sequential, Denoiser::Fp8LinearExperimentalV1) => {
                QwenImage21TransformerFamily::CUDA_FP8_PROFILE_ID
            }
            (
                ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1,
                Denoiser::Fp8LinearExperimentalV1,
            ) => QwenImage21TransformerFamily::CUDA_FP8_FUSED_PROFILE_ID,
            (ImageCudaAttentionPolicy::Sequential, Denoiser::NativeSource) => {
                QwenImage21TransformerFamily::CUDA_NATIVE_BLOCK_PROFILE_ID
            }
            (ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1, Denoiser::NativeSource) => {
                QwenImage21TransformerFamily::CUDA_NATIVE_BLOCK_FUSED_PROFILE_ID
            }
        })?,
        decoder: id(QwenImage21VaeDecoderFamily::CUDA_PROFILE_ID)?,
        euler: Bf16EulerProfile::CudaScalarBf16StagedRneV1,
    })
}

fn contracts(attention: ImageCudaAttentionPolicy) -> Result<Vec<Box<dyn OperationContract>>> {
    let mut contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(silu_bf16_contract()?),
        Box::new(silu_mul_bf16_contract()?),
        Box::new(tanh_bf16_contract()?),
        Box::new(modulation_scale_bf16_contract()?),
        Box::new(gated_residual_bf16_contract()?),
        Box::new(layer_norm_bf16_contract()?),
        Box::new(head_rms_norm_bf16_contract()?),
        Box::new(adjacent_rope_bf16_contract()?),
        Box::new(structured_attention_bf16_contract()?),
        Box::new(zero_centered_rms_norm_bf16_contract()?),
        Box::new(gelu_tanh_bf16_contract()?),
        Box::new(row_gather_bf16_contract()?),
        Box::new(batched_row_concat_bf16_contract()?),
        Box::new(plain_residual_add_bf16_contract()?),
        Box::new(half_split_rope_bf16_contract()?),
        Box::new(structured_gqa_attention_bf16_contract()?),
        Box::new(channel_l2_norm_bf16_contract()?),
        Box::new(silu_nchw_bf16_contract()?),
        Box::new(plain_residual_add_nchw_bf16_contract()?),
        Box::new(channel_l2_norm_image_bf16_contract()?),
        Box::new(spatial_attention_bf16_contract()?),
        Box::new(spatial_upsample_gather_bf16_contract()?),
        Box::new(spatial_group_mean_bf16_contract()?),
        Box::new(philox_normal_bf16_contract()?),
        Box::new(conv2d_bf16_contract()?),
        Box::new(dense_linear_bf16_contract()?),
        Box::new(dense_linear_block_bf16_contract()?),
    ];
    for kind in TensorPrimitiveBf16::ALL {
        contracts.push(Box::new(kind.contract()?));
    }
    if attention == ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1 {
        contracts.push(Box::new(structured_attention_fused_bf16_contract()?));
        contracts.push(Box::new(cross_attention_fused_bf16_contract()?));
    }
    Ok(contracts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_execution::TensorExecutorConfig;

    fn config() -> QwenImage21CudaConfig {
        QwenImage21CudaConfig {
            ordinal: 0,
            memory: ImageCudaMemoryPolicy::ExplicitTotal(NonZeroU64::new(1024 * 1024).unwrap()),
            bootstrap_memory_bytes: 4096,
            stream_memory_bytes: 1024,
            maximum_streams: ImageStreamLimit::Auto,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 1024,
            opaque_memory_profile: "experimental-unqualified-test-v1".into(),
            attention_policy: ImageCudaAttentionPolicy::Sequential,
            precision_policy: ImageCudaPrecisionPolicy::Bf16,
            fp8_host_cache: ImageFp8HostCachePolicy::Off,
            cudnn: CudaCudnnConfig {
                library_path: "/explicit/libcudnn.so.9".into(),
                dependency_libraries: Vec::new(),
                opaque_memory_bytes: 4096,
                workspace_limit_bytes: 0,
                opaque_memory_profile: "experimental-cudnn-test-v1".into(),
            },
            host: HostMemoryLimits::new(
                4096,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 2048,
                    output_bytes: 2048,
                    transfer_staging_bytes: 1024,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
            pipeline: ImageCudaPipelineLimits {
                maximum_user_tokens: 1024,
                maximum_width: 512,
                maximum_height: 512,
                maximum_batch_size: ImagePipelineConfig::DEFAULT_MAXIMUM_BATCH_SIZE,
                decoder_batch_policy: crate::image_generation::ImageDecoderBatchPolicy::default(),
                edit_prefix_policy: crate::image_generation::ImageEditPrefixPolicy::default(),
                plan_cache: crate::image_generation::ImagePlanCachePolicy::default(),
                residency: crate::image_generation::ImageResidencyPolicy::default(),
                dynamic_residency: crate::image_generation::ImageDynamicResidencyPolicy::default(),
                static_initialization: StaticInitializationPolicy::new(1024, 32).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        }
    }

    #[test]
    fn stream_schedule_resolution_drives_native_limit_and_opaque_allowance() {
        let mut c = config();
        let automatic = c.stream_plan().unwrap();
        assert_eq!(
            automatic.source(),
            super::super::ImageStreamLimitSource::Auto
        );
        assert_eq!(automatic.required_streams(), 2);
        assert_eq!(automatic.resolved_streams(), 2);
        let automatic_opaque = c.opaque_bytes().unwrap();
        c.maximum_streams = ImageStreamLimit::Explicit(4);
        c.validate().unwrap();
        let explicit = c.stream_plan().unwrap();
        assert_eq!(explicit.required_streams(), automatic.required_streams());
        assert_eq!(explicit.resolved_streams(), 4);
        assert_eq!(
            c.opaque_bytes().unwrap(),
            automatic_opaque + 2 * c.stream_memory_bytes
        );
    }

    #[test]
    fn numeric_and_host_budgets_validate_without_a_device_or_library_load() {
        config().validate().unwrap();
        let edits: &[fn(&mut QwenImage21CudaConfig)] = &[
            |c: &mut QwenImage21CudaConfig| c.maximum_streams = ImageStreamLimit::Explicit(0),
            |c: &mut QwenImage21CudaConfig| c.maximum_streams = ImageStreamLimit::Explicit(1),
            |c: &mut QwenImage21CudaConfig| c.maximum_in_flight_per_stream = 0,
            |c: &mut QwenImage21CudaConfig| c.maximum_commands_per_submission = 0,
            |c: &mut QwenImage21CudaConfig| c.maximum_commands_per_submission = 31,
            |c: &mut QwenImage21CudaConfig| c.bootstrap_memory_bytes = 0,
            |c: &mut QwenImage21CudaConfig| c.stream_memory_bytes = u64::MAX,
            |c: &mut QwenImage21CudaConfig| {
                c.memory = ImageCudaMemoryPolicy::ExplicitTotal(NonZeroU64::new(9216).unwrap())
            },
            |c: &mut QwenImage21CudaConfig| {
                c.pipeline.static_initialization =
                    StaticInitializationPolicy::new(1025, 32).unwrap()
            },
            |c: &mut QwenImage21CudaConfig| c.cudnn.opaque_memory_bytes = 0,
            |c: &mut QwenImage21CudaConfig| c.cudnn.workspace_limit_bytes = u64::MAX,
            |c: &mut QwenImage21CudaConfig| c.cudnn.library_path = "relative.so".into(),
            |c: &mut QwenImage21CudaConfig| c.opaque_memory_profile = " ".into(),
            |c: &mut QwenImage21CudaConfig| c.cudnn.opaque_memory_profile = "bad\nprofile".into(),
            |c: &mut QwenImage21CudaConfig| c.maximum_in_flight_per_stream = usize::MAX,
        ];
        for edit in edits {
            let mut c = config();
            edit(&mut c);
            assert!(c.validate().is_err());
        }
    }

    #[test]
    fn automatic_policy_has_no_placeholder_capacity_or_second_plan_budget() {
        let mut c = config();
        c.memory = ImageCudaMemoryPolicy::AutoAvailable {
            utilization: CudaMemoryUtilization::new(0.9).unwrap(),
            maximum_bytes: None,
        };
        c.validate().unwrap();
        assert!(c.declared_maximum().is_none());
        // The sole-context observation is injected here only to test the
        // pure configuration bridge. This is not a CUDA observation fixture.
        for observed in [16384, 32768] {
            let bytes = NonZeroU64::new(observed).unwrap();
            assert_eq!(
                c.pipeline
                    .clone()
                    .with_device_capacity(bytes)
                    .device_capacity_bytes,
                observed
            );
        }
        c.memory = ImageCudaMemoryPolicy::AutoAvailable {
            utilization: CudaMemoryUtilization::new(0.9).unwrap(),
            maximum_bytes: NonZeroU64::new(9216),
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn all_registered_contracts_are_unique_and_profiles_are_explicit_cuda() {
        let contracts = contracts(ImageCudaAttentionPolicy::Sequential).unwrap();
        let ids = contracts
            .iter()
            .map(|contract| contract.descriptor().id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(ids.len(), contracts.len());
        for required in [
            PHILOX_NORMAL_BF16_OPERATION_ID,
            DENSE_LINEAR_BF16_OPERATION_ID,
            STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID,
            CONV2D_BF16_OPERATION_ID,
        ] {
            assert!(ids.contains(&OperationId::new(required).unwrap()));
        }
        let profiles = cuda_profiles(ImageCudaAttentionPolicy::Sequential).unwrap();
        assert_eq!(
            profiles.text_encoder.as_str(),
            Qwen3VLTextEncoderFamily::CUDA_PROFILE_ID
        );
        assert_eq!(
            profiles.noise.as_str(),
            CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_ID
        );
        assert_eq!(
            profiles.denoiser.as_str(),
            QwenImage21TransformerFamily::CUDA_PROFILE_ID
        );
        assert_eq!(
            profiles.decoder.as_str(),
            QwenImage21VaeDecoderFamily::CUDA_PROFILE_ID
        );
        assert_eq!(profiles.euler, Bf16EulerProfile::CudaScalarBf16StagedRneV1);
    }

    #[test]
    fn explicit_fused_policy_changes_only_dit_profile_and_charges_native_cache() {
        let sequential = cuda_profiles(ImageCudaAttentionPolicy::Sequential).unwrap();
        let fused = cuda_profiles(ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1).unwrap();
        assert_eq!(sequential.text_encoder, fused.text_encoder);
        assert_eq!(sequential.decoder, fused.decoder);
        assert_eq!(sequential.noise, fused.noise);
        assert_eq!(sequential.euler, fused.euler);
        assert_ne!(sequential.denoiser, fused.denoiser);
        assert_eq!(
            fused.denoiser.as_str(),
            QwenImage21TransformerFamily::CUDA_FUSED_PROFILE_ID
        );
        let legacy = contracts(ImageCudaAttentionPolicy::Sequential).unwrap();
        let fused = contracts(ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1).unwrap();
        assert!(!legacy
            .iter()
            .any(|c| c.descriptor().id.as_str() == CROSS_ATTENTION_BF16_OPERATION_ID));
        assert!(!fused
            .iter()
            .any(|c| c.descriptor().id.as_str() == CROSS_ATTENTION_BF16_OPERATION_ID));
        assert!(!legacy
            .iter()
            .any(|c| c.descriptor().id.as_str() == CROSS_ATTENTION_FUSED_BF16_OPERATION_ID));
        assert_eq!(
            fused
                .iter()
                .filter(|c| c.descriptor().id.as_str() == CROSS_ATTENTION_FUSED_BF16_OPERATION_ID)
                .count(),
            1
        );
        assert!(!legacy
            .iter()
            .any(|c| c.descriptor().id.as_str() == STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID));
        assert_eq!(
            fused
                .iter()
                .filter(
                    |c| c.descriptor().id.as_str() == STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID
                )
                .count(),
            1
        );
        let mut c = config();
        let before = c.opaque_bytes().unwrap();
        c.attention_policy = ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1;
        assert_eq!(c.opaque_bytes().unwrap(), before + 128 * 1024 * 1024);
        assert!(c.validate().is_err());
        c.memory =
            ImageCudaMemoryPolicy::ExplicitTotal(NonZeroU64::new(1024 * 1024 * 1024).unwrap());
        c.validate().unwrap();
    }

    #[test]
    fn every_product_contract_requires_only_declared_cuda_capabilities() {
        // Metadata-only: this does not construct a device, load cuDNN or create
        // a native provider. CapabilityCatalog checks every registered contract,
        // including contracts that no current model plan happens to consume.
        let runtime =
            cuda_tensor_runtime_config(0, DeviceId::new("device.cuda.0").unwrap()).unwrap();
        for policy in [
            ImageCudaAttentionPolicy::Sequential,
            ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1,
        ] {
            for contract in contracts(policy).unwrap() {
                let descriptor = contract.descriptor();
                assert!(
                    descriptor
                        .provider
                        .required_capabilities
                        .is_subset(&runtime.capabilities),
                    "CUDA product contract {} requires unavailable capabilities",
                    descriptor.id,
                );
            }
        }
        // The ordinary online cross provider is Reference-only. Advertising its
        // capability would hide the catalog mismatch without adding a backend.
        assert!(!runtime
            .capabilities
            .contains(&CapabilityId::new(CROSS_ATTENTION_BF16_CAPABILITY_ID).unwrap()));
        assert!(runtime
            .capabilities
            .contains(&CapabilityId::new(CROSS_ATTENTION_FUSED_BF16_CAPABILITY_ID).unwrap()));
    }

    #[test]
    fn cuda_request_cache_requires_an_explicit_vendor_attention_profile() {
        let mut c = config();
        c.memory =
            ImageCudaMemoryPolicy::ExplicitTotal(NonZeroU64::new(1024 * 1024 * 1024).unwrap());
        c.validate().unwrap();
        c.pipeline.edit_prefix_policy = super::super::ImageEditPrefixPolicy::RequestCache;
        assert!(c
            .validate()
            .unwrap_err()
            .to_string()
            .contains("request-cache editing"));
        c.attention_policy = ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1;
        c.validate().unwrap();
    }

    #[test]
    fn fp8_profiles_compose_with_attention_without_changing_noise_decoder_or_euler() {
        for attention in [
            ImageCudaAttentionPolicy::Sequential,
            ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1,
        ] {
            let bf16 = cuda_profiles(attention).unwrap();
            let fp8 = cuda_profiles_with_precision(
                attention,
                ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1,
            )
            .unwrap();
            assert_eq!(fp8.noise, bf16.noise);
            assert_eq!(fp8.decoder, bf16.decoder);
            assert_eq!(fp8.euler, bf16.euler);
            assert_eq!(
                fp8.text_encoder.as_str(),
                Qwen3VLTextEncoderFamily::CUDA_FP8_PROFILE_ID
            );
            assert_eq!(
                fp8.denoiser.as_str(),
                match attention {
                    ImageCudaAttentionPolicy::Sequential => {
                        QwenImage21TransformerFamily::CUDA_FP8_PROFILE_ID
                    }
                    ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1 => {
                        QwenImage21TransformerFamily::CUDA_FP8_FUSED_PROFILE_ID
                    }
                }
            );
            assert_ne!(fp8.text_encoder, bf16.text_encoder);
            assert_ne!(fp8.denoiser, bf16.denoiser);
        }
    }

    #[test]
    fn component_precision_selects_independent_text_and_native_denoiser_profiles() {
        use super::super::{ImageComponentPrecision, ImageDenoiserPrecision, ImageTextPrecision};
        for attention in [
            ImageCudaAttentionPolicy::Sequential,
            ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1,
        ] {
            for text in [
                ImageTextPrecision::Bf16,
                ImageTextPrecision::Fp8LinearExperimentalV1,
            ] {
                let precision = ImageCudaPrecisionPolicy::Components(ImageComponentPrecision {
                    text_encoder: text,
                    denoiser: ImageDenoiserPrecision::NativeSource,
                });
                let profiles = cuda_profiles_with_precision(attention, precision).unwrap();
                let bf16 = cuda_profiles(attention).unwrap();
                assert_eq!(
                    profiles.text_encoder.as_str(),
                    match text {
                        ImageTextPrecision::Bf16 => Qwen3VLTextEncoderFamily::CUDA_PROFILE_ID,
                        ImageTextPrecision::Fp8LinearExperimentalV1 =>
                            Qwen3VLTextEncoderFamily::CUDA_FP8_PROFILE_ID,
                    }
                );
                assert_eq!(
                    profiles.denoiser.as_str(),
                    match attention {
                        ImageCudaAttentionPolicy::Sequential =>
                            QwenImage21TransformerFamily::CUDA_NATIVE_BLOCK_PROFILE_ID,
                        ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1 =>
                            QwenImage21TransformerFamily::CUDA_NATIVE_BLOCK_FUSED_PROFILE_ID,
                    }
                );
                assert_eq!(profiles.decoder, bf16.decoder);
                assert_eq!(profiles.noise, bf16.noise);
                assert_eq!(profiles.euler, bf16.euler);
            }
        }
    }

    #[test]
    fn execution_weight_lower_bound_reserves_room_and_checks_opaque_underflow() {
        let config = config();
        let opaque = config.opaque_bytes().unwrap();
        assert!(config
            .validate_weight_capacity(&[12, 24, 8], opaque + 25)
            .is_ok());
        assert!(config
            .validate_weight_capacity(&[12, 24, 8], opaque + 24)
            .is_err());
        assert!(config
            .validate_weight_capacity(&[12, 24, 8], opaque - 1)
            .is_err());
    }

    #[test]
    fn editing_command_admission_counts_new_stages_and_augmented_text_and_dit() {
        // Real family programs, metadata only: no CUDA context, native query,
        // payload allocation, checkpoint or invented successful provider.
        let raw = serde_json::json!({
            "model_type":"qwen3_vl","architectures":["Qwen3VLForConditionalGeneration"],
            "text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16",
                "hidden_act":"silu","attention_bias":false,"attention_dropout":0.0,
                "num_hidden_layers":3,"hidden_size":8,"intermediate_size":16,
                "num_attention_heads":1,"num_key_value_heads":1,"head_dim":8,
                "vocab_size":32,"max_position_embeddings":1024,"rms_norm_eps":0.000001,
                "rope_theta":100,"rope_scaling":{"rope_type":"default",
                    "mrope_interleaved":true,"mrope_section":[2,1,1]}},
            "vision_config":{"model_type":"qwen3_vl","dtype":"bfloat16",
                "hidden_act":"gelu_pytorch_tanh","depth":3,"hidden_size":8,
                "intermediate_size":12,"num_heads":2,"num_position_embeddings":16,
                "out_hidden_size":8,"deepstack_visual_indexes":[0,1,2],"in_channels":3,
                "patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2}
        });
        let text = Qwen3VLTextConfig::new(
            ferrum_models::qwen3_vl_text::Qwen3VLTextArchitecture::from_hf_value(&raw).unwrap(),
            Qwen3VLTextLayout::new(1, vec![vec![Qwen3VLTextKeyRun::new(1, true)]]).unwrap(),
        )
        .unwrap();
        let dit = QwenImage21TransformerConfig::new(
            ferrum_models::qwen_image21::QwenImage21Config::from_hf_json(
                &serde_json::to_vec(&serde_json::json!({
                    "_class_name":"QwenImage21Transformer2DModel", "num_layers":1,
                    "num_attention_heads":1,"attention_head_dim":8,"context_in_dim":8,
                    "in_channels":64,"out_channels":64,"patch_size":1,"mlp_ratio":2,
                    "axes_dims_rope":[2,2,4],"causal_condition":true,"eps":0.000001
                }))
                .unwrap(),
                ferrum_models::qwen_image21::StructureLimits::default(),
            )
            .unwrap(),
            QwenImage21T2iLayout::new(1, 2, 2, vec![1]).unwrap(),
        )
        .unwrap();
        let vision = Qwen3VLVisionArchitecture::from_hf_value(&raw).unwrap();
        let encoder = QwenImage21VaeEncoderConfig::new(
            64,
            1,
            vec![1, 2, 4, 8, 8],
            2,
            vec![false, true, true, true],
            1,
            16,
            16,
        )
        .unwrap();
        let profiles = cuda_profiles(ImageCudaAttentionPolicy::Sequential).unwrap();
        let ordinary = [
            TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new().unwrap())
                .prepare_with_profile(
                    &serde_json::to_value(&text).unwrap(),
                    &profiles.text_encoder,
                )
                .unwrap(),
            TypedFamilyRegistration::new(QwenImage21TransformerFamily::new().unwrap())
                .prepare_with_profile(&serde_json::to_value(&dit).unwrap(), &profiles.denoiser)
                .unwrap(),
        ];
        let editing = editing_command_families(&text, &dit, &vision, &encoder, &profiles).unwrap();
        assert_eq!(
            editing[0].family_id().as_str(),
            Qwen3VLVisionFamily::FAMILY_ID
        );
        assert_eq!(
            editing[1].family_id().as_str(),
            QwenImage21VaeEncoderFamily::FAMILY_ID
        );
        for (base, edit, added) in [
            (&ordinary[0], &editing[2], 6),
            (&ordinary[1], &editing[3], 3),
        ] {
            let limit = component_command_count(base).unwrap();
            validate_family_command_limits(std::slice::from_ref(base), limit).unwrap();
            assert_eq!(component_command_count(edit).unwrap(), limit + added);
            assert!(validate_family_command_limits(std::slice::from_ref(edit), limit).is_err());
        }
        let required = editing
            .iter()
            .map(|f| component_command_count(f).unwrap())
            .max()
            .unwrap();
        validate_family_command_limits(&editing, required).unwrap();
        assert!(validate_family_command_limits(&editing, required - 1).is_err());
    }
}
