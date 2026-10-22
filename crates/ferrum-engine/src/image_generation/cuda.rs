//! One-context CUDA composition for the experimental cold image pipeline.
//!
//! Prepare `QwenImage21Assets` before calling this module. Construction validates
//! caller budgets and architecture-derived command counts before device work,
//! then creates one runtime, one Host account and one native provider bundle.
//! cuDNN retains its separately admitted opaque owner inside that same runtime.
//! Opaque allowances are declarations, not measured memory qualification.

use super::{ImagePipelineConfig, QwenImage21Assets, QwenImage21Pipeline, QwenImage21Profiles};
use anyhow::{ensure, Context, Result};
use ferrum_diffusion::Bf16EulerProfile;
use ferrum_kernels::backend::cuda::{
    vnext_ops::*,
    vnext_runtime::{CudaDeviceCommand, CudaDeviceRuntime, CudaTensorRuntimeConfig},
    CudaCudnnConfig, CudaMemoryUtilization, CudaTensorMemoryObservation,
    CudaTensorRuntimeAutoConfig,
};
use ferrum_models::normal_noise::{NormalNoiseConfig, NormalNoiseFamily};
use ferrum_models::qwen3_vl_text::{
    Qwen3VLTextConfig, Qwen3VLTextEncoderFamily, Qwen3VLTextKeyRun, Qwen3VLTextLayout,
};
use ferrum_models::qwen_image21::{
    QwenImage21T2iLayout, QwenImage21TransformerConfig, QwenImage21TransformerFamily,
    QwenImage21VaeDecoderFamily,
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
    pub static_initialization: StaticInitializationPolicy,
    pub executor: ferrum_execution::TensorExecutorConfig,
}
impl ImageCudaPipelineLimits {
    pub fn validate(&self) -> Result<()> {
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
            static_initialization: self.static_initialization,
            executor: self.executor,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ImageCudaCompositionReceipt {
    pub memory_policy: &'static str,
    pub device_capacity_bytes: u64,
    pub observed_memory: Option<CudaTensorMemoryObservation>,
    /// Absolute path passed to the admitted native loader. The generated
    /// component plans retain the provider's library fingerprint separately.
    pub cudnn_library: std::path::PathBuf,
    pub cudnn_workspace_limit_bytes: u64,
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
    pub maximum_streams: usize,
    pub maximum_in_flight_per_stream: usize,
    pub maximum_commands_per_submission: usize,
    pub opaque_memory_profile: String,
    pub cudnn: CudaCudnnConfig,
    pub host: HostMemoryLimits,
    pub pipeline: ImageCudaPipelineLimits,
}

impl QwenImage21CudaConfig {
    /// Pure configuration checks. No driver query, context, library load,
    /// account construction, source mapping, or payload allocation occurs.
    /// Request-specific shapes still pass through the pipeline's compilation
    /// and ordinary Host/device admission before their execution.
    pub fn validate(&self) -> Result<()> {
        self.pipeline.validate()?;
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
            self.maximum_streams > 0
                && self.maximum_in_flight_per_stream > 0
                && self.maximum_commands_per_submission > 0
                && self.maximum_commands_per_submission <= u32::MAX as usize,
            "CUDA stream/flight limits must be positive and commands fit u32"
        );
        self.maximum_streams
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
        let profiles = cuda_profiles()?;
        self.validate_assets(&assets, &profiles)?;
        let contracts = contracts()?;
        let materializers = WeightMaterializerRegistry::identity_only()?;
        let device = cuda_tensor_runtime_config(
            self.ordinal,
            DeviceId::new(format!("device.cuda.{}", self.ordinal))?,
        )?;
        let runtime = Arc::new(match self.memory {
            ImageCudaMemoryPolicy::ExplicitTotal(bytes) => {
                CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
                    device,
                    usable_memory_bytes: bytes.get(),
                    bootstrap_memory_bytes: self.bootstrap_memory_bytes,
                    stream_memory_bytes: self.stream_memory_bytes,
                    maximum_streams: self.maximum_streams,
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
                maximum_streams: self.maximum_streams,
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
        self.validate_weight_capacity(&assets, capacity.get())?;
        let receipt = ImageCudaCompositionReceipt {
            memory_policy: match self.memory {
                ImageCudaMemoryPolicy::ExplicitTotal(_) => "explicit_total",
                ImageCudaMemoryPolicy::AutoAvailable { .. } => "auto_available",
            },
            device_capacity_bytes: capacity.get(),
            observed_memory,
            cudnn_library: self.cudnn.library_path.clone(),
            cudnn_workspace_limit_bytes: self.cudnn.workspace_limit_bytes,
            runtime_opaque_profile: self.opaque_memory_profile.clone(),
            cudnn_opaque_profile: self.cudnn.opaque_memory_profile.clone(),
        };
        // ALL includes Philox. Do not call the standalone Philox factory:
        // it would try to acquire a second live native-module permit.
        let mut providers = cuda_vnext_tensor_providers_with_cudnn(
            &runtime,
            &self.cudnn,
            &CudaTensorOperation::ALL,
        )?;
        providers.push(cuda_vnext_dense_bf16_provider(&runtime)?);
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
        // This is the sole Host account. No component receives a fresh ledger.
        let host = HostMemoryAccount::new(self.host);
        let pipeline = QwenImage21Pipeline::new(
            runtime,
            host,
            registry,
            catalog,
            materializers,
            assets,
            profiles,
            self.pipeline.with_device_capacity(capacity),
        )?;
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
    fn validate_weight_capacity(&self, assets: &QwenImage21Assets, capacity: u64) -> Result<()> {
        let largest = assets
            .receipt()
            .required_weight_bytes
            .into_iter()
            .max()
            .unwrap_or(0);
        let opaque = self.opaque_bytes()?;
        ensure!(largest < capacity - opaque, "largest component weights ({largest} bytes) leave no admitted activation capacity under total ceiling {capacity} with declared opaque allowances {opaque}");
        Ok(())
    }

    fn opaque_bytes(&self) -> Result<u64> {
        self.stream_memory_bytes
            .checked_mul(u64::try_from(self.maximum_streams)?)
            .and_then(|n| n.checked_add(self.bootstrap_memory_bytes))
            .and_then(|n| n.checked_add(self.cudnn.opaque_memory_bytes))
            .context("CUDA opaque allowance total overflows")
    }

    fn validate_assets(
        &self,
        assets: &QwenImage21Assets,
        profiles: &QwenImage21Profiles,
    ) -> Result<()> {
        if let Some(capacity) = self.declared_maximum() {
            self.validate_weight_capacity(assets, capacity.get())?;
        }
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
        )?;
        let decoder = assets.decoder(2, 2)?;
        let noise = NormalNoiseConfig::new(64, 2, 2)?;
        let families = [
            TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new()?)
                .prepare_with_profile(&serde_json::to_value(text)?, &profiles.text_encoder)?,
            TypedFamilyRegistration::new(NormalNoiseFamily::new(profiles.noise.as_str())?)
                .prepare_with_profile(&serde_json::to_value(noise)?, &profiles.noise)?,
            TypedFamilyRegistration::new(QwenImage21TransformerFamily::new()?)
                .prepare_with_profile(&serde_json::to_value(dit)?, &profiles.denoiser)?,
            TypedFamilyRegistration::new(QwenImage21VaeDecoderFamily::new()?)
                .prepare_with_profile(&serde_json::to_value(decoder)?, &profiles.decoder)?,
        ];
        for family in families {
            let commands = family
                .program()
                .blocks()
                .iter()
                .try_fold(family.program().inputs().len(), |n, block| {
                    n.checked_add(block.nodes.len())
                })
                .context("component command count overflows")?;
            ensure!(
                commands <= self.maximum_commands_per_submission,
                "{} requires at least {commands} compute/upload commands, exceeding configured {}",
                family.family_id(),
                self.maximum_commands_per_submission
            );
        }
        Ok(())
    }
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

fn cuda_profiles() -> Result<QwenImage21Profiles> {
    let id = |value: &str| NumericalProfileId::new(value).map_err(anyhow::Error::msg);
    Ok(QwenImage21Profiles {
        text_encoder: id(Qwen3VLTextEncoderFamily::CUDA_PROFILE_ID)?,
        noise: id(CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_ID)?,
        denoiser: id(QwenImage21TransformerFamily::CUDA_PROFILE_ID)?,
        decoder: id(QwenImage21VaeDecoderFamily::CUDA_PROFILE_ID)?,
        euler: Bf16EulerProfile::CudaScalarBf16StagedRneV1,
    })
}

fn contracts() -> Result<Vec<Box<dyn OperationContract>>> {
    Ok(vec![
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
        Box::new(philox_normal_bf16_contract()?),
        Box::new(conv2d_bf16_contract()?),
        Box::new(dense_linear_bf16_contract()?),
    ])
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
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 1024,
            opaque_memory_profile: "experimental-unqualified-test-v1".into(),
            cudnn: CudaCudnnConfig {
                library_path: "/explicit/libcudnn.so.9".into(),
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
                static_initialization: StaticInitializationPolicy::new(1024, 32).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        }
    }

    #[test]
    fn numeric_and_host_budgets_validate_without_a_device_or_library_load() {
        config().validate().unwrap();
        let edits: &[fn(&mut QwenImage21CudaConfig)] = &[
            |c: &mut QwenImage21CudaConfig| c.maximum_streams = 0,
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
        let contracts = contracts().unwrap();
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
        let profiles = cuda_profiles().unwrap();
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
}
