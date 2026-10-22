//! Explicit CUDA composition for the serial image pipeline.
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
    CudaCudnnConfig,
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
use std::sync::Arc;

/// Every limit is supplied explicitly. `pipeline.device_capacity_bytes` is the
/// plan's storage ceiling; `usable_memory_bytes` also contains runtime/stream
/// and cuDNN opaque claims. Neither substitutes for actual plan admission.
#[derive(Debug, Clone)]
pub struct QwenImage21CudaConfig {
    pub ordinal: usize,
    pub usable_memory_bytes: u64,
    pub bootstrap_memory_bytes: u64,
    pub stream_memory_bytes: u64,
    pub maximum_streams: usize,
    pub maximum_in_flight_per_stream: usize,
    pub maximum_commands_per_submission: usize,
    pub opaque_memory_profile: String,
    pub cudnn: CudaCudnnConfig,
    pub host: HostMemoryLimits,
    pub pipeline: ImagePipelineConfig,
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
        let overhead = self.opaque_bytes()?;
        ensure!(
            overhead < self.usable_memory_bytes,
            "CUDA runtime/stream/cuDNN opaque allowances leave no storage capacity"
        );
        ensure!(
            self.pipeline.device_capacity_bytes <= self.usable_memory_bytes,
            "pipeline storage ceiling exceeds declared usable device capacity"
        );
        ensure!(
            self.cudnn.workspace_limit_bytes < self.usable_memory_bytes - overhead,
            "cuDNN workspace ceiling leaves no storage for tensors"
        );

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
        self.validate()?;
        let profiles = cuda_profiles()?;
        self.validate_assets(&assets, &profiles)?;
        let contracts = contracts()?;
        let materializers = WeightMaterializerRegistry::identity_only()?;
        let device = cuda_tensor_runtime_config(
            self.ordinal,
            DeviceId::new(format!("device.cuda.{}", self.ordinal))?,
        )?;
        let runtime = Arc::new(CudaDeviceRuntime::new_tensor_eager(
            CudaTensorRuntimeConfig {
                device,
                usable_memory_bytes: self.usable_memory_bytes,
                bootstrap_memory_bytes: self.bootstrap_memory_bytes,
                stream_memory_bytes: self.stream_memory_bytes,
                maximum_streams: self.maximum_streams,
                maximum_in_flight_per_stream: self.maximum_in_flight_per_stream,
                maximum_commands_per_submission: self.maximum_commands_per_submission,
                opaque_memory_profile: self.opaque_memory_profile,
            },
        )?);
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
        QwenImage21Pipeline::new(
            runtime,
            host,
            registry,
            catalog,
            materializers,
            assets,
            profiles,
            self.pipeline,
        )
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
        let largest_weights = assets
            .receipt()
            .required_weight_bytes
            .into_iter()
            .max()
            .unwrap_or(0);
        ensure!(
            largest_weights <= self.pipeline.device_capacity_bytes
                && largest_weights < self.usable_memory_bytes - self.opaque_bytes()?,
            "largest component weights leave no admitted device capacity for activations"
        );
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
            usable_memory_bytes: 1024 * 1024,
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
            pipeline: ImagePipelineConfig {
                device_capacity_bytes: 1024 * 1024,
                maximum_user_tokens: 1024,
                maximum_width: 512,
                maximum_height: 512,
                maximum_batch_size: ImagePipelineConfig::DEFAULT_MAXIMUM_BATCH_SIZE,
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
            |c: &mut QwenImage21CudaConfig| c.usable_memory_bytes = 9216,
            |c: &mut QwenImage21CudaConfig| c.pipeline.device_capacity_bytes = 1024 * 1024 + 1,
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
