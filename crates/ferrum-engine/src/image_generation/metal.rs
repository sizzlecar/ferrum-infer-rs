//! Metal composition for the shared managed image pipeline.
use super::*;
use anyhow::{ensure, Context, Result};
use ferrum_kernels::backend::metal::{
    vnext_ops::tensor::{
        metal_tensor_providers, metal_tensor_runtime_config, MetalTensorOperation,
    },
    vnext_runtime::{
        MetalDeviceRuntime, MetalTensorMemoryObservation, MetalTensorRuntimeAutoConfig,
        MetalTensorRuntimeConfig,
    },
};
use ferrum_models::qwen3_vl_text::{Qwen3VLTextKeyRun, Qwen3VLTextLayout};
use ferrum_models::qwen3_vl_vision::Qwen3VLVisionFamily;
use ferrum_models::qwen_image21::QwenImage21VaeEncoderFamily;
use std::num::NonZeroU64;

#[derive(Debug, Clone, Copy)]
pub enum ImageMetalMemoryPolicy {
    ExplicitTotal(NonZeroU64),
    /// Fraction of the recommended working set, bounded by its remaining
    /// process allowance and an optional caller ceiling; not system free RAM.
    AutoRecommended {
        utilization: f64,
        maximum_bytes: Option<NonZeroU64>,
    },
}

#[derive(Debug, Clone)]
pub struct QwenImage21MetalConfig {
    pub memory: ImageMetalMemoryPolicy,
    pub precision_policy: ImagePrecisionPolicy,
    pub bootstrap_memory_bytes: u64,
    pub stream_memory_bytes: u64,
    pub maximum_streams: ImageStreamLimit,
    pub maximum_in_flight_per_stream: usize,
    pub maximum_commands_per_submission: usize,
    pub opaque_memory_profile: String,
    pub host: HostMemoryLimits,
    pub pipeline: ImagePipelineLimits,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ImageMetalCompositionReceipt {
    pub memory_policy: &'static str,
    pub precision_policy: ImagePrecisionPolicy,
    pub stream_plan: ImageStreamPlan,
    pub device_capacity_bytes: u64,
    pub observed_memory: Option<MetalTensorMemoryObservation>,
    pub required_component_commands: usize,
    pub runtime_opaque_profile: String,
    pub native_block_denoiser: bool,
}

impl QwenImage21MetalConfig {
    pub fn stream_plan(&self) -> Result<ImageStreamPlan> {
        self.maximum_streams
            .resolve(ImageStreamRequirements::for_serial_pipeline(
                self.pipeline.static_initialization,
                self.pipeline.executor,
            ))
    }
    fn explicit(&self, capacity: u64) -> Result<MetalTensorRuntimeConfig> {
        Ok(MetalTensorRuntimeConfig {
            device: metal_tensor_runtime_config(DeviceId::new("device.metal.0")?)?,
            usable_memory_bytes: capacity,
            bootstrap_memory_bytes: self.bootstrap_memory_bytes,
            stream_memory_bytes: self.stream_memory_bytes,
            maximum_streams: self.stream_plan()?.resolved_streams(),
            maximum_in_flight_per_stream: self.maximum_in_flight_per_stream,
            maximum_commands_per_submission: self.maximum_commands_per_submission,
            opaque_memory_profile: self.opaque_memory_profile.clone(),
        })
    }
    fn automatic(
        &self,
        utilization: f64,
        maximum: Option<NonZeroU64>,
    ) -> Result<MetalTensorRuntimeAutoConfig> {
        let explicit = self.explicit(maximum.map_or(u64::MAX, NonZeroU64::get))?;
        Ok(MetalTensorRuntimeAutoConfig {
            device: explicit.device,
            utilization,
            maximum_memory_bytes: maximum.map(NonZeroU64::get),
            bootstrap_memory_bytes: explicit.bootstrap_memory_bytes,
            stream_memory_bytes: explicit.stream_memory_bytes,
            maximum_streams: explicit.maximum_streams,
            maximum_in_flight_per_stream: explicit.maximum_in_flight_per_stream,
            maximum_commands_per_submission: explicit.maximum_commands_per_submission,
            opaque_memory_profile: explicit.opaque_memory_profile,
        })
    }
    pub fn validate(&self) -> Result<()> {
        self.pipeline.validate()?;
        ensure!(!self.precision_policy.uses_fp8(), "Metal image Linear currently supports BF16 or native source blocks; FP8 is not implemented");
        ensure!(
            self.pipeline.edit_prefix_policy == ImageEditPrefixPolicy::Recompute,
            "Metal image editing requires recompute; request-cache attention is not implemented"
        );
        match self.memory {
            ImageMetalMemoryPolicy::ExplicitTotal(capacity) => {
                self.explicit(capacity.get())?.validate()?
            }
            ImageMetalMemoryPolicy::AutoRecommended {
                utilization,
                maximum_bytes,
            } => self.automatic(utilization, maximum_bytes)?.validate()?,
        }
        ensure!(
            self.host.class_bytes(HostMemoryClass::InputPreprocess) >= 16
                && self.host.class_bytes(HostMemoryClass::Output) > 0,
            "Metal needs explicit seed/input and output Host capacity"
        );
        ensure!(
            self.pipeline
                .static_initialization
                .maximum_commands_per_batch()
                <= self.maximum_commands_per_submission,
            "static initialization command batch exceeds Metal submission limit"
        );
        Ok(())
    }
    fn prepare(
        &self,
        assets: &QwenImage21Assets,
        profiles: &QwenImage21Profiles,
        edit_profiles: &QwenImage21EditProfiles,
    ) -> Result<usize> {
        let minimum = assets
            .spatial_scale()?
            .checked_mul(2)
            .context("minimum image size overflows")?;
        ensure!(
            self.pipeline.maximum_width >= minimum && self.pipeline.maximum_height >= minimum,
            "image limits admit no supported even latent grid"
        );
        let text = Qwen3VLTextConfig::new(
            assets.text.clone(),
            Qwen3VLTextLayout::new(1, vec![vec![Qwen3VLTextKeyRun::new(1, true)]])?,
        )?;
        let dit = QwenImage21TransformerConfig::new(
            assets.transformer.clone(),
            QwenImage21T2iLayout::new(1, 2, 2, vec![1])?,
        )?
        .with_weight_encodings(assets.transformer_encodings.clone())?;
        let mut families = vec![
            TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new()?)
                .prepare_with_profile(&serde_json::to_value(&text)?, &profiles.text_encoder)?,
            TypedFamilyRegistration::new(NormalNoiseFamily::new(profiles.noise.as_str())?)
                .prepare_with_profile(
                    &serde_json::to_value(NormalNoiseConfig::new(64, 2, 2)?)?,
                    &profiles.noise,
                )?,
            TypedFamilyRegistration::new(QwenImage21TransformerFamily::new()?)
                .prepare_with_profile(&serde_json::to_value(&dit)?, &profiles.denoiser)?,
            TypedFamilyRegistration::new(QwenImage21VaeDecoderFamily::new()?)
                .prepare_with_profile(
                    &serde_json::to_value(assets.decoder(2, 2)?)?,
                    &profiles.decoder,
                )?,
        ];
        if let Some(edit) = &assets.editing {
            families.extend(super::composition::editing_command_families(
                &text,
                &dit,
                &edit.vision,
                &edit.encoder,
                profiles,
                edit_profiles,
            )?);
        }
        super::composition::validate_family_command_limits(
            &families,
            self.maximum_commands_per_submission,
        )?;
        families
            .iter()
            .map(super::composition::component_command_count)
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .context("missing image components")
    }
    pub fn create_pipeline(
        self,
        assets: QwenImage21Assets,
    ) -> Result<QwenImage21Pipeline<MetalDeviceRuntime>> {
        self.create_pipeline_with_receipt(assets)
            .map(|(pipeline, _)| pipeline)
    }
    pub fn create_pipeline_with_receipt(
        self,
        assets: QwenImage21Assets,
    ) -> Result<(
        QwenImage21Pipeline<MetalDeviceRuntime>,
        ImageMetalCompositionReceipt,
    )> {
        self.validate()?;
        let native_block_denoiser = !assets.transformer_encodings.is_empty();
        self.precision_policy
            .validate_denoiser_source(native_block_denoiser)?;
        let profiles = profiles(native_block_denoiser)?;
        let editing_profiles = editing_profiles()?;
        let required_component_commands = self.prepare(&assets, &profiles, &editing_profiles)?;
        let stream_plan = self.stream_plan()?;
        let (runtime, observed_memory) = match self.memory {
            ImageMetalMemoryPolicy::ExplicitTotal(bytes) => (
                MetalDeviceRuntime::new_tensor_eager(self.explicit(bytes.get())?)?,
                None,
            ),
            ImageMetalMemoryPolicy::AutoRecommended {
                utilization,
                maximum_bytes,
            } => {
                let (runtime, observation) = MetalDeviceRuntime::new_tensor_eager_auto(
                    self.automatic(utilization, maximum_bytes)?,
                )?;
                (runtime, Some(observation))
            }
        };
        let runtime = Arc::new(runtime);
        let capacity = runtime
            .tensor_capacity_bytes()
            .context("Metal runtime has no admitted tensor capacity")?;
        let opaque = self
            .stream_memory_bytes
            .checked_mul(stream_plan.resolved_streams() as u64)
            .and_then(|v| v.checked_add(self.bootstrap_memory_bytes))
            .context("Metal opaque allowance overflows")?;
        let storage = capacity
            .checked_sub(opaque)
            .context("Metal allowances exceed capacity")?;
        for bytes in assets
            .receipt()
            .required_weight_bytes
            .into_iter()
            .chain(assets.editing_weight_bytes().into_iter().flatten())
        {
            ensure!(bytes < storage, "component weight bytes {bytes} leave no activation capacity under Metal storage ceiling {storage}");
        }
        let host = runtime.tensor_host_account(self.host)?;
        let contracts = MetalTensorOperation::ALL
            .into_iter()
            .map(|kind| Ok(Box::new(kind.contract()?) as Box<dyn OperationContract>))
            .collect::<Result<Vec<_>, VNextError>>()?;
        let registry = OperationRuntimeRegistry::new(
            contracts,
            metal_tensor_providers(&runtime, &MetalTensorOperation::ALL)?,
        )?;
        let descriptor = runtime.descriptor();
        let engine = EngineProviderDescriptor::new(
            ProviderId::new("provider.engine.metal.image_generation")?,
            ContractVersion::new(1, 0),
            descriptor.runtime_implementation_fingerprint.clone(),
            descriptor.id.clone(),
            descriptor.capabilities.clone(),
        )?;
        let materializers = WeightMaterializerRegistry::identity_only()?;
        let catalog = materializers
            .augment_catalog(registry.capability_catalog(descriptor.clone(), vec![engine])?)?;
        let editing = assets.editing.is_some();
        let mut pipeline = QwenImage21Pipeline::new(
            runtime,
            host,
            registry,
            catalog,
            materializers,
            assets,
            profiles,
            self.pipeline
                .with_device_capacity(NonZeroU64::new(capacity).context("zero Metal capacity")?),
        )?
        .with_device_opaque_allowance(opaque)?;
        if editing {
            pipeline = pipeline.with_editing_profiles(editing_profiles)?;
        }
        Ok((
            pipeline,
            ImageMetalCompositionReceipt {
                precision_policy: self.precision_policy,
                memory_policy: if observed_memory.is_some() {
                    "auto_recommended"
                } else {
                    "explicit_total"
                },
                stream_plan,
                device_capacity_bytes: capacity,
                observed_memory,
                required_component_commands,
                runtime_opaque_profile: self.opaque_memory_profile,
                native_block_denoiser,
            },
        ))
    }
}
fn profiles(native_block: bool) -> Result<QwenImage21Profiles> {
    let id = |name| NumericalProfileId::new(name).map_err(anyhow::Error::msg);
    Ok(QwenImage21Profiles {
        text_encoder: id(Qwen3VLTextEncoderFamily::METAL_PROFILE_ID)?,
        noise: id(MetalTensorOperation::PhiloxNormal.profile())?,
        denoiser: id(if native_block {
            QwenImage21TransformerFamily::METAL_NATIVE_BLOCK_PROFILE_ID
        } else {
            QwenImage21TransformerFamily::METAL_PROFILE_ID
        })?,
        decoder: id(QwenImage21VaeDecoderFamily::METAL_PROFILE_ID)?,
        euler: Bf16EulerProfile::CudaScalarBf16StagedRneV1,
    })
}
fn editing_profiles() -> Result<QwenImage21EditProfiles> {
    Ok(QwenImage21EditProfiles {
        vision: NumericalProfileId::new(Qwen3VLVisionFamily::METAL_PROFILE_ID)
            .map_err(anyhow::Error::msg)?,
        encoder: NumericalProfileId::new(QwenImage21VaeEncoderFamily::METAL_PROFILE_ID)
            .map_err(anyhow::Error::msg)?,
    })
}
