//! Explicit H3 product document. A public model alias never chooses this
//! architecture or changes its original Comfy weight/activation precision.
use super::configuration::{CudnnConfig, HostConfig, PipelineConfig, ServingConfig};
use anyhow::{ensure, Context, Result};
use ferrum_engine::image_generation::{
    p_h3::{
        PH3ImageArchivePaths, PH3ImageAssetLimits, PH3ImageAssets, PH3ImageInitialNoisePolicy,
        PH3ImageMainVisionReuse, PH3ImagePromptExpansionDiagnosticPolicy,
    },
    ImageDecoderBatchPolicy, ImageEditPrefixPolicy, ImageStreamLimit, ImageStreamRequirements,
};
use ferrum_models::{
    p_h3_image_vae::{PH3ImageVaeSpatialPolicy, PH3ImageVaeTileBatch},
    pipeline_source::PipelineSourceLimits,
};
use serde::Deserialize;
use std::path::{Path, PathBuf};

mod prompt_expansion;
pub use super::trace_artifact::PH3ImageContinuousTraceConfig;
pub use prompt_expansion::{
    PH3ImageCausalModuleConfig, PH3ImagePromptExpansionConfig, PH3ImagePromptExpansionVisionLimits,
    PH3ImagePromptExpansionWeightCacheConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageProductArchitecture {
    PH3Image,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageProductPreset {
    ComfyW4a8F32V1,
}
impl PH3ImageProductPreset {
    pub const fn identity(self) -> &'static str {
        match self {
            Self::ComfyW4a8F32V1 => "comfy_w4a8_f32_v1",
        }
    }
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageProductConfig {
    pub architecture: PH3ImageProductArchitecture,
    pub model: String,
    pub preset: PH3ImageProductPreset,
    /// Initial image/audio only; reference posterior and augmentation retain
    /// their separately declared native CUDA policy.
    #[serde(default)]
    pub initial_noise: PH3ImageInitialNoisePolicy,
    /// Reuse only identical reference-image visual outputs inside one cohort.
    #[serde(default)]
    pub main_vision_reuse: PH3ImageMainVisionReuse,
    pub metadata_directory: PathBuf,
    pub archives: PH3ImageArchivePaths,
    /// Native causal expansion is enabled by default and requires explicit
    /// original sources. Raw prompts require the declared Disabled mode.
    #[serde(default)]
    pub prompt_expansion: PH3ImagePromptExpansionConfig,
    #[serde(default)]
    pub metadata_limits: PH3ImageMetadataReadLimits,
    #[serde(default = "default_spatial_policy")]
    pub vae_spatial_policy: PH3ImageVaeSpatialPolicy,
    /// Maximum real spatial tiles in one decoder execution. This does not
    /// change the number of image requests or the numerical preset.
    #[serde(default)]
    pub vae_decoder_tile_batch: PH3ImageVaeTileBatch,
    pub backend: PH3ImageBackendConfig,
    pub serving: Option<ServingConfig>,
    /// Bounded diagnostic JSONL, separate from image output and HTTP payloads.
    #[serde(default)]
    pub continuous_trace: Option<PH3ImageContinuousTraceConfig>,
    /// Opt-in expansion observations in the same bounded trace. Effective
    /// prompt text is recorded only by the explicit capture mode.
    #[serde(default)]
    pub prompt_expansion_diagnostics: PH3ImagePromptExpansionDiagnosticPolicy,
}
fn default_spatial_policy() -> PH3ImageVaeSpatialPolicy {
    PH3ImageVaeSpatialPolicy::comfy_default(4096)
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageMetadataReadLimits {
    pub configuration_bytes: u64,
    pub tokenizer_bytes: u64,
    pub total_bytes: u64,
}
impl Default for PH3ImageMetadataReadLimits {
    fn default() -> Self {
        let limits = PipelineSourceLimits::default();
        Self {
            configuration_bytes: limits.configuration_bytes(),
            tokenizer_bytes: limits.tokenizer_bytes(),
            total_bytes: limits.total_bytes(),
        }
    }
}
impl PH3ImageMetadataReadLimits {
    pub fn resolved(self) -> Result<PipelineSourceLimits> {
        Ok(PipelineSourceLimits::new(
            self.configuration_bytes,
            self.tokenizer_bytes,
            self.total_bytes,
        )?)
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PH3ImageBackendConfig {
    Cuda(PH3ImageCudaBackendConfig),
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageCudaBackendConfig {
    pub ordinal: usize,
    pub usable_memory_bytes: u64,
    pub bootstrap_memory_bytes: u64,
    pub stream_memory_bytes: u64,
    #[serde(default)]
    pub maximum_streams: ImageStreamLimit,
    /// Retain at most this many retired native stream/BLAS lanes within the
    /// original stream allowance. Omitted documents keep reuse disabled.
    #[serde(default)]
    pub native_lane_reuse_maximum_idle_lanes: Option<std::num::NonZeroUsize>,
    /// Reuse retired native storage, with a fixed reservation inside the same
    /// device ceiling. Every hit still receives fresh Core initialization.
    #[serde(default)]
    pub native_allocation_reuse: Option<PH3ImageNativeAllocationReuseConfig>,
    #[serde(default)]
    pub pinned_weight_upload:
        Option<ferrum_engine::image_generation::p_h3::PH3ImagePinnedWeightUpload>,
    pub maximum_in_flight_per_stream: usize,
    pub maximum_commands_per_submission: usize,
    pub opaque_memory_profile: String,
    #[serde(default)]
    pub prompt_expansion_causal: Option<PH3ImageCausalModuleConfig>,
    pub cudnn: CudnnConfig,
    /// Explicit DiT/refiner arithmetic; omitted documents keep sequential attention.
    #[serde(default)]
    pub attention_policy: ferrum_engine::image_generation::ImageCudaAttentionPolicy,
    /// F32 vision/decoder arithmetic, independent of the DiT attention policy.
    #[serde(default)]
    pub f32_attention_policy: ferrum_engine::image_generation::p_h3::PH3ImageF32AttentionPolicy,
    /// BF16 ConvRot output; independent of attention arithmetic and request batching.
    #[serde(default)]
    pub convrot_output_policy: ferrum_engine::image_generation::p_h3::PH3ImageConvRotOutputPolicy,
    pub host: HostConfig,
    pub pipeline: PipelineConfig,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageNativeAllocationReuseConfig {
    pub maximum_idle_bytes: std::num::NonZeroU64,
    pub maximum_entry_bytes: std::num::NonZeroU64,
    pub maximum_entries: std::num::NonZeroUsize,
}
impl PH3ImageNativeAllocationReuseConfig {
    fn validate(self) -> Result<()> {
        ensure!(
            self.maximum_entry_bytes <= self.maximum_idle_bytes
                && self.maximum_idle_bytes.get() <= isize::MAX as u64
                && self.maximum_entries.get() <= isize::MAX as usize,
            "native allocation reuse needs an addressable entry limit within its fixed byte budget"
        );
        Ok(())
    }
    #[cfg(feature = "cuda")]
    fn resolved(self) -> ferrum_kernels::backend::cuda::vnext_runtime::CudaNativeAllocationReuse {
        ferrum_kernels::backend::cuda::vnext_runtime::CudaNativeAllocationReuse::Bounded {
            maximum_idle_bytes: self.maximum_idle_bytes,
            maximum_entry_bytes: self.maximum_entry_bytes,
            maximum_entries: self.maximum_entries,
        }
    }
}
impl PH3ImageProductConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let path = std::fs::canonicalize(path)
            .with_context(|| format!("resolve H3 product document {}", path.display()))?;
        let bytes = super::source::read_bounded(&path, 1 << 20)?;
        Self::from_bytes(&path, &bytes)
    }
    pub(super) fn from_bytes(path: &Path, bytes: &[u8]) -> Result<Self> {
        let mut config: Self =
            serde_json::from_slice(bytes).context("invalid typed H3 product document")?;
        config.validate_paths()?;
        let base = path.parent().context("H3 product document has no parent")?;
        config.resolve_paths(base);
        config.validate()?;
        let PH3ImageBackendConfig::Cuda(backend) = &config.backend;
        if let Some(serving) = &mut config.serving {
            serving.resolve_pending_bytes(&backend.host)?;
            serving.validate()?;
            ensure!(
                serving.max_num_seqs <= backend.pipeline.maximum_batch_size,
                "H3 serving ceiling exceeds the declared pipeline batch capacity"
            );
        }
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.model.trim().is_empty()
                && self.model.len() <= 160
                && !self.model.chars().any(char::is_control),
            "public H3 model name must be nonempty and bounded"
        );
        self.validate_paths()?;
        self.metadata_limits.resolved()?;
        let PH3ImageBackendConfig::Cuda(backend) = &self.backend;
        self.prompt_expansion.validate()?;
        ensure!(
            self.main_vision_reuse == PH3ImageMainVisionReuse::Off
                || self.prompt_expansion.is_enabled(),
            "main vision cohort reuse requires enabled cohort prompt expansion"
        );
        self.validate_prompt_expansion_diagnostics()?;
        ensure!(self.prompt_expansion.is_enabled() == backend.prompt_expansion_causal.is_some(),
            "enabled H3 prompt expansion requires an explicit causal-module allowance; disabled mode must omit it");
        backend.validate()?;
        if let (PH3ImagePromptExpansionConfig::Enabled(expansion), Some(causal)) =
            (&self.prompt_expansion, &backend.prompt_expansion_causal)
        {
            ensure!(
                causal.precision != ferrum_models::vnext::qwen35::Qwen35DensePrecision::Bf16
                    || expansion.mtp
                        == ferrum_engine::image_generation::p_h3::PH3ImagePromptExpansionMtp::Off,
                "BF16 prompt expansion currently requires MTP off"
            );
        }
        // Tile shape/alignment is checked against the actual VAE architecture
        // by prepare_decoder/encoder. Reject malformed policy values now too.
        if let PH3ImageVaeSpatialPolicy::Tiled {
            tile_pixels,
            minimum_overlap_pixels,
            max_tiles,
        } = self.vae_spatial_policy
        {
            ensure!(
                tile_pixels > 0
                    && minimum_overlap_pixels > 0
                    && minimum_overlap_pixels < tile_pixels
                    && max_tiles > 0,
                "invalid declared H3 VAE spatial policy"
            );
        }
        Ok(())
    }
    fn validate_prompt_expansion_diagnostics(&self) -> Result<()> {
        self.prompt_expansion_diagnostics.validate()?;
        if self.prompt_expansion_diagnostics != PH3ImagePromptExpansionDiagnosticPolicy::Disabled {
            ensure!(
                self.prompt_expansion.is_enabled(),
                "prompt expansion diagnostics require enabled native prompt expansion"
            );
            let trace = self.continuous_trace.as_ref().context(
                "prompt expansion diagnostics require continuous_trace in the same product document",
            )?;
            if let PH3ImagePromptExpansionDiagnosticPolicy::CaptureEffectivePrompt {
                maximum_utf8_bytes,
            } = self.prompt_expansion_diagnostics
            {
                ensure!(
                    maximum_utf8_bytes <= trace.maximum_bytes,
                    "effective prompt capture limit exceeds the continuous trace capacity"
                );
            }
        }
        Ok(())
    }
    fn validate_paths(&self) -> Result<()> {
        self.prompt_expansion.validate_paths()?;
        // Validate before relative resolution can turn an empty path into
        // the configuration directory itself.
        if let Some(trace) = &self.continuous_trace {
            trace.validate()?;
        }
        for path in [
            &self.metadata_directory,
            &self.archives.text_encoder,
            &self.archives.transformer,
            &self.archives.vae,
        ] {
            ensure!(
                !path.as_os_str().is_empty(),
                "H3 metadata and explicit archive paths must be nonempty"
            );
        }
        let PH3ImageBackendConfig::Cuda(backend) = &self.backend;
        ensure!(
            !backend.cudnn.library_path.as_os_str().is_empty()
                && backend
                    .cudnn
                    .dependency_libraries
                    .iter()
                    .all(|p| !p.as_os_str().is_empty()),
            "H3 explicit native library paths must be nonempty"
        );
        Ok(())
    }
    /// Called by the common product factory before creating any device runtime.
    /// Original mapped sources are kept; no BF16/FP8 materializer is substituted.
    pub fn prepare_assets(&self) -> Result<PH3ImageAssets> {
        self.validate()?;
        let PH3ImageBackendConfig::Cuda(backend) = &self.backend;
        let mut expansion = self.prompt_expansion.prepare_sources(backend.ordinal)?;
        if let (Some((_, _, options, _, _)), Some(causal)) =
            (&mut expansion, &backend.prompt_expansion_causal)
        {
            causal.configure_expansion(options);
        }
        let assets = PH3ImageAssets::from_directory(
            &self.metadata_directory,
            self.archives.clone(),
            self.metadata_limits.resolved()?,
            PH3ImageAssetLimits::default(),
        )?;
        let assets = match expansion {
            Some((sources, limits, options, processor, vision_limits)) => assets
                .with_prompt_expansion_visual_sources(
                    sources,
                    limits,
                    options,
                    processor,
                    vision_limits,
                ),
            None => Ok(assets),
        }?;
        self.prompt_expansion
            .apply_weight_configuration(assets)?
            .with_main_vision_reuse(self.main_vision_reuse)
    }
    fn resolve_paths(&mut self, base: &Path) {
        let resolve = |path: &mut PathBuf| {
            if path.is_relative() {
                *path = base.join(&*path);
            }
        };
        resolve(&mut self.metadata_directory);
        resolve(&mut self.archives.text_encoder);
        resolve(&mut self.archives.transformer);
        resolve(&mut self.archives.vae);
        self.prompt_expansion.resolve_paths(base);
        if let Some(trace) = &mut self.continuous_trace {
            trace.resolve_path(base);
        }
        let PH3ImageBackendConfig::Cuda(backend) = &mut self.backend;
        resolve(&mut backend.cudnn.library_path);
        for path in &mut backend.cudnn.dependency_libraries {
            resolve(path);
        }
    }
}
impl PH3ImageCudaBackendConfig {
    /// Pure wire-policy validation. The engine constructor performs complete
    /// native command-record and actual-family checks before its sole context.
    pub fn validate(&self) -> Result<()> {
        let limits = self.pipeline.resolved_limits()?;
        if let Some(policy) = self.pinned_weight_upload {
            policy.validate()?;
        }
        if let Some(reuse) = self.native_allocation_reuse {
            reuse.validate()?;
        }
        ensure!(
            limits.decoder_batch_policy == ImageDecoderBatchPolicy::SingleImage
                && limits.edit_prefix_policy == ImageEditPrefixPolicy::Recompute,
            "H3 requires single-image VAE decoding and recomputed bidirectional editing"
        );
        ensure!(
            self.usable_memory_bytes > 0
                && self.bootstrap_memory_bytes > 0
                && self.stream_memory_bytes > 0,
            "H3 CUDA memory allowances must be positive"
        );
        i32::try_from(self.ordinal).context("CUDA ordinal exceeds driver range")?;
        let requirements = ImageStreamRequirements::for_serial_pipeline(
            limits.static_initialization,
            limits.executor,
        );
        let requirements = if let Some(causal) = &self.prompt_expansion_causal {
            causal.validate()?;
            requirements.with_serial_causal_phase()
        } else {
            requirements
        };
        let streams = self
            .maximum_streams
            .resolve(requirements)?
            .resolved_streams();
        ensure!(
            self.native_lane_reuse_maximum_idle_lanes
                .is_none_or(|maximum| maximum.get() <= streams),
            "H3 native lane reuse exceeds the resolved stream allowance"
        );
        ensure!(
            self.maximum_in_flight_per_stream > 0
                && self.maximum_commands_per_submission > 0
                && self.maximum_commands_per_submission <= u32::MAX as usize,
            "H3 CUDA flight/command ceilings must be positive and bounded"
        );
        streams
            .checked_mul(self.maximum_in_flight_per_stream)
            .and_then(|n| n.checked_mul(self.maximum_commands_per_submission))
            .filter(|&n| n <= isize::MAX as usize)
            .context("H3 command capacity overflows")?;
        ensure!(
            limits.static_initialization.maximum_commands_per_batch()
                <= self.maximum_commands_per_submission,
            "H3 static command batch exceeds submission limit"
        );
        self.host.limits()?;
        ensure!(self.host.weight_backing_bytes>0 && self.host.input_preprocess_bytes>=16
            && self.host.output_bytes>0 && u64::try_from(self.host.transfer_staging_bytes)?
                >=limits.static_initialization.maximum_staging_bytes(),
            "H3 requires explicit source backing, preprocessing, output and static staging Host allowances");
        ensure!(
            self.cudnn.library_path.is_absolute()
                && self.cudnn.dependency_libraries.len() <= 16
                && self
                    .cudnn
                    .dependency_libraries
                    .iter()
                    .all(|p| p.is_absolute()),
            "H3 cuDNN needs absolute library paths with at most 16 dependencies"
        );
        ensure!(
            self.cudnn.opaque_memory_bytes > 0
                && self.cudnn.workspace_limit_bytes <= isize::MAX as u64,
            "H3 cuDNN memory allowances are invalid"
        );
        for value in [
            &self.opaque_memory_profile,
            &self.cudnn.opaque_memory_profile,
        ] {
            ensure!(
                !value.trim().is_empty()
                    && value.len() <= 256
                    && value.is_ascii()
                    && !value.bytes().any(|b| b.is_ascii_control()),
                "H3 opaque-memory profiles must be bounded ASCII"
            );
        }
        let opaque = self
            .stream_memory_bytes
            .checked_add(
                self.prompt_expansion_causal
                    .as_ref()
                    .and_then(|p| p.decode_replay)
                    .map_or(0, |r| r.opaque_memory_bytes_per_stream),
            )
            .and_then(|bytes| bytes.checked_mul(u64::try_from(streams).ok()?))
            .and_then(|n| n.checked_add(self.bootstrap_memory_bytes))
            .and_then(|n| n.checked_add(self.cudnn.opaque_memory_bytes))
            .and_then(|n| n.checked_add(self.attention_policy.additional_opaque_bytes()))
            .and_then(|n| n.checked_add(self.f32_attention_policy.additional_opaque_bytes()))
            .and_then(|n| n.checked_add(self.convrot_output_policy.additional_opaque_bytes()))
            .and_then(|n| {
                n.checked_add(
                    self.prompt_expansion_causal
                        .as_ref()
                        .map_or(0, |c| c.module_memory_bytes),
                )
            })
            .and_then(|n| {
                n.checked_add(
                    self.native_allocation_reuse
                        .map_or(0, |p| p.maximum_idle_bytes.get()),
                )
            })
            .context("H3 opaque memory total overflows")?;
        ensure!(
            opaque < self.usable_memory_bytes,
            "H3 native allowances leave no tensor storage"
        );
        if self.cudnn.convolution_workspace
            == ferrum_engine::image_generation::ImageCudaConvolutionWorkspacePolicy::ConfiguredLimit
        {
            ensure!(
                self.cudnn.workspace_limit_bytes < self.usable_memory_bytes - opaque,
                "H3 cuDNN workspace ceiling leaves no tensor storage"
            );
        }
        Ok(())
    }
    #[cfg(feature = "cuda")]
    pub fn resolved(
        &self,
        spatial_policy: PH3ImageVaeSpatialPolicy,
        decoder_tile_batch: PH3ImageVaeTileBatch,
    ) -> Result<ferrum_engine::image_generation::p_h3::cuda::PH3ImageCudaConfig> {
        use ferrum_engine::image_generation::{
            cuda::ImageCudaMemoryPolicy,
            p_h3::{cuda::PH3ImageCudaConfig, PH3ImagePipelineOptions},
        };
        use ferrum_kernels::backend::cuda::{
            vnext_ops::CUDA_PHILOX_NORMAL_F32_NUMERICAL_PROFILE_ID,
            vnext_runtime::{CudaNativeAllocationReuse, CudaNativeLaneReuse},
            CudaCudnnConfig,
        };
        self.validate()?;
        if let Some(policy) = self.pinned_weight_upload {
            policy.validate()?;
        }
        let mut options = PH3ImagePipelineOptions::comfy_f32(
            ferrum_runtime::NumericalProfileId::new(CUDA_PHILOX_NORMAL_F32_NUMERICAL_PROFILE_ID)
                .map_err(anyhow::Error::msg)?,
            4096,
        )?;
        options.vae_spatial_policy = spatial_policy;
        options.vae_decoder_tile_batch = decoder_tile_batch;
        let config = PH3ImageCudaConfig {
            ordinal: self.ordinal,
            memory: ImageCudaMemoryPolicy::ExplicitTotal(
                std::num::NonZeroU64::new(self.usable_memory_bytes).unwrap(),
            ),
            bootstrap_memory_bytes: self.bootstrap_memory_bytes,
            stream_memory_bytes: self.stream_memory_bytes,
            maximum_streams: self.maximum_streams,
            native_lane_reuse: self
                .native_lane_reuse_maximum_idle_lanes
                .map_or(CudaNativeLaneReuse::Off, |maximum_idle_lanes| {
                    CudaNativeLaneReuse::Bounded { maximum_idle_lanes }
                }),
            native_allocation_reuse: self.native_allocation_reuse.map_or(
                CudaNativeAllocationReuse::Off,
                PH3ImageNativeAllocationReuseConfig::resolved,
            ),
            pinned_weight_upload: self.pinned_weight_upload,
            maximum_in_flight_per_stream: self.maximum_in_flight_per_stream,
            maximum_commands_per_submission: self.maximum_commands_per_submission,
            opaque_memory_profile: self.opaque_memory_profile.clone(),
            prompt_expansion_causal: self
                .prompt_expansion_causal
                .as_ref()
                .map(PH3ImageCausalModuleConfig::resolved)
                .transpose()?,
            cudnn: CudaCudnnConfig {
                library_path: super::library::resolve(Some(&self.cudnn.library_path))?,
                dependency_libraries: self.cudnn.dependency_libraries.clone(),
                opaque_memory_bytes: self.cudnn.opaque_memory_bytes,
                workspace_limit_bytes: self.cudnn.workspace_limit_bytes,
                opaque_memory_profile: self.cudnn.opaque_memory_profile.clone(),
            },
            convolution_workspace: self.cudnn.convolution_workspace,
            attention_policy: self.attention_policy,
            f32_attention_policy: self.f32_attention_policy,
            convrot_output_policy: self.convrot_output_policy,
            host: self.host.limits()?,
            pipeline: self.pipeline.resolved_limits()?,
            options,
        };
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests;
