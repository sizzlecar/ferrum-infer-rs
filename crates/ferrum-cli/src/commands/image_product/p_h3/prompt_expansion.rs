//! Explicit native expansion sources and bounded policies. Source identities
//! remain separate; loading never downloads a model or executes source code.
use anyhow::{ensure, Context, Result};
use ferrum_engine::image_generation::p_h3::{
    PH3ImagePromptExpansionOptions, PH3ImagePromptExpansionSourceBacking,
    PH3ImagePromptExpansionVisionReuse,
};
use ferrum_models::{
    p_h3_image::PH3ImagePromptExpansionLimits,
    qwen3_vl_vision::hf::{Qwen3VLVisionHfLimits, Qwen3VLVisionHfProcessor},
    vnext::{
        open_registered_product_sources, resolve_registered_model_from_sources,
        ProductionModelSourceBundle, ProductionWeightArtifact,
    },
};
mod vision;
mod weight_cache;
use ferrum_runtime::{ElementType, ModelSourceKind, OriginalModelSource, OriginalModelSources};
use serde::Deserialize;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
pub use vision::PH3ImagePromptExpansionVisionLimits;
pub use weight_cache::PH3ImagePromptExpansionWeightCacheConfig;

#[derive(Debug)]
pub enum PH3ImagePromptExpansionConfig {
    Enabled(PH3ImagePromptExpansionEnabledConfig),
    Disabled,
}
impl<'de> Deserialize<'de> for PH3ImagePromptExpansionConfig {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Empty {}
        #[derive(Deserialize)]
        #[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
        enum Wire {
            Enabled(PH3ImagePromptExpansionEnabledConfig),
            Disabled(Empty),
        }
        Ok(match Wire::deserialize(deserializer)? {
            Wire::Enabled(config) => Self::Enabled(config),
            Wire::Disabled(_) => Self::Disabled,
        })
    }
}
impl Default for PH3ImagePromptExpansionConfig {
    fn default() -> Self {
        Self::Enabled(PH3ImagePromptExpansionEnabledConfig::default())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImagePromptExpansionEnabledConfig {
    pub semantic_directory: PathBuf,
    pub tokenizer_directory: PathBuf,
    pub bf16_archive: PathBuf,
    /// Original upsampler processor metadata, selected independently of the
    /// main H3 text encoder's processor and of the public model alias.
    pub processor_config: PathBuf,
    #[serde(default)]
    pub vision_limits: PH3ImagePromptExpansionVisionLimits,
    /// Reuse a completed visual output for an identical image within one
    /// expansion cohort. This does not cache outputs across cohorts.
    #[serde(default)]
    pub vision_reuse: PH3ImagePromptExpansionVisionReuse,
    #[serde(default)]
    pub mtp: ferrum_engine::image_generation::p_h3::PH3ImagePromptExpansionMtp,
    #[serde(default = "default_context")]
    pub max_context_tokens: usize,
    #[serde(default = "default_output")]
    pub max_new_tokens: u32,
    #[serde(default = "default_chunk")]
    pub prefill_chunk_tokens: usize,
    #[serde(default)]
    pub metadata_limits: PH3ImagePromptExpansionMetadataLimits,
    /// Optional process-private cache of completed execution representations.
    #[serde(default)]
    pub weight_cache: PH3ImagePromptExpansionWeightCacheConfig,
    /// Conversion allocation policy for one static installation, not a cache
    /// of completed weights across requests or a change in execution precision.
    #[serde(default)]
    pub source_backing: PH3ImagePromptExpansionSourceBacking,
    /// Maximum workers converting one bounded source range. Values above one
    /// require bounded_streaming; all workers share its existing Host slot.
    #[serde(default = "default_source_conversion_workers")]
    pub source_conversion_workers: std::num::NonZeroUsize,
    /// Target allocation size for causal weight arenas; larger tensors remain
    /// contiguous in their own arena. Omitted keeps the original layout.
    #[serde(default)]
    pub weight_arena_target_bytes: Option<std::num::NonZeroU64>,
}
fn default_context() -> usize {
    8192
}
fn default_output() -> u32 {
    1024
}
fn default_chunk() -> usize {
    256
}
fn default_source_conversion_workers() -> std::num::NonZeroUsize {
    std::num::NonZeroUsize::MIN
}
impl Default for PH3ImagePromptExpansionEnabledConfig {
    fn default() -> Self {
        Self {
            semantic_directory: PathBuf::new(),
            tokenizer_directory: PathBuf::new(),
            bf16_archive: PathBuf::new(),
            processor_config: PathBuf::new(),
            vision_limits: Default::default(),
            vision_reuse: Default::default(),
            mtp: Default::default(),
            max_context_tokens: default_context(),
            max_new_tokens: default_output(),
            prefill_chunk_tokens: default_chunk(),
            metadata_limits: Default::default(),
            weight_cache: Default::default(),
            source_backing: Default::default(),
            source_conversion_workers: default_source_conversion_workers(),
            weight_arena_target_bytes: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PH3ImagePromptExpansionMetadataLimits {
    pub max_config_bytes: usize,
    pub max_processor_bytes: usize,
    pub max_tokenizer_bytes: usize,
    pub max_template_bytes: usize,
    pub max_request_bytes: usize,
    pub max_rendered_bytes: usize,
    pub max_answer_bytes: usize,
    pub template_fuel: u64,
}
impl Default for PH3ImagePromptExpansionMetadataLimits {
    fn default() -> Self {
        Self {
            max_config_bytes: 4 << 20,
            max_processor_bytes: 4 << 20,
            max_tokenizer_bytes: 64 << 20,
            max_template_bytes: 1 << 20,
            max_request_bytes: 1 << 20,
            max_rendered_bytes: 4 << 20,
            max_answer_bytes: 4 << 20,
            template_fuel: 1_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageCausalDecodeReplayConfig {
    pub maximum_executables: usize,
    pub opaque_memory_bytes_per_stream: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageCausalModuleConfig {
    pub module_memory_bytes: u64,
    pub opaque_memory_profile: String,
    #[serde(default)]
    pub decode_replay: Option<PH3ImageCausalDecodeReplayConfig>,
}
impl PH3ImageCausalModuleConfig {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(self.module_memory_bytes > 0 && !self.opaque_memory_profile.trim().is_empty()
            && self.opaque_memory_profile.len() <= 256 && self.opaque_memory_profile.is_ascii()
            && !self.opaque_memory_profile.bytes().any(|b| b.is_ascii_control()),
            "prompt expansion needs a positive explicit causal-module allowance and bounded ASCII profile");
        if let Some(replay) = self.decode_replay {
            ensure!(replay.maximum_executables > 0 && replay.maximum_executables <= u32::MAX as usize && replay.opaque_memory_bytes_per_stream > 0 && replay.opaque_memory_bytes_per_stream <= isize::MAX as u64, "causal decode replay needs bounded positive graph slots and per-stream opaque bytes");
        }
        Ok(())
    }
    pub(super) fn configure_expansion(&self, options: &mut PH3ImagePromptExpansionOptions) {
        if let Some(replay) = self.decode_replay {
            options.engine.backend.enable_reusable_execution = true;
            options.engine.backend.reusable_execution_capture.scope =
                ferrum_types::ReusableExecutionCaptureScope::DecodeOnly;
            options
                .engine
                .backend
                .reusable_execution_capture
                .preparation = ferrum_types::ReusableExecutionPreparationMode::OnDemand;
            options
                .engine
                .backend
                .reusable_execution_capture
                .maximum_executables_per_lane = Some(replay.maximum_executables as u64);
        }
    }
    #[cfg(feature = "cuda")]
    pub(super) fn resolved(
        &self,
    ) -> Result<ferrum_kernels::backend::cuda::CudaDenseF16CausalProfile> {
        self.validate()?;
        let profile = ferrum_kernels::backend::cuda::CudaDenseF16CausalProfile::new(
            self.module_memory_bytes,
            self.opaque_memory_profile.clone(),
        )
        .map_err(anyhow::Error::msg)?;
        match self.decode_replay {
            Some(replay) => profile
                .with_decode_replay(
                    replay.maximum_executables,
                    replay.opaque_memory_bytes_per_stream,
                )
                .map_err(anyhow::Error::msg),
            None => Ok(profile),
        }
    }
}

type PreparedExpansionSources = (
    Arc<ProductionModelSourceBundle>,
    PH3ImagePromptExpansionLimits,
    PH3ImagePromptExpansionOptions,
    Qwen3VLVisionHfProcessor,
    Qwen3VLVisionHfLimits,
);

impl PH3ImagePromptExpansionConfig {
    pub(super) fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled(_))
    }
    pub(super) fn validate_paths(&self) -> Result<()> {
        if let Self::Enabled(config) = self {
            ensure!([&config.semantic_directory, &config.tokenizer_directory, &config.bf16_archive, &config.processor_config]
                .iter().all(|p| !p.as_os_str().is_empty()),
                "enabled H3 prompt expansion requires explicit semantic_directory, tokenizer_directory, bf16_archive and processor_config; raw prompts require mode=disabled");
        }
        Ok(())
    }
    pub(super) fn validate(&self) -> Result<()> {
        self.validate_paths()?;
        if let Self::Enabled(config) = self {
            config.weight_cache.validate()?;
            ferrum_quantization::Bf16F16ConversionPolicy::new(config.source_conversion_workers)?;
            ensure!(
                config.source_conversion_workers.get() == 1
                    || config.source_backing
                        == PH3ImagePromptExpansionSourceBacking::BoundedStreaming,
                "parallel source conversion requires bounded_streaming backing"
            );
            ensure!(
                config.source_backing == PH3ImagePromptExpansionSourceBacking::Fresh
                    || matches!(
                        config.weight_cache,
                        PH3ImagePromptExpansionWeightCacheConfig::Disabled
                    ),
                "installation-local source backing cannot be combined with process-mapped weights"
            );
            let output = usize::try_from(config.max_new_tokens)?;
            ensure!(output > 0 && config.max_context_tokens > output
                && config.prefill_chunk_tokens > 0 && config.prefill_chunk_tokens <= config.max_context_tokens,
                "prompt expansion context must exceed positive output tokens and admit its positive prefill chunk");
            ensure!(
                config
                    .max_context_tokens
                    .checked_mul(4)
                    .is_some_and(|n| n <= isize::MAX as usize),
                "prompt expansion context exceeds address space"
            );
            let limits = config.metadata_limits;
            ensure!(
                [
                    limits.max_config_bytes,
                    limits.max_processor_bytes,
                    limits.max_tokenizer_bytes,
                    limits.max_template_bytes,
                    limits.max_request_bytes,
                    limits.max_rendered_bytes,
                    limits.max_answer_bytes
                ]
                .into_iter()
                .all(|n| n > 0 && n <= isize::MAX as usize)
                    && limits.template_fuel > 0,
                "prompt expansion metadata/request limits must be positive and addressable"
            );
            ensure!(
                config.bf16_archive.extension().and_then(|s| s.to_str()) == Some("safetensors"),
                "prompt expansion bf16_archive must explicitly select one safetensors file"
            );
            config
                .vision_limits
                .resolved(config.metadata_limits.max_processor_bytes)?;
        }
        Ok(())
    }
    pub(super) fn resolve_paths(&mut self, base: &Path) {
        if let Self::Enabled(config) = self {
            config.weight_cache.resolve_paths(base);
            for path in [
                &mut config.semantic_directory,
                &mut config.tokenizer_directory,
                &mut config.bf16_archive,
                &mut config.processor_config,
            ] {
                if path.is_relative() {
                    *path = base.join(&*path);
                }
            }
        }
    }
    pub(super) fn apply_weight_configuration(
        &self,
        assets: ferrum_engine::image_generation::p_h3::PH3ImageAssets,
    ) -> Result<ferrum_engine::image_generation::p_h3::PH3ImageAssets> {
        match self {
            Self::Enabled(config) => config
                .weight_cache
                .apply(assets)?
                .with_prompt_expansion_source_backing(config.source_backing)?
                .with_prompt_expansion_source_conversion_policy(
                    ferrum_quantization::Bf16F16ConversionPolicy::new(
                        config.source_conversion_workers,
                    )?,
                )?
                .with_prompt_expansion_vision_reuse(config.vision_reuse)?
                .with_prompt_expansion_mtp(config.mtp),
            Self::Disabled => Ok(assets),
        }
    }
    pub(super) fn prepare_sources(
        &self,
        ordinal: usize,
    ) -> Result<Option<PreparedExpansionSources>> {
        self.validate()?;
        let Self::Enabled(config) = self else {
            return Ok(None);
        };
        config.validate_metadata_extents()?;
        let (processor, vision_limits) = config.prepare_processor()?;
        let local = |kind, path: &Path| OriginalModelSource {
            kind,
            location: path.display().to_string(),
            requested_revision: None,
        };
        let sources = Arc::new(open_registered_product_sources(
            &config.semantic_directory,
            &config.tokenizer_directory,
            ProductionWeightArtifact::safetensors_file(&config.bf16_archive),
            OriginalModelSources {
                semantic: local(ModelSourceKind::LocalDirectory, &config.semantic_directory),
                tokenizer: local(ModelSourceKind::LocalDirectory, &config.tokenizer_directory),
                weights: local(ModelSourceKind::LocalFile, &config.bf16_archive),
            },
        )?);
        // Validate the declared BF16 storage against only actually executed
        // language schema sources. Nonexecuted vision/MTP inventory is neither
        // loaded as a replacement nor constrained by an unrelated dtype rule.
        let defined = resolve_registered_model_from_sources(&sources)?
            .into_required()?
            .define_from_sources(Arc::clone(&sources))?;
        let profile = ferrum_models::vnext::qwen35::F16_NUMERICAL_PROFILE_ID
            .parse()
            .map_err(anyhow::Error::msg)?;
        let prepared = defined.prepare(&profile)?;
        let archive = sources.weights().open_safetensors_archive()?;
        for component in prepared
            .family()
            .stored_weight_schema()?
            .components
            .iter()
            .filter(|c| c.required)
        {
            for name in &component.external_names {
                ensure!(archive.tensor(name)?.element_type() == Some(ElementType::Bf16),
                    "declared prompt expansion BF16 archive has a non-BF16 required language tensor `{name}`");
            }
        }
        Ok(Some((
            sources,
            config.protocol_limits(),
            config.options(ordinal)?,
            processor,
            vision_limits,
        )))
    }
}

impl PH3ImagePromptExpansionEnabledConfig {
    pub(super) fn options(&self, ordinal: usize) -> Result<PH3ImagePromptExpansionOptions> {
        i32::try_from(ordinal).context("prompt expansion CUDA ordinal exceeds driver range")?;
        let mut engine = ferrum_types::EngineConfig::default();
        engine.backend.device = ferrum_types::Device::CUDA(ordinal);
        engine.runtime.max_model_len = Some(self.max_context_tokens);
        engine.memory.weight_arena_target_bytes = self.weight_arena_target_bytes;
        // Explicit exact profile and causal executor policy; source storage is
        // BF16, execution uses the existing supported F16 ABI.
        engine.numerical_execution = ferrum_models::vnext::qwen35::F16_NUMERICAL_PROFILE_ID
            .parse()
            .map_err(anyhow::Error::msg)?;
        engine.backend.enable_reusable_execution = false;
        Ok(PH3ImagePromptExpansionOptions {
            engine,
            max_new_tokens: self.max_new_tokens,
            prefill_chunk_tokens: self.prefill_chunk_tokens,
        })
    }
    fn protocol_limits(&self) -> PH3ImagePromptExpansionLimits {
        let limits = self.metadata_limits;
        PH3ImagePromptExpansionLimits {
            max_tokenizer_bytes: limits.max_tokenizer_bytes,
            max_template_bytes: limits.max_template_bytes,
            max_request_bytes: limits.max_request_bytes,
            max_rendered_bytes: limits.max_rendered_bytes,
            max_answer_bytes: limits.max_answer_bytes,
            max_generated_tokens: self.max_new_tokens,
            template_fuel: limits.template_fuel,
        }
    }
    fn validate_metadata_extents(&self) -> Result<()> {
        let check = |path: PathBuf, maximum: usize, required: bool| -> Result<()> {
            match std::fs::metadata(&path) {
                Ok(metadata) => ensure!(metadata.is_file() && metadata.len() > 0 && metadata.len() <= u64::try_from(maximum)?,
                    "prompt expansion metadata {} exceeds its declared limit or is not a nonempty file", path.display()),
                Err(error) if !required && error.kind() == std::io::ErrorKind::NotFound => {},
                Err(error) => return Err(error).with_context(|| format!("read original prompt expansion metadata {}", path.display())),
            }
            Ok(())
        };
        let limits = self.metadata_limits;
        check(
            self.semantic_directory.join("config.json"),
            limits.max_config_bytes,
            true,
        )?;
        check(
            self.tokenizer_directory.join("tokenizer.json"),
            limits.max_tokenizer_bytes,
            true,
        )?;
        check(
            self.tokenizer_directory.join("chat_template.jinja"),
            limits.max_template_bytes,
            true,
        )?;
        for name in [
            "tokenizer_config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "chat_template.json",
        ] {
            check(
                self.tokenizer_directory.join(name),
                limits.max_config_bytes,
                name == "tokenizer_config.json",
            )?;
        }
        Ok(())
    }
}
