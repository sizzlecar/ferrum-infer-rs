//! Device-free source and cross-component validation for the native Comfy
//! route. Every archive is supplied explicitly; metadata registrations are
//! checked as data and never followed to download or import executable code.
use super::metadata::ComponentDeclarations;
use crate::image_generation::ImageAssetReceipt;
use anyhow::{ensure, Context, Result};
use ferrum_models::{
    p_h3_image::{
        PH3ImageDenoiserAdapter, PH3ImageDenoiserLimits, PH3ImageSamplingLimits,
        PH3ImageSamplingSchedule, PH3ImageTextConditioningConfig, PH3ImageTextConditioningFamily,
        PH3ImageTextConditioningLimits, PH3ImageTextConnectorCheckpoint,
        PH3ImageTextConnectorCheckpointLayout, PH3ImageTextConnectorConfig,
        PH3ImageTextConnectorProjectionFamily, PH3ImageTextConnectorProjectionLimits,
        PH3ImageTextTokenizer, PH3ImageTextTokenizerLimits, PH3ImageTextWeightAdapter,
        PH3ImageTransformerConfig,
    },
    p_h3_image_vae::{PH3ImageVaeAdapter, PH3ImageVaeAdapterLimits, PH3ImageVaeDecoderConfig},
    qwen3_vl_text::{Qwen3VLTextArchitecture, Qwen3VLTextKeyRun, Qwen3VLTextLayout},
    qwen3_vl_vision::{
        comfy::{
            Qwen3VLVisionComfyConfig, Qwen3VLVisionComfyLimits, Qwen3VLVisionComfyWeightAdapter,
        },
        Qwen3VLVisionArchitecture, Qwen3VLVisionGrid,
    },
};
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use ferrum_runtime::{ModelFamilyProvider, WeightSchema, STRUCTURED_ATTENTION_BF16_MAX_BATCH};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};

/// Original bytes, including the actual `modular_model_index.json` entrypoint.
pub struct PH3ImageMetadata<'a> {
    pub model_index: &'a [u8],
    pub text_encoder: &'a [u8],
    pub text_connector: &'a [u8],
    pub transformer: &'a [u8],
    pub vae: &'a [u8],
    pub scheduler: &'a [u8],
    pub audio_scheduler: &'a [u8],
    pub tokenizer: &'a [u8],
    pub tokenizer_config: &'a [u8],
}

/// The text archive also contains the F32 connector. Each Arc retains one
/// original mapped source; no expanded/dequantized shadow model is produced.
pub struct PH3ImageSources {
    pub text_encoder: Arc<ComfyCanonicalWeightSource>,
    pub transformer: Arc<ComfyCanonicalWeightSource>,
    pub vae: Arc<ComfyCanonicalWeightSource>,
}

#[derive(Clone, Copy, Debug)]
pub struct PH3ImageAssetLimits {
    pub metadata_bytes_per_file: usize,
    pub tokenizer: PH3ImageTextTokenizerLimits,
    pub text: PH3ImageTextConditioningLimits,
    pub vision: Qwen3VLVisionComfyLimits,
    pub connector: PH3ImageTextConnectorProjectionLimits,
    pub denoiser: PH3ImageDenoiserLimits,
    pub vae: PH3ImageVaeAdapterLimits,
    pub sampling: PH3ImageSamplingLimits,
}
impl Default for PH3ImageAssetLimits {
    fn default() -> Self {
        Self {
            metadata_bytes_per_file: 4 << 20,
            tokenizer: PH3ImageTextTokenizerLimits {
                max_source_bytes: 32 << 20,
                max_prompt_bytes: 1 << 20,
                max_tokens_per_prompt: 8192,
                max_batch: STRUCTURED_ATTENTION_BF16_MAX_BATCH,
            },
            text: PH3ImageTextConditioningLimits {
                max_metadata_bytes: 16 << 20,
                max_stored_bytes: 16 << 30,
                max_plan_nodes: 16_384,
            },
            vision: Qwen3VLVisionComfyLimits {
                max_metadata_bytes: 16 << 20,
                max_stored_bytes: 4 << 30,
                max_plan_nodes: 16_384,
            },
            connector: PH3ImageTextConnectorProjectionLimits {
                max_projection_bytes: 1 << 30,
            },
            denoiser: PH3ImageDenoiserLimits {
                max_metadata_bytes: 16 << 20,
                max_stored_bytes: 64 << 30,
                max_plan_nodes: 16_384,
            },
            vae: PH3ImageVaeAdapterLimits {
                max_metadata_bytes: 16 << 20,
                max_stored_bytes: 16 << 30,
                max_plan_nodes: 16_384,
            },
            sampling: PH3ImageSamplingLimits {
                max_metadata_bytes: 4 << 20,
                max_evaluations: 1024,
            },
        }
    }
}

/// Validated architecture and immutable sources. This object grants no device
/// capacity and does not advertise a successfully executed pipeline.
pub struct PH3ImageAssets {
    pub(super) main_vision_reuse: super::PH3ImageMainVisionReuse,
    pub(super) expansion: Option<super::expansion::ConfiguredTextExpansion>,
    pub(super) sources: PH3ImageSources,
    pub(super) text: Qwen3VLTextArchitecture,
    pub(super) text_weights: PH3ImageTextWeightAdapter,
    pub(super) vision: Qwen3VLVisionArchitecture,
    pub(super) vision_weights: Qwen3VLVisionComfyWeightAdapter,
    pub(super) connector: PH3ImageTextConnectorConfig,
    pub(super) tokenizer: PH3ImageTextTokenizer,
    pub(super) denoiser: PH3ImageDenoiserAdapter,
    pub(super) vae: PH3ImageVaeAdapter,
    pub(super) schedule: PH3ImageSamplingSchedule,
    pub(super) declarations: ComponentDeclarations,
    pub(super) limits: PH3ImageAssetLimits,
    receipt: ImageAssetReceipt,
}
impl PH3ImageAssets {
    pub fn prepare(
        metadata: PH3ImageMetadata<'_>,
        sources: PH3ImageSources,
        limits: PH3ImageAssetLimits,
    ) -> Result<Self> {
        let named = [
            ("modular_model_index", metadata.model_index),
            ("text_encoder", metadata.text_encoder),
            ("text_connector", metadata.text_connector),
            ("transformer", metadata.transformer),
            ("vae", metadata.vae),
            ("scheduler", metadata.scheduler),
            ("audio_scheduler", metadata.audio_scheduler),
            ("tokenizer_config", metadata.tokenizer_config),
        ];
        ensure!(
            limits.metadata_bytes_per_file > 0,
            "metadata capacity must be positive"
        );
        for (name, bytes) in named {
            ensure!(
                !bytes.is_empty() && bytes.len() <= limits.metadata_bytes_per_file,
                "{name} metadata exceeds its nonempty declared capacity"
            );
        }
        let declarations =
            ComponentDeclarations::parse(metadata.model_index, limits.metadata_bytes_per_file)?;
        let text = Qwen3VLTextArchitecture::from_hf_value(&serde_json::from_slice(
            metadata.text_encoder,
        )?)?;
        let vision = Qwen3VLVisionArchitecture::from_hf_value(&serde_json::from_slice(
            metadata.text_encoder,
        )?)?;
        ensure!(
            vision.output_size() == text.hidden_size(),
            "visual merger output differs from the text embedding width"
        );
        let vision_config = Qwen3VLVisionComfyConfig::new(
            vision.clone(),
            Qwen3VLVisionGrid::new(1, vision.merge_size(), vision.merge_size())?,
        )?;
        let vision_weights = Qwen3VLVisionComfyWeightAdapter::new(
            &vision_config,
            Arc::clone(&sources.text_encoder),
            limits.vision,
        )?;
        let connector = PH3ImageTextConnectorConfig::from_hf_json(metadata.text_connector)?;
        let transformer = PH3ImageTransformerConfig::from_hf_json(metadata.transformer)?;
        let decoder = PH3ImageVaeDecoderConfig::from_hf_json(metadata.vae)?;
        connector.validate_components(
            u32::try_from(text.layers())?,
            text.hidden_size(),
            &transformer,
        )?;
        ensure!(
            transformer.in_channels() == decoder.latent_channels(),
            "DiT image channels differ from VAE latent channels"
        );
        let tokenizer = PH3ImageTextTokenizer::from_source_bytes(
            metadata.tokenizer,
            metadata.tokenizer_config,
            text.vocabulary_size(),
            limits.tokenizer,
        )?;
        let schedule = PH3ImageSamplingSchedule::parse(
            metadata.model_index,
            metadata.scheduler,
            metadata.audio_scheduler,
            limits.sampling,
        )?;
        // One real token is sufficient to validate the immutable physical
        // text schema; runtime activation geometry is prepared per request.
        let text_config = PH3ImageTextConditioningConfig::from_comfy_checkpoint_with_connector(
            text.clone(),
            Qwen3VLTextLayout::new(1, vec![vec![Qwen3VLTextKeyRun::new(1, true)]])?,
            &sources.text_encoder,
            &connector,
        )?;
        let selected_connector = text_config.connector_config()?;
        // Serialization provenance such as `_diffusers_version` is retained
        // in the metadata digest; only the mathematical contract must match
        // the configuration reconstructed from the selected text encoder.
        ensure!(
            selected_connector.hidden_state_indices() == connector.hidden_state_indices()
                && selected_connector.text_hidden_size() == connector.text_hidden_size()
                && selected_connector.out_channels() == connector.out_channels(),
            "connector metadata differs from the selected encoder taps and projection width"
        );
        let text_weights = PH3ImageTextWeightAdapter::new(
            &text_config,
            Arc::clone(&sources.text_encoder),
            limits.text,
        )?;
        PH3ImageTextConnectorCheckpoint::new_with_layout(
            &connector,
            sources.text_encoder.archive(),
            limits.connector,
            PH3ImageTextConnectorCheckpointLayout::ComfyMergedTextEncoder,
        )?;
        let text_bytes = stored_bytes(
            PH3ImageTextConditioningFamily::new(limits.text)?
                .weights(&text_config)?
                .stored_schema()?,
        )?;
        let projection_bytes = stored_bytes(
            PH3ImageTextConnectorProjectionFamily::new(limits.connector)?
                .weights(&text_config.projection_config()?)?
                .stored_schema()?,
        )?;
        let denoiser = PH3ImageDenoiserAdapter::new(
            transformer,
            Arc::clone(&sources.transformer),
            limits.denoiser,
        )?;
        let vae = PH3ImageVaeAdapter::new(Arc::clone(&sources.vae), metadata.vae, limits.vae)?;
        ensure!(
            vae.latent_channels() == decoder.latent_channels()
                && vae.spatial_scale() == decoder.patch_size(),
            "encoder/decoder/image metadata compression differs"
        );
        let vision_bytes = vision_weights.stored_bytes()?;
        let required_weight_bytes = [
            text_bytes
                .checked_add(projection_bytes)
                .and_then(|n| n.checked_add(vision_bytes))
                .context("conditioning weights overflow")?,
            denoiser.stored_bytes(),
            vae.encoder_stored_bytes()
                .checked_add(vae.decoder_stored_bytes())
                .context("codec weights overflow")?,
        ];
        let mut metadata_sha256 = named
            .into_iter()
            .map(|(name, bytes)| (name.to_owned(), format!("{:x}", Sha256::digest(bytes))))
            .collect::<BTreeMap<_, _>>();
        metadata_sha256.insert(
            "tokenizer".into(),
            format!("{:x}", Sha256::digest(metadata.tokenizer)),
        );
        Ok(Self {
            main_vision_reuse: Default::default(),
            expansion: None,
            sources, text, text_weights, vision, vision_weights, connector, tokenizer, denoiser, vae, schedule, declarations, limits,
            receipt: ImageAssetReceipt {
                metadata_sha256, required_weight_bytes,
                payload_verification: "canonical_quantization_and_physical_schema_checked; source_hashes_require_external_lock",
                lora: None,
            },
        })
    }
    pub fn receipt(&self) -> &ImageAssetReceipt {
        &self.receipt
    }
    /// Reuse only image-only main-TE outputs within one initialization cohort.
    /// Each request still owns its text, reference latent and solver state.
    pub fn with_main_vision_reuse(
        mut self,
        policy: super::PH3ImageMainVisionReuse,
    ) -> Result<Self> {
        ensure!(
            policy == super::PH3ImageMainVisionReuse::Off || self.has_prompt_expansion(),
            "main vision cohort reuse requires enabled cohort prompt expansion"
        );
        self.main_vision_reuse = policy;
        Ok(self)
    }
    /// Attach the immutable original causal source before constructing a
    /// runtime. This performs metadata/mmap validation, never inference.
    pub fn with_prompt_expansion_sources(
        mut self,
        sources: Arc<ferrum_models::vnext::ProductionModelSourceBundle>,
        limits: ferrum_models::p_h3_image::PH3ImagePromptExpansionLimits,
        options: super::PH3ImagePromptExpansionOptions,
    ) -> Result<Self> {
        ensure!(
            self.expansion.is_none()
                && options.max_new_tokens > 0
                && options.prefill_chunk_tokens > 0
                && options.engine.runtime.max_model_len.is_none_or(|v| v > 0),
            "prompt expansion requires one source and positive bounded settings"
        );
        self.expansion = Some(super::expansion::ConfiguredTextExpansion {
            assets: Arc::new(
                super::expansion::NativeTextExpansion::from_sources_with_precision(
                    sources,
                    limits,
                    super::expansion::NativeTextExpansion::precision_for_engine(&options.engine)?,
                )?,
            ),
            options,
        });
        Ok(self)
    }
    pub fn has_prompt_expansion(&self) -> bool {
        self.expansion.is_some()
    }
    /// Select backing for each individual static upsampler installation. A
    /// reusable slot is released after installation and never caches weights
    /// across requests. Configure before sharing these immutable assets.
    pub fn with_prompt_expansion_source_backing(
        mut self,
        policy: super::PH3ImagePromptExpansionSourceBacking,
    ) -> Result<Self> {
        let expansion = self.expansion.as_mut().ok_or_else(|| {
            anyhow::anyhow!("source backing requires configured prompt expansion")
        })?;
        let assets = Arc::get_mut(&mut expansion.assets).ok_or_else(|| {
            anyhow::anyhow!("configure source backing before sharing expansion assets")
        })?;
        assets.configure_source_backing(policy)?;
        Ok(self)
    }
    /// Bound conversion workers within the existing prompt-expansion source
    /// slot. More than one worker requires bounded streaming; configure before
    /// sharing these immutable assets. The default remains one worker.
    pub fn with_prompt_expansion_source_conversion_policy(
        mut self,
        policy: ferrum_quantization::Bf16F16ConversionPolicy,
    ) -> Result<Self> {
        let expansion = self.expansion.as_mut().ok_or_else(|| {
            anyhow::anyhow!("source conversion requires configured prompt expansion")
        })?;
        let assets = Arc::get_mut(&mut expansion.assets).ok_or_else(|| {
            anyhow::anyhow!("configure source conversion before sharing expansion assets")
        })?;
        assets.configure_source_conversion_policy(policy)?;
        Ok(self)
    }
    /// Allow exact-image visual reuse only within one expansion cohort, before
    /// sharing these assets. No output survives into another cohort.
    pub fn with_prompt_expansion_vision_reuse(
        mut self,
        policy: super::PH3ImagePromptExpansionVisionReuse,
    ) -> Result<Self> {
        let expansion = self
            .expansion
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("vision reuse requires configured prompt expansion"))?;
        let assets = Arc::get_mut(&mut expansion.assets).ok_or_else(|| {
            anyhow::anyhow!("configure vision reuse before sharing expansion assets")
        })?;
        assets.configure_vision_reuse(policy);
        Ok(self)
    }

    pub fn with_prompt_expansion_mtp(
        mut self,
        policy: super::PH3ImagePromptExpansionMtp,
    ) -> Result<Self> {
        let expansion = self
            .expansion
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("MTP requires configured prompt expansion"))?;
        let assets = Arc::get_mut(&mut expansion.assets)
            .ok_or_else(|| anyhow::anyhow!("configure MTP before sharing expansion assets"))?;
        assets.configure_mtp(policy)?;
        Ok(self)
    }
    /// Reuse exact execution-precision weights through a bounded private file.
    /// The first admitted expansion builds it; subsequent requests map the same
    /// completed representation. This does not cache prompts, KV, or GPU state.
    pub fn with_prompt_expansion_process_mapped_weights(
        self,
        directory: std::path::PathBuf,
        limits: ferrum_quantization::ProcessMappedDenseLimits,
    ) -> Result<Self> {
        self.with_prompt_expansion_process_mapped_weights_and_upload_policy(
            directory,
            limits,
            ferrum_quantization::ProcessMappedDenseUploadPolicy::default(),
        )
    }
    /// Select whether the immutable cached bytes may use mapped uploads or
    /// must pass through the runtime's original bounded transfer staging.
    pub fn with_prompt_expansion_process_mapped_weights_and_upload_policy(
        mut self,
        directory: std::path::PathBuf,
        limits: ferrum_quantization::ProcessMappedDenseLimits,
        upload_policy: ferrum_quantization::ProcessMappedDenseUploadPolicy,
    ) -> Result<Self> {
        let expansion = self.expansion.as_mut().ok_or_else(|| {
            anyhow::anyhow!("process-mapped weights require configured prompt expansion")
        })?;
        let assets = Arc::get_mut(&mut expansion.assets).ok_or_else(|| {
            anyhow::anyhow!("configure process-mapped weights before sharing expansion assets")
        })?;
        assets.configure_process_mapped_weights(directory, limits, upload_policy)?;
        Ok(self)
    }
    /// Attach the original visual and causal upsampler sources using the
    /// declared CUDA F32 visual arithmetic profile. This validates assets;
    /// it does not allocate GPU resources or claim upstream BF16 equivalence.
    pub fn with_prompt_expansion_visual_sources(
        self,
        sources: Arc<ferrum_models::vnext::ProductionModelSourceBundle>,
        limits: ferrum_models::p_h3_image::PH3ImagePromptExpansionLimits,
        options: super::PH3ImagePromptExpansionOptions,
        processor: ferrum_models::qwen3_vl_vision::hf::Qwen3VLVisionHfProcessor,
        vision_limits: ferrum_models::qwen3_vl_vision::hf::Qwen3VLVisionHfLimits,
    ) -> Result<Self> {
        self.with_prompt_expansion_visual_sources_and_profile(
            sources,
            limits,
            options,
            processor,
            vision_limits,
            ferrum_runtime::NumericalProfileId::new(
                ferrum_models::qwen3_vl_vision::hf::Qwen3VLVisionHfFamily::CUDA_PROFILE_ID,
            )
            .map_err(anyhow::Error::msg)?,
        )
    }
    /// Explicit numerical selection for another declared backend/profile.
    #[allow(clippy::too_many_arguments)]
    pub fn with_prompt_expansion_visual_sources_and_profile(
        mut self,
        sources: Arc<ferrum_models::vnext::ProductionModelSourceBundle>,
        limits: ferrum_models::p_h3_image::PH3ImagePromptExpansionLimits,
        options: super::PH3ImagePromptExpansionOptions,
        processor: ferrum_models::qwen3_vl_vision::hf::Qwen3VLVisionHfProcessor,
        vision_limits: ferrum_models::qwen3_vl_vision::hf::Qwen3VLVisionHfLimits,
        vision_profile: ferrum_runtime::NumericalProfileId,
    ) -> Result<Self> {
        ensure!(
            self.expansion.is_none()
                && options.max_new_tokens > 0
                && options.prefill_chunk_tokens > 0
                && options.engine.runtime.max_model_len.is_none_or(|v| v > 0),
            "prompt expansion requires one source and positive bounded settings"
        );
        let assets = super::expansion::NativeTextExpansion::from_sources_with_precision(
            sources,
            limits,
            super::expansion::NativeTextExpansion::precision_for_engine(&options.engine)?,
        )?
        .with_visual_assets(processor, vision_limits, vision_profile)?;
        self.expansion = Some(super::expansion::ConfiguredTextExpansion {
            assets: Arc::new(assets),
            options,
        });
        Ok(self)
    }
    pub fn has_prompt_expansion_visual(&self) -> bool {
        self.expansion
            .as_ref()
            .is_some_and(|configured| configured.assets.visual_assets().is_some())
    }
    pub(super) fn prompt_expansion_family(&self) -> Option<&ferrum_runtime::PreparedModelFamily> {
        self.expansion
            .as_ref()
            .map(|configured| configured.assets.family())
    }
    pub(super) fn prompt_expansion_visual_causal_family(
        &self,
    ) -> Option<&ferrum_runtime::PreparedModelFamily> {
        self.expansion
            .as_ref()
            .and_then(|configured| configured.assets.visual_causal_family())
    }
    pub(super) fn prompt_expansion_vision_family(
        &self,
    ) -> Option<&ferrum_runtime::PreparedModelFamily> {
        self.expansion
            .as_ref()
            .and_then(|configured| configured.assets.vision_family())
    }
    pub fn latent_channels(&self) -> u64 {
        self.vae.latent_channels()
    }
    pub fn spatial_scale(&self) -> u64 {
        self.vae.spatial_scale()
    }
}
fn stored_bytes(schema: &WeightSchema) -> Result<u64> {
    schema.components.iter().try_fold(0u64, |sum, component| {
        sum.checked_add(component.physical_bytes()?)
            .context("physical component bytes overflow")
    })
}
