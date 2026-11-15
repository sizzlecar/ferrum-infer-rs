//! One immutable checkpoint owner shared by conditioning, denoising and decode.
//! Import bounds limit metadata and mapped source extents; execution backing
//! still requires the caller's existing Host/device admission.
use anyhow::{ensure, Context, Result};
use ferrum_models::{
    fast_h3::{
        audio_vae::FastH3AudioVaeConfig,
        video_vae::{
            FastH3VideoDecoderPackage, FastH3VideoDecoderWeightLimits, FastH3VideoVaeConfig,
        },
        FastH3Checkpoint, FastH3MetadataLimits, FastH3Nvfp4Profile, FastH3PddPlan,
        FastH3TransformerPackage, FastH3WeightLimits,
    },
    qwen3_vl_text::Qwen3VLTextArchitecture,
    qwen3_vl_vision::{
        Qwen3VLVisionArchitecture, Qwen3VLVisionImageProcessor, Qwen3VLVisionProcessorLimits,
        Qwen3VLVisionVideoProcessor,
    },
};
use ferrum_quantization::SafetensorsArchive;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

mod archive;
use archive::{bounded_read, ImportExtent};

/// The chosen file or index is authoritative. No globbing or fallback occurs
/// after selection, and symlinked immutable Hugging Face shards remain valid.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FastH3ArchivePath {
    File { path: PathBuf },
    Index { path: PathBuf },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3AssetPaths {
    pub inference: PathBuf,
    pub transformer_config: PathBuf,
    pub video_scheduler: PathBuf,
    pub audio_scheduler: PathBuf,
    pub transformer_dense: FastH3ArchivePath,
    pub transformer_packed: FastH3ArchivePath,
    pub video_vae_config: PathBuf,
    pub video_encoder: FastH3ArchivePath,
    pub video_decoder: FastH3ArchivePath,
    pub audio_vae_config: PathBuf,
    pub audio_vae: FastH3ArchivePath,
    pub encoder_config: PathBuf,
    /// Vision, token lookup and NVFP4 text layers share this one mapping.
    pub encoder: FastH3ArchivePath,
    pub processor: PathBuf,
    /// Defaults to video_preprocessor_config.json beside the image processor.
    #[serde(default)]
    pub video_processor: Option<PathBuf>,
    pub tokenizer: PathBuf,
}
impl FastH3AssetPaths {
    /// Resolve the original component names once. Explicit field overrides can
    /// select separately deployed components without rewriting model metadata.
    pub fn from_model_directory(root: &Path) -> Self {
        let file = |name: &str| FastH3ArchivePath::File {
            path: root.join(name),
        };
        let index = |name: &str| FastH3ArchivePath::Index {
            path: root.join(name),
        };
        let encoder_index = "text_encoder/model.safetensors.index.json";
        Self {
            inference: root.join("fastvideo_inference.json"),
            transformer_config: root.join("transformer_ref/config.json"),
            video_scheduler: root.join("scheduler/scheduler_config.json"),
            audio_scheduler: root.join("audio_scheduler/scheduler_config.json"),
            transformer_dense: index(
                "transformer_ref/diffusion_pytorch_model.safetensors.index.json",
            ),
            transformer_packed: file("transformer_ref/nvfp4_weights.safetensors"),
            video_vae_config: root.join("vae/config.json"),
            video_encoder: index("vae/diffusion_pytorch_model.safetensors.index.json"),
            video_decoder: file("vae_decoder_nvfp4/decoder-nvfp4.safetensors"),
            audio_vae_config: root.join("audio_vae/config.json"),
            audio_vae: file("audio_vae/diffusion_pytorch_model.safetensors"),
            encoder_config: root.join("text_encoder/config.json"),
            encoder: if root.join(encoder_index).is_file() {
                index(encoder_index)
            } else {
                file("text_encoder/model.safetensors")
            },
            processor: root.join("processor/preprocessor_config.json"),
            video_processor: Some(root.join("processor/video_preprocessor_config.json")),
            tokenizer: root.join("tokenizer/tokenizer.json"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FastH3AssetLimits {
    pub metadata: FastH3MetadataLimits,
    pub maximum_tokenizer_bytes: usize,
    pub maximum_shards_per_archive: usize,
    pub maximum_tensors_per_archive: usize,
    pub maximum_total_source_bytes: u64,
    pub maximum_exported_text_layers: u64,
    pub transformer: FastH3WeightLimits,
    pub video_decoder: FastH3VideoDecoderWeightLimits,
    pub processor: Qwen3VLVisionProcessorLimits,
}

pub struct FastH3Assets {
    pub checkpoint: Arc<FastH3Checkpoint>,
    pub pdd: Arc<FastH3PddPlan>,
    pub transformer: Arc<FastH3TransformerPackage>,
    pub video_vae: Arc<FastH3VideoVaeConfig>,
    pub video_encoder: Arc<SafetensorsArchive>,
    pub video_decoder: Arc<FastH3VideoDecoderPackage>,
    pub audio_vae: Arc<FastH3AudioVaeConfig>,
    pub audio_archive: Arc<SafetensorsArchive>,
    pub encoder_metadata: Arc<serde_json::Value>,
    pub encoder: Arc<SafetensorsArchive>,
    pub text_architecture: Qwen3VLTextArchitecture,
    pub exported_text_layers: u64,
    pub vision_architecture: Qwen3VLVisionArchitecture,
    pub processor: Arc<Qwen3VLVisionImageProcessor>,
    pub video_processor: Arc<Qwen3VLVisionVideoProcessor>,
    pub tokenizer: Arc<tokenizers::Tokenizer>,
    pub source_payload_bytes: u64,
}
impl FastH3Assets {
    pub fn load(paths: &FastH3AssetPaths, limits: FastH3AssetLimits) -> Result<Self> {
        ensure!(
            limits.metadata.max_file_bytes > 0
                && limits.metadata.max_file_bytes < isize::MAX as usize
                && limits.maximum_tokenizer_bytes > 0
                && limits.maximum_tokenizer_bytes < isize::MAX as usize
                && limits.maximum_shards_per_archive > 0
                && limits.maximum_tensors_per_archive > 0
                && limits.maximum_total_source_bytes > 0
                && limits.maximum_exported_text_layers > 0,
            "FastH3 asset import bounds must be positive and addressable"
        );
        let read = |path: &Path| bounded_read(path, limits.metadata.max_file_bytes);
        let checkpoint = Arc::new(FastH3Checkpoint::parse(
            &read(&paths.inference)?,
            &read(&paths.transformer_config)?,
            &read(&paths.video_scheduler)?,
            &read(&paths.audio_scheduler)?,
            limits.metadata,
        )?);
        let pdd = Arc::new(FastH3PddPlan::new(&checkpoint)?);
        let video_vae = Arc::new(FastH3VideoVaeConfig::parse(
            &read(&paths.video_vae_config)?,
            limits.metadata.max_file_bytes,
        )?);
        let audio_vae = Arc::new(FastH3AudioVaeConfig::parse(
            &read(&paths.audio_vae_config)?,
            limits.metadata.max_file_bytes,
        )?);
        let encoder_metadata: serde_json::Value =
            serde_json::from_slice(&read(&paths.encoder_config)?)?;
        let text_architecture = Qwen3VLTextArchitecture::from_hf_value(&encoder_metadata)?;
        let vision_architecture = Qwen3VLVisionArchitecture::from_hf_value(&encoder_metadata)?;
        let exported_text_layers = encoder_metadata
            .get("num_hidden_layers_override")
            .and_then(|v| v.as_u64())
            .context("encoder lacks explicit exported text layer count")?;
        ensure!(
            exported_text_layers > 0
                && exported_text_layers <= text_architecture.layers()
                && exported_text_layers <= limits.maximum_exported_text_layers,
            "exported text prefix exceeds architecture or import capacity"
        );
        ensure!(
            text_architecture.hidden_size() == vision_architecture.output_size()
                && text_architecture.hidden_size() == checkpoint.transformer().text_dim() as u64
                && vision_architecture.deepstack_layers().len() == 3
                && video_vae.latent_channels() == checkpoint.transformer().video_channels()
                && audio_vae.latent_channels() == checkpoint.transformer().audio_channels() as u64,
            "conditioning or VAE dimensions differ from the transformer contract"
        );
        let processor = Arc::new(Qwen3VLVisionImageProcessor::from_source_bytes(
            &read(&paths.processor)?,
            limits.processor,
        )?);
        let video_processor_path = paths.video_processor.clone().unwrap_or_else(|| {
            paths
                .processor
                .with_file_name("video_preprocessor_config.json")
        });
        let video_processor = Arc::new(Qwen3VLVisionVideoProcessor::from_source_bytes(
            &read(&video_processor_path)?,
            limits.processor,
        )?);
        let tokenizer = tokenizers::Tokenizer::from_bytes(bounded_read(
            &paths.tokenizer,
            limits.maximum_tokenizer_bytes,
        )?)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        validate_tokenizer(&tokenizer, &encoder_metadata)?;
        // Every consumer receives one of these same owners. Request-specific
        // source adapters still verify the selected tensor shapes/dtypes before
        // graph compilation and admit their payload through the original Host.
        let mut extent = ImportExtent::new(limits);
        let dense = extent.open(&paths.transformer_dense)?;
        let packed = extent.open(&paths.transformer_packed)?;
        let transformer = Arc::new(FastH3TransformerPackage::new(
            checkpoint.transformer(),
            dense,
            packed,
            FastH3Nvfp4Profile::DiTWithVsa,
            limits.transformer,
        )?);
        let video_encoder = Arc::new(extent.open(&paths.video_encoder)?);
        let video_decoder = Arc::new(FastH3VideoDecoderPackage::new(
            &video_vae,
            extent.open(&paths.video_decoder)?,
            limits.video_decoder,
        )?);
        let audio_archive = Arc::new(extent.open(&paths.audio_vae)?);
        let encoder = Arc::new(extent.open(&paths.encoder)?);
        Ok(Self {
            checkpoint,
            pdd,
            transformer,
            video_vae,
            video_encoder,
            video_decoder,
            audio_vae,
            audio_archive,
            encoder_metadata: Arc::new(encoder_metadata),
            encoder,
            text_architecture,
            exported_text_layers,
            vision_architecture,
            processor,
            video_processor,
            tokenizer: Arc::new(tokenizer),
            source_payload_bytes: extent.payload_bytes(),
        })
    }

    /// Ordered media cardinality comes from the Ref2VA model contract. Actual
    /// text geometry and physical capacity are planned before request admission.
    pub fn validate_request(
        &self,
        request: &crate::video_generation::VideoGenerationRequest,
    ) -> Result<()> {
        super::REFERENCE_LIMITS.validate_kinds(request.references.iter().map(|r| r.kind()))?;
        Ok(())
    }
}

fn validate_tokenizer(
    tokenizer: &tokenizers::Tokenizer,
    metadata: &serde_json::Value,
) -> Result<()> {
    ensure!(
        tokenizer.get_truncation().is_none() && tokenizer.get_padding().is_none(),
        "FastH3 raw presentation does not accept tokenizer truncation or padding"
    );
    for (token, field) in [
        ("<|image_pad|>", "image_token_id"),
        ("<|video_pad|>", "video_token_id"),
        ("<|vision_start|>", "vision_start_token_id"),
        ("<|vision_end|>", "vision_end_token_id"),
    ] {
        let token_id = tokenizer
            .token_to_id(token)
            .with_context(|| format!("tokenizer lacks {token}"))?;
        ensure!(
            metadata.get(field).and_then(|v| v.as_u64()) == Some(u64::from(token_id))
                && tokenizer.id_to_token(token_id).as_deref() == Some(token),
            "tokenizer/checkpoint {field} differs"
        );
    }
    Ok(())
}
