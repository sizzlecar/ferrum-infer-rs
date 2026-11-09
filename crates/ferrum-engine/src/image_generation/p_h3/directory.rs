//! Explicit local metadata and original Comfy archives. This loader performs
//! no network resolution and creates no runtime or Host memory account.
use super::{
    metadata::ComponentDeclarations, PH3ImageAssetLimits, PH3ImageAssets, PH3ImageMetadata,
    PH3ImageSources,
};
use anyhow::{ensure, Context, Result};
use ferrum_models::pipeline_source::PipelineSourceLimits;
use ferrum_quantization::{comfy_convrot::ComfyCanonicalWeightSource, SafetensorsArchive};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Caller-selected single-file archives. These paths may live outside the
/// metadata directory, including explicitly selected local HF cache symlinks.
/// Their files and package metadata must remain immutable while being opened.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageArchivePaths {
    pub text_encoder: PathBuf,
    pub transformer: PathBuf,
    pub vae: PathBuf,
}
impl PH3ImageArchivePaths {
    pub fn new(
        text_encoder: impl Into<PathBuf>,
        transformer: impl Into<PathBuf>,
        vae: impl Into<PathBuf>,
    ) -> Self {
        Self {
            text_encoder: text_encoder.into(),
            transformer: transformer.into(),
            vae: vae.into(),
        }
    }
    /// An explicit filename preset for the published Comfy package. It only
    /// selects paths; it does not infer an architecture, discover alternatives,
    /// download missing files, or select a differently quantized archive.
    pub fn comfy_layout(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        Self::new(
            root.join("text_encoders/p-h3-image_text_encoder_w4a8.safetensors"),
            root.join("diffusion_models/p-h3-image_w4a8.safetensors"),
            root.join("vae/p-h3-image_vae_int8_convrot.safetensors"),
        )
    }
    fn open(&self) -> Result<PH3ImageSources> {
        Ok(PH3ImageSources {
            text_encoder: open_archive("text_encoder", &self.text_encoder)?,
            transformer: open_archive("transformer", &self.transformer)?,
            vae: open_archive("vae", &self.vae)?,
        })
    }
}

impl PH3ImageAssets {
    /// Read the original modular Diffusers metadata layout, then map exactly
    /// the three caller-selected Comfy archives and validate the existing
    /// native component composition. A successful load is source validation,
    /// not device execution or a claim of end-to-end numerical qualification.
    ///
    /// Metadata symlinks must stay inside the canonical metadata root. Archive
    /// paths are separate explicit authorizations, so an external archive is
    /// permitted without relaxing the metadata root boundary. Limits bound
    /// retained file bytes; JSON/tokenizer parser scratch is additional CPU
    /// memory, as with the shared `PipelineSourceLimits` contract.
    pub fn from_directory(
        metadata_root: impl AsRef<Path>,
        archives: PH3ImageArchivePaths,
        read_limits: PipelineSourceLimits,
        asset_limits: PH3ImageAssetLimits,
    ) -> Result<Self> {
        let metadata = LocalMetadata::open(metadata_root.as_ref(), read_limits, asset_limits)?;
        let sources = archives.open()?;
        Self::prepare(metadata.borrowed(), sources, asset_limits)
            .context("validate local H3 component assets")
    }
}

fn open_archive(slot: &str, path: &Path) -> Result<Arc<ComfyCanonicalWeightSource>> {
    let open = || -> Result<_> {
        let basename = path
            .file_name()
            .and_then(|name| name.to_str())
            .context("archive must have a UTF-8 .safetensors basename")?;
        let resolved = fs::canonicalize(path).context("resolve explicit archive")?;
        ensure!(
            fs::metadata(&resolved)?.is_file(),
            "explicit archive must be a regular file"
        );
        let directory = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        // Keep the selected basename rather than the resolved blob basename:
        // explicitly authorized HF-cache symlinks may target extensionless
        // blobs. The existing archive API validates the selected filename and
        // mmaps that exact file without discovering another archive.
        Ok(Arc::new(ComfyCanonicalWeightSource::new(
            SafetensorsArchive::open_single_file(directory, basename)?,
        )?))
    };
    open().with_context(|| format!("open H3 {slot} archive {}", path.display()))
}

struct LocalMetadata {
    model_index: Vec<u8>,
    text_encoder: Vec<u8>,
    text_connector: Vec<u8>,
    transformer: Vec<u8>,
    vae: Vec<u8>,
    scheduler: Vec<u8>,
    audio_scheduler: Vec<u8>,
    tokenizer: Vec<u8>,
    tokenizer_config: Vec<u8>,
}
impl LocalMetadata {
    fn open(
        root: &Path,
        limits: PipelineSourceLimits,
        assets: PH3ImageAssetLimits,
    ) -> Result<Self> {
        ensure!(
            assets.metadata_bytes_per_file > 0 && assets.tokenizer.max_source_bytes > 0,
            "H3 metadata and tokenizer source capacities must be positive"
        );
        let config_limit = limits
            .configuration_bytes()
            .min(u64::try_from(assets.metadata_bytes_per_file)?);
        let mut reader = MetadataReader::new(root, limits.total_bytes())?;
        let model_index = reader.read("modular_model_index.json", config_limit)?;
        ComponentDeclarations::parse(&model_index, assets.metadata_bytes_per_file)
            .context("invalid H3 modular_model_index.json")?;
        let text_encoder = reader.read("text_encoder/config.json", config_limit)?;
        let text_connector = reader.read("text_connector/config.json", config_limit)?;
        let transformer = reader.read("transformer/config.json", config_limit)?;
        let vae = reader.read("vae/config.json", config_limit)?;
        let scheduler = reader.read("scheduler/scheduler_config.json", config_limit)?;
        let audio_scheduler = reader.read("audio_scheduler/scheduler_config.json", config_limit)?;
        let tokenizer_config = reader.read(
            "tokenizer/tokenizer_config.json",
            config_limit.min(u64::try_from(assets.tokenizer.max_source_bytes)?),
        )?;
        // The tokenizer's product capacity covers both original files. Apply
        // the remaining capacity before reading, rather than reading an
        // unrestricted tokenizer and checking its size afterwards.
        let tokenizer_remaining = assets
            .tokenizer
            .max_source_bytes
            .checked_sub(tokenizer_config.len())
            .context("tokenizer/tokenizer_config.json exceeds H3 tokenizer source capacity")?;
        let tokenizer = reader.read(
            "tokenizer/tokenizer.json",
            limits
                .tokenizer_bytes()
                .min(u64::try_from(tokenizer_remaining)?),
        )?;
        let result = Self {
            model_index,
            text_encoder,
            text_connector,
            transformer,
            vae,
            scheduler,
            audio_scheduler,
            tokenizer,
            tokenizer_config,
        };
        // Reject malformed original JSON before opening any weight archive.
        // IgnoredAny checks syntax without constructing a second vocabulary.
        for (path, bytes) in result.files() {
            serde_json::from_slice::<serde::de::IgnoredAny>(bytes)
                .with_context(|| format!("invalid H3 metadata {path}"))?;
        }
        Ok(result)
    }
    fn files(&self) -> [(&'static str, &[u8]); 9] {
        [
            ("modular_model_index.json", &self.model_index),
            ("text_encoder/config.json", &self.text_encoder),
            ("text_connector/config.json", &self.text_connector),
            ("transformer/config.json", &self.transformer),
            ("vae/config.json", &self.vae),
            ("scheduler/scheduler_config.json", &self.scheduler),
            (
                "audio_scheduler/scheduler_config.json",
                &self.audio_scheduler,
            ),
            ("tokenizer/tokenizer.json", &self.tokenizer),
            ("tokenizer/tokenizer_config.json", &self.tokenizer_config),
        ]
    }
    fn borrowed(&self) -> PH3ImageMetadata<'_> {
        PH3ImageMetadata {
            model_index: &self.model_index,
            text_encoder: &self.text_encoder,
            text_connector: &self.text_connector,
            transformer: &self.transformer,
            vae: &self.vae,
            scheduler: &self.scheduler,
            audio_scheduler: &self.audio_scheduler,
            tokenizer: &self.tokenizer,
            tokenizer_config: &self.tokenizer_config,
        }
    }
}

// The shared model-package MetadataReader is private to ferrum-models and its
// PipelineSourceBundle recognizes a different entrypoint/declaration schema.
// Preserve the same local immutable-root, bounded-read rules here without
// changing that existing Qwen package resolver.
struct MetadataReader {
    root: PathBuf,
    remaining: u64,
}
impl MetadataReader {
    fn new(root: &Path, total_bytes: u64) -> Result<Self> {
        let root = fs::canonicalize(root)
            .with_context(|| format!("resolve H3 metadata root {}", root.display()))?;
        ensure!(root.is_dir(), "H3 metadata root must be a directory");
        Ok(Self {
            root,
            remaining: total_bytes,
        })
    }
    fn read(&mut self, relative: &str, per_file_limit: u64) -> Result<Vec<u8>> {
        let path = fs::canonicalize(self.root.join(relative))
            .with_context(|| format!("required H3 metadata {relative}"))?;
        ensure!(
            path.starts_with(&self.root),
            "H3 metadata {relative} resolves outside its source root"
        );
        let metadata =
            fs::metadata(&path).with_context(|| format!("inspect H3 metadata {relative}"))?;
        ensure!(
            metadata.is_file(),
            "H3 metadata {relative} must be a regular file"
        );
        let limit = per_file_limit.min(self.remaining);
        ensure!(
            metadata.len() <= limit,
            "H3 metadata {relative} exceeds its remaining read limit ({limit} bytes)"
        );
        let file = File::open(&path).with_context(|| format!("open H3 metadata {relative}"))?;
        ensure!(
            file.metadata()?.is_file(),
            "opened H3 metadata {relative} must be a regular file"
        );
        let mut bytes = Vec::new();
        file.take(limit + 1)
            .read_to_end(&mut bytes)
            .with_context(|| format!("read H3 metadata {relative}"))?;
        ensure!(
            !bytes.is_empty(),
            "required H3 metadata {relative} is empty"
        );
        let length = u64::try_from(bytes.len())?;
        ensure!(
            length <= limit,
            "H3 metadata {relative} exceeds its remaining read limit ({limit} bytes)"
        );
        self.remaining = self
            .remaining
            .checked_sub(length)
            .context("H3 metadata size accounting overflows")?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests;
