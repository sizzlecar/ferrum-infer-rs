use super::configuration::*;
use anyhow::{ensure, Context, Result};
use ferrum_engine::image_generation::{QwenImage21Assets, QwenImage21Metadata, QwenImage21Sources};
use ferrum_models::pipeline_source::{PipelineArchitecture, PipelineDeclaration};
use ferrum_quantization::{GgufWeightComponentSource, SafetensorsArchive, TensorArchive};
use std::{fs::File, io::Read, path::Path, sync::Arc};

pub(super) fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let limit = u64::try_from(limit)?;
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    ensure!(
        file.metadata()?.len() <= limit,
        "{} exceeds metadata limit",
        path.display()
    );
    let mut bytes = Vec::new();
    file.take(limit.checked_add(1).context("metadata limit overflows")?)
        .read_to_end(&mut bytes)?;
    ensure!(
        !bytes.is_empty() && bytes.len() as u64 <= limit,
        "{} is empty or oversized",
        path.display()
    );
    Ok(bytes)
}

impl WeightSource {
    pub fn open(&self) -> Result<Arc<dyn TensorArchive>> {
        match self {
            Self::Single { directory, file } => Ok(Arc::new(SafetensorsArchive::open_single_file(
                directory, file,
            )?)),
            Self::Index { directory, file } => Ok(Arc::new(SafetensorsArchive::open_with_index(
                directory, file,
            )?)),
            Self::Gguf { directory, file } => {
                ensure!(
                    file.len() > ".gguf".len()
                        && file.ends_with(".gguf")
                        && !file.contains(['/', '\\', ':'])
                        && !file.chars().any(char::is_control),
                    "GGUF source must name one explicit .gguf basename within its directory"
                );
                Ok(Arc::new(GgufWeightComponentSource::open(
                    directory.join(file),
                )?))
            }
        }
    }
}

impl ImageProductConfig {
    /// All metadata is retained and validated before the caller creates a device runtime.
    /// Explicit archive filenames never fall back or discover another source.
    /// An explicit processor path enables editing validation; missing or
    /// incompatible editing assets never silently fall back to generation.
    pub fn prepare_assets(&self) -> Result<QwenImage21Assets> {
        let paths = &self.metadata;
        let model_index = read_bounded(&paths.model_index, 4 * 1024 * 1024)?;
        let declaration = PipelineDeclaration::from_json(&model_index)?;
        match declaration.architecture() {
            PipelineArchitecture::QwenImage21 => {}
        }
        let processor = paths
            .processor
            .as_ref()
            .map(|path| read_bounded(path, 4 * 1024 * 1024))
            .transpose()?;
        let text = read_bounded(&paths.text_encoder, 4 * 1024 * 1024)?;
        let transformer = read_bounded(&paths.transformer, 4 * 1024 * 1024)?;
        let vae = read_bounded(&paths.vae, 4 * 1024 * 1024)?;
        let scheduler = read_bounded(&paths.scheduler, 4 * 1024 * 1024)?;
        let tokenizer = read_bounded(&paths.tokenizer, 32 * 1024 * 1024)?;
        let tokenizer_config = read_bounded(&paths.tokenizer_config, 1024 * 1024)?;
        let chat_template = read_bounded(&paths.chat_template, 64 * 1024)?;
        let assets = QwenImage21Assets::prepare(
            QwenImage21Metadata {
                text_encoder: &text,
                transformer: &transformer,
                vae: &vae,
                model_index: &model_index,
                scheduler: &scheduler,
                tokenizer: &tokenizer,
                tokenizer_config: &tokenizer_config,
                chat_template: &chat_template,
            },
            QwenImage21Sources {
                text_encoder: self.weights.text_encoder.open()?,
                transformer: self.weights.transformer.open()?,
                vae: self.weights.vae.open()?,
            },
        )?;
        match processor {
            Some(processor) => assets.with_image_editing(&text, &vae, &processor),
            None => Ok(assets),
        }
    }
}
