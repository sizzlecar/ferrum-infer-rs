#[cfg(feature = "cuda-cutlass-nvfp4")]
use super::{bounded_read, Config};
#[cfg(feature = "cuda-cutlass-nvfp4")]
use anyhow::Context;
use anyhow::{ensure, Result};
#[cfg(feature = "cuda-cutlass-nvfp4")]
use ferrum_models::{
    fast_h3::{FastH3ImagePresentation, FastH3PresentationLimits, FastH3VisionFeatureView},
    qwen3_vl_text::Qwen3VLTextNvfp4Checkpoint,
    qwen3_vl_vision::{Qwen3VLVisionConfig, Qwen3VLVisionGrid},
};
#[cfg(feature = "cuda-cutlass-nvfp4")]
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
#[cfg(feature = "cuda-cutlass-nvfp4")]
use sha2::{Digest, Sha256};
#[cfg(feature = "cuda-cutlass-nvfp4")]
use std::io::Read;
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceFile {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageForward {
    pub prompt: String,
    pub tokenizer: SourceFile,
    pub vision_manifest: SourceFile,
    pub maximum_tokenizer_bytes: usize,
    pub maximum_prompt_bytes: usize,
    pub maximum_tokens: u32,
    pub output: PathBuf,
}
impl ImageForward {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.maximum_tokenizer_bytes > 0
                && self.maximum_tokenizer_bytes <= isize::MAX as usize
                && self.maximum_prompt_bytes > 0
                && self.prompt.len() <= self.maximum_prompt_bytes
                && self.maximum_tokens > 0
                && self.maximum_tokens <= 1 << 24
                && !self.output.as_os_str().is_empty(),
            "invalid image presentation capacity/output"
        );
        hash(&self.tokenizer.sha256)?;
        hash(&self.vision_manifest.sha256)?;
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize)]
#[cfg(feature = "cuda-cutlass-nvfp4")]
struct Feature {
    value_id: String,
    file: PathBuf,
    element_type: String,
    dimensions: Vec<u64>,
    sha256: String,
}
#[derive(Clone, Debug, Deserialize)]
#[cfg(feature = "cuda-cutlass-nvfp4")]
struct Manifest {
    schema: String,
    grid: [u64; 3],
    merge_size: u64,
    outputs: [Feature; 4],
}
#[cfg(feature = "cuda-cutlass-nvfp4")]
pub struct Prepared {
    pub presentation: FastH3ImagePresentation,
    manifest: Manifest,
}
#[cfg(feature = "cuda-cutlass-nvfp4")]
pub fn prepare(c: &Config, metadata: &[u8]) -> Result<Option<Prepared>> {
    let Some(request) = &c.image_forward else {
        return Ok(None);
    };
    request.validate()?;
    let tokenizer_bytes = checked(&request.tokenizer, request.maximum_tokenizer_bytes)?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_bytes)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let raw: serde_json::Value = serde_json::from_slice(metadata)?;
    for (token, field) in [
        ("<|image_pad|>", "image_token_id"),
        ("<|video_pad|>", "video_token_id"),
        ("<|vision_start|>", "vision_start_token_id"),
        ("<|vision_end|>", "vision_end_token_id"),
    ] {
        ensure!(
            tokenizer.token_to_id(token).map(u64::from) == raw.get(field).and_then(|v| v.as_u64())
                && raw.get(field).is_some(),
            "tokenizer/checkpoint {field} mismatch"
        );
    }
    let manifest: Manifest = serde_json::from_slice(&checked(
        &request.vision_manifest,
        c.maximum_metadata_bytes,
    )?)?;
    ensure!(
        manifest.schema == "ferrum.qwen3_vl.vision_forward.v1",
        "requires actual vision forward manifest"
    );
    let [t, h, w] = manifest.grid;
    let vision =
        Qwen3VLVisionConfig::from_hf_json(metadata, vec![Qwen3VLVisionGrid::new(t, h, w)?])?;
    ensure!(
        manifest.merge_size == vision.architecture().merge_size(),
        "vision merge differs from checkpoint"
    );
    for feature in &manifest.outputs {
        hash(&feature.sha256)?;
        ensure!(
            feature.element_type == "bf16",
            "vision must preserve native BF16"
        );
    }
    let presentation = FastH3ImagePresentation::single_image(
        &tokenizer,
        &request.prompt,
        &vision,
        FastH3PresentationLimits {
            maximum_prompt_bytes: request.maximum_prompt_bytes,
            maximum_tokens: request.maximum_tokens,
        },
    )?;
    Ok(Some(Prepared {
        presentation,
        manifest,
    }))
}
#[cfg(feature = "cuda-cutlass-nvfp4")]
impl Prepared {
    pub fn inputs(
        &self,
        checkpoint: &Qwen3VLTextNvfp4Checkpoint,
        host: &HostMemoryAccount,
    ) -> Result<Vec<ManagedHostTensor>> {
        let [span] = self.presentation.image_spans() else {
            anyhow::bail!("single-image example requires exactly one visual span");
        };
        let rows = span.visual_run().length();
        let width = checkpoint.config().text().architecture().hidden_size();
        let mut features = Vec::with_capacity(4);
        for source in &self.manifest.outputs {
            ensure!(
                source.dimensions == [rows, width],
                "vision feature extent differs"
            );
            let spec = ResolvedTensorSpec::new(
                source.dimensions.clone(),
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?;
            let mut file = std::fs::File::open(&source.file)?;
            let bytes = usize::try_from(spec.minimum_storage_bytes()?)?;
            ensure!(
                file.metadata()?.len() == bytes as u64,
                "vision file extent differs"
            );
            let mut owner = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
            file.read_exact(owner.as_mut())?;
            ensure!(
                format!("{:x}", Sha256::digest(owner.as_ref())) == source.sha256,
                "vision file SHA differs"
            );
            features.push(ManagedHostTensor::new(
                ProgramValueId::new(&source.value_id)?,
                spec,
                owner.freeze(),
            )?);
        }
        let features: [ManagedHostTensor; 4] = features
            .try_into()
            .map_err(|_| anyhow::anyhow!("vision output count differs"))?;
        let views = std::array::from_fn(|i| {
            FastH3VisionFeatureView::new(
                features[i].value_id(),
                features[i].tensor(),
                features[i].bytes(),
            )
        });
        self.presentation.managed_text_inputs(
            checkpoint.config().text(),
            checkpoint.archive(),
            std::slice::from_ref(&views),
            host,
        )
    }
}
fn hash(hash: &str) -> Result<()> {
    ensure!(
        hash.len() == 64
            && hash
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "requires lowercase SHA256"
    );
    Ok(())
}
#[cfg(feature = "cuda-cutlass-nvfp4")]
fn checked(source: &SourceFile, limit: usize) -> Result<Vec<u8>> {
    let bytes = bounded_read(&source.path, limit)
        .with_context(|| format!("source {}", source.path.display()))?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes)) == source.sha256,
        "source SHA256 differs"
    );
    Ok(bytes)
}
