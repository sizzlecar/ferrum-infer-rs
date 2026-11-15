#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::{ensure, Result};
use ferrum_models::qwen_image21::image_input::ImageInputLimits;
use serde::{Deserialize, Serialize};
use std::{
    io::Read,
    path::{Path, PathBuf},
};
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Image {
    Png { path: PathBuf },
    DiagnosticRgb { width: u32, height: u32 },
}
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Presentation {
    #[default]
    ProcessorInput,
    FastH3Reference {
        short_edge: u32,
        /// Original F32 NCHW VAE pixels prepared from the same resized RGB image.
        vae_pixels_output: PathBuf,
    },
}
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Attention {
    /// Explicit sequential F32 reduction for small numerical diagnostics.
    #[default]
    SequentialDiagnostic,
    /// Original BF16 Q/K/V and output with vendor fused reduction.
    Cudnn {
        library: PathBuf,
        dependencies: Vec<PathBuf>,
        opaque_bytes: u64,
        workspace_bytes: u64,
        cache_bytes: u64,
    },
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    pub encoder_dir: PathBuf,
    pub processor: PathBuf,
    pub image: Image,
    #[serde(default)]
    pub presentation: Presentation,
    #[serde(default)]
    pub attention: Attention,
    pub output_dir: PathBuf,
    pub device: usize,
    pub host_bytes: usize,
    pub device_bytes: u64,
    pub bootstrap_bytes: u64,
    pub stream_bytes: u64,
    pub maximum_metadata_bytes: usize,
    pub maximum_shards: usize,
    pub maximum_source_bytes: u64,
    pub maximum_plan_nodes: usize,
    pub maximum_image_encoded_bytes: usize,
    pub maximum_image_pixels: usize,
    pub maximum_image_axis: u32,
    pub image_codec_workspace_bytes: usize,
    pub maximum_patch_rows: u64,
    pub maximum_patch_bytes: usize,
}
impl Configuration {
    pub fn load(path: &Path) -> Result<Self> {
        let c: Self = serde_json::from_slice(&read(path, 1 << 20)?)?;
        ensure!(
            [
                c.host_bytes,
                c.maximum_metadata_bytes,
                c.maximum_shards,
                c.maximum_plan_nodes,
                c.maximum_patch_bytes
            ]
            .into_iter()
            .all(|n| n > 0 && n < isize::MAX as usize),
            "invalid Host/metadata capacity"
        );
        ensure!(
            [
                c.device_bytes,
                c.bootstrap_bytes,
                c.stream_bytes,
                c.maximum_source_bytes,
                c.maximum_patch_rows
            ]
            .into_iter()
            .all(|n| n > 0 && n < isize::MAX as u64),
            "invalid device/source capacity"
        );
        c.image_limits()?;
        if let Attention::Cudnn {
            library,
            opaque_bytes,
            workspace_bytes,
            cache_bytes,
            ..
        } = &c.attention
        {
            ensure!(
                !library.as_os_str().is_empty()
                    && [*opaque_bytes, *workspace_bytes, *cache_bytes]
                        .into_iter()
                        .all(|n| n > 0 && n < isize::MAX as u64),
                "invalid explicit cuDNN capacity/library"
            );
        }
        Ok(c)
    }
    pub fn image_limits(&self) -> Result<ImageInputLimits> {
        Ok(ImageInputLimits::new(
            self.maximum_image_encoded_bytes,
            self.maximum_image_pixels,
            self.maximum_image_axis,
            self.maximum_image_axis,
            self.image_codec_workspace_bytes,
        )?)
    }
}
pub fn read(path: &Path, limit: usize) -> Result<Vec<u8>> {
    ensure!(
        limit > 0 && limit < isize::MAX as usize,
        "invalid finite read bound"
    );
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= limit, "metadata exceeds capacity");
    Ok(bytes)
}
#[cfg(feature = "cuda")]
pub fn archive(c: &Configuration) -> Result<ferrum_quantization::SafetensorsArchive> {
    use ferrum_quantization::SafetensorsArchive;
    use std::{collections::BTreeSet, fs::File};
    let single = c.encoder_dir.join("model.safetensors");
    let index = c.encoder_dir.join("model.safetensors.index.json");
    let indexed = match (single.exists(), index.exists()) {
        (true, false) => false,
        (false, true) => true,
        _ => anyhow::bail!("requires exactly one standard single archive or index"),
    };
    let mut metadata = 0usize;
    let shards = if indexed {
        let b = read(&index, c.maximum_metadata_bytes)?;
        metadata = b.len();
        let v: serde_json::Value = serde_json::from_slice(&b)?;
        let m = v
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .context("missing weight_map")?;
        ensure!(!m.is_empty(), "empty weight_map");
        m.values()
            .map(|v| {
                let n = v.as_str().context("nontext shard")?;
                ensure!(
                    n.ends_with(".safetensors")
                        && !n.contains(['/', '\\', ':'])
                        && !n.chars().any(char::is_control),
                    "nonportable shard"
                );
                Ok(n.to_owned())
            })
            .collect::<Result<BTreeSet<_>>>()?
    } else {
        BTreeSet::from(["model.safetensors".into()])
    };
    ensure!(
        shards.len() <= c.maximum_shards,
        "shard count exceeds capacity"
    );
    for name in shards {
        let mut f = File::open(c.encoder_dir.join(name))?;
        let mut b = [0; 8];
        f.read_exact(&mut b)?;
        let header = usize::try_from(u64::from_le_bytes(b))?;
        metadata = metadata
            .checked_add(header)
            .context("header total overflow")?;
        ensure!(
            header > 0
                && metadata <= c.maximum_metadata_bytes
                && f.metadata()?.len() >= header as u64 + 8,
            "archive header exceeds finite capacity/extent"
        );
    }
    Ok(if indexed {
        SafetensorsArchive::open_with_index(&c.encoder_dir, "model.safetensors.index.json")?
    } else {
        SafetensorsArchive::open_single_file(&c.encoder_dir, "model.safetensors")?
    })
}
