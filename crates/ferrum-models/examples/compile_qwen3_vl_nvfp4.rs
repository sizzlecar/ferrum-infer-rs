//! Compile an explicit checkpoint prefix, optionally execute real token inputs
//! and save its pre-final-norm BF16 features through the same managed session.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
#[cfg(feature = "cuda-cutlass-nvfp4")]
use std::collections::BTreeSet;
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};
#[path = "compile_qwen3_vl_nvfp4/image.rs"]
mod image;
#[path = "support/retained_failure.rs"]
mod retained_failure;
use retained_failure::Failure;
#[cfg(feature = "cuda-cutlass-nvfp4")]
#[path = "compile_qwen3_vl_nvfp4/cuda.rs"]
mod cuda;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    encoder_dir: PathBuf,
    execute_layers: u64,
    #[serde(default)]
    sequence_length: Option<u32>,
    device: usize,
    maximum_metadata_bytes: usize,
    maximum_shards: usize,
    maximum_projections: usize,
    maximum_source_bytes: usize,
    host_bytes: usize,
    device_bytes: u64,
    bootstrap_bytes: u64,
    stream_bytes: u64,
    native_bytes: u64,
    #[serde(default)]
    attention: Attention,
    #[serde(default)]
    forward: Option<Forward>,
    #[serde(default)]
    image_forward: Option<image::ImageForward>,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Attention {
    #[default]
    SequentialDiagnostic,
    CudnnCausal {
        library: PathBuf,
        dependencies: Vec<PathBuf>,
        opaque_bytes: u64,
        workspace_bytes: u64,
        cache_bytes: u64,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Forward {
    token_ids: Vec<u32>,
    /// New raw little-endian BF16 feature file; shape is included in the report.
    output: PathBuf,
}
impl Config {
    fn validate(&self) -> Result<()> {
        if let Attention::CudnnCausal {
            library,
            opaque_bytes,
            workspace_bytes,
            cache_bytes,
            ..
        } = &self.attention
        {
            ensure!(
                !library.as_os_str().is_empty()
                    && *opaque_bytes > 0
                    && *cache_bytes > 0
                    && *workspace_bytes < isize::MAX as u64,
                "cuDNN attention requires a library and bounded owner/cache/workspace allowances"
            );
        }
        ensure!(self.execute_layers > 0, "layer extent must be positive");
        if let Some(image) = &self.image_forward {
            ensure!(self.forward.is_none()&&self.sequence_length.is_none(),"image presentation derives its sequence; token-only forward/explicit sequence must be absent");
            image.validate()?;
        } else {
            ensure!(
                self.sequence_length.is_some_and(|n| n > 0),
                "explicit positive sequence extent required"
            );
        }
        if let Some(forward) = &self.forward {
            ensure!(
                Some(forward.token_ids.len() as u64) == self.sequence_length.map(u64::from)
                    && self.sequence_length.is_some_and(|n| n <= 1 << 24),
                "forward tokens must cover the declared unpadded sequence exactly"
            );
            ensure!(
                !forward.output.as_os_str().is_empty(),
                "forward output path is empty"
            );
        }
        for n in [
            self.maximum_metadata_bytes,
            self.maximum_shards,
            self.maximum_projections,
            self.maximum_source_bytes,
            self.host_bytes,
        ] {
            ensure!(
                n > 0 && n < isize::MAX as usize,
                "Host/source limit is zero or unaddressable"
            );
        }
        for n in [
            self.device_bytes,
            self.bootstrap_bytes,
            self.stream_bytes,
            self.native_bytes,
        ] {
            ensure!(
                n > 0 && n < isize::MAX as u64,
                "device limit is zero or unaddressable"
            );
        }
        Ok(())
    }
}
fn bounded_read(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    ensure!(
        f.metadata()?.len() <= limit as u64,
        "metadata exceeds declared read capacity"
    );
    let mut out = Vec::new();
    f.take(limit as u64 + 1).read_to_end(&mut out)?;
    ensure!(
        out.len() <= limit,
        "metadata grew beyond declared read capacity"
    );
    Ok(out)
}
#[cfg(feature = "cuda-cutlass-nvfp4")]
fn load(
    c: &Config,
) -> Result<(
    ferrum_models::qwen3_vl_text::Qwen3VLTextNvfp4Checkpoint,
    Vec<u8>,
    Option<image::Prepared>,
)> {
    use ferrum_models::qwen3_vl_text::*;
    use ferrum_quantization::SafetensorsArchive;
    c.validate()?;
    let metadata = bounded_read(&c.encoder_dir.join("config.json"), c.maximum_metadata_bytes)?;
    let raw = serde_json::from_slice(&metadata)?;
    let presentation = image::prepare(c, &metadata)?;
    let single = c.encoder_dir.join("model.safetensors");
    let index = c.encoder_dir.join("model.safetensors.index.json");
    let mut metadata_bytes = metadata.len();
    let indexed = match (single.exists(), index.exists()) {
        (true, false) => false,
        (false, true) => true,
        _ => anyhow::bail!(
            "requires exactly one standard single-file or indexed Transformers archive"
        ),
    };
    let shards = if indexed {
        let bytes = bounded_read(&index, c.maximum_metadata_bytes)?;
        metadata_bytes = metadata_bytes
            .checked_add(bytes.len())
            .context("metadata byte overflow")?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        let map = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .context("index weight_map missing")?;
        ensure!(!map.is_empty(), "weight_map is empty");
        map.values()
            .map(|v| {
                let name = v.as_str().context("shard name must be text")?;
                ensure!(
                    name.ends_with(".safetensors")
                        && !name.contains(['/', '\\', ':'])
                        && !name.chars().any(char::is_control),
                    "shard must be a portable basename"
                );
                Ok(name.to_owned())
            })
            .collect::<Result<BTreeSet<_>>>()?
    } else {
        BTreeSet::from(["model.safetensors".to_owned()])
    };
    ensure!(
        shards.len() <= c.maximum_shards,
        "shard count exceeds declared capacity"
    );
    // Header bytes are bounded before the archive parses/mmap-indexes them.
    // Existing immutable HF cache symlinks are permitted by this diagnostic.
    for name in shards {
        let mut file = File::open(c.encoder_dir.join(name))?;
        let mut bytes = [0; 8];
        file.read_exact(&mut bytes)?;
        let header = usize::try_from(u64::from_le_bytes(bytes))?;
        metadata_bytes = metadata_bytes
            .checked_add(header)
            .context("aggregate metadata overflow")?;
        ensure!(
            header > 0 && metadata_bytes <= c.maximum_metadata_bytes,
            "aggregate metadata exceeds declared capacity"
        );
        ensure!(
            file.metadata()?.len()
                >= u64::try_from(header)?
                    .checked_add(8)
                    .context("header overflow")?,
            "header exceeds shard size"
        );
    }
    let archive = if indexed {
        SafetensorsArchive::open_with_index(&c.encoder_dir, "model.safetensors.index.json")?
    } else {
        SafetensorsArchive::open_single_file(&c.encoder_dir, "model.safetensors")?
    };
    let layout = if let Some(p) = &presentation {
        p.presentation.layout()?
    } else {
        let rows = c.sequence_length.context("missing explicit sequence")?;
        Qwen3VLTextLayout::new(rows, vec![vec![Qwen3VLTextKeyRun::new(rows, true)]])?
    };
    let mut checkpoint = Qwen3VLTextNvfp4Checkpoint::new(
        &raw,
        archive,
        layout,
        c.execute_layers,
        Qwen3VLTextNvfp4Limits {
            maximum_projections: c.maximum_projections,
            maximum_source_bytes: c.maximum_source_bytes,
        },
    )?;
    if let Some(p) = &presentation {
        checkpoint = checkpoint.with_deepstack(p.presentation.deepstack()?)?;
    }
    checkpoint = checkpoint.with_attention(match c.attention {
        Attention::SequentialDiagnostic => Qwen3VLTextNvfp4Attention::Sequential,
        Attention::CudnnCausal { .. } => Qwen3VLTextNvfp4Attention::VendorFusedCausal,
    })?;
    Ok((checkpoint, metadata, presentation))
}
fn main() {
    let result = (|| -> std::result::Result<(), Failure> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        if args == ["--help"] {
            println!("compile_qwen3_vl_nvfp4 --config PATH (explicit JSON capacities; optional forward={{token_ids:[...],output:PATH}} or image_forward with prompt, hashed tokenizer/vision manifest and token capacity; image mode derives sequence_length; requires cuda-cutlass-nvfp4)");
            return Ok(());
        }
        if args.len() != 2 || args[0] != "--config" {
            return Err(Failure::retain("arguments", "expected --config PATH", ()));
        }
        let bytes = bounded_read(Path::new(&args[1]), 1 << 20)
            .map_err(|e| Failure::retain("configuration read", e, ()))?;
        let config: Config = serde_json::from_slice(&bytes)
            .map_err(|e| Failure::retain("configuration parse", e, ()))?;
        config
            .validate()
            .map_err(|e| Failure::retain("configuration limits", e, ()))?;
        #[cfg(feature = "cuda-cutlass-nvfp4")]
        {
            let report = cuda::run(config)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&report).map_err(|e| Failure::retain(
                    "report",
                    e,
                    ()
                ))?
            );
            Ok(())
        }
        #[cfg(not(feature = "cuda-cutlass-nvfp4"))]
        {
            let _ = config;
            Err(Failure::retain(
                "feature",
                "requires --features cuda-cutlass-nvfp4",
                (),
            ))
        }
    })();
    if let Err(failure) = result {
        std::mem::forget(failure.owner);
        eprintln!(
            "{}",
            serde_json::json!({"schema":"ferrum.qwen3_vl.nvfp4.compile.failure.v1","stage":failure.stage,"error":failure.message,"retirement":"not claimed; failure owner retained until process exit"})
        );
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qualification_requires_explicit_positive_capacities_and_rejects_unknown_fields() {
        let mut raw = serde_json::json!({"encoder_dir":"example","execute_layers":1,"sequence_length":3,"device":0,
            "maximum_metadata_bytes":1048576,"maximum_shards":16,"maximum_projections":7,"maximum_source_bytes":1073741824,
            "host_bytes":1048576,"device_bytes":1073741824,"bootstrap_bytes":268435456,"stream_bytes":16777216,"native_bytes":67108864});
        let c: Config = serde_json::from_value(raw.clone()).unwrap();
        c.validate().unwrap();
        let mut image = raw.clone();
        image.as_object_mut().unwrap().remove("sequence_length");
        image["image_forward"] = serde_json::json!({"prompt":"cat","tokenizer":{"path":"tokenizer.json","sha256":"a".repeat(64)},
            "vision_manifest":{"path":"manifest.json","sha256":"b".repeat(64)},"maximum_tokenizer_bytes":1<<20,
            "maximum_prompt_bytes":128,"maximum_tokens":128,"output":"features.bf16"});
        serde_json::from_value::<Config>(image.clone())
            .unwrap()
            .validate()
            .unwrap();
        image["sequence_length"] = 3.into();
        assert!(serde_json::from_value::<Config>(image)
            .unwrap()
            .validate()
            .is_err());
        raw["native_bytes"] = 0.into();
        assert!(serde_json::from_value::<Config>(raw.clone())
            .unwrap()
            .validate()
            .is_err());
        raw["unchecked"] = true.into();
        assert!(serde_json::from_value::<Config>(raw).is_err());
    }
}
