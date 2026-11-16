//! Bounded read-only package loading shared by checkpoint and execution probes.
use anyhow::{ensure, Context, Result};
use ferrum_models::fast_h3::{
    FastH3Checkpoint, FastH3MetadataLimits, FastH3Nvfp4Profile, FastH3TransformerPackage,
    FastH3WeightLimits,
};
use ferrum_quantization::SafetensorsArchive;
use std::{collections::BTreeSet, fs::File, io::Read, path::Path};

pub const INDEX: &str = "diffusion_pytorch_model.safetensors.index.json";
pub const PACKED: &str = "nvfp4_weights.safetensors";

pub fn bounded_read(path: &Path, maximum: usize) -> Result<Vec<u8>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    ensure!(
        file.metadata()?.len() <= maximum as u64,
        "metadata {} exceeds capacity",
        path.display()
    );
    let mut bytes = Vec::new();
    file.take(maximum as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= maximum,
        "metadata {} grew beyond capacity",
        path.display()
    );
    Ok(bytes)
}

/// Bound headers, shard names and aggregate payload before mapping archives.
#[allow(dead_code)] // Configuration-only builds use bounded_read without loading weights.
pub fn load(
    model_dir: &Path,
    profile: FastH3Nvfp4Profile,
    maximum_metadata: usize,
    limits: FastH3WeightLimits,
) -> Result<FastH3TransformerPackage> {
    let checkpoint = load_checkpoint(
        model_dir,
        FastH3MetadataLimits {
            max_file_bytes: maximum_metadata,
            max_pdd_steps: limits.maximum_tensors,
        },
    )?;
    let directory = model_dir.join("transformer_ref");
    preflight(&directory, maximum_metadata, limits)?;
    FastH3TransformerPackage::new(
        checkpoint.transformer(),
        SafetensorsArchive::open_with_index(&directory, INDEX)?,
        SafetensorsArchive::open_single_file(&directory, PACKED)?,
        profile,
        limits,
    )
}

pub fn load_checkpoint(model_dir: &Path, limits: FastH3MetadataLimits) -> Result<FastH3Checkpoint> {
    ensure!(
        limits.max_file_bytes > 0 && limits.max_file_bytes < isize::MAX as usize,
        "metadata capacity invalid"
    );
    let read = |name: &str| bounded_read(&model_dir.join(name), limits.max_file_bytes);
    FastH3Checkpoint::parse(
        &read("fastvideo_inference.json")?,
        &read("transformer_ref/config.json")?,
        &read("scheduler/scheduler_config.json")?,
        &read("audio_scheduler/scheduler_config.json")?,
        limits,
    )
}

fn preflight(directory: &Path, maximum_metadata: usize, limits: FastH3WeightLimits) -> Result<()> {
    let raw = bounded_read(&directory.join(INDEX), maximum_metadata)?;
    let index: serde_json::Value = serde_json::from_slice(&raw)?;
    let map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .context("index lacks weight_map")?;
    ensure!(
        !map.is_empty() && map.len() <= limits.maximum_tensors,
        "dense index tensor count exceeds capacity or is empty"
    );
    let mut shards = BTreeSet::new();
    for name in map.values() {
        let name = name.as_str().context("index shard name must be a string")?;
        ensure!(
            name.len() > ".safetensors".len()
                && name.ends_with(".safetensors")
                && !name.contains(['/', '\\', ':'])
                && !name.chars().any(char::is_control),
            "diagnostic requires a portable shard basename: {name}"
        );
        ensure!(
            name != PACKED,
            "dense index must not include packed NVFP4 export"
        );
        shards.insert(name.to_owned());
    }
    shards.insert(PACKED.to_owned());
    let mut source_bytes = 0u64;
    for shard in shards {
        let mut file = File::open(directory.join(&shard))?;
        let mut prefix = [0; 8];
        file.read_exact(&mut prefix)?;
        let header = u64::from_le_bytes(prefix);
        ensure!(
            header > 0 && header <= maximum_metadata as u64,
            "{shard} safetensors header exceeds capacity or is empty"
        );
        let payload = file
            .metadata()?
            .len()
            .checked_sub(header + 8)
            .context("safetensors header exceeds file")?;
        source_bytes = source_bytes
            .checked_add(payload)
            .context("source payload byte count overflow")?;
        ensure!(
            source_bytes <= limits.maximum_source_bytes as u64,
            "source payload bytes exceed capacity"
        );
    }
    Ok(())
}
