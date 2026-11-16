//! CPU import validation of the original full-depth packed video decoder.
//! No forward, CUDA initialization, encoder or audio decoder is executed.
use anyhow::{ensure, Context, Result};
use ferrum_models::fast_h3::video_vae::{
    FastH3VideoDecodePlan, FastH3VideoDecoderPackage, FastH3VideoDecoderWeightLimits,
    FastH3VideoVaeConfig,
};
use ferrum_quantization::SafetensorsArchive;
use std::{collections::BTreeMap, io::Read, path::PathBuf};

fn main() -> Result<()> {
    let mut arguments = BTreeMap::new();
    let mut argv = std::env::args().skip(1);
    while let Some(key) = argv.next() {
        ensure!(
            [
                "--vae-config",
                "--packed-decoder",
                "--max-tensors",
                "--max-source-bytes",
                "--max-metadata-bytes",
                "--latent-frames"
            ]
            .contains(&key.as_str()),
            "unknown argument {key}"
        );
        let value = argv
            .next()
            .with_context(|| format!("missing value for {key}"))?;
        ensure!(
            arguments.insert(key.clone(), value).is_none(),
            "duplicate argument {key}"
        );
    }
    let value = |key: &str| {
        arguments
            .get(key)
            .with_context(|| format!("missing required argument {key}"))
    };
    let positive = |key: &str| -> Result<usize> {
        let n: usize = value(key)?.parse()?;
        ensure!(
            n > 0 && n < isize::MAX as usize,
            "invalid positive capacity {key}"
        );
        Ok(n)
    };
    let maximum_metadata_bytes = positive("--max-metadata-bytes")?;
    let mut bytes = Vec::new();
    std::fs::File::open(value("--vae-config")?)?
        .take((maximum_metadata_bytes as u64) + 1)
        .read_to_end(&mut bytes)?;
    let config = FastH3VideoVaeConfig::parse(&bytes, maximum_metadata_bytes)?;
    let path = PathBuf::from(value("--packed-decoder")?);
    let basename = path
        .file_name()
        .and_then(|s| s.to_str())
        .context("decoder filename is not UTF-8")?;
    let archive = SafetensorsArchive::open_single_file(
        path.parent().context("decoder parent path")?,
        basename,
    )?;
    let package = FastH3VideoDecoderPackage::new(
        &config,
        archive,
        FastH3VideoDecoderWeightLimits {
            maximum_tensors: positive("--max-tensors")?,
            maximum_source_bytes: positive("--max-source-bytes")?,
            maximum_metadata_bytes,
        },
    )?;
    let mut storage = BTreeMap::<String, usize>::new();
    for c in package.components() {
        *storage
            .entry(format!("{:?}", c.physical_element_type()))
            .or_default() += 1;
    }
    let mut reciprocal_profile = None;
    for (prefix, _) in package.matrices() {
        let recipe = package.alpha_derivation(prefix)?;
        recipe.result_bits()?;
        recipe.output_component()?;
        reciprocal_profile = Some(recipe.arithmetic_profile().to_owned());
    }
    let plan = FastH3VideoDecodePlan::new(&config, positive("--latent-frames")?, usize::MAX)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "validation":"original full-depth packed video decoder inventory and payloads; no forward executed",
            "dense_payload_values_scanned":true,"decoder_layers":config.decoder_layers(),
            "nvfp4_matrices":package.matrices().len(),"physical_components":package.components().len(),
            "physical_component_storage_dtypes":storage,"source_payload_bytes":package.source_bytes(),
            "f32_reciprocal_recipe_count":package.matrices().len(),"reciprocal_profile":reciprocal_profile,
            "latent_frames":plan.latent_frames(),"repeat_last_latents":plan.repeat_last_latents(),
            "chunks":plan.chunk_count(),"output_frames":plan.output_frames(),
            "overlap_frames":plan.overlap_frames(),"final_overlap_frames":plan.final_overlap_frames(),
            "encoder_included":false,"audio_decoder_included":false
        }))?
    );
    Ok(())
}
