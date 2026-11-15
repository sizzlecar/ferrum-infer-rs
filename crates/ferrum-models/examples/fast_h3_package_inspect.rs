//! CPU-only checkpoint validation. Does not register or execute a video model.
//!
//! cargo run -p ferrum-models --example fast_h3_package_inspect --
//!   --model-dir /path/to/checkpoint --profile dit-vsa
//!   --max-source-bytes 68719476736 --max-tensors 4096 --max-metadata-bytes 1048576

use std::{collections::BTreeMap, path::PathBuf};

use anyhow::{bail, ensure, Context, Result};
use ferrum_models::fast_h3::{FastH3Nvfp4Profile, FastH3WeightLimits, FastH3WeightStorage};
#[path = "support/fast_h3_package.rs"]
mod package;

struct Args {
    model_dir: PathBuf,
    profile: FastH3Nvfp4Profile,
    max_metadata_bytes: usize,
    limits: FastH3WeightLimits,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut values = BTreeMap::new();
        let mut args = std::env::args().skip(1);
        while let Some(key) = args.next() {
            ensure!(["--model-dir", "--profile", "--max-source-bytes", "--max-tensors", "--max-metadata-bytes"].contains(&key.as_str()),
                "expected --model-dir PATH --profile ffn|dit|dit-vsa --max-source-bytes N --max-tensors N --max-metadata-bytes N; unknown argument {key}");
            let value = args
                .next()
                .with_context(|| format!("missing value for {key}"))?;
            ensure!(
                values.insert(key.clone(), value).is_none(),
                "duplicate argument {key}"
            );
        }
        let value = |key: &str| {
            values
                .get(key)
                .with_context(|| format!("missing required argument {key}"))
        };
        let positive = |key: &str| -> Result<usize> {
            let n: usize = value(key)?
                .parse()
                .with_context(|| format!("{key} must be a positive integer"))?;
            ensure!(
                n > 0 && n < isize::MAX as usize,
                "{key} exceeds address space or is zero"
            );
            Ok(n)
        };
        let profile = match value("--profile")?.as_str() {
            "ffn" => FastH3Nvfp4Profile::FeedForward,
            "dit" => FastH3Nvfp4Profile::DiT,
            "dit-vsa" => FastH3Nvfp4Profile::DiTWithVsa,
            other => bail!("unsupported explicit NVFP4 deployment profile {other}"),
        };
        let max_metadata_bytes = positive("--max-metadata-bytes")?;
        Ok(Self {
            model_dir: PathBuf::from(value("--model-dir")?),
            profile,
            max_metadata_bytes,
            limits: FastH3WeightLimits {
                maximum_tensors: positive("--max-tensors")?,
                maximum_source_bytes: positive("--max-source-bytes")?,
                maximum_skip_metadata_bytes: max_metadata_bytes,
            },
        })
    }
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let package = package::load(
        &args.model_dir,
        args.profile,
        args.max_metadata_bytes,
        args.limits,
    )?;
    let mut storage_dtypes = BTreeMap::<String, usize>::new();
    for component in package.components() {
        *storage_dtypes
            .entry(format!("{:?}", component.physical_element_type()))
            .or_default() += 1;
    }
    let mut compute_dtypes = BTreeMap::<String, usize>::new();
    let (mut dense, mut packed, mut static_inputs) = (0, 0, 0);
    for weight in package.weights() {
        *compute_dtypes
            .entry(format!("{:?}", weight.compute_element_type))
            .or_default() += 1;
        match &weight.storage {
            FastH3WeightStorage::Dense(_) => dense += 1,
            FastH3WeightStorage::Nvfp4 { .. } => {
                let matrix = package.nvfp4_matrix(&weight.name)?;
                packed += 1;
                static_inputs += usize::from(matrix.input_global_scale().is_some());
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "validation": "transformer checkpoint inventory and NVFP4 payload scales; no forward executed",
            "dense_payload_values_scanned": false,
            "profile": format!("{:?}", package.profile()),
            "skip_blocks": package.skip_blocks(),
            "logical_weights": package.weights().len(),
            "dense_weights": dense,
            "nvfp4_matrices": packed,
            "authoritative_f32_alpha_validated": packed,
            "static_input_scales_validated": static_inputs,
            "physical_components": package.components().len(),
            "physical_component_storage_dtypes": storage_dtypes,
            "logical_compute_dtypes": compute_dtypes,
            "source_payload_bytes": package.source_bytes(),
            "retained_dense_originals": package.retained_dense_originals(),
            "metadata_limit_per_file_bytes": args.max_metadata_bytes
        }))?
    );
    Ok(())
}
