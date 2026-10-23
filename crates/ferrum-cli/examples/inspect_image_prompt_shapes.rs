//! Tokenizer-only DiT shape inspection; no model weights or device startup.
//!
//! `image_tokens` is an explicit diagnostic input, not inferred from image
//! dimensions. Its default 4096 represents the 64 x 64 latent grid of the
//! current 1024 x 1024 workload. Tokenizer metadata is outside the Host payload
//! ledger, as in the existing prompt adapter.

use anyhow::{ensure, Context, Result};
use clap::Parser;
use ferrum_bench_core::images::{ImageDataset, ImageResponseLimits};
use ferrum_models::qwen_image21::t2i_prompt::QwenImage21T2iTokenizer;
use serde_json::json;
use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(about = "Inspect T2I prompt and canonical batch shapes without weights or CUDA")]
struct Arguments {
    /// Directory containing tokenizer.json, tokenizer_config.json and chat_template.jinja.
    #[arg(long)]
    tokenizer_dir: PathBuf,
    /// Existing image benchmark dataset; one inspected batch of up to 16 samples.
    #[arg(long)]
    dataset: PathBuf,
    /// Match the product's user-token limit; truncation is never enabled.
    #[arg(long, default_value_t = 8192)]
    maximum_user_tokens: usize,
    /// Explicit latent sequence length used in M = B * (image_tokens + T).
    #[arg(long, default_value_t = 4096)]
    image_tokens: usize,
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() > 0 && metadata.len() <= limit,
        "{} must be a nonempty regular file within {limit} bytes",
        path.display()
    );
    let mut bytes = Vec::new();
    file.take(limit.checked_add(1).context("file read bound overflows")?)
        .read_to_end(&mut bytes)?;
    ensure!(
        !bytes.is_empty() && u64::try_from(bytes.len())? <= limit,
        "{} changed beyond its read bound",
        path.display()
    );
    Ok(bytes)
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    ensure!(args.image_tokens > 0, "image token count must be positive");
    let dataset: ImageDataset =
        serde_json::from_slice(&read_bounded(&args.dataset, 8 * 1024 * 1024)?)?;
    dataset.validate(&ImageResponseLimits::default())?;
    ensure!(
        dataset.samples.len() <= 16,
        "one inspected batch supports at most 16 samples"
    );

    let tokenizer = QwenImage21T2iTokenizer::from_source_bytes(
        &read_bounded(&args.tokenizer_dir.join("tokenizer.json"), 32 * 1024 * 1024)?,
        &read_bounded(
            &args.tokenizer_dir.join("tokenizer_config.json"),
            1024 * 1024,
        )?,
        &read_bounded(&args.tokenizer_dir.join("chat_template.jinja"), 64 * 1024)?,
    )?;
    let prompts: Vec<_> = dataset
        .samples
        .iter()
        .map(|sample| sample.prompt.as_str())
        .collect();
    let (batch, order) = tokenizer.prepare_canonical(&prompts, args.maximum_user_tokens)?;
    let canonical_lengths = batch.valid_feature_lengths();
    let mut original_lengths = vec![0; order.len()];
    for (&original_index, &length) in order.iter().zip(&canonical_lengths) {
        original_lengths[original_index] = length;
    }
    let samples = dataset
        .samples
        .iter()
        .zip(original_lengths)
        .map(|(sample, length)| {
            let m = args
                .image_tokens
                .checked_add(length)
                .context("C1 row count overflows")?;
            Ok(json!({"id": sample.id, "feature_sequence_length": length, "c1_m": m}))
        })
        .collect::<Result<Vec<_>>>()?;
    let joint_sequence = args
        .image_tokens
        .checked_add(batch.feature_sequence_length())
        .context("batch joint sequence length overflows")?;
    let m = batch
        .batch()
        .checked_mul(joint_sequence)
        .context("batch row count overflows")?;
    let hashes = tokenizer.source_sha256();
    let report = json!({
        "schema": "ferrum.image_prompt_shapes.v1",
        "scope": "tokenizer-only; no weight access, device initialization or execution",
        "dataset_fingerprint": dataset.fingerprint()?,
        "tokenizer_source_sha256": {
            "tokenizer.json": hashes[0],
            "tokenizer_config.json": hashes[1],
            "chat_template.jinja": hashes[2]
        },
        "maximum_user_tokens": args.maximum_user_tokens,
        "image_tokens": args.image_tokens,
        "drop_prefix_tokens": batch.drop_prefix_tokens(),
        "samples": samples,
        "canonical_batch": {
            "batch_size": batch.batch(),
            "execution_to_original_index": order,
            "valid_feature_lengths": canonical_lengths,
            "feature_sequence_length": batch.feature_sequence_length(),
            "joint_sequence_length": joint_sequence,
            "m": m
        }
    });
    let mut output = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut output, &report)?;
    output.write_all(b"\n")?;
    Ok(())
}
