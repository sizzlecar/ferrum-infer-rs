//! Execute a complete native H3 block from a real original Comfy checkpoint.
//! Synthetic hidden states are an execution/capacity diagnostic, not an image
//! quality oracle, complete transformer, denoising pipeline or serving benchmark.
#[path = "verify_qwen_image21_projection/artifacts.rs"]
#[cfg(feature = "cuda")]
mod artifacts;
#[path = "verify_qwen_image21_projection/arguments.rs"]
mod backend;
#[cfg(feature = "cuda")]
#[path = "verify_p_h3_image_block/execution.rs"]
mod execution;
use anyhow::{bail, ensure, Context, Result};
use std::{collections::BTreeMap, ffi::OsString, path::PathBuf};

const USAGE: &str = "verify_p_h3_image_block --config TRANSFORMER_CONFIG_JSON --model-dir CHECKPOINT_DIRECTORY --checkpoint-file ORIGINAL_SAFETENSORS_BASENAME --artifact-dir NEW_EXTERNAL_DIRECTORY --host-memory-bytes BYTES --backend cuda [--block-index 0] [--batch 2] [--text-rows 512] [--latent-height 2] [--latent-width 2] [--max-metadata-bytes 1048576] [CUDA budget options from verify_qwen_image21_projection]";
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Arguments {
    config: PathBuf,
    model_dir: PathBuf,
    checkpoint_file: String,
    artifact_dir: PathBuf,
    host_memory_bytes: usize,
    max_metadata_bytes: usize,
    block_index: u64,
    batch: usize,
    text_rows: usize,
    latent_height: usize,
    latent_width: usize,
    cuda: backend::CudaOptions,
}
impl Arguments {
    fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Option<Self>> {
        let mut args = args.into_iter().peekable();
        if args.peek().is_some_and(|v| v == "--help" || v == "-h") {
            args.next();
            ensure!(args.next().is_none(), "--help accepts no other options");
            return Ok(None);
        }
        let mut values = BTreeMap::new();
        while let Some(flag) = args.next() {
            let flag = flag
                .into_string()
                .map_err(|_| anyhow::anyhow!("option requires UTF-8"))?;
            let value = args.next().context("each option requires a value")?;
            ensure!(
                values.insert(flag.clone(), value).is_none(),
                "duplicate option {flag}"
            );
        }
        let config = values.remove("--config").context(USAGE)?.into();
        let model_dir = values.remove("--model-dir").context(USAGE)?.into();
        let artifact_dir = values.remove("--artifact-dir").context(USAGE)?.into();
        let checkpoint_file = values
            .remove("--checkpoint-file")
            .context(USAGE)?
            .into_string()
            .map_err(|_| anyhow::anyhow!("checkpoint basename requires UTF-8"))?;
        let mut number = |name: &str, default: Option<u64>| -> Result<u64> {
            match values.remove(name) {
                Some(v) => Ok(v.to_str().context("integer requires UTF-8")?.parse()?),
                None => default.with_context(|| format!("missing {name}")),
            }
        };
        let host_memory_bytes = usize::try_from(number("--host-memory-bytes", None)?)?;
        let max_metadata_bytes = usize::try_from(number("--max-metadata-bytes", Some(1 << 20))?)?;
        let block_index = number("--block-index", Some(0))?;
        let batch = usize::try_from(number("--batch", Some(2))?)?;
        let text_rows = usize::try_from(number("--text-rows", Some(512))?)?;
        let latent_height = usize::try_from(number("--latent-height", Some(2))?)?;
        let latent_width = usize::try_from(number("--latent-width", Some(2))?)?;
        ensure!(
            [
                host_memory_bytes,
                max_metadata_bytes,
                batch,
                text_rows,
                latent_height,
                latent_width
            ]
            .iter()
            .all(|v| *v > 0 && *v < isize::MAX as usize),
            "all capacities and dimensions must be positive and addressable"
        );
        ensure!(
            batch
                .checked_mul(
                    std::mem::size_of::<ferrum_models::p_h3_image::PH3ImageRequestLayout<'static>>(
                    ) + std::mem::size_of::<ferrum_models::p_h3_image::PH3ImageCurveBlockRequest>(),
                )
                .and_then(|n| {
                    text_rows
                        .checked_mul(std::mem::size_of::<
                            ferrum_models::p_h3_image::PH3ImageModality,
                        >())?
                        .checked_add(n)
                })
                .is_some_and(|n| n <= max_metadata_bytes),
            "request descriptors and token tags exceed declared metadata capacity"
        );
        let backend = backend::Backend::parse_options(&mut values)?;
        backend.ensure_available()?;
        let backend::Backend::Cuda(cuda) = backend else {
            bail!("original W4/W8 execution requires --backend cuda");
        };
        Ok(Some(Self {
            config,
            model_dir,
            checkpoint_file,
            artifact_dir,
            host_memory_bytes,
            max_metadata_bytes,
            block_index,
            batch,
            text_rows,
            latent_height,
            latent_width,
            cuda,
        }))
    }
}
fn main() -> Result<()> {
    let Some(args) = Arguments::parse(std::env::args_os().skip(1))? else {
        println!("{USAGE}\n{}", backend::USAGE);
        return Ok(());
    };
    #[cfg(feature = "cuda")]
    {
        execution::run(args)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = args;
        bail!("requires --features cuda")
    }
}
