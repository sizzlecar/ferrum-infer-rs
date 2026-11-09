//! Real image pipeline node profiling. No GPU work without --run.
use anyhow::{ensure, Result};
use clap::{Parser, ValueEnum};
use ferrum_cli::commands::image_product::ImageProductError;
use std::{num::NonZeroUsize, path::PathBuf};

#[derive(Parser)]
#[command(group(
    clap::ArgGroup::new("source")
        .required(true)
        .args(["model", "image_config"])
))]
struct Arguments {
    #[arg(long)]
    run: bool,
    /// Local Qwen Image directory using the existing FP8 diagnostic preset.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Explicit image configuration; preserves its architecture and numerical preset.
    #[arg(long, value_name = "JSON")]
    image_config: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// One PNG reference per request; omission profiles text-to-image.
    #[arg(long)]
    image: Option<PathBuf>,
    /// Fresh directory for main-vision RGBA and four F32 features, including warmup.
    /// Synchronous capture adds diagnostic overhead to conditioning wall time.
    #[arg(
        long,
        requires_all = ["main_vision_capture_max_bytes", "image", "image_config"],
        conflicts_with = "model"
    )]
    main_vision_capture_dir: Option<PathBuf>,
    /// Cumulative raw payload limit for the entire capture directory; metadata
    /// is excluded. This is not a process RSS or Host allocation limit.
    #[arg(long, requires = "main_vision_capture_dir")]
    main_vision_capture_max_bytes: Option<NonZeroUsize>,
    /// Fresh directory for five raw main-text taps, including shape warmup.
    #[arg(long,
        requires_all = ["main_text_capture_max_bytes", "image", "image_config"],
        conflicts_with_all = ["model", "main_vision_capture_dir"])]
    main_text_capture_dir: Option<PathBuf>,
    /// Cumulative raw F32 payload limit for the entire text capture directory.
    #[arg(long, requires = "main_text_capture_dir")]
    main_text_capture_max_bytes: Option<NonZeroUsize>,
    /// Fresh directory for the completed normalized target and same-job PNG.
    #[arg(long, requires_all = ["final_latent_capture_max_bytes", "image_config"],
        conflicts_with_all = ["model", "main_vision_capture_dir", "main_text_capture_dir"])]
    final_latent_capture_dir: Option<PathBuf>,
    /// Cumulative raw latent byte limit including warmup; excludes PNG and
    /// metadata. PNG uses the ordinary encoder limits and existing Host scratch.
    #[arg(long, requires = "final_latent_capture_dir")]
    final_latent_capture_max_bytes: Option<NonZeroUsize>,
    /// Fresh directory for actual B1 DiT inputs, initial F32 state and same-job PNG.
    #[arg(long, requires_all = ["dit_input_capture_max_bytes", "image_config"],
        conflicts_with_all = ["model", "main_vision_capture_dir", "main_text_capture_dir", "final_latent_capture_dir"])]
    dit_input_capture_dir: Option<PathBuf>,
    /// Cumulative raw input budget across all steps and runs, excluding JSON/PNG.
    #[arg(long, requires = "dit_input_capture_dir")]
    dit_input_capture_max_bytes: Option<NonZeroUsize>,
    /// Fixed batches, each with its own warmup. This is not a serving sweep.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    batch: Vec<usize>,
    #[arg(long, default_value_t = 1)]
    warm_repetitions: usize,
    /// Off provides a matching uninstrumented execution for overhead comparison.
    #[arg(long, value_enum, default_value = "kernel")]
    device_timing: Timing,
    #[arg(long, default_value_t = 1024)]
    width: u32,
    #[arg(long, default_value_t = 1024)]
    height: u32,
    /// Directory memory utilization (default: 0.9); explicit configs own their quota.
    #[arg(long, requires = "model", conflicts_with = "image_config")]
    gpu_memory_utilization: Option<f32>,
    #[arg(long, requires = "model", conflicts_with = "image_config")]
    cudnn_library: Option<PathBuf>,
}

impl Arguments {
    /// Resolve metadata only; neither source loads assets or starts a device here.
    fn source(&self) -> Result<ferrum_cli::commands::image_product::ImageProductSource> {
        use ferrum_cli::commands::image_product::{DirectoryOptions, ImageProductSource};
        use ferrum_engine::image_generation::ImageCudaPrecisionPolicy;

        match (&self.model, &self.image_config) {
            (None, Some(path)) => Ok(ImageProductSource::ExplicitConfig(path.clone())),
            (Some(path), None) => {
                let options = DirectoryOptions::from_cli(
                    self.gpu_memory_utilization.unwrap_or(0.9),
                    None,
                    self.batch.iter().copied().max(),
                    "cuda",
                    None,
                    self.cudnn_library.clone(),
                )?
                .with_precision(ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1)?;
                Ok(ImageProductSource::LocalDirectory {
                    path: path.clone(),
                    options,
                })
            }
            _ => anyhow::bail!("select exactly one of --model or --image-config"),
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Timing {
    Off,
    Kernel,
}

fn validate_arguments(args: &Arguments) -> Result<()> {
    ensure!(
        args.dit_input_capture_dir.is_none() || args.batch == [1],
        "DiT input capture requires one B1 shape"
    );
    ensure!(
        args.run,
        "explicit --run is required; run only while the GPU is exclusively available"
    );
    ensure!(
        !args.batch.is_empty() && args.batch.iter().all(|batch| *batch > 0),
        "batch must be positive"
    );
    ensure!(
        args.warm_repetitions > 0,
        "warm repetitions must be positive"
    );
    ensure!(
        args.main_text_capture_dir.is_none() || args.batch == [1],
        "main text capture requires one B=1 profile shape"
    );
    ensure!(
        args.final_latent_capture_dir.is_none() || args.batch == [1],
        "final latent capture requires one B=1 profile shape"
    );
    Ok(())
}

fn main() -> Result<(), ImageProductError> {
    let args = Arguments::parse();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    validate_arguments(&args)?;
    #[cfg(not(feature = "cuda-cutlass-fp8"))]
    return Err(ImageProductError::Unsupported(
        "build with cuda-cutlass-fp8 and the matching locked native set",
    ));
    #[cfg(feature = "cuda-cutlass-fp8")]
    {
        use ferrum_cli::commands::image_product::{
            profile_image_source, ImageDitInputCaptureOptions, ImageFinalLatentCaptureOptions,
            ImageMainTextCaptureOptions, ImageMainVisionCaptureOptions, ImageProfileWorkload,
        };
        use ferrum_engine::image_generation::ImageGenerationRequest;
        let source = args.source()?;
        let main_vision_capture =
            args.main_vision_capture_dir
                .map(|output_directory| ImageMainVisionCaptureOptions {
                    output_directory,
                    maximum_payload_bytes: args
                        .main_vision_capture_max_bytes
                        .expect("clap requires a capture byte limit"),
                });
        let main_text_capture =
            args.main_text_capture_dir
                .map(|output_directory| ImageMainTextCaptureOptions {
                    output_directory,
                    maximum_payload_bytes: args
                        .main_text_capture_max_bytes
                        .expect("clap requires a text capture byte limit"),
                });
        let final_latent_capture =
            args.final_latent_capture_dir
                .map(|output_directory| ImageFinalLatentCaptureOptions {
                    output_directory,
                    maximum_payload_bytes: args
                        .final_latent_capture_max_bytes
                        .expect("clap requires a final latent byte limit"),
                });
        let request = ImageGenerationRequest {
            prompt: args.prompt,
            width: args.width,
            height: args.height,
            seed: args.seed,
            n: 1,
        };
        let file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(args.output)
            .map_err(anyhow::Error::from)?;
        profile_image_source(
            source,
            ImageProfileWorkload {
                request,
                image: args.image,
                batches: args.batch,
                warm_repetitions: args.warm_repetitions,
                timing: match args.device_timing {
                    Timing::Off => ferrum_runtime::DeviceTimingMode::Off,
                    Timing::Kernel => ferrum_runtime::DeviceTimingMode::Kernel,
                },
                main_vision_capture,
                main_text_capture,
                final_latent_capture,
                dit_input_capture: args.dit_input_capture_dir.map(|output_directory| {
                    ImageDitInputCaptureOptions {
                        output_directory,
                        maximum_payload_bytes: args
                            .dit_input_capture_max_bytes
                            .expect("clap requires a DiT capture byte limit"),
                    }
                }),
            },
            std::io::BufWriter::new(file),
        )
    }
}

#[cfg(test)]
#[path = "profile_image_pipeline/tests.rs"]
mod tests;
