//! Real image pipeline node profiling. No GPU work without --run.
use anyhow::{ensure, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    run: bool,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// One PNG reference per request; omission profiles text-to-image.
    #[arg(long)]
    image: Option<PathBuf>,
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
    #[arg(long, default_value_t = 0.9)]
    gpu_memory_utilization: f32,
    #[arg(long)]
    cudnn_library: Option<PathBuf>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Timing {
    Off,
    Kernel,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
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
    #[cfg(not(feature = "cuda-cutlass-fp8"))]
    anyhow::bail!("build with cuda-cutlass-fp8 and the matching locked native set");
    #[cfg(feature = "cuda-cutlass-fp8")]
    {
        use ferrum_cli::commands::image_product::{
            profile_image_source, DirectoryOptions, ImageProductSource, ImageProfileWorkload,
        };
        use ferrum_engine::image_generation::{ImageCudaPrecisionPolicy, ImageGenerationRequest};
        let options = DirectoryOptions::from_cli(
            args.gpu_memory_utilization,
            None,
            args.batch.iter().copied().max(),
            "cuda",
            None,
            args.cudnn_library,
        )?
        .with_precision(ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1)?;
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
            .open(args.output)?;
        profile_image_source(
            ImageProductSource::LocalDirectory {
                path: args.model,
                options,
            },
            ImageProfileWorkload {
                request,
                image: args.image,
                batches: args.batch,
                warm_repetitions: args.warm_repetitions,
                timing: match args.device_timing {
                    Timing::Off => ferrum_runtime::DeviceTimingMode::Off,
                    Timing::Kernel => ferrum_runtime::DeviceTimingMode::Kernel,
                },
            },
            std::io::BufWriter::new(file),
        )
    }
}
