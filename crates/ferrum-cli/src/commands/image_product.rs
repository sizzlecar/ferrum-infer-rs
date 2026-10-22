//! Shared in-process image product configuration and CLI adapters.
//! Standard local packages and explicit diagnostic configurations share one factory.
//! Architecture selection uses the pipeline declaration,
//! never a public model alias, repository name, or the text-model fallback.

mod configuration;
mod directory;
#[cfg(all(test, feature = "cuda"))]
mod gpu_tests;
mod library;
mod run;
mod serve;
mod source;
#[cfg(test)]
mod tests;

pub use configuration::ImageProductConfig;
pub use directory::{DirectoryOptions, ImageProductSource};
pub use run::execute_run;
pub use run::execute_run_source;
pub use serve::execute_serve;
pub use serve::execute_serve_source;

use clap::{parser::ValueSource, ArgMatches, Args};
use std::{fmt, path::PathBuf};

#[derive(Debug, Args)]
pub struct ImageProductArgs {
    /// Diagnostic image JSON (explicit assets and quotas). A registered local
    /// MODEL directory instead uses experimental_bf16_cold_v1: BF16, cold
    /// batches, 512 default / 1024 maximum pixels per axis, single-image decode.
    #[arg(long, value_name = "JSON")]
    pub image_config: Option<PathBuf>,
    /// Override cuDNN 9 discovery for a local image pipeline directory.
    /// Otherwise use LD_LIBRARY_PATH, the system cache and standard library directories.
    #[arg(long, value_name = "LIBRARY")]
    pub image_cudnn_library: Option<PathBuf>,
    /// Explicit image Linear precision. Default bf16; experimental FP8 changes
    /// TE/DiT projections only. Residency is configured independently.
    #[arg(long, value_name = "bf16|fp8_linear_experimental_v1")]
    pub image_precision: Option<ferrum_engine::image_generation::ImageCudaPrecisionPolicy>,
}

impl ImageProductArgs {
    pub fn source(
        &self,
        model: Option<&str>,
        utilization: f32,
        maximum: Option<std::num::NonZeroUsize>,
        batch: Option<usize>,
        backend: &str,
        gpu_devices: Option<&str>,
    ) -> anyhow::Result<Option<ImageProductSource>> {
        if let Some(path) = &self.image_config {
            anyhow::ensure!(
                model.is_none() && self.image_cudnn_library.is_none() && self.image_precision.is_none(),
                "--image-config cannot be combined with MODEL, --image-cudnn-library, or --image-precision; set precision_policy in the JSON backend"
            );
            return Ok(Some(ImageProductSource::ExplicitConfig(path.clone())));
        }
        let Some(path) = directory::discover(model)? else {
            anyhow::ensure!(
                self.image_cudnn_library.is_none() && self.image_precision.is_none(),
                "--image-cudnn-library and --image-precision require a registered image pipeline directory"
            );
            return Ok(None);
        };
        Ok(Some(ImageProductSource::LocalDirectory {
            path,
            options: DirectoryOptions::from_cli(
                utilization,
                maximum,
                batch,
                backend,
                gpu_devices,
                self.image_cudnn_library.clone(),
            )?
            .with_precision(self.image_precision.unwrap_or_default()),
        }))
    }
}

#[derive(Debug, Args)]
pub struct ImageRunArgs {
    #[command(flatten)]
    pub product: ImageProductArgs,
    /// PNG destination, created exclusively. Required for image generation.
    #[arg(long, value_name = "PNG")]
    pub output: Option<PathBuf>,
    /// Image width in pixels (image default: 512; never rounded).
    #[arg(long)]
    pub width: Option<u32>,
    /// Image height in pixels (image default: 512; never rounded).
    #[arg(long)]
    pub height: Option<u32>,
}

/// Inspect explicit CLI flags rather than treating supplied text defaults as
/// compatible image controls. Local directory recognition reads bounded
/// package metadata; no weight payload or device is opened here.
pub fn validate_cli_matches(
    definition: &clap::Command,
    matches: &ArgMatches,
) -> anyhow::Result<()> {
    let Some((command, args)) = matches.subcommand() else {
        return Ok(());
    };
    if !matches!(command, "run" | "serve") {
        return Ok(());
    }
    let explicit = args
        .try_get_one::<PathBuf>("image_config")
        .ok()
        .flatten()
        .is_some();
    let model = args
        .try_get_one::<String>("model")
        .ok()
        .flatten()
        .or_else(|| args.try_get_one::<String>("model_option").ok().flatten());
    let automatic = !explicit && directory::discover(model.map(String::as_str))?.is_some();
    if !explicit && !automatic {
        for id in [
            "output",
            "width",
            "height",
            "image_cudnn_library",
            "image_precision",
        ] {
            anyhow::ensure!(
                !(args.try_contains_id(id).unwrap_or(false)
                    && args.value_source(id) == Some(ValueSource::CommandLine)),
                "{id} requires a registered local image pipeline directory or --image-config"
            );
        }
        return Ok(());
    }
    let allowed = match command {
        "run" => &[
            "image_config",
            "prompt",
            "output",
            "width",
            "height",
            "seed",
            "verbose",
        ][..],
        "serve" => &[
            "image_config",
            "host",
            "port",
            "served_model_name",
            "verbose",
        ][..],
        _ => unreachable!(),
    };
    let automatic_controls = [
        "model",
        "model_option",
        "backend",
        "gpu_devices",
        "gpu_memory_utilization",
        "runtime_memory_budget_bytes",
        "max_num_seqs",
        "image_cudnn_library",
        "image_precision",
    ];
    let definition = definition
        .find_subcommand(command)
        .ok_or_else(|| anyhow::anyhow!("missing command definition for {command}"))?;
    // Derived flatten groups also occur in ArgMatches::ids(); only actual
    // declared arguments are user controls.
    for argument in definition.get_arguments() {
        let id = argument.get_id();
        if args.value_source(id.as_str()) == Some(ValueSource::CommandLine)
            && !allowed.contains(&id.as_str())
            && !(automatic && automatic_controls.contains(&id.as_str()))
        {
            anyhow::bail!(
                "image generation does not support explicitly supplied text option {}",
                id.as_str()
            );
        }
    }
    if automatic {
        DirectoryOptions::from_cli(
            *args
                .get_one::<f32>("gpu_memory_utilization")
                .ok_or_else(|| anyhow::anyhow!("missing gpu-memory-utilization default"))?,
            args.get_one::<std::num::NonZeroUsize>("runtime_memory_budget_bytes")
                .copied(),
            args.get_one::<usize>("max_num_seqs").copied(),
            args.get_one::<String>("backend")
                .map(String::as_str)
                .unwrap_or("auto"),
            args.get_one::<String>("gpu_devices").map(String::as_str),
            args.get_one::<PathBuf>("image_cudnn_library").cloned(),
        )?;
    }
    if command == "run" {
        anyhow::ensure!(
            args.get_one::<String>("prompt").is_some(),
            "image run requires --prompt"
        );
        anyhow::ensure!(
            args.get_one::<PathBuf>("output").is_some(),
            "image run requires --output"
        );
        for dimension in ["width", "height"] {
            if let Some(value) = args.get_one::<u32>(dimension) {
                anyhow::ensure!(
                    *value > 0 && (!automatic || *value <= 1024),
                    "image {dimension} exceeds the positive preset bound (1024)"
                );
            }
        }
    }
    Ok(())
}

/// The image variant keeps original device recovery owners, not their text.
pub enum ImageProductError {
    Configuration(anyhow::Error),
    Unsupported(&'static str),
    #[cfg(feature = "cuda")]
    Generation {
        source: ferrum_engine::image_generation::ImageGenerationError<
            ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime,
        >,
        pipeline: Box<
            ferrum_engine::image_generation::QwenImage21Pipeline<
                ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime,
            >,
        >,
    },
    #[cfg(feature = "cuda")]
    Server {
        source: anyhow::Error,
        service: ferrum_server::ImageHttpService,
        shutdown: Option<ferrum_engine::image_generation::service::ImageServiceShutdown>,
    },
}
impl From<anyhow::Error> for ImageProductError {
    fn from(value: anyhow::Error) -> Self {
        Self::Configuration(value)
    }
}
impl fmt::Display for ImageProductError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(e) => write!(f, "image configuration/output: {e:#}"),
            Self::Unsupported(reason) => f.write_str(reason),
            #[cfg(feature = "cuda")]
            Self::Generation { source, .. } => write!(f, "{source}"),
            #[cfg(feature = "cuda")]
            Self::Server { source, .. } => write!(f, "image server: {source:#}"),
        }
    }
}
impl fmt::Debug for ImageProductError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl std::error::Error for ImageProductError {}

/// Text construction cleanup retains its existing observation API.
pub enum CommandError {
    Text(ferrum_execution::ExecutorBuildError),
    Image(ImageProductError),
}
impl CommandError {
    pub fn cleanup(&self) -> Option<&ferrum_execution::ExecutorShutdownObserver> {
        match self {
            Self::Text(e) => e.cleanup(),
            Self::Image(_) => None,
        }
    }
}
impl From<ferrum_execution::ExecutorBuildError> for CommandError {
    fn from(e: ferrum_execution::ExecutorBuildError) -> Self {
        Self::Text(e)
    }
}
impl From<ferrum_types::FerrumError> for CommandError {
    fn from(e: ferrum_types::FerrumError) -> Self {
        Self::Text(e.into())
    }
}
impl From<ImageProductError> for CommandError {
    fn from(e: ImageProductError) -> Self {
        Self::Image(e)
    }
}
impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(e) => fmt::Display::fmt(e, f),
            Self::Image(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl fmt::Debug for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl std::error::Error for CommandError {}
