//! Shared in-process image product configuration and CLI adapters.
//! Asset paths are explicit. Architecture selection uses the pipeline declaration,
//! never a public model alias, repository name, or the text-model fallback.

mod configuration;
#[cfg(all(test, feature = "cuda"))]
mod gpu_tests;
mod run;
mod serve;
mod source;
#[cfg(test)]
mod tests;

pub use configuration::ImageProductConfig;
pub use run::execute_run;
pub use serve::execute_serve;

use clap::{parser::ValueSource, ArgMatches, Args};
use std::{fmt, path::PathBuf};

#[derive(Debug, Args)]
pub struct ImageProductArgs {
    /// Explicit local image pipeline JSON (assets, CUDA/Host quotas and limits).
    #[arg(long, value_name = "JSON")]
    pub image_config: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct ImageRunArgs {
    #[command(flatten)]
    pub product: ImageProductArgs,
    /// PNG destination, created exclusively. Required for image generation.
    #[arg(long, value_name = "PNG", requires = "image_config")]
    pub output: Option<PathBuf>,
    /// Image width in pixels (image default: 512; never rounded).
    #[arg(long, requires = "image_config")]
    pub width: Option<u32>,
    /// Image height in pixels (image default: 512; never rounded).
    #[arg(long, requires = "image_config")]
    pub height: Option<u32>,
}

/// Inspect explicit CLI flags rather than treating supplied text defaults as
/// compatible image controls. This runs before configuration/source/device IO.
pub fn validate_cli_matches(
    definition: &clap::Command,
    matches: &ArgMatches,
) -> anyhow::Result<()> {
    let Some((command, args)) = matches.subcommand() else {
        return Ok(());
    };
    if !matches!(command, "run" | "serve")
        || args
            .try_get_one::<PathBuf>("image_config")
            .ok()
            .flatten()
            .is_none()
    {
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
    let definition = definition
        .find_subcommand(command)
        .ok_or_else(|| anyhow::anyhow!("missing command definition for {command}"))?;
    // Derived flatten groups also occur in ArgMatches::ids(); only actual
    // declared arguments are user controls.
    for argument in definition.get_arguments() {
        let id = argument.get_id();
        if args.value_source(id.as_str()) == Some(ValueSource::CommandLine)
            && !allowed.contains(&id.as_str())
        {
            anyhow::bail!(
                "--image-config does not support explicitly supplied text option {}",
                id.as_str()
            );
        }
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
