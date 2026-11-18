//! Video entrypoints use the same model composition and iterative service.
use clap::{parser::ValueSource, ArgMatches, Args};
use std::{any::Any, fmt, path::PathBuf};

#[cfg(feature = "cuda-fast-h3")]
mod configuration;
#[cfg(feature = "cuda-fast-h3")]
mod execution;
#[cfg(any(feature = "cuda-fast-h3", test))]
mod media;
#[cfg(any(feature = "cuda-fast-h3", test))]
mod preset;
#[cfg(test)]
mod tests;

#[derive(Debug, Default, Args)]
pub struct VideoProductArgs {
    /// FastH3 video JSON: explicit assets, CUDA libraries, queue and output limits.
    #[arg(long, value_name = "JSON")]
    pub video_config: Option<PathBuf>,
}

#[derive(Debug, Default, Args)]
pub struct VideoRunArgs {
    #[command(flatten)]
    pub product: VideoProductArgs,
    /// Exact video frame count (FastH3 default: 124). Never rounded or truncated.
    #[arg(long, requires = "video_config")]
    pub num_frames: Option<u32>,
    /// Integer video frame rate (FastH3 default: 24).
    #[arg(long, requires = "video_config")]
    pub fps: Option<u32>,
    /// Ordered reference: image=FILE.png, video=FILE.mp4 or audio=FILE.wav (32 kHz).
    /// Repeat to mix media; use --image for the image-only shorthand.
    #[arg(
        long = "reference",
        value_name = "KIND=FILE",
        requires = "video_config",
        conflicts_with = "image"
    )]
    pub references: Vec<VideoReferencePath>,
}

#[derive(Debug, Clone)]
pub struct VideoReferencePath {
    pub kind: ferrum_engine::video_generation::VideoReferenceKind,
    pub path: PathBuf,
}
impl std::str::FromStr for VideoReferencePath {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        use ferrum_engine::video_generation::VideoReferenceKind;
        let (kind, path) = value
            .split_once('=')
            .ok_or("expected image=FILE, video=FILE or audio=FILE")?;
        let kind = match kind {
            "image" => VideoReferenceKind::Image,
            "audio" => VideoReferenceKind::Audio,
            "video" => VideoReferenceKind::Video,
            _ => return Err("supported reference types are image, video and audio".into()),
        };
        if path.is_empty() {
            return Err("reference path must not be empty".into());
        }
        Ok(Self {
            kind,
            path: path.into(),
        })
    }
}

/// Only controls implemented by this product can be explicitly selected.
/// Text defaults do not silently alter the video inference configuration.
pub fn validate_cli_matches(
    definition: &clap::Command,
    matches: &ArgMatches,
) -> anyhow::Result<bool> {
    let Some((command, args)) = matches.subcommand() else {
        return Ok(false);
    };
    if !matches!(command, "run" | "serve")
        || args
            .try_get_one::<PathBuf>("video_config")
            .ok()
            .flatten()
            .is_none()
    {
        return Ok(false);
    }
    let allowed = match command {
        "run" => &[
            "video_config",
            "prompt",
            "output",
            "image",
            "references",
            "width",
            "height",
            "seed",
            "num_frames",
            "fps",
            "verbose",
        ][..],
        "serve" => &[
            "video_config",
            "host",
            "port",
            "served_model_name",
            "api_key",
            "verbose",
        ][..],
        _ => unreachable!(),
    };
    let definition = definition
        .find_subcommand(command)
        .ok_or_else(|| anyhow::anyhow!("missing {command} command definition"))?;
    for argument in definition.get_arguments() {
        let id = argument.get_id().as_str();
        anyhow::ensure!(
            args.value_source(id) != Some(ValueSource::CommandLine) || allowed.contains(&id),
            "video generation does not support explicitly supplied option {id}; configure its declared video product fields"
        );
    }
    Ok(true)
}

pub struct VideoRunRequest {
    pub config: PathBuf,
    pub parameters: ferrum_engine::video_generation::VideoGenerationParameters,
    pub references: Vec<VideoReferencePath>,
    pub destination: PathBuf,
}

pub async fn execute_run(request: VideoRunRequest) -> Result<(), VideoProductError> {
    #[cfg(feature = "cuda-fast-h3")]
    {
        execution::run(request).await
    }
    #[cfg(not(feature = "cuda-fast-h3"))]
    {
        let _ = request;
        Err(anyhow::anyhow!("FastH3 video generation requires the cuda-fast-h3 feature").into())
    }
}

pub async fn execute_serve(
    config: PathBuf,
    host: String,
    port: u16,
    aliases: Vec<String>,
    api_key: Option<ferrum_server::ApiKey>,
) -> Result<(), VideoProductError> {
    #[cfg(feature = "cuda-fast-h3")]
    {
        execution::serve(config, host, port, aliases, api_key).await
    }
    #[cfg(not(feature = "cuda-fast-h3"))]
    {
        let _ = (config, host, port, aliases, api_key);
        Err(anyhow::anyhow!("FastH3 video serving requires the cuda-fast-h3 feature").into())
    }
}

/// Keep original model/worker/codec retirement owners alive through reporting.
pub enum VideoProductError {
    Configuration(anyhow::Error),
    Retained {
        diagnostic: String,
        owner: Box<dyn Any + Send>,
    },
}
impl VideoProductError {
    #[cfg(feature = "cuda-fast-h3")]
    fn retain<T: Any + Send>(diagnostic: impl Into<String>, owner: T) -> Self {
        Self::Retained {
            diagnostic: diagnostic.into(),
            owner: Box::new(owner),
        }
    }
    #[cfg(feature = "cuda-fast-h3")]
    fn retain_error<E: fmt::Display + Send + 'static>(stage: &'static str, error: E) -> Self {
        let owner = Box::new(error);
        let diagnostic =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| format!("{stage}: {owner}")))
                .unwrap_or_else(|_| format!("{stage} failed; original error owner retained"));
        Self::Retained { diagnostic, owner }
    }
}
impl From<anyhow::Error> for VideoProductError {
    fn from(error: anyhow::Error) -> Self {
        Self::Configuration(error)
    }
}
impl fmt::Display for VideoProductError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(error) => fmt::Display::fmt(error, f),
            Self::Retained { diagnostic, .. } => f.write_str(diagnostic),
        }
    }
}
impl fmt::Debug for VideoProductError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl std::error::Error for VideoProductError {}
