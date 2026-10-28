//! Shared in-process image product configuration and CLI adapters.
//! Standard local packages and explicit diagnostic configurations share one factory.
//! Architecture selection uses the pipeline declaration,
//! never a public model alias, repository name, or the text-model fallback.

mod configuration;
mod directory;
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
mod factory;
#[cfg(all(test, feature = "cuda"))]
mod gpu_tests;
#[cfg(any(feature = "cuda", test))]
mod host_cache;
#[cfg(any(feature = "cuda", test))]
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
pub use run::execute_run_source_with_image;
pub use serve::execute_serve;
pub use serve::execute_serve_source;

use clap::{parser::ValueSource, ArgMatches, Args};
use std::{fmt, path::PathBuf};

#[derive(Debug, Default, Args)]
pub struct ImageProductArgs {
    /// Diagnostic image JSON with explicit assets and quotas. A registered
    /// local MODEL directory uses adaptive residency and capability-derived
    /// defaults, including automatic CUDA batching and single-image decode.
    #[arg(long, value_name = "JSON")]
    pub image_config: Option<PathBuf>,
    /// Override cuDNN 9 discovery for a local image pipeline directory.
    /// Otherwise use LD_LIBRARY_PATH, the system cache and standard library directories.
    #[arg(long, value_name = "LIBRARY")]
    pub image_cudnn_library: Option<PathBuf>,
    /// Image Linear precision: bf16 (default), fp8_linear_experimental_v1, or
    /// {"components":{"text_encoder":"bf16","denoiser":"fp8_linear_experimental_v1"}}.
    /// FP8 Linear is available on CUDA and Metal. An explicit GGUF image
    /// config can select denoiser native_source.
    #[arg(long, value_name = "PRESET|JSON")]
    pub image_precision: Option<ferrum_engine::image_generation::ImagePrecisionPolicy>,
    /// One fixed image adapter: a PEFT directory, or a Diffusers A/B safetensors
    /// file with explicit --image-lora-base-model. Single files use unit export
    /// scaling unless --image-lora-alpha declares alpha/r scaling.
    #[arg(long, value_name = "DIRECTORY|SAFETENSORS")]
    pub image_lora: Option<PathBuf>,
    /// Fixed startup strength; finite negative and zero values are supported.
    #[arg(long, requires = "image_lora", allow_hyphen_values = true)]
    pub image_lora_strength: Option<f32>,
    /// Base-model binding for a single file or missing PEFT metadata.
    #[arg(long, requires = "image_lora")]
    pub image_lora_base_model: Option<String>,
    /// Explicit single-file alpha/r scaling; PEFT uses its own configuration.
    #[arg(long, requires = "image_lora")]
    pub image_lora_alpha: Option<f32>,
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn composed_model_name(base: &str, composition: &serde_json::Value) -> anyhow::Result<String> {
    let Some(adapter) = composition.get("lora") else {
        return Ok(base.to_owned());
    };
    let identity = adapter["identity"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing admitted image LoRA identity"))?;
    anyhow::ensure!(
        identity.len() == 64 && identity.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid image LoRA identity"
    );
    Ok(format!("{base}:lora:{}", &identity[..16]))
}

impl ImageProductArgs {
    fn lora_config(&self) -> anyhow::Result<Option<configuration::ImageLoraConfig>> {
        use ferrum_quantization::linear_adapter::{LinearAdapterScaling, LinearAdapterSource};
        let Some(path) = &self.image_lora else {
            return Ok(None);
        };
        let source = if std::fs::metadata(path)?.is_dir() {
            anyhow::ensure!(
                self.image_lora_alpha.is_none(),
                "PEFT image LoRA uses alpha from adapter_config.json"
            );
            LinearAdapterSource::Peft {
                directory: path.clone(),
                base_model: self.image_lora_base_model.clone(),
            }
        } else {
            LinearAdapterSource::Diffusers {
                file: path.clone(),
                base_model: self.image_lora_base_model.clone().ok_or_else(|| {
                    anyhow::anyhow!("a single-file image LoRA requires --image-lora-base-model")
                })?,
                scaling: self
                    .image_lora_alpha
                    .map_or(LinearAdapterScaling::Unit, |alpha| {
                        LinearAdapterScaling::Alpha { alpha }
                    }),
            }
        };
        let strength = self.image_lora_strength.unwrap_or(1.0);
        anyhow::ensure!(strength.is_finite(), "image LoRA strength must be finite");
        Ok(Some(configuration::ImageLoraConfig { source, strength }))
    }
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
                model.is_none() && self.image_cudnn_library.is_none() && self.image_precision.is_none()
                    && self.image_lora.is_none(),
                "--image-config cannot be combined with MODEL, --image-cudnn-library, or --image-precision; set precision_policy in the JSON backend"
            );
            return Ok(Some(ImageProductSource::ExplicitConfig(path.clone())));
        }
        let Some(path) = directory::discover(model)? else {
            anyhow::ensure!(
                self.image_cudnn_library.is_none() && self.image_precision.is_none() && self.image_lora.is_none(),
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
            .with_precision(self.image_precision.unwrap_or_default())?
            .with_lora(self.lora_config()?)?,
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
    /// Edit one static 8-bit RGB/RGBA PNG. No masks or multiple input images.
    #[arg(long, value_name = "PNG")]
    pub image: Option<PathBuf>,
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
            "image",
            "width",
            "height",
            "image_cudnn_library",
            "image_precision",
            "image_lora",
            "image_lora_strength",
            "image_lora_base_model",
            "image_lora_alpha",
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
            "image",
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
        "image_lora",
        "image_lora_strength",
        "image_lora_base_model",
        "image_lora_alpha",
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
    let maximum_dimension = if automatic {
        let options = DirectoryOptions::from_cli(
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
        )?
        .with_precision(
            args.get_one::<ferrum_engine::image_generation::ImagePrecisionPolicy>(
                "image_precision",
            )
            .copied()
            .unwrap_or_default(),
        )?;
        options.pipeline().maximum_width
    } else {
        u32::MAX
    };
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
                    *value > 0 && *value <= maximum_dimension,
                    "image {dimension} exceeds the positive preset bound ({maximum_dimension})"
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
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    Generation {
        source: Box<dyn std::error::Error + Send>,
        pipeline: Box<dyn ferrum_engine::image_generation::service::ImageGenerator>,
    },
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    Server {
        source: anyhow::Error,
        service: ferrum_server::ImageHttpService,
        shutdown: Option<ferrum_engine::image_generation::service::ImageServiceShutdown>,
    },
}
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
impl ImageProductError {
    fn generation<R: ferrum_runtime::DeviceRuntime>(
        source: ferrum_engine::image_generation::ImageGenerationError<R>,
        pipeline: Box<ferrum_engine::image_generation::QwenImage21Pipeline<R>>,
    ) -> Self {
        // Erase only the backend type: both original recovery owners survive.
        Self::Generation {
            source: Box::new(source),
            pipeline,
        }
    }
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
            #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
            Self::Generation { source, .. } => write!(f, "{source}"),
            #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
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
