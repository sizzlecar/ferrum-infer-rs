//! A bounded, adaptive-residency preset for standard local pipeline packages.
//! Limits are admission bounds, not measured memory peaks or performance claims.
use super::configuration::{HostConfig, PipelineConfig, ServingConfig};
use anyhow::{ensure, Context, Result};
use ferrum_models::pipeline_source::{
    PipelineArchitecture, PipelineSourceBundle, PipelineSourceLimits,
};
use std::{
    num::NonZeroU64,
    path::{Path, PathBuf},
};

pub const PRESET_NAME: &str = "bf16_adaptive_v1";
const MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageBackendKind {
    Cuda,
    Metal,
}

#[derive(Debug, Clone)]
pub struct DirectoryOptions {
    pub backend: ImageBackendKind,
    pub utilization: f64,
    pub maximum_memory_bytes: Option<NonZeroU64>,
    pub maximum_batch_size: usize,
    pub batching: ferrum_types::ImageBatchPolicy,
    pub ordinal: usize,
    pub cudnn_library: Option<PathBuf>,
    pub precision: ferrum_engine::image_generation::ImageCudaPrecisionPolicy,
}
impl DirectoryOptions {
    /// A direct invocation has exactly one image request, regardless of the
    /// serving capacity ceiling supplied on the command line.
    pub fn single_request(&self) -> Self {
        let mut options = self.clone();
        options.maximum_batch_size = 1;
        options
    }

    pub fn from_cli(
        utilization: f32,
        maximum: Option<std::num::NonZeroUsize>,
        batch: Option<usize>,
        backend: &str,
        gpu_devices: Option<&str>,
        cudnn_library: Option<PathBuf>,
    ) -> Result<Self> {
        let backend = match backend {
            "auto" if cfg!(all(feature = "metal", target_os = "macos")) => ImageBackendKind::Metal,
            "auto" | "cuda" => ImageBackendKind::Cuda,
            "metal" => ImageBackendKind::Metal,
            _ => anyhow::bail!("image generation supports --backend auto, cuda or metal"),
        };
        ensure!(
            backend != ImageBackendKind::Metal || gpu_devices.is_none(),
            "Metal uses the default Apple GPU; --gpu-devices is a CUDA ordinal option"
        );
        let ordinal = match gpu_devices {
            None => 0,
            Some(value) => {
                ensure!(
                    !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()),
                    "image generation requires exactly one CUDA ordinal in --gpu-devices"
                );
                value
                    .parse::<usize>()
                    .context("invalid image CUDA ordinal")?
            }
        };
        let result = Self {
            backend,
            utilization: f64::from(utilization),
            maximum_memory_bytes: maximum
                .map(|v| u64::try_from(v.get()).map(|n| NonZeroU64::new(n).unwrap()))
                .transpose()?,
            maximum_batch_size: batch.unwrap_or(match backend {
                ImageBackendKind::Cuda => ferrum_runtime::STRUCTURED_ATTENTION_BF16_MAX_BATCH,
                ImageBackendKind::Metal => 1,
            }),
            batching: match backend {
                ImageBackendKind::Cuda => ferrum_types::ImageBatchPolicy::Auto,
                ImageBackendKind::Metal => ferrum_types::ImageBatchPolicy::Configured,
            },
            ordinal,
            cudnn_library,
            precision: Default::default(),
        };
        result.validate()?;
        Ok(result)
    }
    pub fn with_precision(
        mut self,
        precision: ferrum_engine::image_generation::ImageCudaPrecisionPolicy,
    ) -> Result<Self> {
        self.precision = precision;
        self.validate()?;
        Ok(self)
    }
    pub const fn preset_name(&self) -> &'static str {
        match self.precision {
            ferrum_engine::image_generation::ImageCudaPrecisionPolicy::Bf16 => PRESET_NAME,
            ferrum_engine::image_generation::ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1 => {
                "fp8_linear_adaptive_v1"
            }
            ferrum_engine::image_generation::ImageCudaPrecisionPolicy::Components(_) => {
                "component_precision_adaptive_v1"
            }
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.backend == ImageBackendKind::Metal {
            ensure!(
                self.ordinal == 0 && self.cudnn_library.is_none(),
                "Metal does not accept CUDA ordinals or cuDNN libraries"
            );
        }
        ensure!(
            self.utilization.is_finite() && self.utilization > 0.0 && self.utilization <= 1.0,
            "gpu-memory-utilization must be finite and in (0, 1]"
        );
        ensure!(
            (1..=ferrum_runtime::STRUCTURED_ATTENTION_BF16_MAX_BATCH)
                .contains(&self.maximum_batch_size),
            "max-num-seqs exceeds supported image batch capacity"
        );
        i32::try_from(self.ordinal).context("CUDA ordinal exceeds driver range")?;
        Ok(())
    }
    pub fn serving(&self) -> Result<ServingConfig> {
        self.serving_with_output_bytes(self.host().output_bytes)
    }
    pub(super) fn serving_with_output_bytes(&self, output_bytes: usize) -> Result<ServingConfig> {
        self.validate()?;
        let host = self.host();
        let pending_bytes = host.default_pending_bytes()?;
        let automatic_cuda = self.backend == ImageBackendKind::Cuda
            && self.batching == ferrum_types::ImageBatchPolicy::Auto;
        let pending = if automatic_cuda {
            pending_bytes
                .get()
                .checked_div(
                    ferrum_engine::image_generation::service::ImageService::minimum_pending_request_bytes(),
                )
                .filter(|count| *count > 0)
                .context("Host pending budget cannot admit one image request")?
        } else {
            self.maximum_batch_size.max(4)
        };
        let active_sessions = pending
            .checked_add(self.maximum_batch_size)
            .context("image output session count overflows")?;
        let max_sessions = if automatic_cuda {
            use ferrum_output::png::{encoded_rgba8_png_bytes, PngDimensions, PngLimits};
            // Every successful response owns at least a 1x1 PNG's bytes.
            // JSON/base64 and physical alignment only increase that cost. This
            // is a count ceiling, not a reservation or a chosen workload size.
            let minimum_output_bytes = usize::try_from(encoded_rgba8_png_bytes(
                PngDimensions::new(1, 1)?,
                PngLimits::default(),
            )?)?;
            let completed_sessions = output_bytes / minimum_output_bytes;
            let sessions = active_sessions
                .checked_add(completed_sessions)
                .context("automatic image output session count overflows")?;
            ensure!(
                sessions <= tokio::sync::Semaphore::MAX_PERMITS,
                "automatic image output session count exceeds semaphore capacity"
            );
            sessions
                .checked_mul(ferrum_output::TERMINAL_ERROR_BYTES)
                .context("automatic image output error allowance exceeds address space")?;
            sessions
        } else {
            active_sessions
        };
        let serving = ServingConfig {
            batching: self.batching,
            maximum_pending: pending,
            maximum_pending_bytes: Some(pending_bytes),
            maximum_prompt_bytes: (1024 * 1024 / self.maximum_batch_size).min(65536),
            max_num_seqs: self.maximum_batch_size,
            request_batch_max_wait_ms: if self.batching == ferrum_types::ImageBatchPolicy::Auto
                || self.maximum_batch_size == 1
            {
                0
            } else {
                200
            },
            default_width: 512,
            default_height: 512,
            default_seed: 0,
            rgba_scratch_bytes: 16 * 1024,
            output: ferrum_types::OutputConfig {
                max_sessions,
                ..Default::default()
            },
        };
        serving.validate()?;
        Ok(serving)
    }
    pub fn host(&self) -> HostConfig {
        HostConfig {
            total_bytes: 1024 * MIB as usize,
            input_preprocess_bytes: 768 * MIB as usize,
            transfer_staging_bytes: if self.backend == ImageBackendKind::Cuda {
                16 * MIB as usize
            } else {
                0
            },
            output_bytes: 256 * MIB as usize,
            weight_backing_bytes: if self.precision.uses_fp8() {
                128 * MIB as usize
            } else {
                0
            },
        }
    }
    pub fn pipeline(&self) -> PipelineConfig {
        let maximum_dimension = match self.backend {
            ImageBackendKind::Cuda => 2048,
            ImageBackendKind::Metal => 1024,
        };
        PipelineConfig {
            maximum_user_tokens: 1024,
            maximum_width: maximum_dimension,
            maximum_height: maximum_dimension,
            maximum_batch_size: self.maximum_batch_size,
            decoder_batch_policy:
                ferrum_engine::image_generation::ImageDecoderBatchPolicy::SingleImage,
            edit_prefix_policy: match self.backend {
                ImageBackendKind::Cuda => {
                    ferrum_engine::image_generation::ImageEditPrefixPolicy::RequestCache
                }
                ImageBackendKind::Metal => {
                    ferrum_engine::image_generation::ImageEditPrefixPolicy::Recompute
                }
            },
            plan_cache: match self.backend {
                ImageBackendKind::Cuda => {
                    ferrum_engine::image_generation::ImagePlanCachePolicy::MultiShape {
                        maximum_entries_per_stage: 16,
                        maximum_encoded_metadata_bytes: 128 * MIB as usize,
                    }
                }
                ImageBackendKind::Metal => {
                    ferrum_engine::image_generation::ImagePlanCachePolicy::LastPerStage
                }
            },
            residency: ferrum_engine::image_generation::ImageResidencyPolicy::Adaptive,
            dynamic_residency:
                ferrum_engine::image_generation::ImageDynamicResidencyPolicy::Adaptive,
            maximum_staging_bytes: 16 * MIB,
            maximum_initialization_commands: 64,
            maximum_backing_maintenance_passes: 8,
        }
    }
    #[cfg(feature = "cuda")]
    pub fn cuda(&self) -> Result<ferrum_engine::image_generation::cuda::QwenImage21CudaConfig> {
        use ferrum_engine::image_generation::cuda::{ImageCudaMemoryPolicy, QwenImage21CudaConfig};
        self.validate()?;
        ensure!(
            self.backend == ImageBackendKind::Cuda,
            "image backend is not CUDA"
        );
        Ok(QwenImage21CudaConfig {
            ordinal: self.ordinal,
            memory: ImageCudaMemoryPolicy::AutoAvailable {
                utilization: ferrum_kernels::backend::cuda::CudaMemoryUtilization::new(
                    self.utilization,
                )
                .map_err(anyhow::Error::msg)?,
                maximum_bytes: self.maximum_memory_bytes,
            },
            bootstrap_memory_bytes: 1024 * MIB,
            stream_memory_bytes: 64 * MIB,
            maximum_streams: ferrum_engine::image_generation::ImageStreamLimit::Auto,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 2048,
            opaque_memory_profile: "experimental-unqualified-runtime-v1".into(),
            attention_policy:
                ferrum_engine::image_generation::ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1,
            precision_policy: self.precision,
            fp8_host_cache: ferrum_engine::image_generation::ImageFp8HostCachePolicy::Off,
            convolution_workspace:
                ferrum_engine::image_generation::ImageCudaConvolutionWorkspacePolicy::Auto,
            cudnn: ferrum_kernels::backend::cuda::CudaCudnnConfig {
                library_path: super::library::resolve(self.cudnn_library.as_deref())?,
                dependency_libraries: Vec::new(),
                opaque_memory_bytes: 64 * MIB,
                workspace_limit_bytes: 16 * MIB,
                opaque_memory_profile: "experimental-unqualified-cudnn-v1".into(),
            },
            host: self.host().limits()?,
            pipeline: self.pipeline().resolved_limits()?,
        })
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn metal(&self) -> Result<ferrum_engine::image_generation::metal::QwenImage21MetalConfig> {
        use super::configuration::{MetalConfig, MetalMemoryConfig};
        self.validate()?;
        ensure!(
            self.backend == ImageBackendKind::Metal,
            "image backend is not Metal"
        );
        MetalConfig {
            memory: MetalMemoryConfig::AutoRecommended {
                utilization: self.utilization,
                maximum_bytes: self.maximum_memory_bytes,
            },
            precision_policy: self.precision,
            attention_policy: ferrum_engine::image_generation::ImageMetalAttentionPolicy::Reference,
            bootstrap_memory_bytes: 128 * MIB,
            stream_memory_bytes: 2 * MIB,
            maximum_streams: ferrum_engine::image_generation::ImageStreamLimit::Auto,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 2048,
            opaque_memory_profile: "metal.image.declared-runtime-v1".into(),
            host: self.host(),
            pipeline: self.pipeline(),
        }
        .resolved()
    }
}

/// No model-name heuristic and no weight reads. A local pipeline declaration
/// commits to pipeline preflight: malformed or incomplete packages never fall
/// through to the text loader. A normal text directory keeps its original path.
pub fn discover(model: Option<&str>) -> Result<Option<PathBuf>> {
    let Some(model) = model else {
        return Ok(None);
    };
    let path = Path::new(model);
    if !path.is_dir() {
        return Ok(None);
    }
    match std::fs::symlink_metadata(path.join("model_index.json")) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("inspect local pipeline declaration"),
        Ok(_) => {}
    }
    let bundle = PipelineSourceBundle::open(path, PipelineSourceLimits::default())
        .context("local pipeline package metadata preflight")?;
    match bundle.declaration().architecture() {
        PipelineArchitecture::QwenImage21 => {}
    }
    Ok(Some(bundle.root().to_owned()))
}

#[derive(Debug, Clone)]
pub enum ImageProductSource {
    ExplicitConfig(PathBuf),
    LocalDirectory {
        path: PathBuf,
        options: DirectoryOptions,
    },
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) struct PreparedProduct {
    pub model: String,
    pub profile: &'static str,
    pub serving: Option<ServingConfig>,
    pub backend: super::factory::PreparedBackend,
    pub assets: ferrum_engine::image_generation::QwenImage21Assets,
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
impl ImageProductSource {
    pub(super) fn prepare_for_run(&self) -> Result<PreparedProduct> {
        match self {
            Self::LocalDirectory { path, options } => {
                // `run` has one request. Do not size initialization or resources
                // for the serving ceiling even when the user supplied one.
                Self::LocalDirectory {
                    path: path.clone(),
                    options: options.single_request(),
                }
                .prepare()
            }
            Self::ExplicitConfig(_) => self.prepare(),
        }
    }

    pub(super) fn prepare(&self) -> Result<PreparedProduct> {
        match self {
            Self::ExplicitConfig(path) => {
                let config = super::ImageProductConfig::load(path)?;
                let backend = super::factory::PreparedBackend::from_config(&config.backend)?;
                let assets = config.prepare_assets()?;
                Ok(PreparedProduct {
                    model: config.model,
                    profile: config.execution_profile.as_str(),
                    serving: config.serving,
                    backend,
                    assets,
                })
            }
            Self::LocalDirectory { path, options } => {
                options.validate()?;
                super::factory::PreparedBackend::ensure_supported(options.backend)?;
                // Complete metadata and actual archive headers precede any
                // device context, provider, or native library construction.
                let assets = ferrum_engine::image_generation::QwenImage21Assets::from_directory(
                    path,
                    PipelineSourceLimits::default(),
                )?;
                let backend = super::factory::PreparedBackend::from_directory(options, &assets)?;
                let model = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .context("model directory needs a UTF-8 public name")?
                    .to_owned();
                ensure!(!model.is_empty() && model.len() <= 256 && !model.chars().any(char::is_control), "invalid model directory public name; rename it or use explicit image configuration");
                Ok(PreparedProduct {
                    model,
                    profile: options.preset_name(),
                    serving: Some(options.serving_with_output_bytes(backend.host_output_bytes())?),
                    backend,
                    assets,
                })
            }
        }
    }
}
