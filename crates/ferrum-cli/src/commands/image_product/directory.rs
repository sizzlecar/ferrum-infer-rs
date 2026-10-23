//! A named, bounded cold-execution preset for standard local pipeline packages.
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

pub const PRESET_NAME: &str = "experimental_bf16_cold_v1";
const MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone)]
pub struct DirectoryOptions {
    pub utilization: f64,
    pub maximum_memory_bytes: Option<NonZeroU64>,
    pub maximum_batch_size: usize,
    pub ordinal: usize,
    pub cudnn_library: Option<PathBuf>,
    pub precision: ferrum_engine::image_generation::ImageCudaPrecisionPolicy,
}
impl DirectoryOptions {
    pub fn from_cli(
        utilization: f32,
        maximum: Option<std::num::NonZeroUsize>,
        batch: Option<usize>,
        backend: &str,
        gpu_devices: Option<&str>,
        cudnn_library: Option<PathBuf>,
    ) -> Result<Self> {
        ensure!(
            matches!(backend, "auto" | "cuda"),
            "image generation currently supports --backend auto or cuda"
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
            utilization: f64::from(utilization),
            maximum_memory_bytes: maximum
                .map(|v| u64::try_from(v.get()).map(|n| NonZeroU64::new(n).unwrap()))
                .transpose()?,
            maximum_batch_size: batch.unwrap_or(1),
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
    ) -> Self {
        self.precision = precision;
        self
    }
    pub const fn preset_name(&self) -> &'static str {
        match self.precision {
            ferrum_engine::image_generation::ImageCudaPrecisionPolicy::Bf16 => PRESET_NAME,
            ferrum_engine::image_generation::ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1 => {
                "experimental_fp8_linear_cold_v1"
            }
        }
    }
    pub fn validate(&self) -> Result<()> {
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
        self.validate()?;
        let pending = self.maximum_batch_size.max(4);
        let serving = ServingConfig {
            maximum_pending: pending,
            maximum_prompt_bytes: (1024 * 1024 / self.maximum_batch_size).min(65536),
            max_num_seqs: self.maximum_batch_size,
            request_batch_max_wait_ms: if self.maximum_batch_size == 1 { 0 } else { 200 },
            default_width: 512,
            default_height: 512,
            default_seed: 0,
            rgba_scratch_bytes: 16 * 1024,
            output: ferrum_types::OutputConfig {
                max_sessions: pending + self.maximum_batch_size,
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
            transfer_staging_bytes: 16 * MIB as usize,
            output_bytes: 256 * MIB as usize,
            weight_backing_bytes: match self.precision {
                ferrum_engine::image_generation::ImageCudaPrecisionPolicy::Bf16 => 0,
                ferrum_engine::image_generation::ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1 => 128 * MIB as usize,
            },
        }
    }
    pub fn pipeline(&self) -> PipelineConfig {
        PipelineConfig {
            maximum_user_tokens: 1024,
            maximum_width: 1024,
            maximum_height: 1024,
            maximum_batch_size: self.maximum_batch_size,
            decoder_batch_policy:
                ferrum_engine::image_generation::ImageDecoderBatchPolicy::SingleImage,
            edit_prefix_policy: ferrum_engine::image_generation::ImageEditPrefixPolicy::default(),
            plan_cache: ferrum_engine::image_generation::ImagePlanCachePolicy::LastPerStage,
            residency: ferrum_engine::image_generation::ImageResidencyPolicy::Cold,
            dynamic_residency:
                ferrum_engine::image_generation::ImageDynamicResidencyPolicy::Release,
            maximum_staging_bytes: 16 * MIB,
            maximum_initialization_commands: 64,
            maximum_backing_maintenance_passes: 8,
        }
    }
    #[cfg(feature = "cuda")]
    pub fn cuda(&self) -> Result<ferrum_engine::image_generation::cuda::QwenImage21CudaConfig> {
        use ferrum_engine::image_generation::cuda::{ImageCudaMemoryPolicy, QwenImage21CudaConfig};
        self.validate()?;
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
            attention_policy: ferrum_engine::image_generation::ImageCudaAttentionPolicy::Sequential,
            precision_policy: self.precision,
            fp8_host_cache: ferrum_engine::image_generation::ImageFp8HostCachePolicy::Off,
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

#[cfg(feature = "cuda")]
pub(super) struct PreparedProduct {
    pub model: String,
    pub profile: &'static str,
    pub serving: Option<ServingConfig>,
    pub cuda: ferrum_engine::image_generation::cuda::QwenImage21CudaConfig,
    pub assets: ferrum_engine::image_generation::QwenImage21Assets,
}

#[cfg(feature = "cuda")]
impl ImageProductSource {
    pub(super) fn prepare(&self) -> Result<PreparedProduct> {
        match self {
            Self::ExplicitConfig(path) => {
                let config = super::ImageProductConfig::load(path)?;
                let super::configuration::ImageBackendConfig::Cuda(cuda) = &config.backend;
                let mut cuda = cuda.resolved()?;
                cuda.cudnn.library_path = super::library::resolve(Some(&cuda.cudnn.library_path))?;
                cuda.validate()?;
                let assets = config.prepare_assets()?;
                Ok(PreparedProduct {
                    model: config.model,
                    profile: config.execution_profile.as_str(),
                    serving: config.serving,
                    cuda,
                    assets,
                })
            }
            Self::LocalDirectory { path, options } => {
                options.validate()?;
                // Complete metadata and actual archive headers precede any
                // device context, provider, or native library construction.
                let assets = ferrum_engine::image_generation::QwenImage21Assets::from_directory(
                    path,
                    PipelineSourceLimits::default(),
                )?;
                let cuda = options.cuda()?;
                cuda.validate()?;
                let model = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .context("model directory needs a UTF-8 public name")?
                    .to_owned();
                ensure!(!model.is_empty() && model.len() <= 256 && !model.chars().any(char::is_control), "invalid model directory public name; rename it or use explicit image configuration");
                Ok(PreparedProduct {
                    model,
                    profile: options.preset_name(),
                    serving: Some(options.serving()?),
                    cuda,
                    assets,
                })
            }
        }
    }
}
